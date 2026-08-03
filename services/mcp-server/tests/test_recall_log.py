"""Tests for the best-effort recall-event logger (``app.recall_log``).

Mock-based, mirroring ``tests/test_capture.py``: no live Postgres.  The single
``log_recall_event`` helper takes the DB session as a plain argument, so a bare
``unittest.mock.MagicMock`` stands in for the session — we assert on the
``execute``/``commit`` calls and the bound INSERT params, never touching a real
table.  The contract under test is twofold: (1) a happy-path INSERT + commit,
and (2) best-effort swallowing — a failing ``execute`` must NOT propagate, so a
logging failure can never break the recall that triggered it.
"""

import os
import uuid
from unittest.mock import MagicMock

import pytest
from app.recall_log import log_recall_event
from sqlalchemy import create_engine, text


def test_log_recall_event_inserts_row():
    """Happy path: one INSERT with the expected params, then a single commit."""
    db = MagicMock()

    log_recall_event(
        db,
        org_id="test-org",
        agent_id="agent-1",
        project_id="proj-9",
        query_text="auth flow",
        result_count=3,
        source="mcp",
    )

    db.execute.assert_called_once()
    # Bound params are the second positional arg to ``Session.execute``.
    params = db.execute.call_args.args[1]
    assert params["org_id"] == "test-org"
    assert params["agent_id"] == "agent-1"
    assert params["project_id"] == "proj-9"
    assert params["query_text"] == "auth flow"
    assert params["result_count"] == 3
    assert params["source"] == "mcp"
    assert params["id"]  # an app-minted UUID is present
    db.commit.assert_called_once()


def test_log_recall_event_swallows_errors():
    """A failing execute is caught and swallowed: rolls back, never raises, never commits."""
    db = MagicMock()
    db.execute.side_effect = RuntimeError("boom")

    # Must not raise — logging a recall can never break the recall.
    log_recall_event(
        db,
        org_id="test-org",
        agent_id=None,
        project_id=None,
        query_text=None,
        result_count=0,
        source="hook",
    )

    db.rollback.assert_called_once()
    db.commit.assert_not_called()


# ---------------------------------------------------------------------------
# Recall-log query redaction (Task 3)
# ---------------------------------------------------------------------------


def test_log_recall_event_redacts_query_text():
    """query_text containing a secret must be redacted before the INSERT params.

    The raw Stripe test key must never appear in the params dict passed to
    db.execute — it must be replaced by the [REDACTED:…] placeholder.
    """

    class _Rec:
        def __init__(self):
            self.params = None

        def execute(self, sql, params=None):
            self.params = params

        def commit(self):
            pass

    db = _Rec()
    log_recall_event(
        db,
        org_id="org-1",
        agent_id="a",
        project_id=None,
        query_text=(
            "find sk_test_51TcfsTP4cXbUl5jnQJG2Ou9BUcBChSVqEDNj7d65uQjZXtAVzI1XR2"
            "eewh5KDq8GmCfrr1a1xRNcSdB1c7TtWrU00GsK8y8Je"
        ),
        result_count=0,
        source="mcp",
    )
    assert "sk_test_51Tcfs" not in (db.params or {}).get("query_text", "")


# ---------------------------------------------------------------------------
# The composition the mocks cannot see: a poisoned session (real Postgres)
# ---------------------------------------------------------------------------
#
# ``log_recall_event`` swallows its own failures by design, so the ONLY way to
# observe it failing is to count rows. A DB error inside recall's graph
# expansion used to abort the transaction and leave the session in
# ``InFailedSqlTransaction``; recall returned its rows, the logger's INSERT
# then failed, the logger swallowed it, and the run wrote zero audit rows —
# silently, with billing riding on that table.

_DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",
)


def _db_reachable() -> bool:
    try:
        engine = create_engine(_DB_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _db_reachable(), reason="Postgres not reachable")
def test_the_audit_row_lands_even_when_graph_expansion_hit_a_db_error():
    from unittest.mock import AsyncMock, patch

    import anyio
    from shared import graph_store
    from shared.memory import retrieve_memories_scoped

    org = f"org_{uuid.uuid4().hex[:8]}"
    engine = create_engine(_DB_URL)
    conn = engine.connect()
    try:
        # Stage A must return something, or expansion is skipped entirely and
        # this test proves nothing.
        conn.execute(
            text(
                """
                INSERT INTO session_memories (
                    id, session_id, org_id, agent_id, memory_type, content,
                    embedding, scope, status, metadata, created_at, updated_at
                ) VALUES (
                    :id, 'sess-1', :org, 'agent-1', 'fact', 'an entry row',
                    :embedding, 'org', 'active', '{}'::jsonb, NOW(), NOW()
                )
                """
            ),
            {
                "id": str(uuid.uuid4()),
                "org": org,
                "embedding": "[" + ",".join(["0.1"] * 1024) + "]",
            },
        )
        conn.commit()

        def _bad_sql(db_session, **kwargs):
            db_session.execute(text("SELECT * FROM a_table_that_does_not_exist"))

        with (
            patch.object(graph_store, "neighbors", _bad_sql),
            patch(
                "shared.memory.embed_text",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
            ),
        ):
            found = anyio.run(
                lambda: retrieve_memories_scoped(conn, org, "query", limit=5),
            )
        assert len(found) == 1, "control: stage A returned a row to expand from"

        log_recall_event(
            conn,
            org_id=org,
            agent_id="agent-1",
            project_id=None,
            query_text="query",
            result_count=len(found),
            source="mcp",
        )

        written = conn.execute(
            text("SELECT count(*) FROM brain_recall_events WHERE org_id = :o"),
            {"o": org},
        ).scalar()
        assert written == 1, (
            "the recall audit event was silently dropped: graph expansion left "
            "the session in a failed transaction"
        )
    finally:
        conn.rollback()
        conn.execute(
            text("DELETE FROM brain_recall_events WHERE org_id = :o"), {"o": org}
        )
        conn.execute(text("DELETE FROM session_memories WHERE org_id = :o"), {"o": org})
        conn.commit()
        conn.close()
        engine.dispose()
