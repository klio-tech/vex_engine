"""Coexistence guard tests: verification vs. Klio brain non-contamination.

Vex's existing org-scoped *verification* memory and the additive *Klio brain*
share one physical table (``session_memories``). The safety contract this suite
pins is that the two read paths are mutually non-contaminating:

* Verification reads via :func:`shared.memory.retrieve_memories`, which is
  **session-scoped** (``WHERE session_id = :session_id AND org_id = :org_id``)
  and was NOT altered by the brain work. It can only ever return rows for the
  exact ``session_id`` it was asked for.
* Brain reads via :func:`shared.memory.retrieve_memories_scoped`, which spans
  sessions but filters on the ``scope`` *column* (``scope = 'org'`` for the
  default), which EXCLUDES verification's ``scope='session'`` rows even though
  they live in the same table.

These tests are mock-based (mirroring ``tests/test_memory.py``): a
``MagicMock`` db_session captures the executed statements, ``embed_text`` is
patched so no embedding model / LLM is called, and the rendered SQL is inspected
via ``str(call.args[0])`` with bind params via ``call.args[1]``. We assert on
the SQL *predicates* rather than DB results so the guard holds without infra and
catches a regression the moment the query text drifts.

The brain session-id-namespace guard (deliverable 1d) lives in the mcp-server
suite (``services/mcp-server/tests/test_coexistence.py``) because the helpers it
covers (``app.session.brain_session_id`` / ``app.curator``) resolve cleanly only
under the mcp-server import root; importing them here would require sys.path
gymnastics. See that file for the namespace assertions.
"""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.memory import retrieve_memories, retrieve_memories_scoped

# A short deterministic vector; its exact value is irrelevant because we never
# touch a real DB — we only inspect the SQL the functions render.
SAMPLE_EMBEDDING = [0.1, 0.2, 0.3, 0.4, 0.5]


def _executed_statements(db_session: MagicMock) -> list[tuple[str, dict]]:
    """Return [(rendered_sql, bind_params), ...] for every execute() call.

    ``str(call.args[0])`` renders the SQLAlchemy ``text()`` clause back to its
    SQL string; ``call.args[1]`` is the bound-parameter dict. This mirrors the
    inspection style used in ``tests/test_memory.py``.
    """
    statements: list[tuple[str, dict]] = []
    for call in db_session.execute.call_args_list:
        sql = str(call.args[0])
        params = call.args[1] if len(call.args) > 1 else {}
        statements.append((sql, params))
    return statements


def _row_statements(db_session: MagicMock) -> list[tuple[str, dict]]:
    """The statements that can actually return a memory.

    Recall also issues a one-line ``websearch_to_tsquery`` probe that reads no
    table, to decide whether the lexical arm has anything to search for. The
    counts below are about how many ways rows can reach the caller, so the
    probe is filtered out by the only property that matters here: it does not
    name the table.
    """
    return [
        (sql, params)
        for sql, params in _executed_statements(db_session)
        if "session_memories" in sql
    ]


# ---------------------------------------------------------------------------
# (a) retrieve_memories is still session-scoped (regression pin)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_memories_still_session_scoped():
    """Prevents brain→verification contamination.

    Pins that the brain work did NOT turn the verification read path into a
    cross-session/brain query. Every statement ``retrieve_memories`` executes
    must filter ``session_id = :session_id AND org_id = :org_id`` and must NOT
    contain the brain ``scope = 'org'`` predicate. If this fails, verification
    could surface brain (or other-session) rows it must never see.
    """
    db_session = MagicMock()
    db_session.execute = MagicMock(
        side_effect=[
            MagicMock(fetchall=MagicMock(return_value=[])),
            MagicMock(fetchall=MagicMock(return_value=[])),
        ]
    )

    with patch(
        "shared.memory.embed_text",
        new_callable=AsyncMock,
        return_value=SAMPLE_EMBEDDING,
    ):
        await retrieve_memories(
            db_session=db_session,
            session_id="real-conversation-id",
            org_id="org-1",
            query_text="what happened?",
        )

    statements = _executed_statements(db_session)
    assert statements, "retrieve_memories must execute at least one query"

    for sql, params in statements:
        # Session scoping is present on every query.
        assert "session_id = :session_id" in sql, (
            f"verification query lost session scoping:\n{sql}"
        )
        assert "org_id = :org_id" in sql, f"verification query lost org scoping:\n{sql}"
        # It must NOT have been turned into a brain (org-scope) query.
        assert "scope = 'org'" not in sql, (
            "verification query was accidentally turned into a brain "
            f"org-scoped query (would leak brain rows):\n{sql}"
        )
        # The exact session/org it was asked for is what gets bound.
        assert params.get("session_id") == "real-conversation-id"
        assert params.get("org_id") == "org-1"


def test_retrieve_memories_signature_unchanged():
    """Pins the verification read API: ``(db_session, session_id, org_id, ...)``.

    A signature change (e.g. dropping ``session_id`` or reordering params) would
    silently let callers widen the scope. The first three positional params
    must remain exactly as the verification path depends on them.
    """
    params = list(inspect.signature(retrieve_memories).parameters)
    assert params[:3] == ["db_session", "session_id", "org_id"], (
        f"retrieve_memories signature drifted: {params}"
    )


# ---------------------------------------------------------------------------
# (b) retrieve_memories_scoped(scope='org') excludes verification rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_memories_scoped_org_excludes_session_rows():
    """Prevents verification→brain contamination (the prior-bug guard).

    The default org-scope brain read must filter on the ``scope`` *column*
    (``scope = 'org'``) so verification's ``scope='session'`` rows — which live
    in the same table — are excluded. A prior bug left ``scope`` unfiltered,
    leaking session rows into brain recall; this asserts that filter is back.
    """
    db_session = MagicMock()
    db_session.execute = MagicMock(
        return_value=MagicMock(fetchall=MagicMock(return_value=[]))
    )

    with patch(
        "shared.memory.embed_text",
        new_callable=AsyncMock,
        return_value=SAMPLE_EMBEDDING,
    ):
        await retrieve_memories_scoped(
            db_session=db_session,
            org_id="org-1",
            query_text="recall org facts",
            scope="org",
        )

    statements = _row_statements(db_session)
    assert len(statements) == 1, "org-scope recall should execute exactly one query"
    sql, params = statements[0]

    assert "scope = 'org'" in sql, (
        "brain org-scope recall lost its scope filter — verification "
        f"scope='session' rows would leak into brain recall:\n{sql}"
    )
    # Tenancy is still enforced.
    assert "org_id = :org_id" in sql
    assert params.get("org_id") == "org-1"
    # agent_id must not be bound on a pure org query.
    assert "agent_id" not in params


# ---------------------------------------------------------------------------
# (c) retrieve_memories_scoped(scope='agent') visibility ladder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_memories_scoped_agent_ladder():
    """Prevents agent→agent contamination (cross-private leakage).

    An agent-scope recall must admit org-shared rows PLUS the calling agent's
    own private rows, and NOTHING from another agent. The predicate
    ``scope = 'org' OR (scope = 'agent' AND agent_id = :agent_id)`` encodes
    exactly that ladder; we pin the predicate and the bound agent_id.
    """
    db_session = MagicMock()
    db_session.execute = MagicMock(
        return_value=MagicMock(fetchall=MagicMock(return_value=[]))
    )

    with patch(
        "shared.memory.embed_text",
        new_callable=AsyncMock,
        return_value=SAMPLE_EMBEDDING,
    ):
        await retrieve_memories_scoped(
            db_session=db_session,
            org_id="org-1",
            query_text="recall my notes",
            scope="agent",
            agent_id="agent-7",
        )

    statements = _row_statements(db_session)
    assert len(statements) == 1
    sql, params = statements[0]

    assert "scope = 'org' OR (scope = 'agent' AND agent_id = :agent_id)" in sql, (
        "agent-scope recall lost the visibility-ladder predicate — an agent "
        f"could see another agent's private rows or none of the org's:\n{sql}"
    )
    assert params.get("agent_id") == "agent-7"
    assert params.get("org_id") == "org-1"
    # Verification session rows are still excluded: the predicate only admits
    # scope IN ('org','agent'), never scope='session'.
    assert "scope = 'session'" not in sql
