"""Tests for ``retrieve_memories_scoped`` (Klio brain recall).

Brain recall reads across sessions at org / agent / space / project scope. This
is DB-logic exercising pgvector cosine search, so the tests run against real
Postgres (mirroring ``tests/test_brain.py`` and
``services/migrations/tests/test_migration_020.py``) while MOCKING
``embed_text`` so retrieval ranking is deterministic and no LLM is called.

Rows are seeded directly via SQL with a fixed 1024-dim vector literal so cosine
ordering is controlled without a real embedding model. Each test uses a unique
``org_id`` for isolation on the shared dev DB, and the fixture deletes every
seeded ``session_memories`` row on teardown.
"""

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

from shared.memory import (
    MemoryRecord,
    retrieve_memories,
    retrieve_memories_scoped,
)

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",
)


def _db_reachable() -> bool:
    """True if the test Postgres is reachable. Brain recall exercises real
    pgvector cosine search that cannot be mocked, so when no DB is available
    (e.g. CI runs the unit-test matrix without Postgres) the whole module is
    skipped rather than erroring."""
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_reachable(),
    reason="Postgres not reachable; brain recall DB tests require the live test DB",
)

EMBED_DIM = 1024

# A single deterministic 1024-dim embedding shared by the query and every seeded
# row. Because all rows have the same vector, cosine distance to the query is 0
# for all of them, so they all rank equally — the tests then assert which rows
# the WHERE/scope/filter logic admits, not relative ranking.
FIXED_EMBEDDING = [0.1] * EMBED_DIM
FIXED_EMBEDDING_LITERAL = "[" + ",".join(str(x) for x in FIXED_EMBEDDING) + "]"


def _unique_org() -> str:
    """Generate a unique org id so tests don't collide on the shared dev DB."""
    return f"org_{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def db_session():
    """Yield a real SQLAlchemy Connection and clean up seeded rows afterward.

    A ``Connection`` in SQLAlchemy 2.x exposes ``.execute()`` and ``.commit()``,
    matching the synchronous interface used by ``shared.memory``. Tracked org
    ids have their ``session_memories`` rows deleted on teardown.
    """
    engine = create_engine(DB_URL)
    conn = engine.connect()
    created_org_ids: set[str] = set()
    conn.created_org_ids = created_org_ids  # type: ignore[attr-defined]
    try:
        yield conn
    finally:
        try:
            for org_id in created_org_ids:
                conn.execute(
                    text("DELETE FROM session_memories WHERE org_id = :o"),
                    {"o": org_id},
                )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _track(db_session, org_id: str) -> str:
    """Register an org id for teardown cleanup and return it."""
    db_session.created_org_ids.add(org_id)
    return org_id


def _seed_memory(
    db_session,
    *,
    org_id: str,
    content: str,
    scope: str = "org",
    agent_id: str = "agent-1",
    session_id: str | None = None,
    space_id: str | None = None,
    project_id: str | None = None,
    status: str = "active",
    superseded_by: str | None = None,
    memory_type: str = "fact",
) -> str:
    """Insert one ``session_memories`` row with the fixed embedding literal.

    ``scope`` is written explicitly to the ``scope`` column so tests can seed a
    MIX of ``'session'`` (verification), ``'org'`` (shared brain), and
    ``'agent'`` (agent-private) rows in the same org and prove that
    ``retrieve_memories_scoped`` admits/excludes the right rows. Defaults to
    ``'org'`` so brain-recall seeds pass the new scope predicate by default;
    pass ``scope='session'`` to simulate a verification row.

    Returns the new row id. Uses the same vector for every row so cosine
    ordering is neutral and the scope/filter clauses are what's under test.
    """
    memory_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    db_session.execute(
        text("""
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, execution_id,
                memory_type, content, embedding, scope, status,
                superseded_by, space_id, project_id, metadata,
                created_at, updated_at
            ) VALUES (
                :id, :session_id, :org_id, :agent_id, :execution_id,
                :memory_type, :content, :embedding, :scope, :status,
                :superseded_by, :space_id, :project_id, '{}'::jsonb,
                :created_at, :updated_at
            )
        """),
        {
            "id": memory_id,
            "session_id": session_id or f"sess-{uuid.uuid4().hex[:6]}",
            "org_id": org_id,
            "agent_id": agent_id,
            "execution_id": f"exec-{uuid.uuid4().hex[:6]}",
            "memory_type": memory_type,
            "content": content,
            "embedding": FIXED_EMBEDDING_LITERAL,
            "scope": scope,
            "status": status,
            "superseded_by": superseded_by,
            "space_id": space_id,
            "project_id": project_id,
            "created_at": now,
            "updated_at": now,
        },
    )
    db_session.commit()
    return memory_id


def _mock_embed():
    """Patch ``shared.memory.embed_text`` to return the fixed query vector."""
    return patch(
        "shared.memory.embed_text",
        new_callable=AsyncMock,
        return_value=FIXED_EMBEDDING,
    )


# ---------------------------------------------------------------------------
# Org isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scoped_recall_org_isolation(db_session):
    """Org A's recall never returns org B's memory, even with identical content."""
    org_a = _track(db_session, _unique_org())
    org_b = _track(db_session, _unique_org())

    _seed_memory(db_session, org_id=org_a, content="shared secret content")
    _seed_memory(db_session, org_id=org_b, content="shared secret content")

    with _mock_embed():
        results = await retrieve_memories_scoped(
            db_session, org_a, "shared secret content", limit=10
        )

    assert len(results) == 1
    assert all(r.org_id == org_a for r in results)
    assert all(isinstance(r, MemoryRecord) for r in results)


@pytest.mark.asyncio
async def test_scoped_recall_org_spans_agents(db_session):
    """scope='org' returns org-scope memories from multiple agents within the org."""
    org_id = _track(db_session, _unique_org())

    _seed_memory(
        db_session, org_id=org_id, scope="org", agent_id="agent-A", content="fact A"
    )
    _seed_memory(
        db_session, org_id=org_id, scope="org", agent_id="agent-B", content="fact B"
    )

    with _mock_embed():
        results = await retrieve_memories_scoped(
            db_session, org_id, "any query", scope="org", limit=10
        )

    agent_ids = {r.agent_id for r in results}
    assert agent_ids == {"agent-A", "agent-B"}


# ---------------------------------------------------------------------------
# Scope-column isolation (the visibility ladder)
#
# These are the regression tests for the BLOCKER: verification rows live in the
# same table with scope='session'; the brain must filter on the scope column,
# not just on a Python switch.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_org_recall_excludes_verification_session_rows(db_session):
    """scope='org' recall returns the brain (org) row, NOT verification's
    scope='session' row living in the same org/table.

    REGRESSION TEST for the BLOCKER: without a scope-column predicate, an org
    recall returns every active row in the org — including verification's
    scope='session' rows — contaminating the brain.
    """
    org_id = _track(db_session, _unique_org())

    _seed_memory(
        db_session, org_id=org_id, scope="session", content="verification fact"
    )
    _seed_memory(db_session, org_id=org_id, scope="org", content="brain fact")

    with _mock_embed():
        results = await retrieve_memories_scoped(
            db_session, org_id, "any query", scope="org", limit=10
        )

    contents = {r.content for r in results}
    assert contents == {"brain fact"}
    assert "verification fact" not in contents
    assert all(r.scope == "org" for r in results)


@pytest.mark.asyncio
async def test_org_recall_excludes_agent_private_rows(db_session):
    """scope='org' recall returns only org-scope rows, NOT agent-private rows."""
    org_id = _track(db_session, _unique_org())

    _seed_memory(db_session, org_id=org_id, scope="org", content="org fact")
    _seed_memory(
        db_session,
        org_id=org_id,
        scope="agent",
        agent_id="agent-A",
        content="agentA private",
    )

    with _mock_embed():
        results = await retrieve_memories_scoped(
            db_session, org_id, "any query", scope="org", limit=10
        )

    contents = {r.content for r in results}
    assert contents == {"org fact"}
    assert "agentA private" not in contents
    assert all(r.scope == "org" for r in results)


# ---------------------------------------------------------------------------
# Agent narrowing (org + own agent-private, never another agent's private)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_recall_returns_org_plus_own_agent(db_session):
    """scope='agent' returns the team's org rows PLUS the agent's own private
    rows, but NOT another agent's private rows."""
    org_id = _track(db_session, _unique_org())

    _seed_memory(db_session, org_id=org_id, scope="org", content="org fact")
    _seed_memory(
        db_session,
        org_id=org_id,
        scope="agent",
        agent_id="agent-A",
        content="agentA private",
    )
    _seed_memory(
        db_session,
        org_id=org_id,
        scope="agent",
        agent_id="agent-B",
        content="agentB private",
    )

    with _mock_embed():
        results = await retrieve_memories_scoped(
            db_session,
            org_id,
            "any query",
            scope="agent",
            agent_id="agent-A",
            limit=10,
        )

    contents = {r.content for r in results}
    assert contents == {"org fact", "agentA private"}
    assert "agentB private" not in contents


@pytest.mark.asyncio
async def test_agent_scope_requires_agent_id(db_session):
    """scope='agent' with agent_id=None raises ValueError before any DB/embed work."""
    org_id = _track(db_session, _unique_org())

    with pytest.raises(ValueError):
        await retrieve_memories_scoped(
            db_session, org_id, "any query", scope="agent", agent_id=None
        )


# ---------------------------------------------------------------------------
# Unsupported / unknown scope values raise (no silent org-wide read)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_scope_raises(db_session):
    """scope='session' is unsupported here and raises (use retrieve_memories)."""
    org_id = _track(db_session, _unique_org())

    with pytest.raises(ValueError):
        await retrieve_memories_scoped(db_session, org_id, "any query", scope="session")


@pytest.mark.asyncio
async def test_unknown_scope_raises(db_session):
    """An unknown scope value raises ValueError."""
    org_id = _track(db_session, _unique_org())

    with pytest.raises(ValueError):
        await retrieve_memories_scoped(db_session, org_id, "any query", scope="bogus")


# ---------------------------------------------------------------------------
# Project filter (NULL-surfacing)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scoped_recall_project_filter_surfaces_null(db_session):
    """project_id=X returns project-X rows AND project_id-NULL rows, not Y."""
    org_id = _track(db_session, _unique_org())

    # Create two real project rows so the FK on session_memories is satisfied.
    project_x = str(uuid.uuid4())
    project_y = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    for pid, name in ((project_x, "Project X"), (project_y, "Project Y")):
        db_session.execute(
            text("""
                INSERT INTO projects (
                    id, org_id, display_name, repo_root_path,
                    created_at, last_seen_at
                ) VALUES (
                    :id, :org_id, :name, :path, :now, :now
                )
            """),
            {
                "id": pid,
                "org_id": org_id,
                "name": name,
                "path": f"/repo/{name}",
                "now": now,
            },
        )
    db_session.commit()

    # Seed as scope='org' so rows pass the new scope-column predicate; the
    # project filter (NULL-surfacing) is what's under test here.
    _seed_memory(
        db_session,
        org_id=org_id,
        scope="org",
        content="x-tagged",
        project_id=project_x,
    )
    _seed_memory(
        db_session,
        org_id=org_id,
        scope="org",
        content="y-tagged",
        project_id=project_y,
    )
    _seed_memory(
        db_session, org_id=org_id, scope="org", content="unfiled", project_id=None
    )

    try:
        with _mock_embed():
            results = await retrieve_memories_scoped(
                db_session, org_id, "any query", project_id=project_x, limit=10
            )

        contents = {r.content for r in results}
        assert contents == {"x-tagged", "unfiled"}
        assert "y-tagged" not in contents

        # Returned records carry the project_id field through _row_to_record.
        by_content = {r.content: r for r in results}
        assert str(by_content["x-tagged"].project_id) == project_x
        assert by_content["unfiled"].project_id is None
    finally:
        db_session.execute(
            text("DELETE FROM session_memories WHERE org_id = :o"), {"o": org_id}
        )
        db_session.execute(
            text("DELETE FROM projects WHERE org_id = :o"), {"o": org_id}
        )
        db_session.commit()


# ---------------------------------------------------------------------------
# Embedding failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scoped_recall_returns_empty_when_embedding_none(db_session):
    """embed_text returning None yields an empty list (no DB query)."""
    org_id = _track(db_session, _unique_org())
    _seed_memory(db_session, org_id=org_id, content="present")

    with patch("shared.memory.embed_text", new_callable=AsyncMock, return_value=None):
        results = await retrieve_memories_scoped(
            db_session, org_id, "any query", limit=10
        )

    assert results == []


# ---------------------------------------------------------------------------
# Regression: existing retrieve_memories untouched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existing_retrieve_memories_unchanged(db_session):
    """The session-scoped retrieve_memories still works end-to-end.

    Regression guard proving the additive change did not break the existing
    session-scoped read path: a session's row is found by semantic search.
    """
    org_id = _track(db_session, _unique_org())
    session_id = f"sess-{uuid.uuid4().hex[:8]}"
    _seed_memory(
        db_session,
        org_id=org_id,
        session_id=session_id,
        content="session-scoped fact",
        memory_type="fact",
    )

    with _mock_embed():
        results = await retrieve_memories(
            db_session,
            session_id=session_id,
            org_id=org_id,
            query_text="anything",
            limit=10,
        )

    contents = {r.content for r in results}
    assert "session-scoped fact" in contents
    assert all(r.session_id == session_id for r in results)


# ---------------------------------------------------------------------------
# recall_hidden filter (Task 7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_sql_excludes_hidden_and_inactive(monkeypatch):
    """The generated recall SQL must filter recall_hidden and status.

    This is a unit-level SQL-shape test: no live DB is required. A fake DB
    session captures the rendered SQL string, which must contain both
    ``recall_hidden = FALSE`` and ``status = 'active'``.  The test is
    intentionally isolated from Postgres so it runs in every CI environment.
    """
    import shared.memory as memory_mod

    captured = {}

    async def _embed(_):
        return [0.1, 0.2]

    monkeypatch.setattr(memory_mod, "embed_text", _embed)

    class _DB:
        def execute(self, sql, params=None):
            captured["sql"] = str(sql)

            class R:
                def fetchall(self_inner):
                    return []

            return R()

    await memory_mod.retrieve_memories_scoped(_DB(), "org-1", "query")
    assert "recall_hidden = FALSE" in captured["sql"]
    assert "status = 'active'" in captured["sql"]
