"""Adversarial tests for :func:`shared.memory.retrieve_memories`. Real Postgres.

``retrieve_memories`` is the *session-scoped* read used by the sync-gateway
verification path. That path passes ``event.session_id`` straight off the wire
(``sync-gateway/app/routes.py``), and the MCP surface mints reserved buckets of
the form ``klio-brain:<org_id>:<agent_id>`` from a client-declared
``X-Vex-Agent`` header. So the session id is *attacker-chosen*: one
authenticated caller in an org can name any bucket in that org.

That makes the session id worthless as a privacy boundary on its own. The rows
sharing a bucket may include another user's ``scope='private'`` rows and
``recall_hidden`` curator fuel, and whatever this function returns is handed to
the consistency LLM. The two predicates under test — ``scope = 'session'`` and
``recall_hidden = FALSE`` — are the only things standing between a named bucket
and that disclosure.

Real Postgres rather than a mocked session for the same reason as the rest of
the tenancy suites: the boundary IS the SQL. ``embed_text`` is mocked so every
row carries one identical vector and cosine ranking cannot hide an admission
bug — what is under test is which rows the WHERE clause admits, never order.

Every test carries a positive control: an identically-seeded row that the
caller IS entitled to, proving the fixture is capable of returning anything at
all and that a passing assertion is not vacuous.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

from shared.memory import retrieve_memories
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the session-scoped retrieval boundary tests")

EMBED_DIM = 1024
FIXED_EMBEDDING = [0.1] * EMBED_DIM
FIXED_EMBEDDING_LITERAL = "[" + ",".join(str(x) for x in FIXED_EMBEDDING) + "]"

PRIYA = "11111111-2222-3333-4444-555555555555"


def _unique_org() -> str:
    return f"org_{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def db():
    """A real Connection, with every row this test created removed afterwards."""
    engine = create_engine(DB_URL)
    conn = engine.connect()
    orgs: set[str] = set()
    conn.created_org_ids = orgs  # type: ignore[attr-defined]
    try:
        yield conn
    finally:
        try:
            for org_id in orgs:
                conn.execute(
                    text("DELETE FROM session_memories WHERE org_id = :o"),
                    {"o": org_id},
                )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _org(db) -> str:
    org_id = _unique_org()
    db.created_org_ids.add(org_id)
    return org_id


def _seed(
    db,
    *,
    org_id: str,
    session_id: str,
    content: str,
    scope: str = "session",
    user_id: str | None = None,
    recall_hidden: bool = False,
    memory_type: str = "fact",
    sequence_number: int | None = None,
    agent_id: str = "agent-1",
) -> str:
    memory_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    db.execute(
        text("""
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, execution_id,
                memory_type, content, embedding, scope, status,
                user_id, recall_hidden, sequence_number, metadata,
                created_at, updated_at
            ) VALUES (
                :id, :session_id, :org_id, :agent_id, NULL,
                :memory_type, :content, :embedding, :scope, 'active',
                :user_id, :recall_hidden, :sequence_number, '{}'::jsonb,
                :created_at, :updated_at
            )
        """),
        {
            "id": memory_id,
            "session_id": session_id,
            "org_id": org_id,
            "agent_id": agent_id,
            "memory_type": memory_type,
            "content": content,
            "embedding": FIXED_EMBEDDING_LITERAL,
            "scope": scope,
            "user_id": user_id,
            "recall_hidden": recall_hidden,
            "sequence_number": sequence_number,
            "created_at": now,
            "updated_at": now,
        },
    )
    db.commit()
    return memory_id


def _mock_embed():
    return patch(
        "shared.memory.embed_text",
        new_callable=AsyncMock,
        return_value=FIXED_EMBEDDING,
    )


def _brain_session(org_id: str, agent_id: str = "agent-1") -> str:
    """The reserved bucket the MCP surface mints from a client-declared header."""
    return f"klio-brain:{org_id}:{agent_id}"


# ---------------------------------------------------------------------------
# ATTACK 1 — name the reserved brain bucket, read another user's private rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_named_brain_session_does_not_return_private_rows(db):
    """A caller who names ``klio-brain:<org>:<agent>`` gets no private row.

    The attacker supplies the session id; the only thing that can refuse them
    is ``scope = 'session'``.
    """
    org = _org(db)
    session = _brain_session(org)

    _seed(
        db,
        org_id=org,
        session_id=session,
        content="priya is interviewing at a competitor",
        scope="private",
        user_id=PRIYA,
    )
    # Positive control: a genuinely session-scoped row in the same bucket.
    control = _seed(
        db,
        org_id=org,
        session_id=session,
        content="the deploy target is staging",
        scope="session",
    )

    with _mock_embed():
        records = await retrieve_memories(
            db_session=db,
            session_id=session,
            org_id=org,
            query_text="anything",
        )

    contents = [r.content for r in records]
    assert "priya is interviewing at a competitor" not in contents
    assert [str(r.id) for r in records] == [control], (
        "positive control missing: the fixture returned nothing at all, so the "
        "privacy assertion above proves nothing"
    )


# ---------------------------------------------------------------------------
# ATTACK 2 — recall_hidden curator fuel must never leave the database
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_hidden_rows_are_not_returned(db):
    """``recall_hidden`` rows are curator fuel, excluded from every recall."""
    org = _org(db)
    session = _brain_session(org)

    _seed(
        db,
        org_id=org,
        session_id=session,
        content="hidden raw observation with a customer name in it",
        scope="session",
        recall_hidden=True,
    )
    control = _seed(
        db,
        org_id=org,
        session_id=session,
        content="visible session note",
        scope="session",
        recall_hidden=False,
    )

    with _mock_embed():
        records = await retrieve_memories(
            db_session=db,
            session_id=session,
            org_id=org,
            query_text="anything",
        )

    contents = [r.content for r in records]
    assert "hidden raw observation with a customer name in it" not in contents
    assert [str(r.id) for r in records] == [control], (
        "positive control missing: recall_hidden=FALSE rows must still return"
    )


# ---------------------------------------------------------------------------
# ATTACK 3 — the summary arm is a second query and needs the same predicates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_arm_excludes_private_and_hidden(db):
    """Both queries in ``retrieve_memories`` carry both predicates.

    The recency arm selects by ``sequence_number``, not by vector, so a fix
    applied only to the semantic arm still discloses everything here.
    """
    org = _org(db)
    session = _brain_session(org)

    _seed(
        db,
        org_id=org,
        session_id=session,
        content="private summary of priya's 1:1",
        scope="private",
        user_id=PRIYA,
        memory_type="summary",
        sequence_number=90,
    )
    _seed(
        db,
        org_id=org,
        session_id=session,
        content="hidden summary awaiting distillation",
        scope="session",
        recall_hidden=True,
        memory_type="summary",
        sequence_number=91,
    )
    control = _seed(
        db,
        org_id=org,
        session_id=session,
        content="ordinary session summary",
        scope="session",
        memory_type="summary",
        sequence_number=92,
    )

    with _mock_embed():
        records = await retrieve_memories(
            db_session=db,
            session_id=session,
            org_id=org,
            query_text="anything",
        )

    contents = [r.content for r in records]
    assert "private summary of priya's 1:1" not in contents
    assert "hidden summary awaiting distillation" not in contents
    assert [str(r.id) for r in records] == [control], (
        "positive control missing: an ordinary session summary must return"
    )


# ---------------------------------------------------------------------------
# ATTACK 4 — every non-session scope is refused, not just 'private'
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_org_and_project_scopes_are_not_session_reads(db):
    """``scope = 'session'`` is an allowlist, not a ``scope <> 'private'`` denylist.

    Brain rows land in the same reserved bucket at ``agent`` and ``org`` scope.
    Session-scoped retrieval is not the surface that decides who may read those.
    """
    org = _org(db)
    session = _brain_session(org)

    for scope in ("agent", "org"):
        _seed(
            db,
            org_id=org,
            session_id=session,
            content=f"{scope}-scoped brain row",
            scope=scope,
        )
    control = _seed(
        db,
        org_id=org,
        session_id=session,
        content="genuinely session-scoped row",
        scope="session",
    )

    with _mock_embed():
        records = await retrieve_memories(
            db_session=db,
            session_id=session,
            org_id=org,
            query_text="anything",
        )

    assert [str(r.id) for r in records] == [control], (
        f"expected only the session-scoped row, got {[r.content for r in records]}"
    )
