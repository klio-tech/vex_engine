"""The lexical recall arm, against a real Postgres.

The premise of the whole feature is a *negative result*, so it is asserted
directly rather than assumed: a corpus is seeded in which one row contains a
rare literal and five topically similar rows do not, and the dense arm ranks
the row containing the literal LAST. See ``lexical_fixtures`` for why the
embedding stand-in models that honestly rather than arranging it.

Real Postgres and not a fake session, for the same reason the tenancy suites
use one: the behaviour under test is what ``to_tsvector``, ``ts_rank`` and
``websearch_to_tsquery`` actually do, and a fake that records the emitted SQL
proves only that a string was written.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

import lexical_fixtures as corpus
from shared.memory import retrieve_memories_scoped
from shared.testing import database_url, requires_db

pytestmark = requires_db("the lexical recall arm tests")

DB_URL = database_url()


@pytest.fixture()
def db():
    """A real Connection whose rows are removed afterwards, org by org."""
    engine = create_engine(DB_URL)
    conn = engine.connect()
    orgs: set[str] = set()
    conn.created_org_ids = orgs  # type: ignore[attr-defined]
    try:
        yield conn
    finally:
        try:
            conn.rollback()
            for org_id in orgs:
                conn.execute(
                    text(
                        """
                        DELETE FROM project_members WHERE project_id IN (
                            SELECT id FROM projects WHERE org_id = :o)
                        """
                    ),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM session_memories WHERE org_id = :o"),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM projects WHERE org_id = :o"), {"o": org_id}
                )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _org(db) -> str:
    org_id = f"org_{uuid.uuid4().hex[:8]}"
    db.created_org_ids.add(org_id)
    return org_id


def _seed(
    db,
    *,
    org_id: str,
    content: str,
    scope: str = "org",
    user_id: str | None = None,
    project_id: str | None = None,
    recall_hidden: bool = False,
    agent_id: str = "agent-1",
) -> str:
    """Insert one memory whose embedding is the stand-in's vector for it."""
    memory_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    db.execute(
        text("""
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, execution_id,
                memory_type, content, embedding, scope, status,
                user_id, provenance, project_id, recall_hidden, metadata,
                created_at, updated_at
            ) VALUES (
                :id, :session_id, :org_id, :agent_id, NULL,
                'fact', :content, :embedding, :scope, 'active',
                :user_id, 'EXTRACTED', :project_id, :recall_hidden, '{}'::jsonb,
                :created_at, :updated_at
            )
        """),
        {
            "id": memory_id,
            "session_id": f"sess-{uuid.uuid4().hex[:6]}",
            "org_id": org_id,
            "agent_id": agent_id,
            "content": content,
            "embedding": corpus.embedding_literal(content),
            "scope": scope,
            "user_id": user_id,
            "project_id": project_id,
            "recall_hidden": recall_hidden,
            "created_at": now,
            "updated_at": now,
        },
    )
    db.commit()
    return memory_id


def _seed_corpus(db, org_id: str) -> str:
    """The target row plus its five topical decoys. Returns the target's id."""
    target_id = _seed(db, org_id=org_id, content=corpus.TARGET_CONTENT)
    for content in corpus.DISTRACTOR_CONTENTS:
        _seed(db, org_id=org_id, content=content)
    return target_id


def _embed_as(text_value: str):
    """Patch ``embed_text`` with the stand-in, which is what seeded the rows."""
    return patch(
        "shared.memory.embed_text",
        new_callable=AsyncMock,
        return_value=corpus.embed(text_value),
    )


# ── the premise, and the fix ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_exact_literal_ranks_first(db):
    """A query for a rare literal returns the row containing it, at rank 1.

    This is the observed RED: before the lexical arm existed the dense arm
    ranked the target sixth of six and ``limit=5`` dropped it entirely.
    """
    org = _org(db)
    target_id = _seed_corpus(db, org)

    with _embed_as(corpus.LITERAL):
        results = await retrieve_memories_scoped(db, org, corpus.LITERAL, limit=5)

    assert results, "recall returned nothing at all"
    assert str(results[0].id) == target_id, (
        "the row containing the literal is not first; order was: "
        + " | ".join(r.content[:40] for r in results)
    )


@pytest.mark.asyncio
async def test_target_row_is_otherwise_findable(db):
    """Positive control. Without this, the test above could pass against a
    corpus where the target is the only row that exists at all."""
    org = _org(db)
    target_id = _seed_corpus(db, org)

    probe = "Railway deploy note the platform team rotated the secret"
    with _embed_as(probe):
        results = await retrieve_memories_scoped(db, org, probe, limit=6)

    assert target_id in {str(r.id) for r in results}
