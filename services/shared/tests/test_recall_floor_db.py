"""The relevance floor and the candidate pool, wired into real recall.

:mod:`test_ranking` proves the floor's arithmetic over plain objects. This
proves it is actually connected: that ``retrieve_memories_scoped`` can return
an empty list, that it does so BEFORE graph expansion so a rejected row cannot
drag its neighbours in behind it, and that the deeper candidate pool reaches
the statement.
"""

import uuid

import pytest
from sqlalchemy import create_engine, text

from shared import memory, ranking
from shared.testing import database_url, requires_db

pytestmark = requires_db("the recall floor tests")

DB_URL = database_url()
EMBED_DIM = 1024


def _unit(first: float) -> list[float]:
    """A unit vector leaning ``first`` along axis 0, the rest on axis 1."""
    rest = (1.0 - first * first) ** 0.5
    return [first, rest] + [0.0] * (EMBED_DIM - 2)


@pytest.fixture()
def db():
    engine = create_engine(DB_URL)
    conn = engine.connect()
    conn.org_id = f"org_floor_{uuid.uuid4().hex[:8]}"  # type: ignore[attr-defined]
    memory.reset_embedding_space_probe()
    try:
        yield conn
    finally:
        try:
            conn.rollback()
            conn.execute(
                text("DELETE FROM session_memories WHERE org_id = :o"),
                {"o": conn.org_id},
            )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _seed(db, content: str, vector: list[float]) -> str:
    memory_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, memory_type, content,
                embedding, scope, status, created_at, updated_at,
                recall_hidden
            ) VALUES (
                :id, 's-floor', :org, 'agent-1', 'fact', :content,
                :embedding, 'org', 'active', NOW(), NOW(), FALSE
            )
            """
        ),
        {
            "id": memory_id,
            "org": db.org_id,
            "content": content,
            "embedding": str(vector),
        },
    )
    db.commit()
    return memory_id


@pytest.fixture()
def query_vector(monkeypatch):
    """Embed every query as the axis-0 unit vector, so a row's cosine against
    it is exactly the ``first`` component it was seeded with."""

    async def _embed(text_input, prefix=""):
        return _unit(1.0)

    monkeypatch.setattr(memory, "_aembedding_call", _embed)
    memory.clear_embedding_memo()


@pytest.mark.asyncio
async def test_recall_returns_nothing_when_nothing_clears_the_floor(
    db, monkeypatch, query_vector
):
    """The behaviour the whole change is for.

    Every seeded row is a real, findable memory. None of them is relevant
    enough to be worth putting in front of an agent, so none of them is. This
    is the case that used to return four confident rows at 0.85.
    """
    monkeypatch.setenv(ranking.MIN_RELEVANCE_ENV, "0.90")
    monkeypatch.setenv(ranking.RELATIVE_FLOOR_ENV, "1.0")
    monkeypatch.setenv("KLIO_LEXICAL_RECALL_ENABLED", "0")
    _seed(db, "vaguely related note one", _unit(0.5))
    _seed(db, "vaguely related note two", _unit(0.45))

    assert await memory.retrieve_memories_scoped(db, db.org_id, "anything") == []


@pytest.mark.asyncio
async def test_a_genuine_match_still_comes_back(db, monkeypatch, query_vector):
    """Positive control. Without this the test above passes against a floor
    that rejects everything, which is a different bug wearing the same face."""
    monkeypatch.setenv(ranking.MIN_RELEVANCE_ENV, "0.90")
    monkeypatch.setenv(ranking.RELATIVE_FLOOR_ENV, "1.0")
    monkeypatch.setenv("KLIO_LEXICAL_RECALL_ENABLED", "0")
    wanted = _seed(db, "the row that is actually about this", _unit(0.99))
    _seed(db, "vaguely related note", _unit(0.5))

    results = await memory.retrieve_memories_scoped(db, db.org_id, "anything")
    assert [str(r.id) for r in results] == [wanted]


@pytest.mark.asyncio
async def test_the_relative_floor_trims_the_tail_of_a_good_result(
    db, monkeypatch, query_vector
):
    monkeypatch.setenv(ranking.MIN_RELEVANCE_ENV, "0.0")
    monkeypatch.setenv(ranking.RELATIVE_FLOOR_ENV, "0.05")
    monkeypatch.setenv("KLIO_LEXICAL_RECALL_ENABLED", "0")
    best = _seed(db, "the best row", _unit(0.99))
    near = _seed(db, "nearly as good", _unit(0.96))
    _seed(db, "much worse", _unit(0.40))

    results = await memory.retrieve_memories_scoped(db, db.org_id, "anything")
    assert {str(r.id) for r in results} == {best, near}


@pytest.mark.asyncio
async def test_the_candidate_pool_reaches_the_statement(db, monkeypatch, query_vector):
    """The pool default moved from ``limit * 4`` to ``max(64, limit * 8)``.

    Asserted at the bind parameter rather than through a result, because the
    symptom of a too-shallow pool is not a wrong row — it is a re-rank that
    silently had nothing to re-rank.
    """
    captured: list[dict] = []
    real_execute = db.execute

    def _spy(statement, params=None, *args, **kwargs):
        if params and "candidate_pool" in params:
            captured.append(dict(params))
        return real_execute(statement, params, *args, **kwargs)

    monkeypatch.setattr(db, "execute", _spy)
    _seed(db, "a row", _unit(0.99))
    await memory.retrieve_memories_scoped(db, db.org_id, "anything", limit=8)

    assert captured, "the recall statement was never issued"
    assert captured[0]["candidate_pool"] == ranking.candidate_pool(8)
    assert captured[0]["candidate_pool"] > 8 * 4
