"""Dual-write and the read-path switch, against a real Postgres.

A fake session cannot answer the questions here. "Does a write populate both
vector columns", "does the read path actually change column", and "does a
service that comes up before migration 046 still work" are all questions about
what Postgres was asked, and a mock that records SQL proves only that a string
was written.
"""

import uuid

import pytest
from sqlalchemy import create_engine, text

from shared import embedding_space, memory
from shared.testing import database_url, requires_db

pytestmark = requires_db("the embedding-space migration tests")

DB_URL = database_url()
EMBED_DIM = 1024


def _vector(seed: float) -> list[float]:
    """A unit-ish vector that differs per seed, so v1 and v2 are separable."""
    return [seed] + [0.0] * (EMBED_DIM - 1)


@pytest.fixture()
def db():
    engine = create_engine(DB_URL)
    conn = engine.connect()
    org_id = f"org_espace_{uuid.uuid4().hex[:8]}"
    conn.org_id = org_id  # type: ignore[attr-defined]
    memory.reset_embedding_space_probe()
    try:
        yield conn
    finally:
        try:
            conn.rollback()
            conn.execute(
                text("DELETE FROM session_memories WHERE org_id = :o"), {"o": org_id}
            )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()
            memory.reset_embedding_space_probe()


@pytest.fixture()
def distinguishable_embeddings(monkeypatch):
    """Return a different vector per (prefix) so the two columns cannot agree.

    If dual-write ever memoised across spaces — the bug the prefix in the memo
    key exists to prevent — both columns would hold the same vector and every
    assertion below would still pass with a naive stand-in. This one makes that
    failure visible.
    """
    calls: list[str] = []

    async def _embed(text_input, prefix=""):
        calls.append(prefix)
        return _vector(0.5 if prefix else 0.25)

    monkeypatch.setattr(memory, "_aembedding_call", _embed)
    memory.clear_embedding_memo()
    return calls


async def _write(db, content: str = "a durable note"):
    return await memory.write_memory(
        db,
        session_id=f"s-{uuid.uuid4().hex[:8]}",
        org_id=db.org_id,
        agent_id="agent-1",
        execution_id=None,
        memory_type="note",
        content=content,
    )


def _columns(db, memory_id) -> tuple[bool, bool]:
    row = db.execute(
        text(
            "SELECT embedding IS NOT NULL, embedding_v2 IS NOT NULL "
            "FROM session_memories WHERE id = :id"
        ),
        {"id": str(memory_id)},
    ).first()
    return bool(row[0]), bool(row[1])


# ── dual write ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_write_populates_both_vector_columns(db, distinguishable_embeddings):
    """The property the whole migration rests on.

    If writes during the backfill window only filled ``embedding``, the switch
    to v2 would land on a corpus missing every memory written since the
    backfill ran — and it would be missing exactly the newest ones, which are
    the ones a session is most likely to want.
    """
    record = await _write(db)
    assert record is not None
    assert _columns(db, record.id) == (True, True)


@pytest.mark.asyncio
async def test_the_two_columns_hold_different_vectors(db, distinguishable_embeddings):
    """Same content, two spaces, two prefixes, two vectors.

    Identical vectors in both columns would mean the memo collapsed the pair
    and the migration is a very convincing no-op.
    """
    record = await _write(db)
    row = db.execute(
        text(
            "SELECT embedding::text = embedding_v2::text "
            "FROM session_memories WHERE id = :id"
        ),
        {"id": str(record.id)},
    ).first()
    assert row[0] is False
    assert sorted(distinguishable_embeddings) == ["", "passage: "]


@pytest.mark.asyncio
async def test_the_write_still_lands_when_the_inactive_space_fails(db, monkeypatch):
    """Best-effort means best-effort.

    A row missing its INACTIVE vector is fully readable today and will be
    picked up by the backfill. Failing the write over it would trade a working
    memory system for a tidier column.
    """
    calls = {"n": 0}

    async def _embed(text_input, prefix=""):
        calls["n"] += 1
        if prefix:  # the v2 passage
            raise RuntimeError("gateway down for the inactive space")
        return _vector(0.25)

    monkeypatch.setattr(memory, "_aembedding_call", _embed)
    memory.clear_embedding_memo()

    record = await _write(db)
    assert record is not None
    assert _columns(db, record.id) == (True, False)


@pytest.mark.asyncio
async def test_the_write_is_refused_when_the_active_space_fails(db, monkeypatch):
    """The historical contract, unchanged: no vector for the space recall
    reads means no row, rather than a row nothing can ever find."""

    async def _embed(text_input, prefix=""):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(memory, "_aembedding_call", _embed)
    memory.clear_embedding_memo()
    assert await _write(db) is None


# ── the read switch ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recall_reads_the_v2_column_when_switched(
    db, monkeypatch, distinguishable_embeddings
):
    """The switch is a column change, and this is the only way to see it.

    The seeded row's two columns hold different vectors, so a query embedded
    for v2 can only match it through ``embedding_v2``.
    """
    record = await _write(db, "the deploy topology is Railway from GitHub source")
    assert record is not None
    db.execute(
        text("UPDATE session_memories SET scope = 'org' WHERE id = :id"),
        {"id": str(record.id)},
    )
    db.commit()

    monkeypatch.setenv(embedding_space.SPACE_ENV, "v2")
    monkeypatch.setenv("KLIO_RECALL_MIN_RELEVANCE", "0.0")
    monkeypatch.setenv("KLIO_RECALL_RELATIVE_FLOOR", "1.0")
    memory.reset_embedding_space_probe()

    results = await memory.retrieve_memories_scoped(db, db.org_id, "deploy topology")
    assert [str(r.id) for r in results] == [str(record.id)]
    # ``passage: `` for the seed, ``query: `` for the recall.
    assert "query: " in distinguishable_embeddings


@pytest.mark.asyncio
async def test_a_missing_column_degrades_to_v1_rather_than_failing(db, monkeypatch):
    """A service that comes up before migration 046, or an env var set too
    early, must cost ranking quality and not every recall in the org."""
    monkeypatch.setenv(embedding_space.SPACE_ENV, "v2")
    memory.reset_embedding_space_probe()
    monkeypatch.setattr(memory, "_v2_column_present", lambda _db: False)
    assert memory.effective_space(db) == embedding_space.SPACE_V1
