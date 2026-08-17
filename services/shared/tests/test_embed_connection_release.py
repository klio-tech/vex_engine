"""The pooled connection must be released before the embed is awaited.

SQLAlchemy 2.0 autobegins a transaction on the first ``execute()`` and holds
the checked-out connection until commit/rollback/close. Every write path opens
with synchronous DB work — a quota check, project resolution, space resolution
— and *then* awaits an embedding. So each in-flight write pinned one pooled
connection for the whole duration of a network call to a third party.

Measured against the production pool shape (``pool_size=5, max_overflow=10``
-> 15 checkouts) with a 1 s embed::

    concurrency= 15 hold_conn_across_embed=True  -> ok= 15 failed=  0 wall=  2.18s
    concurrency= 16 hold_conn_across_embed=True  -> ok= 15 failed=  1 TimeoutError
    concurrency= 40 hold_conn_across_embed=True  -> ok= 15 failed= 25 TimeoutError
    concurrency= 40 hold_conn_across_embed=False -> ok= 40 failed=  0 wall=  2.07s
    concurrency=200 hold_conn_across_embed=False -> ok=200 failed=  0 wall=  2.24s

These tests use a pool of exactly one connection, so "holds it across the
await" and "does not" are the difference between a deadlock and a pass.
"""

import asyncio
import uuid

import pytest
from shared import memory
from shared.testing import database_url, requires_db
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

pytestmark = requires_db("the pooled connection is released across the embed")


def _vector():
    return [0.1] * 1024


@pytest.fixture
def single_slot_pool():
    """An engine with exactly one connection and no overflow."""
    engine = create_engine(database_url(), pool_size=1, max_overflow=0, pool_timeout=3)
    try:
        yield engine, sessionmaker(bind=engine)
    finally:
        engine.dispose()


@pytest.fixture(autouse=True)
def _clear_memo():
    memory.clear_embedding_memo()
    yield
    memory.clear_embedding_memo()


@pytest.mark.asyncio
async def test_two_concurrent_dedup_probes_share_one_connection(
    single_slot_pool, monkeypatch
):
    """Before: the second call queued on the pool and raised ``TimeoutError``."""
    engine, SessionLocal = single_slot_pool

    async def _slow_embed(**kwargs):
        await asyncio.sleep(0.3)
        return type("R", (), {"data": [{"embedding": _vector()}]})()

    monkeypatch.setattr(memory, "aembedding", _slow_embed)

    async def one(tag: str):
        db = SessionLocal()
        try:
            # Stand in for the quota check / project resolution that opens
            # every write: first execute -> checkout + autobegin.
            db.execute(text("SELECT 1"))
            return await memory.find_similar_memory(
                db, session_id=f"s-{tag}", org_id="org-1", content=f"note {tag}"
            )
        finally:
            db.close()

    # Two writers, one connection slot. This can only pass if neither holds
    # the connection across its embed.
    await asyncio.gather(one("a"), one("b"))


@pytest.mark.asyncio
async def test_the_connection_is_not_checked_out_during_the_embed(
    single_slot_pool, monkeypatch
):
    """Direct observation, not an inference from a timeout."""
    engine, SessionLocal = single_slot_pool
    observed = []

    async def _observing_embed(**kwargs):
        observed.append(engine.pool.checkedout())
        return type("R", (), {"data": [{"embedding": _vector()}]})()

    monkeypatch.setattr(memory, "aembedding", _observing_embed)

    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        assert engine.pool.checkedout() == 1, "positive control: it was checked out"
        await memory.find_similar_memory(
            db, session_id="s-1", org_id="org-1", content="a note"
        )
    finally:
        db.close()

    assert observed == [0], (
        f"connection was still checked out during the embed: {observed}"
    )


@pytest.mark.asyncio
async def test_write_memory_does_not_hold_its_caller_s_connection(
    single_slot_pool, monkeypatch
):
    """``write_memory`` embeds first; the session handed to it may be open."""
    engine, SessionLocal = single_slot_pool
    observed = []

    async def _observing_embed(**kwargs):
        observed.append(engine.pool.checkedout())
        return type("R", (), {"data": [{"embedding": _vector()}]})()

    monkeypatch.setattr(memory, "aembedding", _observing_embed)

    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        record = await memory.write_memory(
            db,
            session_id="s-release",
            org_id="org-release",
            agent_id="agent-1",
            execution_id=None,
            memory_type="note",
            content="a note that must not pin a connection",
        )
    finally:
        db.close()

    assert record is not None, "positive control: the write actually happened"
    assert observed == [0], (
        f"connection was still checked out during the embed: {observed}"
    )


@pytest.mark.asyncio
async def test_releasing_does_not_lose_work_already_done(single_slot_pool, monkeypatch):
    """The release must commit pending work, not discard it."""
    engine, SessionLocal = single_slot_pool

    async def _embed(**kwargs):
        return type("R", (), {"data": [{"embedding": _vector()}]})()

    monkeypatch.setattr(memory, "aembedding", _embed)

    # Unique per run: these rows are written to a shared database.
    session_id = f"s-durable-{uuid.uuid4()}"

    db = SessionLocal()
    try:
        first = await memory.write_memory(
            db,
            session_id=session_id,
            org_id="org-durable",
            agent_id="agent-1",
            execution_id=None,
            memory_type="note",
            content="written before the release",
        )
        assert first is not None
        # A second write on the same session releases again; the first row
        # must still be there afterwards.
        await memory.write_memory(
            db,
            session_id=session_id,
            org_id="org-durable",
            agent_id="agent-1",
            execution_id=None,
            memory_type="note",
            content="written after the release",
        )
    finally:
        db.close()

    verify = SessionLocal()
    try:
        count = verify.execute(
            text(
                "SELECT count(*) FROM session_memories "
                "WHERE org_id = 'org-durable' AND session_id = :sid"
            ),
            {"sid": session_id},
        ).scalar()
    finally:
        verify.close()
    assert count == 2, f"expected both rows to be durable, found {count}"


@pytest.mark.asyncio
async def test_release_tolerates_a_session_that_cannot_commit(monkeypatch):
    """Non-Session callers (raw connections, fakes) must not start failing."""

    class NoCommit:
        def execute(self, *a, **k):
            raise AssertionError("not reached")

    # No commit attribute at all: releasing is a no-op, not an AttributeError.
    memory.release_connection(NoCommit())
