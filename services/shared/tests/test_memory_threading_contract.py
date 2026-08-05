"""Blocking DB work in ``shared.memory`` must not run on the event loop.

``shared.projects`` states the contract the engine is supposed to follow:

    Synchronous (blocking psycopg2 I/O), matching the house pattern in
    :mod:`shared.brain`. Call from async handlers via ``anyio.to_thread.run_sync``.

``capture.py`` honours it at 14 sites. The memory primitives did not: they are
``async def`` whose bodies issue psycopg2 statements inline, so every caller —
including the MCP tool surface — blocked the loop for the duration.

Measured with 6 statements at a 10 ms round trip plus one 100 ms embed::

    inline      concurrency= 1  rps=  5.5  worst_loop_lag=   75.7ms
    inline      concurrency=20  rps= 11.5  worst_loop_lag= 1626.4ms
    inline      concurrency=50  rps= 11.9  worst_loop_lag= 4092.2ms
    to_thread   concurrency= 1  rps=  5.9  worst_loop_lag=    1.3ms
    to_thread   concurrency=20  rps= 80.5  worst_loop_lag=    2.0ms
    to_thread   concurrency=50  rps=126.3  worst_loop_lag=    1.4ms

Throughput plateaus at ~12 rps inline no matter how much concurrency is
offered, because the loop — not the pool — is the first ceiling.

These tests assert the mechanism directly: the thread the statement runs on is
not the thread running the event loop.
"""

import asyncio
import threading

import pytest
from shared import memory


@pytest.fixture
def loop_thread_ident():
    """The identity of the thread the test's event loop runs on."""
    return threading.get_ident()


class _ThreadRecordingSession:
    """A session that records which thread each call arrived on."""

    def __init__(self, fetch_result=None):
        self.threads: list[int] = []
        self._fetch_result = fetch_result

    def execute(self, *args, **kwargs):
        self.threads.append(threading.get_ident())

        class _Result:
            def __init__(self, row):
                self._row = row

            def fetchone(self):
                return self._row

            def fetchall(self):
                return [] if self._row is None else [self._row]

        return _Result(self._fetch_result)

    def commit(self):
        self.threads.append(threading.get_ident())


@pytest.fixture(autouse=True)
def _stub_embed(monkeypatch):
    async def _embed(**kwargs):
        return type("R", (), {"data": [{"embedding": [0.1] * 8}]})()

    monkeypatch.setattr(memory, "aembedding", _embed)
    memory.clear_embedding_memo()
    yield
    memory.clear_embedding_memo()


@pytest.mark.asyncio
async def test_find_similar_memory_runs_its_query_off_the_loop():
    """The dedup SELECT is a vector scan; it must not block the loop."""
    loop_ident = threading.get_ident()
    db = _ThreadRecordingSession(fetch_result=None)

    await memory.find_similar_memory(
        db, session_id="s-1", org_id="org-1", content="a note"
    )

    assert db.threads, "positive control: the session was actually used"
    offloaded = [t for t in db.threads if t != loop_ident]
    assert offloaded, f"every call ran on the event loop thread: {db.threads}"


@pytest.mark.asyncio
async def test_write_memory_runs_its_insert_off_the_loop():
    loop_ident = threading.get_ident()
    db = _ThreadRecordingSession()

    record = await memory.write_memory(
        db,
        session_id="s-1",
        org_id="org-1",
        agent_id="agent-1",
        execution_id=None,
        memory_type="note",
        content="a note",
    )

    assert record is not None, "positive control: the write returned a record"
    insert_threads = [t for t in db.threads if t != loop_ident]
    assert insert_threads, f"every call ran on the event loop thread: {db.threads}"


@pytest.mark.asyncio
async def test_the_loop_stays_responsive_during_a_slow_insert():
    """The end the offload exists for, not just the mechanism."""
    ticks = 0
    stop = False

    async def heartbeat():
        nonlocal ticks
        while not stop:
            await asyncio.sleep(0.005)
            ticks += 1

    class _SlowSession:
        def execute(self, *a, **k):
            import time

            time.sleep(0.25)

            class _R:
                def fetchone(self):
                    return None

            return _R()

        def commit(self):
            pass

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.02)
    ticks = 0

    await memory.write_memory(
        _SlowSession(),
        session_id="s-1",
        org_id="org-1",
        agent_id="agent-1",
        execution_id=None,
        memory_type="note",
        content="slow",
    )
    stop = True
    await beat

    # 250 ms of blocking work; a 5 ms heartbeat should have ticked ~50 times.
    assert ticks > 10, (
        f"the loop ticked only {ticks} times during a 250ms insert; it was blocked"
    )
