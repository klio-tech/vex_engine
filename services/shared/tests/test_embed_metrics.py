"""The embedding path reports what it is doing.

Four signals, each of which was an outage that had to be diagnosed by hand:

* how many embeds succeed, time out, or fail — a timeout is a slow backend, an
  error is a wrong one, and they call for different responses;
* how long they take;
* the memo hit rate — a single write embeds the same string twice, so a hit
  rate far below 50% means the memo has silently stopped working;
* whether the concurrency cap is saturated — the difference between "the
  backend is slow" and "we are queueing behind our own limit".
"""

import asyncio

import pytest
from shared import memory, metrics


@pytest.fixture(autouse=True)
def _clean():
    memory.clear_embedding_memo()
    memory.reset_embed_concurrency_limiter()
    metrics.reset_for_test()
    yield
    memory.clear_embedding_memo()
    memory.reset_embed_concurrency_limiter()
    metrics.reset_for_test()


def _vector(seed: float = 0.1) -> list[float]:
    return [seed] * 8


class _Response:
    def __init__(self, vector):
        self.data = [{"embedding": vector}]


# ── outcomes ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_successful_embed_is_counted_and_timed(monkeypatch):
    async def ok(**_kwargs):
        return _Response(_vector())

    monkeypatch.setattr(memory, "aembedding", ok)
    assert await memory.embed_text("hello") is not None

    assert metrics.EMBEDDING_CALLS.value(outcome="ok") == 1.0
    assert metrics.EMBEDDING_SECONDS.snapshot().count == 1


@pytest.mark.asyncio
async def test_a_timeout_is_counted_as_a_timeout_not_an_error(monkeypatch):
    """``timeout`` and ``error`` are separate values on purpose."""
    monkeypatch.setenv("KLIO_EMBED_TIMEOUT_S", "0.05")

    async def hang(**_kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(memory, "aembedding", hang)
    assert await memory.embed_text("hello") is None

    assert metrics.EMBEDDING_CALLS.value(outcome="timeout") == 1.0
    assert metrics.EMBEDDING_CALLS.value(outcome="error") == 0.0


@pytest.mark.asyncio
async def test_a_backend_error_is_counted_as_an_error(monkeypatch):
    async def boom(**_kwargs):
        raise RuntimeError("backend said no")

    monkeypatch.setattr(memory, "aembedding", boom)
    await memory.embed_text("hello")

    assert metrics.EMBEDDING_CALLS.value(outcome="error") >= 1.0
    assert metrics.EMBEDDING_CALLS.value(outcome="timeout") == 0.0


@pytest.mark.asyncio
async def test_a_timed_out_embed_is_still_timed(monkeypatch):
    """Latency of a failure is latency; dropping it hides the incident."""
    monkeypatch.setenv("KLIO_EMBED_TIMEOUT_S", "0.05")

    async def hang(**_kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(memory, "aembedding", hang)
    await memory.embed_text("hello")

    assert metrics.EMBEDDING_SECONDS.snapshot().count == 1


# ── memo hit rate ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_memo_records_a_miss_then_a_hit(monkeypatch):
    calls = []

    async def ok(**_kwargs):
        calls.append(1)
        return _Response(_vector())

    monkeypatch.setattr(memory, "aembedding", ok)
    await memory.embed_text("same string")
    await memory.embed_text("same string")

    assert len(calls) == 1, "the memo did not collapse the second call"
    assert metrics.EMBEDDING_MEMO.value(result="miss") == 1.0
    assert metrics.EMBEDDING_MEMO.value(result="hit") == 1.0


@pytest.mark.asyncio
async def test_distinct_content_records_two_misses(monkeypatch):
    """Positive control: the hit counter is not incremented unconditionally."""

    async def ok(**_kwargs):
        return _Response(_vector())

    monkeypatch.setattr(memory, "aembedding", ok)
    await memory.embed_text("first")
    await memory.embed_text("second")

    assert metrics.EMBEDDING_MEMO.value(result="miss") == 2.0
    assert metrics.EMBEDDING_MEMO.value(result="hit") == 0.0


# ── concurrency-cap saturation ────────────────────────────────────────


@pytest.mark.asyncio
async def test_saturating_the_cap_is_counted(monkeypatch):
    """With the cap at 1, the second concurrent embed must queue — and say so."""
    monkeypatch.setenv("KLIO_EMBED_MAX_CONCURRENCY", "1")
    memory.reset_embed_concurrency_limiter()

    release = asyncio.Event()

    async def slow(**_kwargs):
        await release.wait()
        return _Response(_vector())

    monkeypatch.setattr(memory, "aembedding", slow)

    first = asyncio.create_task(memory.embed_text("one"))
    await asyncio.sleep(0.05)
    second = asyncio.create_task(memory.embed_text("two"))
    await asyncio.sleep(0.05)

    saturated = metrics.EMBEDDING_CONCURRENCY_SATURATED.value()
    release.set()
    await asyncio.gather(first, second)

    assert saturated == 1.0, "queueing behind the cap was not reported"


@pytest.mark.asyncio
async def test_a_lone_embed_at_cap_one_is_not_saturated(monkeypatch):
    """The cap must be read *before* it is acquired, not after.

    With the cap at 1, a single uncontended call holds the only permit — so
    checking ``locked()`` after acquiring reports every call as saturated, and
    the signal becomes a constant. Only a check taken before the acquire can
    tell "I had to queue" from "I took the last permit".
    """
    monkeypatch.setenv("KLIO_EMBED_MAX_CONCURRENCY", "1")
    memory.reset_embed_concurrency_limiter()

    async def ok(**_kwargs):
        return _Response(_vector())

    monkeypatch.setattr(memory, "aembedding", ok)
    await memory.embed_text("alone at cap one")

    assert metrics.EMBEDDING_CONCURRENCY_SATURATED.value() == 0.0


@pytest.mark.asyncio
async def test_an_uncontended_embed_is_not_reported_as_saturated(monkeypatch):
    """Positive control: the saturation counter is not just always incremented."""
    monkeypatch.setenv("KLIO_EMBED_MAX_CONCURRENCY", "32")
    memory.reset_embed_concurrency_limiter()

    async def ok(**_kwargs):
        return _Response(_vector())

    monkeypatch.setattr(memory, "aembedding", ok)
    await memory.embed_text("alone")

    assert metrics.EMBEDDING_CONCURRENCY_SATURATED.value() == 0.0


@pytest.mark.asyncio
async def test_in_flight_returns_to_zero(monkeypatch):
    """The gauge must be released on the failure path too, or it ratchets up."""

    async def boom(**_kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(memory, "aembedding", boom)
    await memory.embed_text("hello")

    assert metrics.EMBEDDING_IN_FLIGHT.value() == 0.0


@pytest.mark.asyncio
async def test_in_flight_is_nonzero_while_a_call_is_running(monkeypatch):
    """Positive control for the test above: the gauge does move."""
    observed = []
    release = asyncio.Event()

    async def slow(**_kwargs):
        observed.append(metrics.EMBEDDING_IN_FLIGHT.value())
        await release.wait()
        return _Response(_vector())

    monkeypatch.setattr(memory, "aembedding", slow)
    task = asyncio.create_task(memory.embed_text("hello"))
    await asyncio.sleep(0.05)
    release.set()
    await task

    assert observed == [1.0]


# ── instrumentation must never break the thing it measures ────────────


@pytest.mark.asyncio
async def test_a_broken_metric_does_not_fail_the_embed(monkeypatch):
    """If instrumentation can take a write down, it is worse than no metrics."""

    async def ok(**_kwargs):
        return _Response(_vector())

    def explode(*_args, **_kwargs):
        raise RuntimeError("metrics backend on fire")

    monkeypatch.setattr(memory, "aembedding", ok)
    monkeypatch.setattr(metrics.EMBEDDING_CALLS, "inc", explode)

    assert await memory.embed_text("hello") is not None
