"""The embedding call must be bounded, memoized, and must not pin a connection.

Three defects on the write path, all on the same await:

1. **No timeout.** ``litellm.aembedding`` was awaited with no bound at all, so a
   hung embedding backend held the caller forever.
2. **Two embeds per write.** ``find_similar_memory`` embeds the content to look
   for a duplicate, then ``write_memory`` embeds the *same* content again to
   store it. Every write paid the backend twice.
3. **The pooled connection was held across both awaits.** SQLAlchemy 2.0
   autobegins on the first ``execute()`` and holds the checked-out connection
   until commit/rollback/close, so the quota check at the top of a write pinned
   a connection for the whole embed. Measured against the real pool
   (``pool_size=5, max_overflow=10`` -> 15 checkouts): the **16th** concurrent
   write queued and then raised ``TimeoutError``; at concurrency 40, 25 of 40
   failed. Releasing before the await took 200 concurrent writes to zero
   failures.

The concurrency cap is separate from the pool: it bounds how many calls this
process fans out at the embedding *backend*, which the pool never did.
"""

import asyncio
import os
import time

import pytest
from shared import memory


@pytest.fixture(autouse=True)
def _clear_embed_state():
    """Each test starts with an empty memo and no env knobs set."""
    memory.clear_embedding_memo()
    saved = {
        k: os.environ.pop(k, None)
        for k in ("KLIO_EMBED_TIMEOUT_S", "KLIO_EMBED_MAX_CONCURRENCY")
    }
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
    memory.clear_embedding_memo()


def _vector(seed: float = 0.1) -> list[float]:
    return [seed] * 8


# ── 1. Bounded timeout ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_hung_embedding_backend_does_not_hang_the_caller(monkeypatch):
    """Unbounded before: this await never returned."""
    monkeypatch.setenv("KLIO_EMBED_TIMEOUT_S", "0.05")

    async def _hang(**kwargs):
        await asyncio.sleep(30)

    monkeypatch.setattr(memory, "aembedding", _hang)

    started = time.monotonic()
    result = await memory.embed_text("anything")
    elapsed = time.monotonic() - started

    assert result is None, "a timed-out embed degrades to None, like any backend error"
    assert elapsed < 5.0, f"returned in {elapsed:.2f}s; the timeout was not applied"


@pytest.mark.asyncio
async def test_the_timeout_is_configurable(monkeypatch):
    """Positive control for the knob: a longer bound lets a slow call finish."""
    monkeypatch.setenv("KLIO_EMBED_TIMEOUT_S", "5")

    async def _slow(**kwargs):
        await asyncio.sleep(0.05)
        return type("R", (), {"data": [{"embedding": _vector()}]})()

    monkeypatch.setattr(memory, "aembedding", _slow)
    assert await memory.embed_text("anything") == _vector()


@pytest.mark.asyncio
async def test_a_malformed_timeout_knob_falls_back_to_the_default(monkeypatch):
    """``KLIO_EMBED_TIMEOUT_S='10s'`` must not raise out of every write."""
    monkeypatch.setenv("KLIO_EMBED_TIMEOUT_S", "10s")

    async def _ok(**kwargs):
        return type("R", (), {"data": [{"embedding": _vector()}]})()

    monkeypatch.setattr(memory, "aembedding", _ok)
    assert await memory.embed_text("anything") == _vector()


@pytest.mark.asyncio
async def test_a_timeout_is_not_mistaken_for_an_over_length_rejection(monkeypatch):
    """A timeout must NOT trigger the progressive-truncation retry loop.

    Retrying a hung backend four times at shrinking sizes turns one stall into
    four, and would silently store a truncated vector if the stall cleared.
    """
    monkeypatch.setenv("KLIO_EMBED_TIMEOUT_S", "0.05")
    calls = []

    async def _hang(**kwargs):
        calls.append(kwargs["input"][0])
        await asyncio.sleep(30)

    monkeypatch.setattr(memory, "aembedding", _hang)
    assert await memory.embed_text("x" * 4000) is None
    assert len(calls) == 1, f"expected one attempt, got {len(calls)}"


# ── 2. Memoization: one embed per distinct content ───────────────────


@pytest.mark.asyncio
async def test_identical_content_is_embedded_once_not_twice(monkeypatch):
    """The dedup probe and the insert embed the same string; pay once."""
    calls = []

    async def _count(**kwargs):
        calls.append(kwargs["input"][0])
        return type("R", (), {"data": [{"embedding": _vector()}]})()

    monkeypatch.setattr(memory, "aembedding", _count)

    first = await memory.embed_text("the same note")
    second = await memory.embed_text("the same note")

    assert first == second == _vector()
    assert len(calls) == 1, f"content was embedded {len(calls)} times"


@pytest.mark.asyncio
async def test_different_content_is_not_collapsed(monkeypatch):
    """Positive control: the memo must not serve one note's vector for another."""
    seeds = iter([0.1, 0.2])

    async def _distinct(**kwargs):
        return type("R", (), {"data": [{"embedding": _vector(next(seeds))}]})()

    monkeypatch.setattr(memory, "aembedding", _distinct)
    a = await memory.embed_text("note one")
    b = await memory.embed_text("note two")
    assert a != b


@pytest.mark.asyncio
async def test_a_different_model_is_not_served_from_the_memo(monkeypatch):
    """The vector is model-specific; the key must say so."""
    calls = []

    async def _count(**kwargs):
        calls.append(kwargs["model"])
        return type("R", (), {"data": [{"embedding": _vector()}]})()

    monkeypatch.setattr(memory, "aembedding", _count)
    monkeypatch.setenv("MEMORY_EMBEDDING_MODEL", "model-a")
    await memory.embed_text("same text")
    monkeypatch.setenv("MEMORY_EMBEDDING_MODEL", "model-b")
    await memory.embed_text("same text")
    assert calls == ["model-a", "model-b"]


@pytest.mark.asyncio
async def test_a_failure_is_never_memoized(monkeypatch):
    """A transient backend error must not poison the content for its whole TTL."""
    attempts = []

    async def _fail_then_succeed(**kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("gateway 502")
        return type("R", (), {"data": [{"embedding": _vector()}]})()

    monkeypatch.setattr(memory, "aembedding", _fail_then_succeed)
    assert await memory.embed_text("flaky") is None
    assert await memory.embed_text("flaky") == _vector()


@pytest.mark.asyncio
async def test_the_memo_is_bounded(monkeypatch):
    """It is a within-request collapse, not a cache: it must not grow forever."""

    async def _ok(**kwargs):
        return type("R", (), {"data": [{"embedding": _vector()}]})()

    monkeypatch.setattr(memory, "aembedding", _ok)
    for i in range(memory.EMBED_MEMO_MAX_ENTRIES * 2):
        await memory.embed_text(f"note {i}")
    assert memory.embedding_memo_size() <= memory.EMBED_MEMO_MAX_ENTRIES


@pytest.mark.asyncio
async def test_the_memo_expires(monkeypatch):
    """A stale entry must not outlive its TTL."""
    calls = []

    async def _count(**kwargs):
        calls.append(1)
        return type("R", (), {"data": [{"embedding": _vector()}]})()

    monkeypatch.setattr(memory, "aembedding", _count)
    monkeypatch.setattr(memory, "EMBED_MEMO_TTL_S", 0.0)
    await memory.embed_text("expiring")
    await memory.embed_text("expiring")
    assert len(calls) == 2


# ── 3. Concurrency cap at the embedding backend ──────────────────────


@pytest.mark.asyncio
async def test_embeds_are_capped_in_flight(monkeypatch):
    """The pool never bounded backend fan-out; this does."""
    monkeypatch.setenv("KLIO_EMBED_MAX_CONCURRENCY", "3")
    memory.reset_embed_concurrency_limiter()
    in_flight = 0
    peak = 0

    async def _track(**kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.02)
            return type("R", (), {"data": [{"embedding": _vector()}]})()
        finally:
            in_flight -= 1

    monkeypatch.setattr(memory, "aembedding", _track)
    await asyncio.gather(*(memory.embed_text(f"note {i}") for i in range(20)))
    assert peak <= 3, f"peak in-flight embeds was {peak}, cap was 3"


@pytest.mark.asyncio
async def test_the_cap_does_not_serialize_everything(monkeypatch):
    """Positive control: the cap must permit real concurrency below it."""
    monkeypatch.setenv("KLIO_EMBED_MAX_CONCURRENCY", "8")
    memory.reset_embed_concurrency_limiter()
    in_flight = 0
    peak = 0

    async def _track(**kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.05)
            return type("R", (), {"data": [{"embedding": _vector()}]})()
        finally:
            in_flight -= 1

    monkeypatch.setattr(memory, "aembedding", _track)
    await asyncio.gather(*(memory.embed_text(f"note {i}") for i in range(8)))
    assert peak > 1, "the cap serialized calls that should have run concurrently"
