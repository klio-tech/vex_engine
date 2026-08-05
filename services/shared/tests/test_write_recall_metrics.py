"""Writes and recalls are counted, timed, and classified by outcome.

``ok`` / ``quota`` / ``error`` are three values rather than two because they
demand three different responses: ``quota`` is a product signal (someone should
upgrade), ``error`` is an operational one (someone should be paged), and folding
them together makes a pricing success look like an incident.
"""

import asyncio

import pytest
from shared import metrics


@pytest.fixture(autouse=True)
def _clean():
    metrics.reset_for_test()
    yield
    metrics.reset_for_test()


# ── the decorator that carries the classification ─────────────────────


@pytest.mark.asyncio
async def test_a_successful_call_counts_ok_and_is_timed():
    @metrics.tracked(metrics.MEMORY_WRITES, metrics.MEMORY_WRITE_SECONDS)
    async def write():
        return "written"

    assert await write() == "written"
    assert metrics.MEMORY_WRITES.value(outcome="ok") == 1.0
    assert metrics.MEMORY_WRITE_SECONDS.snapshot(outcome="ok").count == 1


@pytest.mark.asyncio
async def test_a_raising_call_counts_error_and_still_raises():
    @metrics.tracked(metrics.MEMORY_WRITES, metrics.MEMORY_WRITE_SECONDS)
    async def write():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await write()

    assert metrics.MEMORY_WRITES.value(outcome="error") == 1.0
    assert metrics.MEMORY_WRITES.value(outcome="ok") == 0.0


@pytest.mark.asyncio
async def test_a_failure_is_still_timed():
    """Dropping the latency of failures is how a slow error path stays hidden."""

    @metrics.tracked(metrics.MEMORY_WRITES, metrics.MEMORY_WRITE_SECONDS)
    async def write():
        await asyncio.sleep(0.01)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await write()

    assert metrics.MEMORY_WRITE_SECONDS.snapshot(outcome="error").count == 1


@pytest.mark.asyncio
async def test_the_decorator_preserves_the_wrapped_function_identity():
    @metrics.tracked(metrics.MEMORY_RECALLS, metrics.MEMORY_RECALL_SECONDS)
    async def recall(a, b=2):
        """A recall."""
        return a + b

    assert recall.__name__ == "recall"
    assert recall.__doc__ == "A recall."
    assert await recall(1, b=5) == 6


@pytest.mark.asyncio
async def test_broken_instrumentation_does_not_break_the_call(monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("metrics on fire")

    monkeypatch.setattr(metrics.MEMORY_WRITES, "inc", explode)

    @metrics.tracked(metrics.MEMORY_WRITES, metrics.MEMORY_WRITE_SECONDS)
    async def write():
        return "written"

    assert await write() == "written"


# ── the real functions carry it ───────────────────────────────────────


def test_write_memory_is_tracked():
    from shared import memory

    assert getattr(memory.write_memory, "_klio_tracked", None) == (
        "klio_memory_writes_total"
    )


@pytest.mark.parametrize("name", ["retrieve_memories", "retrieve_memories_scoped"])
def test_the_recall_paths_are_tracked(name):
    from shared import memory

    assert getattr(getattr(memory, name), "_klio_tracked", None) == (
        "klio_memory_recalls_total"
    )


def test_the_tracked_marker_is_not_universal():
    """Positive control: the marker means something, it is not on everything."""
    from shared import memory

    assert getattr(memory.embed_text, "_klio_tracked", None) is None
