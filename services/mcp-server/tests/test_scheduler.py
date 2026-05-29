"""Tests for the in-process curator scheduler (``app.scheduler``).

No real DB / curator work: ``app.scheduler.run_curator_once`` is patched at the
module boundary so the loop mechanics (interval cadence, crash isolation,
start/stop lifecycle) are asserted in isolation.  A tiny interval keeps the
async tests fast.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.scheduler import _curator_loop, start_curator, stop_curator


def _fake_app():
    """A stand-in app exposing the ``.state`` attribute the scheduler writes."""
    return SimpleNamespace(state=SimpleNamespace())


@pytest.mark.asyncio
async def test_loop_runs_after_interval():
    """The loop sleeps then runs a pass, repeatedly, until cancelled."""
    run = AsyncMock(return_value={"orgs": 0, "synthesized_total": 0})
    with patch("app.scheduler.run_curator_once", run):
        task = asyncio.create_task(_curator_loop(lambda: MagicMock(), 0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert run.await_count >= 1


@pytest.mark.asyncio
async def test_loop_survives_pass_failure():
    """A failing pass is swallowed; the loop keeps running subsequent passes."""
    calls = {"n": 0}

    async def flaky(_factory):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return {"orgs": 0}

    with patch("app.scheduler.run_curator_once", flaky):
        task = asyncio.create_task(_curator_loop(lambda: MagicMock(), 0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    # >= 2 proves the first failure did not kill the loop.
    assert calls["n"] >= 2


@pytest.mark.asyncio
async def test_start_curator_disabled_creates_no_task():
    """When disabled, no task is created and state reflects that."""
    app = _fake_app()
    task = start_curator(app, enabled=False)
    assert task is None
    assert app.state.curator_task is None


@pytest.mark.asyncio
async def test_start_then_stop_curator_lifecycle():
    """start_curator spawns a task; stop_curator cancels and clears it."""
    app = _fake_app()
    with patch("app.scheduler.run_curator_once", AsyncMock(return_value={})):
        task = start_curator(
            app,
            enabled=True,
            interval_seconds=0.01,
            session_factory=lambda: MagicMock(),
        )
        assert task is not None
        assert app.state.curator_task is task
        await asyncio.sleep(0.03)
        await stop_curator(app)

    assert app.state.curator_task is None
    assert task.done()


@pytest.mark.asyncio
async def test_stop_curator_is_noop_without_task():
    """stop_curator tolerates being called when nothing was started."""
    app = _fake_app()
    app.state.curator_task = None
    await stop_curator(app)  # must not raise
    assert app.state.curator_task is None
