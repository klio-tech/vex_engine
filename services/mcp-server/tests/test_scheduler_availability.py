"""The artifact GC sweep must be bounded and must not block the event loop.

``collect_artifacts_once`` is ``async def`` doing purely blocking work: sync
psycopg2, a commit per row, and boto3 ``delete_object`` (5 s connect / 30 s
read / 3 adaptive retries). Normal case that is ~2.5 s of wedged loop per hour.
With a blackholed object store it is ~105 s **per row** — 50 rows wedges the
loop for roughly an hour and a half, during which no request is served.

``collect_purgeable`` also had no LIMIT: retention or a mass ``forget`` can
make it return every dead row in the deployment, and the sweep then tries to
process all of them in one tick.

Note what is NOT the defect: pending rows are capped at
``MAX_PENDING_PER_ORG`` (32) per org, so this is not an unbounded-memory
problem. It is a loop-blocking and tick-duration problem, and that is what
these tests pin.
"""

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from app.scheduler import ARTIFACT_GC_ROW_LIMIT, collect_artifacts_once


def _rows(n, prefix="a"):
    return [
        {"id": f"{prefix}{i}", "org_id": "acme", "version": 1, "status": "retracted"}
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_the_sweep_runs_off_the_event_loop():
    """Direct observation: the blocking work is not on the loop thread."""
    loop_ident = threading.get_ident()
    seen = []
    db = MagicMock()

    with (
        patch("app.scheduler.artifacts") as af,
        patch("app.scheduler.artifact_storage") as st,
    ):
        af.collect_purgeable.return_value = []
        af.collect_abandoned.return_value = _rows(2)
        st.object_key.side_effect = lambda o, a, v: f"{o}/{a}/v{v}"
        st.get_artifact_s3_client.return_value = MagicMock()
        st.delete.side_effect = lambda s3, key: seen.append(threading.get_ident())

        collected = await collect_artifacts_once(lambda: db)

    assert collected == 2, "positive control: the sweep did its work"
    assert seen, "positive control: delete was actually called"
    assert all(t != loop_ident for t in seen), (
        f"object deletes ran on the event loop thread: {seen}"
    )


@pytest.mark.asyncio
async def test_a_slow_object_store_does_not_wedge_the_loop():
    """The end the offload exists for: 105s/row against a blackholed endpoint."""
    ticks = 0
    stop = False

    async def heartbeat():
        nonlocal ticks
        while not stop:
            await asyncio.sleep(0.005)
            ticks += 1

    db = MagicMock()
    with (
        patch("app.scheduler.artifacts") as af,
        patch("app.scheduler.artifact_storage") as st,
    ):
        af.collect_purgeable.return_value = []
        af.collect_abandoned.return_value = _rows(3)
        st.object_key.side_effect = lambda o, a, v: f"{o}/{a}/v{v}"
        st.get_artifact_s3_client.return_value = MagicMock()
        st.delete.side_effect = lambda s3, key: time.sleep(0.1)

        beat = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.02)
        ticks = 0
        await collect_artifacts_once(lambda: db)
        stop = True
        await beat

    # 300 ms of blocking deletes; a 5 ms heartbeat should tick ~60 times.
    assert ticks > 10, (
        f"the loop ticked only {ticks} times during 300ms of object deletes"
    )


@pytest.mark.asyncio
async def test_each_tick_is_bounded_by_a_row_limit():
    """A mass ``forget`` must not turn one tick into an unbounded sweep."""
    db = MagicMock()
    with (
        patch("app.scheduler.artifacts") as af,
        patch("app.scheduler.artifact_storage") as st,
    ):
        af.collect_purgeable.return_value = []
        af.collect_abandoned.return_value = []
        st.get_artifact_s3_client.return_value = MagicMock()
        await collect_artifacts_once(lambda: db)

        assert af.collect_abandoned.call_args.kwargs.get("limit") == (
            ARTIFACT_GC_ROW_LIMIT
        )
        assert af.collect_purgeable.call_args.kwargs.get("limit") == (
            ARTIFACT_GC_ROW_LIMIT
        )


def test_the_row_limit_is_a_real_bound():
    assert 0 < ARTIFACT_GC_ROW_LIMIT <= 10_000
