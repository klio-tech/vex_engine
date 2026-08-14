"""Tests for app.main consumer-loop orchestration.

H1 regression coverage: an unset KLIO_TRACE_BUCKET must only disable the
traces consumer (sessions.captured). It must never prevent the legacy
executions.raw/executions.verified consumers from starting.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import main


async def test_consume_traces_skips_when_bucket_unset(monkeypatch, caplog):
    """_consume_traces refuses to start (logs, returns) without raising."""
    monkeypatch.delenv("KLIO_TRACE_BUCKET", raising=False)
    redis_client = MagicMock()
    redis_client.xreadgroup = AsyncMock()

    with caplog.at_level("ERROR"):
        await main._consume_traces(redis_client, s3_client=MagicMock())

    # Must return without ever entering the read loop.
    redis_client.xreadgroup.assert_not_called()
    assert any("traces-consumer" in rec.message for rec in caplog.records)


async def test_consume_traces_enters_loop_when_bucket_set(monkeypatch):
    """With the bucket set, _consume_traces proceeds into its read loop
    (as opposed to returning immediately)."""
    monkeypatch.setenv("KLIO_TRACE_BUCKET", "klio-traces")
    redis_client = MagicMock()
    redis_client.xreadgroup = AsyncMock(side_effect=RuntimeError("stop-loop"))

    with patch.object(main.asyncio, "sleep", new=AsyncMock(side_effect=RuntimeError("stop-loop"))):
        with pytest.raises(RuntimeError, match="stop-loop"):
            await main._consume_traces(redis_client, s3_client=MagicMock())

    redis_client.xreadgroup.assert_called_once()


async def test_run_starts_legacy_consumers_when_trace_bucket_unset(monkeypatch):
    """run() with KLIO_TRACE_BUCKET unset: legacy consumers start, the
    traces consumer runs but returns early (guard trips), and run() does
    not raise.
    """
    monkeypatch.delenv("KLIO_TRACE_BUCKET", raising=False)

    fake_redis_client = MagicMock()
    fake_redis_client.xgroup_create = AsyncMock()
    fake_redis_client.xgroup_destroy = AsyncMock()
    fake_redis_client.xreadgroup = AsyncMock()

    raw_started = AsyncMock()
    verified_started = AsyncMock()

    with patch.object(main.aioredis, "from_url", return_value=fake_redis_client), \
         patch.object(main, "get_s3_client", return_value=MagicMock()), \
         patch.object(main, "_consume_raw", new=raw_started), \
         patch.object(main, "_consume_verified", new=verified_started):
        # _consume_traces is left as the real implementation so H1's guard
        # is actually exercised end-to-end through run().
        await main.run()

    raw_started.assert_called_once()
    verified_started.assert_called_once()
    # The real _consume_traces ran, tripped the bucket guard, and returned
    # before ever calling xreadgroup on the traces stream.
    fake_redis_client.xreadgroup.assert_not_called()


async def test_run_does_not_raise_with_trace_bucket_set(monkeypatch):
    """Sanity check: run() still wires up all three consumers when the
    bucket *is* set (using stubs so the loops don't actually block).
    """
    monkeypatch.setenv("KLIO_TRACE_BUCKET", "klio-traces")

    fake_redis_client = MagicMock()
    fake_redis_client.xgroup_create = AsyncMock()
    fake_redis_client.xgroup_destroy = AsyncMock()
    fake_redis_client.xreadgroup = AsyncMock()

    raw_started = AsyncMock()
    verified_started = AsyncMock()
    traces_started = AsyncMock()

    with patch.object(main.aioredis, "from_url", return_value=fake_redis_client), \
         patch.object(main, "get_s3_client", return_value=MagicMock()), \
         patch.object(main, "_consume_raw", new=raw_started), \
         patch.object(main, "_consume_verified", new=verified_started), \
         patch.object(main, "_consume_traces", new=traces_started):
        await main.run()

    raw_started.assert_called_once()
    verified_started.assert_called_once()
    traces_started.assert_called_once()
