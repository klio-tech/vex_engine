"""Tests for app.trace_consumer (sessions.captured -> S3/DB -> sessions.stored).

Mocked db/s3/redis throughout; only entry-shape and call-shape are asserted.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import trace_consumer


def _entry(**overrides):
    payload = {
        "org_id": "o",
        "session_id": "s",
        "project_id": None,
        "user_id": None,
        "agent_id": None,
        "messages": [{"role": "assistant", "content": "done"}],
        "tool_calls": [],
    }
    payload.update(overrides)
    return {"payload": json.dumps(payload)}


@pytest.mark.anyio
async def test_handle_captured_puts_s3_then_row_then_emits():
    db, s3 = MagicMock(), MagicMock()
    redis = MagicMock()
    redis.xadd = AsyncMock()
    entry = _entry()
    with (
        patch.object(trace_consumer, "_bucket", return_value="klio-traces"),
        patch.object(
            trace_consumer.traces,
            "upsert_trace",
            return_value={"id": "tid", "previous_ref": None},
        ) as up,
    ):
        ok = await trace_consumer.handle_captured(db, s3, redis, entry)
    assert ok
    body = s3.put_object.call_args.kwargs["Body"]
    assert b"done" in body
    assert up.call_args.kwargs["message_count"] == 1
    stored = redis.xadd.call_args.args
    assert stored[0] == trace_consumer.SESSIONS_STORED
    assert json.loads(redis.xadd.call_args.args[1]["payload"])["session_id"] == "s"
    # Fresh insert (no previous_ref) — nothing to delete.
    s3.delete_object.assert_not_called()


@pytest.mark.anyio
async def test_bad_payload_is_skipped_not_fatal():
    ok = await trace_consumer.handle_captured(
        MagicMock(), MagicMock(), MagicMock(), {"payload": "not json"}
    )
    assert ok is False


@pytest.mark.anyio
async def test_upsert_lost_deletes_fresh_object_but_still_emits_stored():
    """A LOST richer-wins race still means the session is stored — a richer
    trace already exists — so sessions.stored is emitted regardless, but the
    object THIS call just wrote (which no row points at) must be cleaned up."""
    db, s3 = MagicMock(), MagicMock()
    redis = MagicMock()
    redis.xadd = AsyncMock()
    entry = _entry()
    with (
        patch.object(trace_consumer, "_bucket", return_value="klio-traces"),
        patch.object(trace_consumer.traces, "upsert_trace", return_value=None),
    ):
        ok = await trace_consumer.handle_captured(db, s3, redis, entry)
    assert ok is True
    fresh_key = s3.put_object.call_args.kwargs["Key"]
    s3.delete_object.assert_called_once_with(Bucket="klio-traces", Key=fresh_key)
    redis.xadd.assert_awaited_once()
    assert redis.xadd.call_args.args[0] == trace_consumer.SESSIONS_STORED


@pytest.mark.anyio
async def test_upsert_won_deletes_previous_ref_object():
    """A WON upsert with a differing previous_ref means the row used to point
    at an older, poorer object — that object is now an orphan and must be
    deleted."""
    db, s3 = MagicMock(), MagicMock()
    redis = MagicMock()
    redis.xadd = AsyncMock()
    entry = _entry()
    old_ref = "s3://klio-traces/traces/o/s/old-uuid.json"
    with (
        patch.object(trace_consumer, "_bucket", return_value="klio-traces"),
        patch.object(
            trace_consumer.traces,
            "upsert_trace",
            return_value={"id": "tid", "previous_ref": old_ref},
        ),
    ):
        ok = await trace_consumer.handle_captured(db, s3, redis, entry)
    assert ok is True
    s3.delete_object.assert_called_once_with(
        Bucket="klio-traces", Key="traces/o/s/old-uuid.json"
    )
    redis.xadd.assert_awaited_once()
