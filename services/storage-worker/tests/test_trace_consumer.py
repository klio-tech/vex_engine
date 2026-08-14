"""Tests for app.trace_consumer (sessions.captured -> S3/DB -> sessions.stored).

Mocked db/s3/redis throughout; only entry-shape and call-shape are asserted.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import trace_consumer


@pytest.mark.anyio
async def test_handle_captured_puts_s3_then_row_then_emits():
    db, s3 = MagicMock(), MagicMock()
    redis = MagicMock()
    redis.xadd = AsyncMock()
    entry = {"payload": json.dumps({
        "org_id": "o", "session_id": "s", "project_id": None, "user_id": None,
        "agent_id": None,
        "messages": [{"role": "assistant", "content": "done"}],
        "tool_calls": [],
    })}
    with patch.object(trace_consumer, "_bucket", return_value="klio-traces"), \
         patch.object(trace_consumer.traces, "upsert_trace", return_value="tid") as up:
        ok = await trace_consumer.handle_captured(db, s3, redis, entry)
    assert ok
    body = s3.put_object.call_args.kwargs["Body"]
    assert b"done" in body
    assert up.call_args.kwargs["message_count"] == 1
    stored = redis.xadd.call_args.args
    assert stored[0] == trace_consumer.SESSIONS_STORED
    assert json.loads(redis.xadd.call_args.args[1]["payload"])["session_id"] == "s"


@pytest.mark.anyio
async def test_bad_payload_is_skipped_not_fatal():
    ok = await trace_consumer.handle_captured(
        MagicMock(), MagicMock(), MagicMock(), {"payload": "not json"}
    )
    assert ok is False
