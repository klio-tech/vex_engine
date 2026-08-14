"""Tests for the evidence-stream producer contract (``shared.evidence_stream``).

Mock-based: no live Redis, no real redaction fixtures beyond the module's own
``redact``. Mirrors the mocked-pool policy used elsewhere in this repo — these
assert call shapes and payload contents, not Redis wire behaviour.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from shared import evidence_stream as es


class _Clean:
    """Stand-in for ``shared.redaction.redact``'s return value."""

    def __init__(self, clean_text: str) -> None:
        self.clean_text = clean_text


def test_payload_redacts_messages_and_tool_args():
    """Every message content and tool-call args go through redact()."""
    with patch.object(es, "redact", return_value=_Clean("CLEAN")):
        p = es.build_captured_payload(
            org_id="o",
            session_id="s",
            messages=[{"role": "user", "content": "sk-live-9"}],
            tool_calls=[{"tool": "Bash", "args": {"command": "curl -H sk-live-9"}}],
        )
    assert p["messages"][0]["content"] == "CLEAN"
    assert "sk-live-9" not in str(p["tool_calls"])
    assert p["org_id"] == "o" and p["session_id"] == "s"


def test_payload_none_when_redaction_fails():
    """A redactor failure drops the whole payload — nothing unscreened leaves."""
    with patch.object(es, "redact", side_effect=RuntimeError("down")):
        p = es.build_captured_payload(
            org_id="o",
            session_id="s",
            messages=[{"role": "user", "content": "x"}],
            tool_calls=[],
        )
    assert p is None


def test_payload_carries_optional_identity_fields():
    """project_id/user_id/agent_id pass through untouched when supplied."""
    with patch.object(es, "redact", return_value=_Clean("CLEAN")):
        p = es.build_captured_payload(
            org_id="o",
            session_id="s",
            messages=[],
            tool_calls=[],
            project_id="p1",
            user_id="u1",
            agent_id="a1",
        )
    assert p == {
        "org_id": "o",
        "session_id": "s",
        "project_id": "p1",
        "user_id": "u1",
        "agent_id": "a1",
        "messages": [],
        "tool_calls": [],
    }


def test_payload_tail_trims_oldest_messages_to_fit_cap():
    """Messages exceeding the 256 KB cap are trimmed from the FRONT (oldest first)."""
    big = "x" * 50_000
    messages = [{"role": "user", "content": f"{big}-{i}"} for i in range(10)]
    with patch.object(es, "redact", side_effect=lambda t: _Clean(t)):
        p = es.build_captured_payload(
            org_id="o", session_id="s", messages=messages, tool_calls=[]
        )
    assert p is not None
    size = len(json.dumps(p, ensure_ascii=False).encode("utf-8"))
    assert size <= es._MAX_PAYLOAD_BYTES
    # The most recent message must survive the trim; the oldest must not.
    assert p["messages"][-1]["content"].endswith("-9")
    assert not any(m["content"].endswith("-0") for m in p["messages"])
    assert len(p["messages"]) < len(messages)


def test_payload_does_not_trim_tool_calls_when_messages_alone_fit_cap():
    """When trimming messages already gets under the cap, tool_calls are untouched."""
    tool_calls = [{"tool": "Bash", "args": {"command": f"cmd-{i}"}} for i in range(5)]
    with patch.object(es, "redact", side_effect=lambda t: _Clean(t)):
        p = es.build_captured_payload(
            org_id="o",
            session_id="s",
            messages=[{"role": "user", "content": "hi"}],
            tool_calls=tool_calls,
        )
    assert len(p["tool_calls"]) == 5


def test_payload_falls_back_to_trimming_tool_calls_when_messages_are_not_enough():
    """Tiny messages + huge tool_calls: once messages are fully trimmed and the
    payload is still over the cap, tool_calls are trimmed oldest-first too,
    and the newest tool call survives."""
    big = "y" * 60_000
    messages = [{"role": "user", "content": "hi"}]
    tool_calls = [
        {"tool": "Bash", "args": {"command": f"{big}-{i}"}} for i in range(10)
    ]
    with patch.object(es, "redact", side_effect=lambda t: _Clean(t)):
        p = es.build_captured_payload(
            org_id="o", session_id="s", messages=messages, tool_calls=tool_calls
        )
    assert p is not None
    size = len(json.dumps(p, ensure_ascii=False).encode("utf-8"))
    assert size <= es._MAX_PAYLOAD_BYTES
    assert len(p["tool_calls"]) < len(tool_calls)
    # The newest tool call must survive; the oldest must not. ``args`` is a
    # JSON-encoded string (e.g. '...-9"}'), so match with the closing
    # delimiter rather than a bare endswith.
    assert "-9\"}" in p["tool_calls"][-1]["args"]
    assert not any("-0\"}" in t["args"] for t in p["tool_calls"])


@pytest.mark.asyncio
async def test_publish_captured_xadds_to_sessions_captured_stream():
    """publish_captured XADDs the JSON-encoded payload to SESSIONS_CAPTURED."""
    fake_redis = AsyncMock()
    with patch.object(es, "_get_redis", return_value=fake_redis):
        await es.publish_captured(payload={"org_id": "o", "session_id": "s"})

    assert fake_redis.xadd.await_args.args[0] == es.SESSIONS_CAPTURED
    fields = fake_redis.xadd.await_args.args[1]
    assert json.loads(fields["payload"]) == {"org_id": "o", "session_id": "s"}


@pytest.mark.asyncio
async def test_publish_captured_raises_on_redis_failure():
    """A Redis failure propagates — publish is NOT silently swallowed here.

    Best-effort behaviour (never fail the capture request) is the caller's
    job; this function must surface the failure so it can be observed/handled.
    """
    fake_redis = AsyncMock()
    fake_redis.xadd.side_effect = RuntimeError("redis down")
    with patch.object(es, "_get_redis", return_value=fake_redis):
        with pytest.raises(RuntimeError):
            await es.publish_captured(payload={"org_id": "o"})
