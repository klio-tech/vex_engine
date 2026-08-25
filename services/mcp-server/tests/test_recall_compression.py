"""recall returns compressed payloads, without breaking its contract.

recall's output is a public API. The compression is additive: information is
preserved, the shape gains an optional ``common`` block, and every existing
consumer can still find what it needs.

Mirrors ``test_recall_tool.py``'s harness exactly — the tool callable is
reached via the FastMCP registry and the DB / retrieval seams are patched at
the ``app.mcp_app`` module boundary, so no Postgres or embeddings are touched.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.mcp_app import build_mcp_server
from shared.memory import MemoryRecord


@pytest.fixture(autouse=True)
def _no_metering():
    with (
        patch("app.mcp_app.check_memory_quota", return_value=False),
        patch("app.mcp_app.record_usage"),
    ):
        yield


def _fake_ctx(headers: dict | None = None):
    request = SimpleNamespace(headers=headers if headers is not None else {})
    request_context = SimpleNamespace(request=request)
    return SimpleNamespace(request_context=request_context)


def _recall_fn():
    server = build_mcp_server()
    tool = server._tool_manager.get_tool("recall")
    assert tool is not None, "tool 'recall' not registered"
    return tool.fn


def _make_record(*, content: str, index: int) -> MemoryRecord:
    """A record shaped like ``retrieve_memories_scoped`` returns.

    Everything except id / content / created_at / relevance is deliberately
    identical across records: that is the redundancy the compressor exists to
    remove, and it is what a real single-project org-scope recall looks like.
    """
    created = datetime(2026, 7, 27, tzinfo=timezone.utc) + timedelta(hours=index)
    return MemoryRecord(
        id=uuid.uuid4(),
        session_id="klio-brain:org-x:agent-y",
        org_id="org-x",
        agent_id="agent-y",
        memory_type="fact",
        content=content,
        execution_id=None,
        embedding=[0.1, 0.2, 0.3],
        confidence=None,
        scope="org",
        status="active",
        metadata={"source": "hook-transcript"},
        created_at=created,
        updated_at=created,
        space_id=None,
        project_id="93e3f7df-788a-4de0-9c3f-654e1259ce00",
        relevance=round(0.9 - index * 0.03, 3),
    )


CONTENTS = [
    "Team uses pgvector for recall",
    "Recall relevance threshold is 0.85",
    "Klio is the memory product brand",
    "MCP is the primary protocol surface",
    "Migration head is 033",
]


async def _run_recall():
    records = [_make_record(content=c, index=i) for i, c in enumerate(CONTENTS)]
    retrieve_mock = AsyncMock(return_value=records)
    with (
        patch(
            "app.mcp_app.resolve_active_principal",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        return records, await _recall_fn()(_fake_ctx(), "q")


@pytest.mark.asyncio
async def test_recall_payload_is_compressed():
    """The envelope shrinks.

    Asserted against the payload recall would have returned uncompressed —
    the same records through the same serializer — so this measures the
    compressor's effect and not the size of the fixture.
    """
    from app.mcp_app import RECALL_WRITE_NUDGE, _serialize_record

    records, result = await _run_recall()

    uncompressed = {
        "memories": [_serialize_record(r) for r in records],
        "count": len(records),
        "guidance": RECALL_WRITE_NUDGE,
    }
    before = len(json.dumps(uncompressed, default=str))
    after = len(json.dumps(result, default=str))
    assert after < before, f"no saving: {before} -> {after}"

    # Nulls are gone and the genuinely constant fields were hoisted once
    # instead of repeated on every row.
    assert result["common"]["memory_type"] == "fact"
    assert all("memory_type" not in m for m in result["memories"])
    assert '"confidence": null' not in json.dumps(result, default=str)

    # `scope` is deliberately NOT among them. It is the field a consumer is
    # most likely to branch on, and hoisting it left every row reporting
    # `scope: None`.
    assert "scope" not in result["common"]
    assert all(m["scope"] == "org" for m in result["memories"])


@pytest.mark.asyncio
async def test_recall_still_returns_every_content_and_id():
    """Compression must not cost the caller any information."""
    records, result = await _run_recall()

    # Proof that this is asserting against a *compressed* payload: without
    # this the test would pass against a recall that never compressed at all.
    assert "common" in result, "payload was not compressed"

    blob = json.dumps(result, default=str)
    for record in records:
        assert record.content in blob, f"lost content: {record.content}"
        assert str(record.id) in blob, f"lost id: {record.id}"

    # Per-row varying fields stay per-row, and the count is untouched.
    assert result["count"] == len(records)
    assert len(result["memories"]) == len(records)
    assert [m["content"] for m in result["memories"]] == list(CONTENTS)

    # Every field of the original serialization is still reachable, either on
    # the row or in `common`. Nothing was dropped that was not None.
    from app.mcp_app import _serialize_record

    for record, row in zip(records, result["memories"]):
        original = _serialize_record(record)
        merged = {**result.get("common", {}), **row}
        for key, value in original.items():
            if value is None:
                continue
            assert merged[key] == value, f"field {key} changed or was lost"


@pytest.mark.asyncio
async def test_recall_guidance_survives_compression():
    """The write nudge is on the envelope and must not be compressed away."""
    from app.mcp_app import RECALL_WRITE_NUDGE

    _records, result = await _run_recall()
    assert "common" in result, "payload was not compressed"
    assert result["guidance"] == RECALL_WRITE_NUDGE


@pytest.mark.asyncio
async def test_recall_empty_result_is_untouched():
    """No rows means nothing to compress; the shape must not gain a `common`."""
    import app.mcp_app as mcp_app

    retrieve_mock = AsyncMock(return_value=[])
    spy = MagicMock(side_effect=mcp_app.compress_payload)
    with (
        patch(
            "app.mcp_app.resolve_active_principal",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
        patch("app.mcp_app.compress_payload", spy),
    ):
        result = await _recall_fn()(_fake_ctx(), "q")

    # The compressor is on the path even for an empty result — it simply has
    # nothing to do. Without this the assertions below prove nothing.
    spy.assert_called_once()
    assert result["memories"] == []
    assert result["count"] == 0
    assert "common" not in result


@pytest.mark.asyncio
async def test_compression_failure_does_not_break_recall():
    """Fail-open: a compressor that raises must not fail the request."""
    records = [_make_record(content=c, index=i) for i, c in enumerate(CONTENTS)]
    retrieve_mock = AsyncMock(return_value=records)
    with (
        patch(
            "app.mcp_app.resolve_active_principal",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
        patch(
            "app.mcp_app.compress_payload",
            side_effect=RuntimeError("compressor exploded"),
        ),
    ):
        result = await _recall_fn()(_fake_ctx(), "q")

    assert result["count"] == len(records)
    assert [m["content"] for m in result["memories"]] == list(CONTENTS)
