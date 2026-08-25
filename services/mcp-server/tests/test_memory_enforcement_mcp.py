"""MCP write/recall enforce memory quota."""

from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from shared.usage import MemoryQuotaExceeded

from app import mcp_app


def _ctx():
    ctx = MagicMock()
    ctx.request_context.request = MagicMock()
    return ctx


@pytest.mark.asyncio
async def test_write_blocked_returns_error_dict():
    with (
        patch.object(
            mcp_app,
            "resolve_active_principal",
            return_value=("org-1", "a1", "free", None),
        ),
        patch.object(
            mcp_app,
            "check_memory_quota",
            side_effect=MemoryQuotaExceeded(
                metric="memory", limit=1000, plan_used=1000, suggested_plan="starter"
            ),
        ),
    ):
        result = await mcp_app._perform_write(_ctx(), content="x", memory_type="memory")
    assert result["stored"] is False
    assert result["error"] == "quota_exceeded"
    assert "detail" in result
    assert "limit" in result["detail"].lower()
    assert result["suggested_plan"] == "starter"


@pytest.mark.asyncio
async def test_write_records_only_when_stored():
    with (
        patch.object(
            mcp_app,
            "resolve_active_principal",
            return_value=("org-1", "a1", "free", None),
        ),
        patch.object(mcp_app, "check_memory_quota", return_value=False),
        patch.object(mcp_app, "record_usage") as rec,
        patch.object(
            mcp_app.brain,
            "write_scoped_memory",
            new=AsyncMock(return_value={"stored": True, "id": "m1"}),
        ),
        patch.object(mcp_app, "SessionLocal", return_value=MagicMock()),
    ):
        await mcp_app._perform_write(_ctx(), content="x", memory_type="memory")
    rec.assert_called_once_with(ANY, "org-1", "memory", over_cap=False)


@pytest.mark.asyncio
async def test_write_skips_record_when_deduped():
    with (
        patch.object(
            mcp_app,
            "resolve_active_principal",
            return_value=("org-1", "a1", "free", None),
        ),
        patch.object(mcp_app, "check_memory_quota", return_value=False),
        patch.object(mcp_app, "record_usage") as rec,
        patch.object(
            mcp_app.brain,
            "write_scoped_memory",
            new=AsyncMock(return_value={"stored": False, "deduplicated": True}),
        ),
        patch.object(mcp_app, "SessionLocal", return_value=MagicMock()),
    ):
        await mcp_app._perform_write(_ctx(), content="x", memory_type="memory")
    rec.assert_not_called()


@pytest.mark.asyncio
async def test_recall_blocked_returns_error_dict():
    with (
        patch.object(
            mcp_app,
            "resolve_active_principal",
            return_value=("org-1", "a1", "free", None),
        ),
        patch.object(
            mcp_app,
            "check_memory_quota",
            side_effect=MemoryQuotaExceeded(
                metric="recall", limit=10000, plan_used=10000, suggested_plan="starter"
            ),
        ),
    ):
        result = await mcp_app._perform_recall(
            _ctx(), query="q", scope="org", space=None, project=None
        )
    assert result["error"] == "quota_exceeded"
    assert "detail" in result
    assert "limit" in result["detail"].lower()
    assert result["suggested_plan"] == "starter"


@pytest.mark.asyncio
async def test_recall_records_usage_on_success():
    with (
        patch.object(
            mcp_app,
            "resolve_active_principal",
            return_value=("org-1", "a1", "free", None),
        ),
        patch.object(mcp_app, "check_memory_quota", return_value=False),
        patch.object(mcp_app, "SessionLocal", return_value=MagicMock()),
        patch.object(
            mcp_app, "retrieve_memories_scoped", new=AsyncMock(return_value=[])
        ),
        patch.object(mcp_app, "log_recall_event"),
        patch.object(mcp_app, "record_usage") as rec,
    ):
        await mcp_app._perform_recall(
            _ctx(), query="q", scope="org", space=None, project=None
        )
    rec.assert_called_once_with(ANY, "org-1", "recall", over_cap=False)
