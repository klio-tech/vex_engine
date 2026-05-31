"""MCP write/recall enforce memory quota."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.usage import MemoryQuotaExceeded

from app import mcp_app


def _ctx():
    ctx = MagicMock()
    ctx.request_context.request = MagicMock()
    return ctx


@pytest.mark.asyncio
async def test_write_blocked_returns_error_dict():
    with patch.object(
        mcp_app, "resolve_principal_plan", return_value=("org-1", "a1", "free")
    ), patch.object(
        mcp_app,
        "check_memory_quota",
        side_effect=MemoryQuotaExceeded(metric="memory", limit=1000, plan_used=1000),
    ):
        result = await mcp_app._perform_write(
            _ctx(), content="x", memory_type="memory"
        )
    assert result["stored"] is False
    assert result["error"] == "quota_exceeded"


@pytest.mark.asyncio
async def test_write_records_only_when_stored():
    with patch.object(
        mcp_app, "resolve_principal_plan", return_value=("org-1", "a1", "free")
    ), patch.object(mcp_app, "check_memory_quota", return_value=False), patch.object(
        mcp_app, "record_usage"
    ) as rec, patch.object(
        mcp_app.brain,
        "write_scoped_memory",
        new=AsyncMock(return_value={"stored": True, "id": "m1"}),
    ), patch.object(mcp_app, "SessionLocal", return_value=MagicMock()):
        await mcp_app._perform_write(_ctx(), content="x", memory_type="memory")
    rec.assert_called_once()


@pytest.mark.asyncio
async def test_write_skips_record_when_deduped():
    with patch.object(
        mcp_app, "resolve_principal_plan", return_value=("org-1", "a1", "free")
    ), patch.object(mcp_app, "check_memory_quota", return_value=False), patch.object(
        mcp_app, "record_usage"
    ) as rec, patch.object(
        mcp_app.brain,
        "write_scoped_memory",
        new=AsyncMock(return_value={"stored": False, "deduplicated": True}),
    ), patch.object(mcp_app, "SessionLocal", return_value=MagicMock()):
        await mcp_app._perform_write(_ctx(), content="x", memory_type="memory")
    rec.assert_not_called()


@pytest.mark.asyncio
async def test_recall_blocked_returns_error_dict():
    with patch.object(
        mcp_app, "resolve_principal_plan", return_value=("org-1", "a1", "free")
    ), patch.object(
        mcp_app,
        "check_memory_quota",
        side_effect=MemoryQuotaExceeded(metric="recall", limit=10000, plan_used=10000),
    ):
        result = await mcp_app._perform_recall(
            _ctx(), query="q", scope="org", space=None, project=None
        )
    assert result["error"] == "quota_exceeded"
