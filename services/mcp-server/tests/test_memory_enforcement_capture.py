"""REST capture endpoints enforce memory quota → HTTP 402."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from shared.usage import MemoryQuotaExceeded

from app import capture


@pytest.mark.asyncio
async def test_capture_event_quota_exceeded_raises_402():
    req = MagicMock()
    req.memory_type = "observation"
    req.metadata = None
    info = MagicMock(plan="free")
    with patch.object(
        capture, "check_memory_quota",
        side_effect=MemoryQuotaExceeded(metric="memory", limit=1000, plan_used=1000),
    ), patch.object(capture, "SessionLocal", return_value=MagicMock()), patch.object(
        capture, "_ensure_project_id", return_value=None
    ):
        with pytest.raises(HTTPException) as exc:
            await capture.capture_event(req, principal=("org-1", "a1"), key_info=info)
    assert exc.value.status_code == 402


@pytest.mark.asyncio
async def test_capture_event_records_when_stored():
    req = MagicMock()
    req.memory_type = "observation"
    req.metadata = None
    info = MagicMock(plan="free")
    with patch.object(capture, "check_memory_quota", return_value=False), patch.object(
        capture, "record_usage"
    ) as rec, patch.object(
        capture.brain, "write_scoped_memory",
        new=AsyncMock(return_value={"stored": True, "id": "m1"}),
    ), patch.object(capture, "SessionLocal", return_value=MagicMock()), patch.object(
        capture, "_ensure_project_id", return_value=None
    ):
        await capture.capture_event(req, principal=("org-1", "a1"), key_info=info)
    rec.assert_called_once()
