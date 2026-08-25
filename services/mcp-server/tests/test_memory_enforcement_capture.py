"""REST capture endpoints enforce memory quota → HTTP 402."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from shared.usage import MemoryQuotaExceeded

from app import capture


def _request():
    """A stand-in for the Starlette request the handler now takes.

    Phase 2 gave the capture handlers a ``Request`` so they can resolve the
    caller's identity (``resolve_user_id``) and route the write to the right
    scope. These tests are about quota, not identity, so the request only has
    to exist and carry no OAuth principal.
    """
    req = MagicMock()
    req.state = SimpleNamespace()
    req.headers = {}
    return req


@pytest.mark.asyncio
async def test_capture_event_quota_exceeded_raises_402():
    req = MagicMock()
    req.memory_type = "observation"
    req.metadata = None
    info = MagicMock(plan="free", plan_overrides=None)
    with (
        patch.object(
            capture,
            "check_memory_quota",
            side_effect=MemoryQuotaExceeded(
                metric="memory", limit=1000, plan_used=1000, suggested_plan="starter"
            ),
        ),
        patch.object(capture, "SessionLocal", return_value=MagicMock()),
        patch.object(capture, "_ensure_project_id", return_value=None),
        patch.object(
            capture,
            "resolve_active_principal",
            return_value=("org-1", "a1", "free", None),
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            await capture.capture_event(_request(), req, key_info=info)
    assert exc.value.status_code == 402
    assert exc.value.detail["suggested_plan"] == "starter"


@pytest.mark.asyncio
async def test_capture_event_records_when_stored():
    req = MagicMock()
    req.memory_type = "observation"
    req.metadata = None
    info = MagicMock(plan="free", plan_overrides=None)
    with (
        patch.object(capture, "check_memory_quota", return_value=False),
        patch.object(capture, "record_usage") as rec,
        patch.object(
            capture.brain,
            "write_scoped_memory",
            new=AsyncMock(return_value={"stored": True, "id": "m1"}),
        ),
        patch.object(capture, "SessionLocal", return_value=MagicMock()),
        patch.object(capture, "_ensure_project_id", return_value=None),
        patch.object(
            capture,
            "resolve_active_principal",
            return_value=("org-1", "a1", "free", None),
        ),
    ):
        await capture.capture_event(_request(), req, key_info=info)
    rec.assert_called_once()
