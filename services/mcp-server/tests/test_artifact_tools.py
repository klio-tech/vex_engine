"""Artifact tool bodies, exercised through the FastMCP registry.

Same seams as test_write_tools.py: resolve_principal_plan, SessionLocal, and
brain.write_scoped_memory are patched at the app.mcp_app boundary, so no DB,
no embeddings and no S3 are touched.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.mcp_app import build_mcp_server

MB = 1024 * 1024


def _ctx():
    request = SimpleNamespace(headers={"x-vex-agent": "agent-1"})
    return SimpleNamespace(request_context=SimpleNamespace(request=request))


def _tool(name):
    return build_mcp_server()._tool_manager.get_tool(name).fn


@pytest.fixture
def seams():
    """Patch every external dependency of the artifact tools."""
    plan = SimpleNamespace(
        artifact_max_file_bytes=10 * MB,
        artifact_storage_bytes=100 * MB,
    )
    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("acme", "agent-1", "free", None),
        ),
        patch("app.mcp_app.get_plan_config", return_value=plan),
        patch("app.mcp_app.SessionLocal", MagicMock()) as session_local,
        patch("app.mcp_app.artifacts") as af,
        patch("app.mcp_app.brain") as brain,
        patch("app.mcp_app.resolve_project_from_request", return_value=None),
    ):
        af.INLINE_MAX_BYTES = 65_536
        af.human_size.return_value = "3 B"
        af.card_content.return_value = "[artifact] t — s (document, 3 B)"
        af.org_storage_bytes.return_value = 0
        af.find_active_by_checksum.return_value = None
        af.create_pending.return_value = "new-artifact-id"
        af.activate.return_value = {
            "id": "new-artifact-id",
            "size_bytes": 3,
            "checksum": "abc",
        }
        brain.write_scoped_memory = AsyncMock(return_value={"stored": True})
        brain.ensure_space.return_value = None
        # ``SessionLocal()`` (called once per tool invocation) always yields
        # this same fake session, so tests can inspect raw ``db.execute`` calls
        # (e.g. the supersede/hide-card path) without a real DB.
        yield SimpleNamespace(artifacts=af, brain=brain, db=session_local.return_value)


@pytest.mark.asyncio
async def test_inline_put_activates_and_writes_a_card(seams):
    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="document", content="hey"
    )

    assert result["stored"] is True
    assert result["artifact_id"] == "new-artifact-id"
    assert result["status"] == "active"

    # The card is the discovery mechanism; without it the artifact is invisible.
    seams.brain.write_scoped_memory.assert_awaited_once()
    kwargs = seams.brain.write_scoped_memory.await_args.kwargs
    assert kwargs["memory_type"] == "artifact"
    assert kwargs["metadata"]["artifact_id"] == "new-artifact-id"
    assert kwargs["content"].startswith("[artifact] ")


@pytest.mark.asyncio
async def test_inline_put_stores_text_inline_not_in_object_storage(seams):
    await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="document", content="hey"
    )

    kwargs = seams.artifacts.activate.call_args.kwargs
    assert kwargs["inline_text"] == "hey"
    assert kwargs["storage_uri"] is None
    # Measured server-side, never taken from the caller.
    assert kwargs["size_bytes"] == 3


@pytest.mark.asyncio
async def test_content_over_the_inline_cap_is_rejected_with_a_pointer(seams):
    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="document", content="x" * 65_537
    )

    assert result["stored"] is False
    assert result["error"] == "too_large_for_inline"
    seams.brain.write_scoped_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_over_pool_write_is_rejected_before_any_work(seams):
    seams.artifacts.org_storage_bytes.return_value = 100 * MB

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="document", content="hey"
    )

    assert result["stored"] is False
    assert result["error"] == "storage_quota_exceeded"
    seams.artifacts.create_pending.assert_not_called()
    seams.brain.write_scoped_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_bytes_return_the_existing_artifact(seams):
    seams.artifacts.find_active_by_checksum.return_value = {"id": "old", "version": 1}

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="document", content="hey"
    )

    assert result["artifact_id"] == "old"
    assert result["deduplicated"] is True
    seams.artifacts.create_pending.assert_not_called()


@pytest.mark.asyncio
async def test_auth_failure_returns_a_clean_error_not_an_exception():
    from fastapi import HTTPException

    with patch(
        "app.mcp_app.resolve_principal_plan",
        side_effect=HTTPException(status_code=401, detail="bad key"),
    ):
        result = await _tool("artifact_put")(
            _ctx(), title="t", summary="s", kind="document", content="hey"
        )

    assert result["stored"] is False
    assert result["error"] == "unauthorized"


@pytest.mark.asyncio
async def test_supersedes_marks_old_artifact_superseded_and_hides_its_card(seams):
    result = await _tool("artifact_put")(
        _ctx(),
        title="t",
        summary="s",
        kind="document",
        content="hey",
        supersedes="old-id",
    )

    # The new artifact is still created and carded normally in this call.
    assert result["stored"] is True
    assert result["artifact_id"] == "new-artifact-id"
    seams.brain.write_scoped_memory.assert_awaited_once()

    # The old artifact is superseded using the resolved org_id — never a
    # caller-supplied one.
    seams.artifacts.mark_superseded.assert_called_once()
    supersede_kwargs = seams.artifacts.mark_superseded.call_args.kwargs
    assert supersede_kwargs["org_id"] == "acme"
    assert supersede_kwargs["artifact_id"] == "old-id"

    # Its card is hidden from recall (raw UPDATE against session_memories,
    # since there is no shared.artifacts helper for hiding a card).
    hide_call = next(
        call
        for call in seams.db.execute.call_args_list
        if "recall_hidden" in str(call.args[0])
    )
    params = hide_call.args[1]
    assert params["org_id"] == "acme"
    assert params["artifact_id"] == "old-id"


@pytest.mark.asyncio
async def test_card_write_failure_retracts_the_artifact_and_returns_an_error(seams):
    """A card-write failure must not leave a paid-for, invisible artifact.

    ``artifacts.create_pending`` / ``activate`` each commit independently and
    only then does the card get written, so a failure here (e.g. an embedding
    service hiccup) would otherwise leave the artifact ``status='active'`` with
    no discovery card: billed, listed by any future artifacts UI, and
    permanently invisible to recall. The fix compensates by retracting the
    just-activated artifact and surfacing an error rather than a false
    success.
    """
    seams.brain.write_scoped_memory.side_effect = RuntimeError("embedding service down")

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="document", content="hey"
    )

    assert result["stored"] is False
    assert result["error"] == "card_write_failed"

    seams.artifacts.retract.assert_called_once()
    retract_kwargs = seams.artifacts.retract.call_args.kwargs
    assert retract_kwargs["org_id"] == "acme"
    assert retract_kwargs["artifact_id"] == "new-artifact-id"
