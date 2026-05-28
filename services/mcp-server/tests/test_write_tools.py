"""Tests for the Klio write-tool wiring (remember/observe/plan/decide/note).

These tests exercise the *tool bodies* in :mod:`app.mcp_app` without a live
DB or real embeddings. They reach the registered tool callables via the
FastMCP tool registry (``server._tool_manager.get_tool(name).fn``) and invoke
them with a hand-built fake :class:`Context` whose
``.request_context.request.headers`` is a dict-like supporting ``.get(...)``
— the only surface the tool bodies touch on the request.

The seams patched at the ``app.mcp_app`` module boundary:

* ``resolve_principal`` → returns a fixed ``(org_id, agent_id)`` (or raises
  to drive the auth-failure path).
* ``brain.write_scoped_memory`` → an ``AsyncMock`` that captures kwargs and
  returns a canned result, so no DB/embedding is touched.
* ``SessionLocal`` → a ``MagicMock`` so ``SessionLocal()`` yields a fake
  session that is ``.close()``-d in the ``finally``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import app.mcp_app as mcp_app
from app.mcp_app import build_mcp_server

# Tool name -> expected memory_type the body must map it to.
_TOOL_MEMORY_TYPE = {
    "remember": "memory",
    "observe": "observation",
    "plan": "plan",
    "decide": "decision",
    "note": "note",
}


def _fake_ctx(headers: dict | None = None):
    """Build a minimal stand-in for the FastMCP Context.

    The tool bodies only read ``ctx.request_context.request`` and pass it to
    ``resolve_principal`` / the session-id helper, which in turn read
    ``request.headers.get(...)``. A nested ``SimpleNamespace`` with a real
    dict for ``headers`` satisfies both without a real ASGI scope.
    """
    request = SimpleNamespace(headers=headers if headers is not None else {})
    request_context = SimpleNamespace(request=request)
    return SimpleNamespace(request_context=request_context)


def _get_tool_fn(name: str):
    """Return the registered tool's underlying callable from a fresh server."""
    server = build_mcp_server()
    tool = server._tool_manager.get_tool(name)
    assert tool is not None, f"tool {name!r} not registered"
    return tool.fn


def _canned_result(memory_type: str) -> dict:
    return {
        "stored": True,
        "deduplicated": False,
        "id": "00000000-0000-0000-0000-000000000001",
        "memory_type": memory_type,
        "scope": "org",
        "space_id": None,
        "project_id": None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name,expected_type", list(_TOOL_MEMORY_TYPE.items()))
async def test_write_tool_maps_memory_type_and_scopes_org(tool_name, expected_type):
    """Each write tool resolves the principal, opens a session, and calls
    write_scoped_memory with the correct memory_type; result scope is org."""
    fn = _get_tool_fn(tool_name)
    write_mock = AsyncMock(return_value=_canned_result(expected_type))
    session = MagicMock()
    session_local = MagicMock(return_value=session)

    with (
        patch(
            "app.mcp_app.resolve_principal",
            return_value=("org-x", "agent-y"),
        ),
        patch("app.mcp_app.SessionLocal", session_local),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
    ):
        result = await fn(_fake_ctx({"mcp-session-id": "conn-1"}), "some content")

    write_mock.assert_awaited_once()
    kwargs = write_mock.await_args.kwargs
    assert kwargs["org_id"] == "org-x"
    assert kwargs["agent_id"] == "agent-y"
    assert kwargs["memory_type"] == expected_type
    assert kwargs["content"] == "some content"
    # Stable per-connection session id derived from the mcp-session-id header.
    assert kwargs["session_id"] == "klio-brain:conn-1"
    assert result["scope"] == "org"
    # The DB session is always closed.
    session.close.assert_called_once()


@pytest.mark.asyncio
async def test_decide_with_rationale_sets_metadata():
    """decide folds rationale into metadata while preserving the source tag."""
    fn = _get_tool_fn("decide")
    write_mock = AsyncMock(return_value=_canned_result("decision"))

    with (
        patch("app.mcp_app.resolve_principal", return_value=("org-x", "agent-y")),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
    ):
        await fn(_fake_ctx(), "ship it", rationale="team aligned")

    kwargs = write_mock.await_args.kwargs
    assert kwargs["memory_type"] == "decision"
    # rationale is added alongside the provenance tag, not in place of it,
    # so decide stays consistent with the other write tools.
    assert kwargs["metadata"] == {"source": "mcp", "rationale": "team aligned"}


@pytest.mark.asyncio
async def test_decide_without_rationale_omits_rationale_metadata():
    """decide without a rationale does not inject a rationale into metadata."""
    fn = _get_tool_fn("decide")
    write_mock = AsyncMock(return_value=_canned_result("decision"))

    with (
        patch("app.mcp_app.resolve_principal", return_value=("org-x", "agent-y")),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
    ):
        await fn(_fake_ctx(), "ship it")

    kwargs = write_mock.await_args.kwargs
    meta = kwargs.get("metadata")
    assert not (meta and "rationale" in meta)


@pytest.mark.asyncio
async def test_session_id_falls_back_to_principal_without_header():
    """Absent the mcp-session-id header, session_id falls back to org:agent."""
    fn = _get_tool_fn("remember")
    write_mock = AsyncMock(return_value=_canned_result("memory"))

    with (
        patch("app.mcp_app.resolve_principal", return_value=("org-x", "agent-y")),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
    ):
        await fn(_fake_ctx({}), "a durable fact")

    assert write_mock.await_args.kwargs["session_id"] == "klio-brain:org-x:agent-y"


@pytest.mark.asyncio
async def test_write_tool_no_request_context_returns_unauthorized():
    """When the transport attaches no request, the tool denies the write
    rather than guessing a principal."""
    fn = _get_tool_fn("remember")
    write_mock = AsyncMock()
    session_local = MagicMock()
    ctx = SimpleNamespace(request_context=SimpleNamespace(request=None))

    with (
        patch("app.mcp_app.resolve_principal") as resolve_mock,
        patch("app.mcp_app.SessionLocal", session_local),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
    ):
        result = await fn(ctx, "content")

    assert result["stored"] is False
    assert result["error"] == "unauthorized"
    resolve_mock.assert_not_called()
    write_mock.assert_not_called()
    session_local.assert_not_called()


@pytest.mark.asyncio
async def test_write_tool_auth_failure_returns_clean_error():
    """When resolve_principal raises HTTPException, the tool returns a clean
    unauthorized dict and never calls write_scoped_memory."""
    fn = _get_tool_fn("remember")
    write_mock = AsyncMock()
    session = MagicMock()
    session_local = MagicMock(return_value=session)

    with (
        patch(
            "app.mcp_app.resolve_principal",
            side_effect=HTTPException(
                status_code=401, detail="Missing X-Vex-Key header"
            ),
        ),
        patch("app.mcp_app.SessionLocal", session_local),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
    ):
        result = await fn(_fake_ctx({}), "content")

    assert result["stored"] is False
    assert result["error"] == "unauthorized"
    assert "Missing X-Vex-Key header" in result["detail"]
    write_mock.assert_not_called()
    # No DB session should be opened when auth fails before session creation.
    session_local.assert_not_called()
