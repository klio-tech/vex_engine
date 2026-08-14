"""Tests for the Klio ``claim`` / ``release`` tool wiring and the
``active_claims`` injection into ``recall`` responses (Build 4).

Mirrors the mock style of ``test_write_tools.py`` / ``test_recall_tool.py``:
tool callables are reached via the FastMCP registry
(``server._tool_manager.get_tool(name).fn``) and invoked with a hand-built
fake :class:`Context`. The DB / embedding / storage seams are patched at the
``app.mcp_app`` module boundary so no real Postgres or embeddings are
touched:

* ``resolve_principal`` -> a fixed ``(org_id, agent_id)`` (or raises to
  drive the auth-failure path).
* ``SessionLocal`` -> a ``MagicMock`` whose instance is ``.close()``-d.
* ``resolve_project_from_request`` -> controls the resolved project filter.
* ``claims.create_claim`` / ``claims.release_claim`` / ``claims.live_claims``
  -> patched directly so the storage primitives themselves (covered by
  ``services/shared/tests/test_claims.py``) are never re-tested here.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.mcp_app import build_mcp_server


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


def _get_tool_fn(name: str):
    server = build_mcp_server()
    tool = server._tool_manager.get_tool(name)
    assert tool is not None, f"tool {name!r} not registered"
    return tool.fn


# ---------------------------------------------------------------------------
# claim
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claim_happy_path_forwards_principal_and_project():
    fn = _get_tool_fn("claim")
    session = MagicMock()
    session_local = MagicMock(return_value=session)
    create_mock = AsyncMock(
        return_value={"claim_id": "c-1", "collisions": [{"id": "c-old"}]}
    )

    with (
        patch("app.mcp_app.resolve_principal", return_value=("org-x", "agent-y")),
        patch("app.mcp_app.resolve_request_user_id", return_value="user-1"),
        patch("app.mcp_app.SessionLocal", session_local),
        patch("app.mcp_app.resolve_project_from_request", return_value="proj-7"),
        patch("app.mcp_app.claims.create_claim", create_mock),
    ):
        result = await fn(_fake_ctx(), "fix login token expiry")

    assert result == {"claim_id": "c-1", "collisions": [{"id": "c-old"}]}
    create_mock.assert_awaited_once()
    _, kwargs = create_mock.call_args
    assert kwargs["org_id"] == "org-x"
    assert kwargs["agent_id"] == "agent-y"
    assert kwargs["user_id"] == "user-1"
    assert kwargs["project_id"] == "proj-7"
    assert kwargs["description"] == "fix login token expiry"
    assert kwargs["ttl_minutes"] == 15
    session.close.assert_called_once()


@pytest.mark.asyncio
async def test_claim_forwards_custom_ttl():
    fn = _get_tool_fn("claim")
    session_local = MagicMock(return_value=MagicMock())
    create_mock = AsyncMock(return_value={"claim_id": "c-1", "collisions": []})

    with (
        patch("app.mcp_app.resolve_principal", return_value=("org-x", "agent-y")),
        patch("app.mcp_app.resolve_request_user_id", return_value=None),
        patch("app.mcp_app.SessionLocal", session_local),
        patch("app.mcp_app.resolve_project_from_request", return_value=None),
        patch("app.mcp_app.claims.create_claim", create_mock),
    ):
        await fn(_fake_ctx(), "do the thing", 30)

    assert create_mock.call_args.kwargs["ttl_minutes"] == 30


@pytest.mark.asyncio
async def test_claim_auth_failure_returns_clean_error():
    fn = _get_tool_fn("claim")
    with patch(
        "app.mcp_app.resolve_principal",
        side_effect=HTTPException(status_code=401, detail="bad key"),
    ):
        result = await fn(_fake_ctx(), "fix login token expiry")

    assert result["error"] == "unauthorized"
    assert result["claim_id"] is None


@pytest.mark.asyncio
async def test_claim_no_request_context_returns_unauthorized():
    fn = _get_tool_fn("claim")
    ctx = SimpleNamespace(request_context=SimpleNamespace(request=None))
    result = await fn(ctx, "fix login token expiry")
    assert result["error"] == "unauthorized"
    assert result["claim_id"] is None


@pytest.mark.asyncio
async def test_claim_storage_failure_returns_clean_error():
    fn = _get_tool_fn("claim")
    session_local = MagicMock(return_value=MagicMock())

    with (
        patch("app.mcp_app.resolve_principal", return_value=("org-x", "agent-y")),
        patch("app.mcp_app.resolve_request_user_id", return_value=None),
        patch("app.mcp_app.SessionLocal", session_local),
        patch("app.mcp_app.resolve_project_from_request", return_value=None),
        patch(
            "app.mcp_app.claims.create_claim",
            AsyncMock(side_effect=RuntimeError("db down")),
        ),
    ):
        result = await fn(_fake_ctx(), "fix login token expiry")

    assert result == {"claim_id": None, "error": "claim_failed"}


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_release_happy_path_forwards_principal_and_claim_id():
    fn = _get_tool_fn("release")
    session = MagicMock()
    session_local = MagicMock(return_value=session)

    with (
        patch("app.mcp_app.resolve_principal", return_value=("org-x", "agent-y")),
        patch("app.mcp_app.SessionLocal", session_local),
        patch("app.mcp_app.claims.release_claim", return_value=True) as release_mock,
    ):
        result = await fn(_fake_ctx(), "c-1")

    assert result == {"released": True}
    release_mock.assert_called_once_with(
        session, org_id="org-x", claim_id="c-1", agent_id="agent-y"
    )
    session.close.assert_called_once()


@pytest.mark.asyncio
async def test_release_not_owned_returns_false():
    fn = _get_tool_fn("release")
    session_local = MagicMock(return_value=MagicMock())

    with (
        patch("app.mcp_app.resolve_principal", return_value=("org-x", "agent-y")),
        patch("app.mcp_app.SessionLocal", session_local),
        patch("app.mcp_app.claims.release_claim", return_value=False),
    ):
        result = await fn(_fake_ctx(), "c-1")

    assert result == {"released": False}


@pytest.mark.asyncio
async def test_release_auth_failure_returns_clean_error():
    fn = _get_tool_fn("release")
    with patch(
        "app.mcp_app.resolve_principal",
        side_effect=HTTPException(status_code=401, detail="bad key"),
    ):
        result = await fn(_fake_ctx(), "c-1")

    assert result["error"] == "unauthorized"
    assert result["released"] is False


@pytest.mark.asyncio
async def test_release_no_request_context_returns_unauthorized():
    fn = _get_tool_fn("release")
    ctx = SimpleNamespace(request_context=SimpleNamespace(request=None))
    result = await fn(ctx, "c-1")
    assert result["error"] == "unauthorized"
    assert result["released"] is False


# ---------------------------------------------------------------------------
# active_claims injected into recall
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_injects_active_claims():
    server = build_mcp_server()
    recall_fn = server._tool_manager.get_tool("recall").fn

    session = MagicMock()
    session_local = MagicMock(return_value=session)
    live_claims_mock = MagicMock(
        return_value=[{"id": "c-1", "description": "fix login", "agent_id": "B"}]
    )

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", session_local),
        patch("app.mcp_app.retrieve_memories_scoped", AsyncMock(return_value=[])),
        patch("app.mcp_app.resolve_project_from_request", return_value=None),
        patch("app.mcp_app.claims.live_claims", live_claims_mock),
    ):
        result = await recall_fn(_fake_ctx(), "what did we decide")

    assert result["active_claims"] == [
        {"id": "c-1", "description": "fix login", "agent_id": "B"}
    ]
    live_claims_mock.assert_called_once_with(session, org_id="org-x", project_id=None)


@pytest.mark.asyncio
async def test_recall_active_claims_failure_does_not_break_recall():
    """live_claims raising must never be the reason a recall fails -- it
    degrades to an empty active_claims list, same as compress_payload's own
    fail-open contract just above it."""
    server = build_mcp_server()
    recall_fn = server._tool_manager.get_tool("recall").fn

    session_local = MagicMock(return_value=MagicMock())

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", session_local),
        patch("app.mcp_app.retrieve_memories_scoped", AsyncMock(return_value=[])),
        patch("app.mcp_app.resolve_project_from_request", return_value=None),
        patch(
            "app.mcp_app.claims.live_claims",
            MagicMock(side_effect=RuntimeError("db down")),
        ),
    ):
        result = await recall_fn(_fake_ctx(), "what did we decide")

    assert result["active_claims"] == []
    assert result["count"] == 0
