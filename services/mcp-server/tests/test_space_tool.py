"""Tests for the Klio ``space`` management-tool wiring.

Mirrors the mock style of ``tests/test_recall_tool.py``: the tool callable is
reached via the FastMCP registry (``server._tool_manager.get_tool(name).fn``)
and invoked with a hand-built fake :class:`Context`. The DB seams are patched
at the ``app.mcp_app`` module boundary so no real Postgres is touched:

* ``resolve_principal`` → a fixed ``(org_id, agent_id)`` (or raises to drive
  the auth-failure path).
* ``SessionLocal`` → a ``MagicMock`` whose instance is ``.close()``-d.
* ``brain.list_spaces`` / ``brain.get_space`` / ``brain.ensure_space`` →
  control listing, lookup, and create without a DB.

Tenant-isolation invariant asserted throughout: the ``org_id`` reaching every
brain call is the one from ``resolve_principal`` — never a tool argument.
"""

import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import app.mcp_app as mcp_app
from app.mcp_app import build_mcp_server


def _fake_ctx(headers: dict | None = None):
    """Minimal stand-in for the FastMCP Context (only request.headers is read)."""
    request = SimpleNamespace(headers=headers if headers is not None else {})
    request_context = SimpleNamespace(request=request)
    return SimpleNamespace(request_context=request_context)


def _space_fn():
    """Return the registered ``space`` tool's underlying callable."""
    server = build_mcp_server()
    tool = server._tool_manager.get_tool("space")
    assert tool is not None, "tool 'space' not registered"
    return tool.fn


def _fake_space(*, id=None, name="Research", slug="research", embedding_model=None):
    """Build a brain-layer space dict like list_spaces/get_space return.

    Uses a real ``uuid`` string for id and a real ``datetime`` for created_at so
    the serialization (and its json-safety) is exercised against the genuine
    types the brain helpers hand back.
    """
    return {
        "id": id if id is not None else str(uuid.uuid4()),
        "name": name,
        "slug": slug,
        "embedding_model": embedding_model,
        "created_at": datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc),
    }


def _patches(*, resolve=("org-x", "agent-y", "free", None), session=None):
    """Return the common (resolve_principal, SessionLocal) patch context pair."""
    session = session if session is not None else MagicMock()
    return (
        patch("app.mcp_app.resolve_active_principal", return_value=resolve),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=session)),
    )


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_space_list_serializes_and_is_json_safe():
    """list returns count==2, each serialized space has created_at as str|None,
    and the whole payload survives stdlib json (locks json-safety)."""
    fn = _space_fn()
    spaces = [_fake_space(slug="research"), _fake_space(slug="personal")]
    list_mock = MagicMock(return_value=spaces)
    resolve_p, session_p = _patches()

    with resolve_p, session_p, patch.object(mcp_app.brain, "list_spaces", list_mock):
        result = await fn(_fake_ctx(), "list")

    assert result["action"] == "list"
    assert result["count"] == 2
    assert len(result["spaces"]) == 2
    for s in result["spaces"]:
        assert isinstance(s["id"], str)
        assert isinstance(s["created_at"], (str, type(None)))
    # The whole payload must survive stdlib json (not just FastMCP coercion).
    json.dumps(result)


@pytest.mark.asyncio
async def test_space_list_org_isolation_uses_resolved_principal():
    """The org_id forwarded to list_spaces is exactly the one from
    resolve_principal — never anything caller-supplied."""
    fn = _space_fn()
    list_mock = MagicMock(return_value=[])
    resolve_p, session_p = _patches(resolve=("org-secure", "agent-z", "free", None))

    with resolve_p, session_p, patch.object(mcp_app.brain, "list_spaces", list_mock):
        await fn(_fake_ctx(), "list")

    list_mock.assert_called_once()
    assert list_mock.call_args.kwargs["org_id"] == "org-secure"


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_space_create_happy_calls_ensure_with_resolved_org():
    """create with name+slug calls ensure_space with the resolved org_id +
    name + slug, and returns created==True with the new id."""
    fn = _space_fn()
    ensure_mock = MagicMock(return_value="sp-1")
    resolve_p, session_p = _patches(resolve=("org-x", "agent-y", "free", None))

    with resolve_p, session_p, patch.object(mcp_app.brain, "ensure_space", ensure_mock):
        result = await fn(_fake_ctx(), "create", name="Research", slug="research")

    ensure_mock.assert_called_once()
    kwargs = ensure_mock.call_args.kwargs
    assert kwargs["org_id"] == "org-x"
    assert kwargs["name"] == "Research"
    assert kwargs["slug"] == "research"
    assert result == {
        "action": "create",
        "created": True,
        "id": "sp-1",
        "slug": "research",
    }


@pytest.mark.asyncio
async def test_space_create_missing_name_does_not_call_ensure():
    """create without name returns the required-fields error and never touches
    ensure_space."""
    fn = _space_fn()
    ensure_mock = MagicMock()
    resolve_p, session_p = _patches()

    with resolve_p, session_p, patch.object(mcp_app.brain, "ensure_space", ensure_mock):
        result = await fn(_fake_ctx(), "create", name=None, slug="research")

    assert result == {"action": "create", "error": "name and slug are required"}
    ensure_mock.assert_not_called()


@pytest.mark.asyncio
async def test_space_create_missing_slug_does_not_call_ensure():
    """create without slug returns the required-fields error and never touches
    ensure_space."""
    fn = _space_fn()
    ensure_mock = MagicMock()
    resolve_p, session_p = _patches()

    with resolve_p, session_p, patch.object(mcp_app.brain, "ensure_space", ensure_mock):
        result = await fn(_fake_ctx(), "create", name="Research", slug=None)

    assert result == {"action": "create", "error": "name and slug are required"}
    ensure_mock.assert_not_called()


@pytest.mark.asyncio
async def test_space_create_empty_strings_do_not_call_ensure():
    """Empty (whitespace-only) name/slug are treated as missing."""
    fn = _space_fn()
    ensure_mock = MagicMock()
    resolve_p, session_p = _patches()

    with resolve_p, session_p, patch.object(mcp_app.brain, "ensure_space", ensure_mock):
        result = await fn(_fake_ctx(), "create", name="   ", slug="research")

    assert result == {"action": "create", "error": "name and slug are required"}
    ensure_mock.assert_not_called()


@pytest.mark.asyncio
async def test_space_create_then_list_shows_it():
    """The plan's named create->list test: after a create (ensure_space->"sp-9"),
    a subsequent list whose list_spaces returns that space includes it.

    NOTE: this drives the two tool calls with mocked brain helpers — a real-DB
    create->list round-trip (proving the row actually persists) is deferred to
    the F2 e2e task.
    """
    fn = _space_fn()
    ensure_mock = MagicMock(return_value="sp-9")
    resolve_p, session_p = _patches()

    with resolve_p, session_p, patch.object(mcp_app.brain, "ensure_space", ensure_mock):
        create_result = await fn(_fake_ctx(), "create", name="Ledger", slug="ledger")

    assert create_result["created"] is True
    assert create_result["id"] == "sp-9"

    # Now the same space surfaces in a subsequent list call.
    listed = [_fake_space(id="sp-9", name="Ledger", slug="ledger")]
    list_mock = MagicMock(return_value=listed)
    resolve_p2, session_p2 = _patches()

    with resolve_p2, session_p2, patch.object(mcp_app.brain, "list_spaces", list_mock):
        list_result = await fn(_fake_ctx(), "list")

    slugs = {s["slug"] for s in list_result["spaces"]}
    ids = {s["id"] for s in list_result["spaces"]}
    assert "ledger" in slugs
    assert "sp-9" in ids


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_space_info_found_returns_serialized_space():
    """info on a known slug returns the serialized space; payload is json-safe."""
    fn = _space_fn()
    space = _fake_space(id="sp-7", slug="research")
    get_mock = MagicMock(return_value=space)
    resolve_p, session_p = _patches()

    with resolve_p, session_p, patch.object(mcp_app.brain, "get_space", get_mock):
        result = await fn(_fake_ctx(), "info", slug="research")

    assert result["action"] == "info"
    assert result["space"]["id"] == "sp-7"
    assert result["space"]["slug"] == "research"
    assert isinstance(result["space"]["created_at"], (str, type(None)))
    json.dumps(result)
    # Org-scoped lookup with the resolved principal.
    assert get_mock.call_args.kwargs["org_id"] == "org-x"
    assert get_mock.call_args.kwargs["slug"] == "research"


@pytest.mark.asyncio
async def test_space_info_unknown_returns_warning():
    """info on an unknown slug returns a warning, no space."""
    fn = _space_fn()
    get_mock = MagicMock(return_value=None)
    resolve_p, session_p = _patches()

    with resolve_p, session_p, patch.object(mcp_app.brain, "get_space", get_mock):
        result = await fn(_fake_ctx(), "info", slug="ghost")

    assert result == {"action": "info", "warning": "unknown space: ghost"}


@pytest.mark.asyncio
async def test_space_info_missing_slug_errors_without_lookup():
    """info without a slug returns the required error and never looks up."""
    fn = _space_fn()
    get_mock = MagicMock()
    resolve_p, session_p = _patches()

    with resolve_p, session_p, patch.object(mcp_app.brain, "get_space", get_mock):
        result = await fn(_fake_ctx(), "info", slug=None)

    assert result == {"action": "info", "error": "slug is required"}
    get_mock.assert_not_called()


# ---------------------------------------------------------------------------
# switch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_space_switch_found_returns_space_and_note():
    """switch on a known slug validates existence and returns a note instructing
    the caller to pass space=<slug> to recall (the server holds no active-space
    state)."""
    fn = _space_fn()
    space = _fake_space(id="sp-3", slug="research")
    get_mock = MagicMock(return_value=space)
    resolve_p, session_p = _patches()

    with resolve_p, session_p, patch.object(mcp_app.brain, "get_space", get_mock):
        result = await fn(_fake_ctx(), "switch", slug="research")

    assert result["action"] == "switch"
    assert result["space"]["slug"] == "research"
    assert "space='research'" in result["note"]
    assert "recall" in result["note"]
    json.dumps(result)


@pytest.mark.asyncio
async def test_space_switch_unknown_returns_warning():
    """switch on an unknown slug returns a warning, no note/space."""
    fn = _space_fn()
    get_mock = MagicMock(return_value=None)
    resolve_p, session_p = _patches()

    with resolve_p, session_p, patch.object(mcp_app.brain, "get_space", get_mock):
        result = await fn(_fake_ctx(), "switch", slug="ghost")

    assert result == {"action": "switch", "warning": "unknown space: ghost"}


@pytest.mark.asyncio
async def test_space_switch_missing_slug_errors_without_lookup():
    """switch without a slug returns the required error and never looks up."""
    fn = _space_fn()
    get_mock = MagicMock()
    resolve_p, session_p = _patches()

    with resolve_p, session_p, patch.object(mcp_app.brain, "get_space", get_mock):
        result = await fn(_fake_ctx(), "switch", slug=None)

    assert result == {"action": "switch", "error": "slug is required"}
    get_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Tenant isolation across info/switch.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_space_info_org_isolation_uses_resolved_principal():
    """get_space receives the resolved org_id, never a tool arg."""
    fn = _space_fn()
    get_mock = MagicMock(return_value=None)
    resolve_p, session_p = _patches(resolve=("org-secure", "agent-z", "free", None))

    with resolve_p, session_p, patch.object(mcp_app.brain, "get_space", get_mock):
        await fn(_fake_ctx(), "info", slug="research")

    assert get_mock.call_args.kwargs["org_id"] == "org-secure"


# ---------------------------------------------------------------------------
# Auth failures.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_space_auth_failure_returns_clean_error():
    """resolve_principal raising HTTPException yields an unauthorized dict and
    never calls any brain helper or opens a DB session."""
    fn = _space_fn()
    list_mock = MagicMock()
    session_local = MagicMock()

    with (
        patch(
            "app.mcp_app.resolve_active_principal",
            side_effect=HTTPException(
                status_code=401, detail="Missing X-Vex-Key header"
            ),
        ),
        patch("app.mcp_app.SessionLocal", session_local),
        patch.object(mcp_app.brain, "list_spaces", list_mock),
    ):
        result = await fn(_fake_ctx({}), "list")

    assert result["error"] == "unauthorized"
    assert "Missing X-Vex-Key header" in result["detail"]
    list_mock.assert_not_called()
    # The session opens before resolution now (the active-node resolver
    # needs it); the guarantee on a refusal is that it was CLOSED, unused.
    session_local.assert_called_once()
    session_local.return_value.close.assert_called_once()


@pytest.mark.asyncio
async def test_space_no_request_context_returns_unauthorized():
    """When the transport attaches no request, space denies rather than guess."""
    fn = _space_fn()
    list_mock = MagicMock()
    session_local = MagicMock()
    ctx = SimpleNamespace(request_context=SimpleNamespace(request=None))

    with (
        patch("app.mcp_app.resolve_active_principal") as resolve_mock,
        patch("app.mcp_app.SessionLocal", session_local),
        patch.object(mcp_app.brain, "list_spaces", list_mock),
    ):
        result = await fn(ctx, "list")

    assert result["error"] == "unauthorized"
    resolve_mock.assert_not_called()
    list_mock.assert_not_called()
    session_local.assert_not_called()
