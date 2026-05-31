"""Tests for the Klio ``recall`` read-tool wiring.

Mirrors the mock style of ``tests/test_write_tools.py``: the tool callable is
reached via the FastMCP registry (``server._tool_manager.get_tool(name).fn``)
and invoked with a hand-built fake :class:`Context`. The DB / embedding seams
are patched at the ``app.mcp_app`` module boundary so no real Postgres or
embeddings are touched:

* ``resolve_principal_plan`` → a fixed ``(org_id, agent_id, plan,
  plan_overrides)`` (or raises to drive the auth-failure path).
* ``SessionLocal`` → a ``MagicMock`` whose instance is ``.close()``-d.
* ``brain.find_space_id`` / ``brain.find_project_id`` → control filter
  resolution.
* ``retrieve_memories_scoped`` (looked up in ``app.mcp_app``) → an
  ``AsyncMock`` returning canned :class:`MemoryRecord`s, so the serialization
  and org-isolation behaviour can be asserted without a DB.
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import app.mcp_app as mcp_app
from app.mcp_app import build_mcp_server
from shared.memory import MemoryRecord


@pytest.fixture(autouse=True)
def _no_metering():
    """Isolate recall tool tests from quota enforcement.

    These tests focus on tool wiring and payload mapping, not metering.
    Patch quota checks to always pass (no-op) so tests stay focused.
    """
    with patch("app.mcp_app.check_memory_quota", return_value=False), \
         patch("app.mcp_app.record_usage"):
        yield


def _fake_ctx(headers: dict | None = None):
    """Minimal stand-in for the FastMCP Context (only request.headers is read)."""
    request = SimpleNamespace(headers=headers if headers is not None else {})
    request_context = SimpleNamespace(request=request)
    return SimpleNamespace(request_context=request_context)


def _recall_fn():
    """Return the registered ``recall`` tool's underlying callable."""
    server = build_mcp_server()
    tool = server._tool_manager.get_tool("recall")
    assert tool is not None, "tool 'recall' not registered"
    return tool.fn


def _make_record(
    *, content="a fact", memory_type="memory", scope="org"
) -> MemoryRecord:
    """Build a MemoryRecord populated like retrieve_memories_scoped returns,
    including an embedding so we can assert it is NOT serialized."""
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        session_id="klio-brain:org-x:agent-y",
        org_id="org-x",
        agent_id="agent-y",
        memory_type=memory_type,
        content=content,
        execution_id=None,
        embedding=[0.1, 0.2, 0.3],  # must NOT appear in the serialized output
        confidence=0.8,
        scope=scope,
        status="active",
        metadata={"source": "mcp"},
        created_at=now,
        updated_at=now,
        space_id="space-9",
        project_id="proj-7",
    )


# ---------------------------------------------------------------------------
# Happy path: serialization excludes embedding, created_at is isoformat str.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_happy_path_serializes_without_embedding():
    """recall returns count==2, each dict omits `embedding`, and created_at is
    a string (isoformat) or None."""
    fn = _recall_fn()
    records = [_make_record(content="first"), _make_record(content="second")]
    retrieve_mock = AsyncMock(return_value=records)
    session = MagicMock()
    session_local = MagicMock(return_value=session)

    with (
        patch("app.mcp_app.resolve_principal_plan", return_value=("org-x", "agent-y", "free", None)),
        patch("app.mcp_app.SessionLocal", session_local),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        result = await fn(_fake_ctx(), "what did we decide")

    assert result["count"] == 2
    assert len(result["memories"]) == 2
    for mem in result["memories"]:
        assert "embedding" not in mem
        assert isinstance(mem["id"], str)
        assert isinstance(mem["created_at"], (str, type(None)))
        # Spot-check the serialized shape.
        assert set(mem.keys()) == {
            "id",
            "memory_type",
            "content",
            "scope",
            "confidence",
            "space_id",
            "project_id",
            "created_at",
            "metadata",
        }
    # The DB session is always closed.
    session.close.assert_called_once()


@pytest.mark.asyncio
async def test_recall_stringifies_uuid_space_and_project_ids():
    """space_id/project_id are sa.UUID() columns, so a raw text() SELECT yields
    uuid.UUID objects. The serialized result must stringify them (like `id`) so
    the payload is plain-json-serializable, not just pydantic-coercible."""
    import json

    rec = _make_record()
    rec.space_id = uuid.uuid4()
    rec.project_id = uuid.uuid4()
    retrieve_mock = AsyncMock(return_value=[rec])

    with (
        patch("app.mcp_app.resolve_principal_plan", return_value=("org-x", "agent-y", "free", None)),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        result = await _recall_fn()(_fake_ctx(), "q")

    mem = result["memories"][0]
    assert isinstance(mem["space_id"], str)
    assert isinstance(mem["project_id"], str)
    assert mem["space_id"] == str(rec.space_id)
    assert mem["project_id"] == str(rec.project_id)
    # The whole payload must survive stdlib json (not just FastMCP coercion).
    json.dumps(result)


@pytest.mark.asyncio
async def test_recall_serializes_null_space_and_project_ids():
    """When space_id/project_id are None they stay None (no 'None' string)."""
    rec = _make_record()
    rec.space_id = None
    rec.project_id = None
    retrieve_mock = AsyncMock(return_value=[rec])

    with (
        patch("app.mcp_app.resolve_principal_plan", return_value=("org-x", "agent-y", "free", None)),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        result = await _recall_fn()(_fake_ctx(), "q")

    mem = result["memories"][0]
    assert mem["space_id"] is None
    assert mem["project_id"] is None


@pytest.mark.asyncio
async def test_recall_serializes_null_created_at():
    """A record with created_at=None serializes created_at as None, not a crash."""
    fn = _recall_fn()
    rec = _make_record()
    rec.created_at = None
    retrieve_mock = AsyncMock(return_value=[rec])

    with (
        patch("app.mcp_app.resolve_principal_plan", return_value=("org-x", "agent-y", "free", None)),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        result = await fn(_fake_ctx(), "q")

    assert result["count"] == 1
    assert result["memories"][0]["created_at"] is None


# ---------------------------------------------------------------------------
# Recall-event logging (best-effort): the tool logs source='mcp'.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_logs_event():
    """recall logs a brain_recall_event tagged source='mcp' with the resolved
    principal and the returned result_count, after retrieving the records."""
    fn = _recall_fn()
    records = [_make_record(content="first"), _make_record(content="second")]
    retrieve_mock = AsyncMock(return_value=records)
    log_mock = MagicMock()
    session = MagicMock()

    with (
        patch("app.mcp_app.resolve_principal_plan", return_value=("org-x", "agent-y", "free", None)),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=session)),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
        patch("app.mcp_app.log_recall_event", log_mock),
    ):
        result = await fn(_fake_ctx(), "what did we decide")

    assert result["count"] == 2
    log_mock.assert_called_once()
    kw = log_mock.call_args.kwargs
    assert kw["source"] == "mcp"
    assert kw["result_count"] == 2
    assert kw["org_id"] == "org-x"
    assert kw["agent_id"] == "agent-y"
    assert kw["query_text"] == "what did we decide"
    # The log call receives the same DB session the tool opened.
    assert log_mock.call_args.args[0] is session


# ---------------------------------------------------------------------------
# Org isolation: the org passed to retrieve comes from resolve_principal.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_org_isolation_uses_resolved_principal():
    """The org_id forwarded to retrieve_memories_scoped is exactly the one from
    resolve_principal — never anything caller-supplied."""
    fn = _recall_fn()
    retrieve_mock = AsyncMock(return_value=[])

    with (
        patch("app.mcp_app.resolve_principal_plan", return_value=("org-secure", "agent-z", "free", None)),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        await fn(_fake_ctx(), "q")

    retrieve_mock.assert_awaited_once()
    call = retrieve_mock.await_args
    # org_id is the 2nd positional arg (after db_session).
    assert call.args[1] == "org-secure"
    assert call.kwargs["scope"] == "org"


# ---------------------------------------------------------------------------
# scope='agent' forwards agent_id.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_agent_scope_forwards_agent_id():
    """scope='agent' forwards the resolved agent_id to retrieve_memories_scoped."""
    fn = _recall_fn()
    retrieve_mock = AsyncMock(return_value=[])

    with (
        patch("app.mcp_app.resolve_principal_plan", return_value=("org-x", "agent-y", "free", None)),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        await fn(_fake_ctx(), "q", scope="agent")

    call = retrieve_mock.await_args
    assert call.kwargs["scope"] == "agent"
    assert call.kwargs["agent_id"] == "agent-y"


# ---------------------------------------------------------------------------
# Space filter resolution.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_resolves_space_filter():
    """A known space slug resolves to a space_id forwarded to retrieve."""
    fn = _recall_fn()
    retrieve_mock = AsyncMock(return_value=[])

    with (
        patch("app.mcp_app.resolve_principal_plan", return_value=("org-x", "agent-y", "free", None)),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "find_space_id", return_value="space-42"),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        await fn(_fake_ctx(), "q", space="research")

    assert retrieve_mock.await_args.kwargs["space_id"] == "space-42"


@pytest.mark.asyncio
async def test_recall_unknown_space_returns_warning_without_retrieve():
    """An unknown space slug short-circuits: empty result + warning, no retrieve."""
    fn = _recall_fn()
    retrieve_mock = AsyncMock(return_value=[])

    with (
        patch("app.mcp_app.resolve_principal_plan", return_value=("org-x", "agent-y", "free", None)),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "find_space_id", return_value=None),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        result = await fn(_fake_ctx(), "q", space="ghost")

    assert result == {"memories": [], "count": 0, "warning": "unknown space: ghost"}
    retrieve_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Project filter resolution.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_resolves_project_filter():
    """A known project identifier resolves to a project_id forwarded to retrieve."""
    fn = _recall_fn()
    retrieve_mock = AsyncMock(return_value=[])

    with (
        patch("app.mcp_app.resolve_principal_plan", return_value=("org-x", "agent-y", "free", None)),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "find_project_id", return_value="proj-99"),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        await fn(_fake_ctx(), "q", project="acme/app")

    assert retrieve_mock.await_args.kwargs["project_id"] == "proj-99"


@pytest.mark.asyncio
async def test_recall_unknown_project_returns_warning_without_retrieve():
    """An unknown project short-circuits: empty result + warning, no retrieve."""
    fn = _recall_fn()
    retrieve_mock = AsyncMock(return_value=[])

    with (
        patch("app.mcp_app.resolve_principal_plan", return_value=("org-x", "agent-y", "free", None)),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "find_project_id", return_value=None),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        result = await fn(_fake_ctx(), "q", project="ghost-proj")

    assert result == {
        "memories": [],
        "count": 0,
        "warning": "unknown project: ghost-proj",
    }
    retrieve_mock.assert_not_called()


@pytest.mark.asyncio
async def test_recall_project_any_is_cross_project_no_lookup():
    """project='any' means cross-project: no resolution, project_id stays None."""
    fn = _recall_fn()
    retrieve_mock = AsyncMock(return_value=[])
    find_project = MagicMock()

    with (
        patch("app.mcp_app.resolve_principal_plan", return_value=("org-x", "agent-y", "free", None)),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "find_project_id", find_project),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        await fn(_fake_ctx(), "q", project="any")

    # No project resolution attempted for the "any" sentinel.
    find_project.assert_not_called()
    retrieve_mock.assert_awaited_once()
    assert retrieve_mock.await_args.kwargs["project_id"] is None


@pytest.mark.asyncio
async def test_recall_project_any_case_insensitive():
    """project='ANY' (any case) is treated as the cross-project sentinel."""
    fn = _recall_fn()
    retrieve_mock = AsyncMock(return_value=[])
    find_project = MagicMock()

    with (
        patch("app.mcp_app.resolve_principal_plan", return_value=("org-x", "agent-y", "free", None)),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "find_project_id", find_project),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        await fn(_fake_ctx(), "q", project="ANY")

    find_project.assert_not_called()
    assert retrieve_mock.await_args.kwargs["project_id"] is None


# ---------------------------------------------------------------------------
# No filters: space_id / project_id stay None.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_no_filters_passes_none():
    """Without space/project args, the corresponding ids stay None (no filter)."""
    fn = _recall_fn()
    retrieve_mock = AsyncMock(return_value=[])

    with (
        patch("app.mcp_app.resolve_principal_plan", return_value=("org-x", "agent-y", "free", None)),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        await fn(_fake_ctx(), "q")

    kwargs = retrieve_mock.await_args.kwargs
    assert kwargs["space_id"] is None
    assert kwargs["project_id"] is None


# ---------------------------------------------------------------------------
# Auth failures.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_auth_failure_returns_clean_error():
    """resolve_principal_plan raising HTTPException yields an unauthorized dict
    and never calls retrieve_memories_scoped."""
    fn = _recall_fn()
    retrieve_mock = AsyncMock()
    session_local = MagicMock()

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            side_effect=HTTPException(
                status_code=401, detail="Missing X-Vex-Key header"
            ),
        ),
        patch("app.mcp_app.SessionLocal", session_local),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        result = await fn(_fake_ctx({}), "q")

    assert result["error"] == "unauthorized"
    assert "Missing X-Vex-Key header" in result["detail"]
    retrieve_mock.assert_not_called()
    session_local.assert_not_called()


@pytest.mark.asyncio
async def test_recall_no_request_context_returns_unauthorized():
    """When the transport attaches no request, recall denies rather than guess."""
    fn = _recall_fn()
    retrieve_mock = AsyncMock()
    session_local = MagicMock()
    ctx = SimpleNamespace(request_context=SimpleNamespace(request=None))

    with (
        patch("app.mcp_app.resolve_principal_plan") as resolve_mock,
        patch("app.mcp_app.SessionLocal", session_local),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        result = await fn(ctx, "q")

    assert result["error"] == "unauthorized"
    resolve_mock.assert_not_called()
    retrieve_mock.assert_not_called()
    session_local.assert_not_called()
