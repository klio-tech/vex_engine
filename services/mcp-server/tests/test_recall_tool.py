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
    with (
        patch("app.mcp_app.check_memory_quota", return_value=False),
        patch("app.mcp_app.record_usage"),
    ):
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
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", session_local),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        result = await fn(_fake_ctx(), "what did we decide")

    assert result["count"] == 2
    assert len(result["memories"]) == 2
    # Structural compression hoists fields identical on every row into
    # `common` and drops nulls, so a field lives either on the row or on the
    # envelope. The assertion is over the merged view, which is what the
    # serialized shape now means. See test_recall_compression.py.
    common = result.get("common", {})
    assert "embedding" not in common
    for mem in result["memories"]:
        merged = {**common, **mem}
        assert "embedding" not in mem
        assert isinstance(merged["id"], str)
        assert isinstance(merged["created_at"], (str, type(None)))
        # Spot-check the serialized shape. 'relevance' and 'project' (added
        # in Task 10) are None on this fixture and nulls are dropped, so they
        # are absent here; both are still asserted directly in
        # test_recall_result_includes_relevance_and_project_name.
        assert set(merged.keys()) == {
            "id",
            "memory_type",
            "content",
            "scope",
            "confidence",
            # Added in phase 2: EXTRACTED / INFERRED / AMBIGUOUS, so a caller
            # can tell what the user said from what a model deduced.
            "provenance",
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
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
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
    """When space_id/project_id are None they never become the string 'None'.

    Structural compression represents null by absence, so the assertion is
    that the key is gone rather than that it is present-and-None. The bug this
    guards — a UUID or None stringified into "None" — would show up as a
    present key with a junk value, which is still caught.
    """
    rec = _make_record()
    rec.space_id = None
    rec.project_id = None
    retrieve_mock = AsyncMock(return_value=[rec])

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        result = await _recall_fn()(_fake_ctx(), "q")

    import json

    mem = {**result.get("common", {}), **result["memories"][0]}
    assert mem.get("space_id") is None
    assert mem.get("project_id") is None
    # The failure mode being guarded: stringified into "None" rather than
    # dropped as a genuine null.
    assert "None" not in json.dumps(result)


@pytest.mark.asyncio
async def test_recall_serializes_null_created_at():
    """A record with created_at=None serializes as a null, not a crash.

    Structural compression represents null by absence, so the key is dropped
    rather than present-and-None. The behaviour under test — no exception, and
    never the string "None" — is unchanged.
    """
    fn = _recall_fn()
    rec = _make_record()
    rec.created_at = None
    retrieve_mock = AsyncMock(return_value=[rec])

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        result = await fn(_fake_ctx(), "q")

    import json

    assert result["count"] == 1
    mem = {**result.get("common", {}), **result["memories"][0]}
    assert mem.get("created_at") is None
    assert "None" not in json.dumps(result)


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
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
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
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-secure", "agent-z", "free", None),
        ),
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
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
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
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
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
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "find_space_id", return_value=None),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        result = await fn(_fake_ctx(), "q", space="ghost")

    from app.mcp_app import RECALL_WRITE_NUDGE

    assert result == {
        "memories": [],
        "count": 0,
        "warning": "unknown space: ghost",
        "guidance": RECALL_WRITE_NUDGE,
    }
    retrieve_mock.assert_not_called()


@pytest.mark.asyncio
async def test_recall_space_slug_is_stripped():
    """A padded recall space slug is stripped before lookup, so it converges with
    the write path (which stores the stripped slug): recall(space='  research  ')
    resolves slug 'research'."""
    fn = _recall_fn()
    retrieve_mock = AsyncMock(return_value=[])
    find_mock = MagicMock(return_value="space-42")

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "find_space_id", find_mock),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        await fn(_fake_ctx(), "q", space="  research  ")

    assert find_mock.call_args.kwargs["slug"] == "research"
    assert retrieve_mock.await_args.kwargs["space_id"] == "space-42"


@pytest.mark.asyncio
async def test_recall_blank_space_resolves_to_org_wide():
    """A blank / whitespace-only recall space is treated as no filter (org-wide,
    space_id=None) — mirroring the write side — rather than erroring 'unknown
    space'. find_space_id is never consulted for an empty slug."""
    fn = _recall_fn()
    retrieve_mock = AsyncMock(return_value=[])
    find_mock = MagicMock()

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "find_space_id", find_mock),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        result = await fn(_fake_ctx(), "q", space="   ")

    # No unknown-space short-circuit: retrieval ran org-wide with space_id=None.
    find_mock.assert_not_called()
    retrieve_mock.assert_awaited_once()
    assert retrieve_mock.await_args.kwargs["space_id"] is None
    assert "warning" not in result


# ---------------------------------------------------------------------------
# Project filter resolution.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_resolves_project_filter():
    """A known project identifier resolves to a project_id forwarded to retrieve."""
    fn = _recall_fn()
    retrieve_mock = AsyncMock(return_value=[])

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
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
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "find_project_id", return_value=None),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        result = await fn(_fake_ctx(), "q", project="ghost-proj")

    from app.mcp_app import RECALL_WRITE_NUDGE

    assert result == {
        "memories": [],
        "count": 0,
        "warning": "unknown project: ghost-proj",
        "guidance": RECALL_WRITE_NUDGE,
    }
    retrieve_mock.assert_not_called()


@pytest.mark.asyncio
async def test_recall_project_any_is_cross_project_no_lookup():
    """project='any' means cross-project: no resolution, project_id stays None."""
    fn = _recall_fn()
    retrieve_mock = AsyncMock(return_value=[])
    find_project = MagicMock()

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
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
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
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
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
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


# ---------------------------------------------------------------------------
# Task 10: relevance + project label in serialized output
# ---------------------------------------------------------------------------


def test_recall_result_includes_relevance_and_project_name():
    """_serialize_record carries relevance and project label fields.

    The project label uses the key ``"id"`` (not ``"slug"``) because
    project_id is a UUID, not a slug.
    """
    import uuid
    from datetime import datetime, timezone

    from app.mcp_app import _serialize_record
    from shared.memory import MemoryRecord

    record = MemoryRecord(
        id=uuid.uuid4(),
        session_id="s",
        org_id="o",
        agent_id="a",
        memory_type="fact",
        content="x",
        created_at=datetime.now(timezone.utc),
        relevance=0.873,
    )
    out = _serialize_record(record, project_label={"id": "widgets", "name": "Widgets"})
    assert out["relevance"] == 0.873
    assert out["project"] == {"id": "widgets", "name": "Widgets"}


# ---------------------------------------------------------------------------
# Task 10: recall tool schema advertises `limit`
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_schema_has_limit_field():
    """The recall tool schema must expose a `limit` input field (1-50, default 10)."""
    from mcp.shared.memory import create_connected_server_and_client_session

    mcp = build_mcp_server()
    async with create_connected_server_and_client_session(mcp) as session:
        result = await session.list_tools()

    recall_tool = next(t for t in result.tools if t.name == "recall")
    schema = recall_tool.inputSchema
    assert "limit" in schema.get("properties", {}), (
        "recall schema missing 'limit' property"
    )


@pytest.mark.asyncio
async def test_recall_limit_description_admits_graph_expansion():
    """The schema text a model reads must match what recall actually returns.

    Recall runs a graph-expansion stage that adds up to GRAPH_EXPANSION_CAP
    rows ON TOP of ``limit`` — ``limit=1`` really can return 6. Describing
    ``limit`` as "max memories to return" is therefore false, and a caller
    that sizes its context window on it can be blown out by rows it never
    asked for. The expansion is the feature; the description is the defect.
    """
    from mcp.shared.memory import create_connected_server_and_client_session
    from shared.memory import GRAPH_EXPANSION_CAP

    mcp = build_mcp_server()
    async with create_connected_server_and_client_session(mcp) as session:
        result = await session.list_tools()

    recall_tool = next(t for t in result.tools if t.name == "recall")
    described = recall_tool.inputSchema["properties"]["limit"].get("description", "")

    # The true ceiling must be stated as a number, not left to be inferred.
    true_max = 50 + GRAPH_EXPANSION_CAP
    assert str(true_max) in described, (
        f"limit description must state the true maximum ({true_max}); got {described!r}"
    )
    # And it must say WHY more rows can come back, so the extra rows do not
    # read as a bug to whatever is on the other end.
    assert any(
        word in described.lower() for word in ("connected", "expansion", "related")
    ), f"limit description must explain the extra rows; got {described!r}"
    # A description that no longer bounds the vector stage is not a fix.
    assert "50" in described, f"limit description lost its own bound; got {described!r}"


# ---------------------------------------------------------------------------
# Task 1: RECALL_WRITE_NUDGE constant + guidance on every recall return
# ---------------------------------------------------------------------------


def test_recall_write_nudge_constant_steers_durable_not_noise():
    from app.mcp_app import RECALL_WRITE_NUDGE

    low = RECALL_WRITE_NUDGE.lower()
    assert "durable" in low
    # Must explicitly steer AWAY from tool-call/step noise (protects the
    # signal-quality cleanup we just shipped).
    assert "tool call" in low or "tool calls" in low
    assert "remember" in low


def test_recall_serializer_unchanged_and_guidance_is_separate():
    """guidance is a top-level recall-result field, not per-memory."""
    import uuid
    from datetime import datetime, timezone

    from app.mcp_app import _serialize_record
    from shared.memory import MemoryRecord

    out = _serialize_record(
        MemoryRecord(
            id=uuid.uuid4(),
            session_id="s",
            org_id="o",
            agent_id="a",
            memory_type="fact",
            content="x",
            created_at=datetime.now(timezone.utc),
        )
    )
    assert "guidance" not in out  # nudge lives on the envelope, not each record


def test_serialize_record_carries_the_graph_marker():
    """A row that rode in on graph expansion must SAY so on the wire.

    ``relevance=None`` was its only marker, and structural compression strips
    null keys row by row — so the agent received a row with no way to tell a
    semantic match from a one-hop graph walk. The marker is only a fix if the
    serializer forwards it.
    """
    import uuid as _uuid

    from app.mcp_app import _serialize_record
    from shared.memory import VIA_GRAPH, MemoryRecord

    expanded = MemoryRecord(
        id=_uuid.uuid4(),
        session_id="s",
        org_id="o",
        agent_id="a",
        memory_type="fact",
        content="rode in on an edge",
        via=VIA_GRAPH,
    )
    semantic = MemoryRecord(
        id=_uuid.uuid4(),
        session_id="s",
        org_id="o",
        agent_id="a",
        memory_type="fact",
        content="matched the query",
        relevance=0.873,
    )

    assert _serialize_record(expanded)["via"] == "graph"
    assert _serialize_record(semantic)["via"] is None
