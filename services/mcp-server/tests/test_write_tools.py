"""Tests for the Klio write-tool wiring (remember/observe/plan/decide/note).

These tests exercise the *tool bodies* in :mod:`app.mcp_app` without a live
DB or real embeddings. They reach the registered tool callables via the
FastMCP tool registry (``server._tool_manager.get_tool(name).fn``) and invoke
them with a hand-built fake :class:`Context` whose
``.request_context.request.headers`` is a dict-like supporting ``.get(...)``
— the only surface the tool bodies touch on the request.

The seams patched at the ``app.mcp_app`` module boundary:

* ``resolve_principal_plan`` → returns a fixed ``(org_id, agent_id, plan,
  plan_overrides)`` (or raises to drive the auth-failure path).
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


@pytest.fixture(autouse=True)
def _no_metering():
    """Isolate write tool tests from quota enforcement.

    These tests focus on tool wiring and payload mapping, not metering.
    Patch quota checks to always pass (no-op) so tests stay focused.
    """
    with (
        patch("app.mcp_app.check_memory_quota", return_value=False),
        patch("app.mcp_app.record_usage"),
    ):
        yield


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
    ``resolve_principal_plan`` / the session-id helper, which in turn read
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
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
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
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
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
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
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
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
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
        patch("app.mcp_app.resolve_principal_plan") as resolve_mock,
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
    """When resolve_principal_plan raises HTTPException, the tool returns a clean
    unauthorized dict and never calls write_scoped_memory."""
    fn = _get_tool_fn("remember")
    write_mock = AsyncMock()
    session = MagicMock()
    session_local = MagicMock(return_value=session)

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
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


@pytest.mark.asyncio
async def test_write_tool_scope_agent_forwarded():
    """Passing scope='agent' to a write tool flows to write_scoped_memory."""
    fn = _get_tool_fn("remember")
    write_mock = AsyncMock(return_value=_canned_result("memory"))

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
    ):
        await fn(_fake_ctx({}), "a private fact", scope="agent")

    assert write_mock.await_args.kwargs["scope"] == "agent"


@pytest.mark.asyncio
async def test_write_tool_scope_defaults_to_org():
    """A write tool with no scope arg writes scope='org' (backward compatible)."""
    fn = _get_tool_fn("observe")
    write_mock = AsyncMock(return_value=_canned_result("observation"))

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
    ):
        await fn(_fake_ctx({}), "an observation")

    assert write_mock.await_args.kwargs["scope"] == "org"


# ---------------------------------------------------------------------------
# Task 10 (Step F): header-project filing on writes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_files_under_header_project(monkeypatch):
    """A write with X-Vex-Project resolved files the row under that project.

    Monkeypatches resolve_project_from_request on the mcp_app module to return
    "proj-9", then confirms write_scoped_memory is called with project_id="proj-9".
    """

    fn = _get_tool_fn("remember")
    seen = {}

    async def _fake_write(db, **kwargs):
        seen.update(kwargs)
        return {"stored": True, "deduplicated": False, "id": "x", "scope": "org"}

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "write_scoped_memory", _fake_write),
        patch("app.mcp_app.resolve_project_from_request", return_value="proj-9"),
    ):
        await fn(
            _fake_ctx({"X-Vex-Project": "git@github.com:acme/widgets.git"}),
            "a fact about the project",
        )

    assert seen.get("project_id") == "proj-9"


# ---------------------------------------------------------------------------
# D1 (space-on-write): a caller-supplied space slug is resolved to a space_id
# via ensure_space (get-or-create, under the caller's org) and threaded into
# write_scoped_memory. Omitting it keeps today's org-brain behavior.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name,expected_type", list(_TOOL_MEMORY_TYPE.items()))
async def test_write_with_space_resolves_and_threads_space_id(tool_name, expected_type):
    """Each write tool resolves space= to a space_id via ensure_space (get-or-
    create under the caller's org) and threads it into write_scoped_memory; the
    resolved id is echoed back on the result.

    The shared ``_perform_write`` body implements this once, so the behavior is
    identical across all five write tools (parametrized here to lock that).
    """
    fn = _get_tool_fn(tool_name)

    async def _fake_write(db, **kwargs):
        # Echo the scope columns like the real write_scoped_memory result so the
        # assertion on the returned space_id is end-to-end-ish, not mock-only.
        return {
            "stored": True,
            "deduplicated": False,
            "id": "00000000-0000-0000-0000-000000000001",
            "memory_type": kwargs.get("memory_type"),
            "scope": kwargs.get("scope", "org"),
            "space_id": kwargs.get("space_id"),
            "project_id": kwargs.get("project_id"),
        }

    write_mock = AsyncMock(side_effect=_fake_write)
    ensure_mock = MagicMock(return_value="sp-42")

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
        patch.object(mcp_app.brain, "ensure_space", ensure_mock),
    ):
        result = await fn(_fake_ctx({}), "some content", space="spicychat-platform")

    # Space resolved under the caller's org (never caller-supplied org input).
    ensure_mock.assert_called_once()
    ekw = ensure_mock.call_args.kwargs
    assert ekw["org_id"] == "org-x"
    # The provided space string is used verbatim as the slug (mirrors recall,
    # which uses `space` as the slug) with name==slug (no separate name input).
    assert ekw["slug"] == "spicychat-platform"
    assert ekw["name"] == "spicychat-platform"
    # Resolved id threaded into the write and echoed on the result.
    assert write_mock.await_args.kwargs["space_id"] == "sp-42"
    assert result["space_id"] == "sp-42"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["remember", "note"])
async def test_write_without_space_leaves_space_id_none(tool_name):
    """Omitting space keeps today's behavior: space_id is None and ensure_space
    is never called (backward compatible org-brain write)."""
    fn = _get_tool_fn(tool_name)
    write_mock = AsyncMock(return_value=_canned_result("memory"))
    ensure_mock = MagicMock()

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
        patch.object(mcp_app.brain, "ensure_space", ensure_mock),
    ):
        await fn(_fake_ctx({}), "a durable fact")

    assert write_mock.await_args.kwargs["space_id"] is None
    ensure_mock.assert_not_called()


@pytest.mark.asyncio
async def test_write_space_resolved_under_resolved_principal_org():
    """The org_id reaching ensure_space is the one from resolve_principal_plan —
    never caller-supplied — so a write can only ever target its own org's space
    (tenant-safe get-or-create)."""
    fn = _get_tool_fn("note")
    write_mock = AsyncMock(return_value=_canned_result("note"))
    ensure_mock = MagicMock(return_value="sp-secure")

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-secure", "agent-z", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
        patch.object(mcp_app.brain, "ensure_space", ensure_mock),
    ):
        await fn(_fake_ctx({}), "content", space="research")

    assert ensure_mock.call_args.kwargs["org_id"] == "org-secure"


@pytest.mark.asyncio
async def test_write_same_space_slug_is_idempotent_reuses_space():
    """Two writes with the same space slug both resolve to the SAME space_id
    (ensure_space is get-or-create), so a second write reuses the space rather
    than creating a duplicate."""
    fn = _get_tool_fn("remember")
    write_mock = AsyncMock(return_value=_canned_result("memory"))
    # ensure_space returns the same id for a repeat (org, slug) — its ON CONFLICT
    # get-or-create semantics; the fixed return value models that idempotency.
    ensure_mock = MagicMock(return_value="sp-shared")

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
        patch.object(mcp_app.brain, "ensure_space", ensure_mock),
    ):
        await fn(_fake_ctx({}), "first fact", space="ledger")
        first_space_id = write_mock.await_args.kwargs["space_id"]
        await fn(_fake_ctx({}), "second fact", space="ledger")
        second_space_id = write_mock.await_args.kwargs["space_id"]

    assert first_space_id == second_space_id == "sp-shared"
    assert ensure_mock.call_count == 2
    for call in ensure_mock.call_args_list:
        assert call.kwargs["slug"] == "ledger"
        assert call.kwargs["org_id"] == "org-x"


@pytest.mark.asyncio
async def test_write_blank_space_treated_as_no_space():
    """A blank/whitespace-only space is treated as omitted: space_id stays None
    and no stray empty-slug space is created."""
    fn = _get_tool_fn("remember")
    write_mock = AsyncMock(return_value=_canned_result("memory"))
    ensure_mock = MagicMock()

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
        patch.object(mcp_app.brain, "ensure_space", ensure_mock),
    ):
        await fn(_fake_ctx({}), "content", space="   ")

    assert write_mock.await_args.kwargs["space_id"] is None
    ensure_mock.assert_not_called()


@pytest.mark.asyncio
async def test_write_space_slug_is_stripped():
    """A padded space slug is stripped before the get-or-create, so the space is
    filed under the trimmed slug (name==slug=="spicy") and a later
    recall(space="spicy") converges on the same space rather than being unable to
    find a padded "  spicy  " slug."""
    fn = _get_tool_fn("remember")
    write_mock = AsyncMock(return_value=_canned_result("memory"))
    ensure_mock = MagicMock(return_value="sp-spicy")

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
        patch.object(mcp_app.brain, "ensure_space", ensure_mock),
    ):
        await fn(_fake_ctx({}), "content", space="  spicy  ")

    ensure_mock.assert_called_once()
    ekw = ensure_mock.call_args.kwargs
    assert ekw["slug"] == "spicy"
    assert ekw["name"] == "spicy"
    assert ekw["org_id"] == "org-x"
    assert write_mock.await_args.kwargs["space_id"] == "sp-spicy"


# ---------------------------------------------------------------------------
# CAS (Task 8): base_revision / memory_id on the four re-assertable tools
# ---------------------------------------------------------------------------
#
# Absent base_revision is asserted ALREADY, indirectly, by every test above:
# none of them pass it, and every one still reaches write_scoped_memory
# exactly as before. These tests cover the branch those never touch.

_CAS_TOOLS = ["remember", "observe", "decide", "note"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", _CAS_TOOLS)
async def test_base_revision_present_skips_plain_write_and_calls_cas(tool_name):
    """base_revision routes to supersede_with_revision, never write_scoped_memory."""
    fn = _get_tool_fn(tool_name)
    write_mock = AsyncMock(return_value=_canned_result("memory"))
    cas_mock = MagicMock(
        return_value={
            "conflict": False, "stored": True, "new_id": "new-1",
            "new_revision": "rev-2", "redacted": True,
            "redaction_kinds": ["stripe_secret_key"],
        }
    )

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
        patch.object(mcp_app, "supersede_with_revision", cas_mock),
    ):
        result = await fn(
            _fake_ctx(), "corrected content",
            memory_id="mem-1", base_revision="rev-1",
        )

    write_mock.assert_not_called()
    cas_mock.assert_called_once()
    ckw = cas_mock.call_args.kwargs
    assert ckw["memory_id"] == "mem-1"
    assert ckw["base_revision"] == "rev-1"
    assert ckw["new_content"] == "corrected content"
    assert ckw["org_id"] == "org-x"
    assert ckw["agent_id"] == "agent-y"
    # user_id is forwarded too — the tool layer resolves it from the request
    # (see resolve_request_user_id, unpatched here so it resolves to None),
    # and supersede_with_revision needs it for the authorization ladder.
    assert "user_id" in ckw

    assert result["stored"] is True
    assert result["id"] == "new-1"
    assert result["revision"] == "rev-2"
    assert result["superseded_id"] == "mem-1"
    assert result["redacted"] is True
    assert result["redaction_kinds"] == ["stripe_secret_key"]


@pytest.mark.asyncio
async def test_base_revision_auth_refusal_is_passed_through_as_stored_false():
    """not_found / forbidden / invalid_id from the authorization ladder (H1)
    surface as a plain stored:False error, never the revision_conflict shape
    — that shape is reserved for "the row is real, you may act on it, but it
    moved", and these callers may not act on it at all."""
    fn = _get_tool_fn("remember")
    write_mock = AsyncMock()
    cas_mock = MagicMock(
        return_value={"conflict": False, "stored": False, "error": "not_found"}
    )

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
        patch.object(mcp_app, "supersede_with_revision", cas_mock),
        patch("app.mcp_app.record_usage") as usage_mock,
    ):
        result = await fn(
            _fake_ctx(), "attacker's overwrite",
            memory_id="mem-1", base_revision="whatever",
        )

    write_mock.assert_not_called()
    assert result == {"stored": False, "error": "not_found"}
    # A refused write stores nothing and must not consume the org's
    # allowance, same gate as a conflict.
    usage_mock.assert_not_called()


@pytest.mark.asyncio
async def test_base_revision_conflict_returns_tool_error_payload():
    """A stale base_revision returns the documented revision_conflict shape,
    not a generic write_failed."""
    fn = _get_tool_fn("remember")
    write_mock = AsyncMock()
    cas_mock = MagicMock(
        return_value={
            "conflict": True,
            "current": {
                "id": "mem-1", "content": "someone else's edit",
                "revision": "rev-9", "user_id": "u2",
            },
        }
    )

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
        patch.object(mcp_app, "supersede_with_revision", cas_mock),
        patch("app.mcp_app.record_usage") as usage_mock,
    ):
        result = await fn(
            _fake_ctx(), "my edit", memory_id="mem-1", base_revision="stale-rev",
        )

    write_mock.assert_not_called()
    assert result == {
        "error": "revision_conflict",
        "current": {
            "id": "mem-1", "content": "someone else's edit",
            "revision": "rev-9", "user_id": "u2",
        },
        "hint": "belief changed while you worked — re-read and re-assert",
    }
    # A rejected CAS write stores nothing, so it must not consume the org's
    # monthly memory allowance — same gate as a dedup skip on the plain path.
    usage_mock.assert_not_called()


@pytest.mark.asyncio
async def test_base_revision_without_memory_id_is_refused_before_any_write():
    """base_revision alone (no memory_id naming the row) is a clean error, not
    a call into supersede_with_revision with a guaranteed-wrong guard."""
    fn = _get_tool_fn("remember")
    write_mock = AsyncMock()
    cas_mock = MagicMock()

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
        patch.object(mcp_app, "supersede_with_revision", cas_mock),
    ):
        result = await fn(_fake_ctx(), "content", base_revision="rev-1")

    write_mock.assert_not_called()
    cas_mock.assert_not_called()
    assert result["stored"] is False
    assert result["error"] == "missing_memory_id"


@pytest.mark.asyncio
async def test_absent_base_revision_is_unaffected_by_the_cas_wiring():
    """Zero-regression check: with base_revision omitted, supersede_with_revision
    is never even imported into the call path — write_scoped_memory runs
    exactly as it did before Task 8."""
    fn = _get_tool_fn("remember")
    write_mock = AsyncMock(return_value=_canned_result("memory"))
    cas_mock = MagicMock()

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
        patch.object(mcp_app, "supersede_with_revision", cas_mock),
    ):
        result = await fn(_fake_ctx(), "plain content")

    cas_mock.assert_not_called()
    write_mock.assert_awaited_once()
    assert result["stored"] is True
