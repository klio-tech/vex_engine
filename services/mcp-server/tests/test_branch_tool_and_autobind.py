"""The ``branch`` MCP tool and the hook auto-bind (Track C, Task C3).

Two surfaces, one file, because they are the two halves of the same contract:
the tool is how a person drives a branch's lifecycle, and the auto-bind is how
a hook capture lands on the branch the person is already standing on.

Mock-based, mirroring ``test_recall_branch_wiring.py`` (tool half — the SQL
lives in ``services/shared/tests/test_branch_merge_discard.py``) and
``test_capture.py`` (auto-bind half). Under test:

* The tool is member-gated like the project tools: a non-member gets
  ``not_found``, never a confirmation the project exists. Merge/discard are
  additionally gated by ``projects.can_write``.
* Auto-bind: a transcript carrying ``git_branch`` lands its writes on that
  branch when (and only when) a project resolved AND the caller is
  attributable. ``main``/``master`` (case-insensitive) mean NULL — the repo's
  default branch IS main — and an unattributable caller never auto-creates a
  branch.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.capture as capture
import app.mcp_app as mcp_app
from app.auth import verify_api_key
from app.main import create_app
from app.mcp_app import build_mcp_server
from fastapi.testclient import TestClient
from shared.auth import KeyInfo

USER = "aaaaaaaa-1111-2222-3333-444444444444"

BRANCH_ROW = {
    "id": "beefbeef-1111-2222-3333-444444444444",
    "org_id": "org-x",
    "project_id": "proj-1",
    "name": "feat/x",
    "status": "open",
    "created_by": USER,
    "created_at": None,
    "closed_at": None,
}


# ---------------------------------------------------------------------------
# The branch tool
# ---------------------------------------------------------------------------


def _fake_ctx(headers: dict | None = None):
    request = SimpleNamespace(headers=headers if headers is not None else {})
    return SimpleNamespace(request_context=SimpleNamespace(request=request))


def _branch_fn():
    server = build_mcp_server()
    tool = server._tool_manager.get_tool("branch")
    assert tool is not None, "tool 'branch' not registered"
    return tool.fn


def _principal():
    return patch(
        "app.mcp_app.resolve_active_principal",
        return_value=("org-x", "agent-y", "free", None),
    )


def _tool_patches(*, member=True, writer=True, user_id=USER):
    return (
        _principal(),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.resolve_request_user_id", return_value=user_id),
        patch.object(mcp_app.brain, "find_project_id", return_value="proj-1"),
        patch.object(mcp_app.projects, "is_member", return_value=member),
        patch.object(mcp_app.projects, "can_write", return_value=writer),
    )


@pytest.mark.asyncio
async def test_non_member_merge_is_not_found():
    """Non-member gets not_found — same disclosure rule as project_members:
    confirming the project (or its branches) exist is itself a leak."""
    fn = _branch_fn()
    merge_mock = MagicMock()

    principal, db, user, find, member, writer = _tool_patches(member=False)
    with (
        principal,
        db,
        user,
        find,
        member,
        writer,
        patch.object(mcp_app.branches, "merge_branch", merge_mock),
    ):
        result = await fn(_fake_ctx(), action="merge", name="feat/x", project="klio")

    assert result["error"] == "not_found"
    merge_mock.assert_not_called()


@pytest.mark.asyncio
async def test_non_member_list_is_not_found():
    fn = _branch_fn()
    list_mock = MagicMock()

    principal, db, user, find, member, writer = _tool_patches(member=False)
    with (
        principal,
        db,
        user,
        find,
        member,
        writer,
        patch.object(mcp_app.branches, "list_branches", list_mock),
    ):
        result = await fn(_fake_ctx(), action="list", project="klio")

    assert result["error"] == "not_found"
    list_mock.assert_not_called()


@pytest.mark.asyncio
async def test_read_only_member_merge_is_forbidden():
    """A member without write may LOOK but not merge — projects.can_write is
    the gate, and a read-only member already knows the project exists so
    forbidden discloses nothing new."""
    fn = _branch_fn()
    merge_mock = MagicMock()

    principal, db, user, find, member, writer = _tool_patches(writer=False)
    with (
        principal,
        db,
        user,
        find,
        member,
        writer,
        patch.object(mcp_app.branches, "merge_branch", merge_mock),
    ):
        result = await fn(_fake_ctx(), action="merge", name="feat/x", project="klio")

    assert result["error"] == "forbidden"
    merge_mock.assert_not_called()


@pytest.mark.asyncio
async def test_member_merge_resolves_and_merges():
    fn = _branch_fn()
    merge_mock = MagicMock(
        return_value={"merged": True, "promoted": 3, "superseded": 1}
    )

    principal, db, user, find, member, writer = _tool_patches()
    with (
        principal,
        db,
        user,
        find,
        member,
        writer,
        patch.object(mcp_app.branches, "get_branch", return_value=BRANCH_ROW),
        patch.object(mcp_app.branches, "merge_branch", merge_mock),
    ):
        result = await fn(_fake_ctx(), action="merge", name="feat/x", project="klio")

    assert result == {"merged": True, "promoted": 3, "superseded": 1}
    merge_mock.assert_called_once()
    kwargs = merge_mock.call_args.kwargs
    assert kwargs["org_id"] == "org-x"
    assert kwargs["project_id"] == "proj-1"
    assert kwargs["branch_id"] == BRANCH_ROW["id"]
    assert kwargs["merged_by"] == USER


@pytest.mark.asyncio
async def test_merge_of_a_closed_name_reports_noop_not_error():
    """Idempotence at the tool surface: the name no longer resolves as OPEN,
    but a closed branch by that name answers already_<status>, not an error —
    retrying a merge must stay safe."""
    fn = _branch_fn()
    closed = dict(BRANCH_ROW, status="merged")

    principal, db, user, find, member, writer = _tool_patches()
    with (
        principal,
        db,
        user,
        find,
        member,
        writer,
        patch.object(mcp_app.branches, "get_branch", return_value=None),
        patch.object(mcp_app.branches, "list_branches", return_value=[closed]),
    ):
        result = await fn(_fake_ctx(), action="merge", name="feat/x", project="klio")

    assert result == {"merged": False, "reason": "already_merged"}


@pytest.mark.asyncio
async def test_member_discard_resolves_and_discards():
    fn = _branch_fn()
    discard_mock = MagicMock(return_value={"discarded": True, "retracted": 2})

    principal, db, user, find, member, writer = _tool_patches()
    with (
        principal,
        db,
        user,
        find,
        member,
        writer,
        patch.object(mcp_app.branches, "get_branch", return_value=BRANCH_ROW),
        patch.object(mcp_app.branches, "discard_branch", discard_mock),
    ):
        result = await fn(_fake_ctx(), action="discard", name="feat/x", project="klio")

    assert result == {"discarded": True, "retracted": 2}
    kwargs = discard_mock.call_args.kwargs
    assert kwargs["org_id"] == "org-x"
    assert kwargs["branch_id"] == BRANCH_ROW["id"]
    assert kwargs["discarded_by"] == USER


@pytest.mark.asyncio
async def test_create_refuses_invalid_name():
    fn = _branch_fn()

    principal, db, user, find, member, writer = _tool_patches()
    with (
        principal,
        db,
        user,
        find,
        member,
        writer,
        patch.object(
            mcp_app.branches,
            "ensure_branch",
            MagicMock(side_effect=ValueError("'main' is not a branch")),
        ),
    ):
        result = await fn(_fake_ctx(), action="create", name="main", project="klio")

    assert result["error"] == "invalid_name"


@pytest.mark.asyncio
async def test_unknown_project_is_not_found():
    fn = _branch_fn()

    with (
        _principal(),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.resolve_request_user_id", return_value=USER),
        patch.object(mcp_app.brain, "find_project_id", return_value=None),
    ):
        result = await fn(_fake_ctx(), action="list", project="ghost")

    assert result["error"] == "not_found"


@pytest.mark.asyncio
async def test_no_project_resolvable_is_an_explicit_refusal():
    """A branch always belongs to a project (global constraint): with neither
    a project arg nor a resolvable header there is nothing to operate on."""
    fn = _branch_fn()

    with (
        _principal(),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.resolve_request_user_id", return_value=USER),
        patch("app.mcp_app.resolve_project_from_request", return_value=None),
    ):
        result = await fn(_fake_ctx(), action="list")

    assert result["error"] == "project_required"


# ---------------------------------------------------------------------------
# Hook auto-bind: /capture/transcript and /capture/event carry git_branch
# ---------------------------------------------------------------------------


def _client(user_id=None):
    app = create_app()
    app.dependency_overrides[verify_api_key] = lambda: KeyInfo(
        org_id="test-org", key_id="test-key", scopes=["ingest"], user_id=user_id
    )
    return TestClient(app)


def _capture_patches(write, ensure_branch):
    session = MagicMock()
    return (
        patch("app.capture.SessionLocal", MagicMock(return_value=session)),
        patch.object(capture.brain, "write_scoped_memory", write),
        patch.object(capture.brain, "ensure_project", MagicMock(return_value="proj-1")),
        patch.object(
            capture.project_scope,
            "resolve_capture_scope",
            MagicMock(return_value="private"),
        ),
        patch.object(capture.branches, "ensure_branch", ensure_branch),
        patch("app.capture.generate_summary", AsyncMock(return_value="a summary")),
        patch("app.capture.extract_typed_memories", AsyncMock(return_value=[])),
        patch("app.capture.enqueue_trace_outbox", AsyncMock()),
    )


def _post_transcript(client, **extra):
    payload = {
        "messages": [{"role": "user", "content": "hello there"}],
        "git_remote": "git@example.com:klio.git",
        **extra,
    }
    return client.post("/capture/transcript", json=payload)


def test_transcript_git_branch_binds_writes_to_the_branch():
    write = AsyncMock(return_value={"stored": True})
    ensure_branch = MagicMock(return_value=dict(BRANCH_ROW))

    patches = _capture_patches(write, ensure_branch)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
    ):
        resp = _post_transcript(_client(user_id=USER), git_branch="feat/x")

    assert resp.status_code == 200
    ensure_branch.assert_called_once()
    kwargs = ensure_branch.call_args.kwargs
    assert kwargs["org_id"] == "test-org"
    assert kwargs["project_id"] == "proj-1"
    assert kwargs["name"] == "feat/x"
    assert kwargs["created_by"] == USER
    for call in write.await_args_list:
        assert call.kwargs["branch_id"] == BRANCH_ROW["id"]
        # Branch rows carry scope='project' (global constraint): a branch is
        # pushed-branch visible to project members, not a private draft.
        assert call.kwargs["scope"] == "project"


@pytest.mark.parametrize("default_branch", ["main", "Main", "master", "MASTER"])
def test_transcript_default_branch_means_main(default_branch):
    """``main``/``master`` (case-insensitive) map to NULL: the repo's default
    branch IS main, and a registry row for it would fork the meaning of NULL."""
    write = AsyncMock(return_value={"stored": True})
    ensure_branch = MagicMock()

    patches = _capture_patches(write, ensure_branch)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
    ):
        resp = _post_transcript(_client(user_id=USER), git_branch=default_branch)

    assert resp.status_code == 200
    ensure_branch.assert_not_called()
    for call in write.await_args_list:
        assert call.kwargs["branch_id"] is None


def test_transcript_unattributable_caller_never_auto_creates_a_branch(client):
    """The default client fixture carries no user: unattributable. Auto-bind
    must not mint branch rows for a caller nobody owns."""
    write = AsyncMock(return_value={"stored": True})
    ensure_branch = MagicMock()

    patches = _capture_patches(write, ensure_branch)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
    ):
        resp = _post_transcript(client, git_branch="feat/x")

    assert resp.status_code == 200
    ensure_branch.assert_not_called()
    for call in write.await_args_list:
        assert call.kwargs["branch_id"] is None


def test_transcript_without_a_project_never_auto_creates_a_branch():
    """No project resolved → no branch (global constraint): the write falls
    through to its normal scope on main."""
    write = AsyncMock(return_value={"stored": True})
    ensure_branch = MagicMock()

    session = MagicMock()
    with (
        patch("app.capture.SessionLocal", MagicMock(return_value=session)),
        patch.object(capture.brain, "write_scoped_memory", write),
        patch.object(capture.brain, "ensure_project", MagicMock(return_value=None)),
        patch.object(capture.branches, "ensure_branch", ensure_branch),
        patch("app.capture.generate_summary", AsyncMock(return_value="a summary")),
        patch("app.capture.extract_typed_memories", AsyncMock(return_value=[])),
        patch("app.capture.enqueue_trace_outbox", AsyncMock()),
    ):
        resp = _client(user_id=USER).post(
            "/capture/transcript",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "git_branch": "feat/x",
            },
        )

    assert resp.status_code == 200
    ensure_branch.assert_not_called()
    for call in write.await_args_list:
        assert call.kwargs["branch_id"] is None


def test_event_git_branch_binds_the_write():
    write = AsyncMock(return_value={"stored": True})
    ensure_branch = MagicMock(return_value=dict(BRANCH_ROW))

    patches = _capture_patches(write, ensure_branch)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
    ):
        resp = _client(user_id=USER).post(
            "/capture/event",
            json={
                "content": "decided to use the pooled endpoint",
                "memory_type": "decision",
                "git_remote": "git@example.com:klio.git",
                "git_branch": "feat/x",
            },
        )

    assert resp.status_code == 200
    ensure_branch.assert_called_once()
    write.assert_awaited_once()
    assert write.await_args.kwargs["branch_id"] == BRANCH_ROW["id"]
    assert write.await_args.kwargs["scope"] == "project"


def test_event_without_git_branch_is_byte_identical(client):
    """No field ⇒ no behaviour change: branch_id is None and the registry is
    never consulted (the untouched-suite constraint, asserted locally)."""
    write = AsyncMock(return_value={"stored": True})
    ensure_branch = MagicMock()

    session = MagicMock()
    with (
        patch("app.capture.SessionLocal", MagicMock(return_value=session)),
        patch.object(capture.brain, "write_scoped_memory", write),
        patch.object(capture.brain, "ensure_project", MagicMock(return_value=None)),
        patch.object(capture.branches, "ensure_branch", ensure_branch),
    ):
        resp = client.post("/capture/event", json={"content": "do the thing"})

    assert resp.status_code == 200
    ensure_branch.assert_not_called()
    assert write.await_args.kwargs["branch_id"] is None
