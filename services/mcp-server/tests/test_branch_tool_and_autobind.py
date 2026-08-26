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


def test_transcript_branch_never_widens_a_private_capture():
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
    # The fixture's project resolves capture scope to 'private' (the 047
    # default). A git branch must NOT widen that: nearly every developer
    # is on a feature branch nearly all the time, and forcing 'project'
    # here silently publishes private hook captures to the whole team.
    ensure_branch.assert_not_called()
    for call in write.await_args_list:
        assert call.kwargs["branch_id"] is None
        assert call.kwargs["scope"] == "private"


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

    patches = list(_capture_patches(write, ensure_branch))
    patches[5] = patch.object(
        capture.project_scope,
        "resolve_capture_scope",
        MagicMock(return_value="project"),
    )
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


def test_transcript_binds_the_branch_when_the_project_opted_into_project_scope():
    """The legitimate case: default_scope='project' (the project opted in via
    migration 047), so the capture is project-visible anyway and the branch
    binds -- a draft of the project, on the project's context branch."""
    write = AsyncMock(return_value={"stored": True})
    ensure_branch = MagicMock(return_value=BRANCH_ROW)

    patches = list(_capture_patches(write, ensure_branch))
    patches[5] = patch.object(
        capture.project_scope,
        "resolve_capture_scope",
        MagicMock(return_value="project"),
    )
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
    assert ensure_branch.call_args.kwargs["name"] == "feat/x"
    for call in write.await_args_list:
        assert call.kwargs["branch_id"] == BRANCH_ROW["id"]
        assert call.kwargs["scope"] == "project"


def test_event_branch_never_widens_a_private_capture():
    """The event-route twin of the transcript privacy fence. The two routes
    have separate gate sites in capture.py; a mutation at either one must be
    caught by its own test, not its sibling's."""
    write = AsyncMock(return_value={"stored": True})
    ensure_branch = MagicMock(return_value=BRANCH_ROW)

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
    ensure_branch.assert_not_called()
    for call in write.await_args_list:
        assert call.kwargs["branch_id"] is None
        assert call.kwargs["scope"] == "private"


# ---------------------------------------------------------------------------
# Migration 051: a work-item branch files its captures under that work item
# ---------------------------------------------------------------------------


def _project_scope_capture():
    """Patches identical to :func:`_capture_patches` but capture lands on the
    project, which is the only scope a branch is ever bound under."""
    session = MagicMock()
    return {
        "session": patch("app.capture.SessionLocal", MagicMock(return_value=session)),
        "ensure_project": patch.object(
            capture.brain, "ensure_project", MagicMock(return_value="proj-1")
        ),
        "scope": patch.object(
            capture.project_scope,
            "resolve_capture_scope",
            MagicMock(return_value="project"),
        ),
        "summary": patch("app.capture.generate_summary", AsyncMock(return_value="a summary")),
        "typed": patch("app.capture.extract_typed_memories", AsyncMock(return_value=[])),
        "outbox": patch("app.capture.enqueue_trace_outbox", AsyncMock()),
    }


def _run_transcript(branch_row, *, git_branch="feat/x"):
    write = AsyncMock(return_value={"stored": True})
    ensure_branch = MagicMock(return_value=branch_row)
    p = _project_scope_capture()
    with (
        p["session"],
        patch.object(capture.brain, "write_scoped_memory", write),
        p["ensure_project"],
        p["scope"],
        patch.object(capture.branches, "ensure_branch", ensure_branch),
        p["summary"],
        p["typed"],
        p["outbox"],
    ):
        resp = _post_transcript(_client(user_id=USER), git_branch=git_branch)
    return resp, write


def test_a_capture_on_a_work_item_branch_inherits_the_work_item():
    """The point of a feature branch: bind once, then just work.

    Every memory the hooks capture while standing on a branch bound to a work
    item is filed under that work item, without the hook — which knows only a
    git branch name — ever naming it.
    """
    row = dict(BRANCH_ROW, work_item_id="cafe1234-1111-2222-3333-444444444444")
    resp, write = _run_transcript(row)

    assert resp.status_code == 200
    assert write.await_args_list, "the transcript wrote nothing to assert on"
    for call in write.await_args_list:
        assert call.kwargs["branch_id"] == BRANCH_ROW["id"]
        assert call.kwargs["work_item_id"] == "cafe1234-1111-2222-3333-444444444444"


def test_a_capture_on_a_plain_branch_files_no_work_item():
    """A branch with no work item must not invent one."""
    row = dict(BRANCH_ROW)
    row.pop("work_item_id", None)
    resp, write = _run_transcript(row)

    assert resp.status_code == 200
    for call in write.await_args_list:
        assert call.kwargs["branch_id"] == BRANCH_ROW["id"]
        assert call.kwargs["work_item_id"] is None


def test_a_capture_on_main_files_no_work_item():
    """No branch means no inherited filing, even on a project that has one."""
    write = AsyncMock(return_value={"stored": True})
    ensure_branch = MagicMock()
    p = _project_scope_capture()
    with (
        p["session"],
        patch.object(capture.brain, "write_scoped_memory", write),
        p["ensure_project"],
        p["scope"],
        patch.object(capture.branches, "ensure_branch", ensure_branch),
        p["summary"],
        p["typed"],
        p["outbox"],
    ):
        resp = _post_transcript(_client(user_id=USER), git_branch="main")

    assert resp.status_code == 200
    ensure_branch.assert_not_called()
    for call in write.await_args_list:
        assert call.kwargs["branch_id"] is None
        assert call.kwargs["work_item_id"] is None
