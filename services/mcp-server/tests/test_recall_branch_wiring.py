"""The ``recall`` tool's context-branch wiring (Track C, Task C2).

Mirrors the mock style of ``tests/test_recall_tool.py``: the tool callable is
reached via the FastMCP registry and invoked with a hand-built fake
:class:`Context`; the DB / resolution seams are patched at the
``app.mcp_app`` module boundary.

Under test is the RESOLUTION contract, not the SQL (that lives in
``services/shared/tests/test_branch_reads.py``):

* ``branch`` resolves via ``branches.get_branch`` against the RESOLVED
  project — no project, no branch (warn-and-empty), unknown branch
  warn-and-empty, exactly like an unknown space. Strict on purpose: a recall
  that silently dropped the branch filter would answer from main while the
  caller believes it is reading the branch.
* ``main`` (case-insensitive) and blank are the DEFAULT read, not a lookup:
  main is the absence of a branch (``branch_id IS NULL``), so naming it must
  neither error nor consult the registry.
* Each returned memory whose row carries the branch's id is tagged
  ``source_branch: <name>``; main rows are not tagged, and the B3 ``source``
  tag is untouched.
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.mcp_app as mcp_app
from app.mcp_app import RECALL_WRITE_NUDGE, build_mcp_server
from shared.memory import MemoryRecord


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


def _recall_fn():
    server = build_mcp_server()
    tool = server._tool_manager.get_tool("recall")
    assert tool is not None, "tool 'recall' not registered"
    return tool.fn


def _record(*, content: str, branch_id: str | None = None) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        session_id="klio-brain:org-x:agent-y",
        org_id="org-x",
        agent_id="agent-y",
        memory_type="memory",
        content=content,
        scope="project",
        status="active",
        created_at=now,
        updated_at=now,
        project_id="proj-1",
        branch_id=branch_id,
    )


def _principal():
    return patch(
        "app.mcp_app.resolve_active_principal",
        return_value=("org-x", "agent-y", "free", None),
    )


BRANCH_ROW = {
    "id": "beefbeef-1111-2222-3333-444444444444",
    "org_id": "org-x",
    "project_id": "proj-1",
    "name": "feat/x",
    "status": "open",
    "created_by": None,
    "created_at": None,
    "closed_at": None,
}


# ---------------------------------------------------------------------------
# Resolution: needs a project, unknown warns, known forwards the id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_branch_without_a_project_warns_and_returns_empty():
    """A branch always belongs to a project (global constraint): naming one
    with no project resolvable is warn-and-empty, never a silent main read."""
    fn = _recall_fn()
    retrieve_mock = AsyncMock(return_value=[])

    with (
        _principal(),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.resolve_project_from_request", return_value=None),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        result = await fn(_fake_ctx(), "q", branch="feat/x")

    assert result == {
        "memories": [],
        "count": 0,
        "warning": (
            "branch 'feat/x' needs a project: pass project= or send the "
            "X-Vex-Project header"
        ),
        "guidance": RECALL_WRITE_NUDGE,
    }
    retrieve_mock.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_branch_warns_and_returns_empty():
    fn = _recall_fn()
    retrieve_mock = AsyncMock(return_value=[])

    with (
        _principal(),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "find_project_id", return_value="proj-1"),
        patch.object(mcp_app.branches, "get_branch", return_value=None),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        result = await fn(_fake_ctx(), "q", project="klio", branch="ghost")

    assert result == {
        "memories": [],
        "count": 0,
        "warning": "unknown branch: ghost",
        "guidance": RECALL_WRITE_NUDGE,
    }
    retrieve_mock.assert_not_called()


@pytest.mark.asyncio
async def test_known_branch_resolves_against_the_project_and_forwards_its_id():
    fn = _recall_fn()
    retrieve_mock = AsyncMock(return_value=[])
    get_branch = MagicMock(return_value=BRANCH_ROW)

    with (
        _principal(),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "find_project_id", return_value="proj-1"),
        patch.object(mcp_app.branches, "get_branch", get_branch),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        await fn(_fake_ctx(), "q", project="klio", branch="  feat/x  ")

    # Stripped, and resolved against the RESOLVED project in the caller's org.
    assert get_branch.call_args.kwargs["org_id"] == "org-x"
    assert get_branch.call_args.kwargs["project_id"] == "proj-1"
    assert get_branch.call_args.kwargs["name"] == "feat/x"
    assert retrieve_mock.await_args.kwargs["branch_id"] == BRANCH_ROW["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("named", ["main", "MAIN", "  ", None])
async def test_main_and_blank_are_the_default_read(named):
    """``main`` is the absence of a branch: no registry lookup, no warning,
    branch_id=None — byte-identical to not naming a branch at all."""
    fn = _recall_fn()
    retrieve_mock = AsyncMock(return_value=[])
    get_branch = MagicMock()

    with (
        _principal(),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "find_project_id", return_value="proj-1"),
        patch.object(mcp_app.branches, "get_branch", get_branch),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        result = await fn(_fake_ctx(), "q", project="klio", branch=named)

    get_branch.assert_not_called()
    retrieve_mock.assert_awaited_once()
    assert retrieve_mock.await_args.kwargs["branch_id"] is None
    assert "warning" not in result


@pytest.mark.asyncio
async def test_no_branch_argument_forwards_none():
    """The untouched call shape: branch_id=None reaches retrieve, so the
    shared layer applies its main-only default."""
    fn = _recall_fn()
    retrieve_mock = AsyncMock(return_value=[])

    with (
        _principal(),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.resolve_project_from_request", return_value=None),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        await fn(_fake_ctx(), "q")

    assert retrieve_mock.await_args.kwargs["branch_id"] is None


# ---------------------------------------------------------------------------
# Tagging: branch rows carry source_branch, main rows do not
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_branch_rows_are_tagged_source_branch_and_main_rows_are_not():
    fn = _recall_fn()
    on_branch = _record(content="branch draft", branch_id=BRANCH_ROW["id"])
    on_main = _record(content="main truth")
    retrieve_mock = AsyncMock(return_value=[on_branch, on_main])

    with (
        _principal(),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch.object(mcp_app.brain, "find_project_id", return_value="proj-1"),
        patch.object(mcp_app.branches, "get_branch", return_value=BRANCH_ROW),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        result = await fn(_fake_ctx(), "q", project="klio", branch="feat/x")

    common = result.get("common", {})
    assert "source_branch" not in common  # not identical on every row
    by_content = {m["content"]: m for m in result["memories"]}
    assert by_content["branch draft"].get("source_branch") == "feat/x"
    assert "source_branch" not in by_content["main truth"]


@pytest.mark.asyncio
async def test_no_branch_recall_payload_is_untagged():
    """No header/param ⇒ byte-identical behaviour: without a named branch no
    memory carries source_branch, even if a record (impossibly) had one."""
    fn = _recall_fn()
    retrieve_mock = AsyncMock(return_value=[_record(content="main truth")])

    with (
        _principal(),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.resolve_project_from_request", return_value=None),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve_mock),
    ):
        result = await fn(_fake_ctx(), "q")

    for mem in result["memories"]:
        assert "source_branch" not in mem
    assert "source_branch" not in result.get("common", {})
