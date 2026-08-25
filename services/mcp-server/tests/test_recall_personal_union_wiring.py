"""``_perform_recall`` stacks the caller's personal node under the active org.

Task B3 of the context-network wiring, tool-layer half: the personal org is
passed to ``retrieve_memories_scoped`` as ``extra_org_id`` under exactly three
conditions — the caller is attributable, a personal org exists
(``organizations.personal_owner_user_id``), and the active org is not already
the personal org — and every returned memory is tagged ``source:
"personal" | "active"`` from its ``org_id``.

Seams are the B1 conventions (``test_active_principal_wiring.py``): identity
patched at ``app.identity`` over real workspace-shaped tables in a rolled-back
transaction; retrieval patched at ``app.mcp_app`` and asserted on what it was
handed.
"""

import contextlib
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from shared.memory import MemoryRecord
from shared.testing import database_url, requires_db

from app import identity as identity_module
from app import mcp_app

DB_URL = database_url()

SUBJECT_USER = "44444444-4444-4444-4444-444444444444"

KEY_ORG = "org_union_team"
PERSONAL_ORG = "org_union_personal"

KEY_WORKSPACE_ID = "dddddddd-0000-0000-0000-000000000001"
PERSONAL_WORKSPACE_ID = "dddddddd-0000-0000-0000-000000000002"


class _State:
    pass


class _Req:
    def __init__(self, headers=None):
        self.headers = headers or {}
        self.state = _State()


def _ctx(request):
    ctx = MagicMock()
    ctx.request_context.request = request
    return ctx


class _KeepOpen:
    """The fixture connection with ``close()`` neutered (see B1)."""

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):  # noqa: D102 - the whole point of the wrapper
        pass


@pytest.fixture()
def wired(monkeypatch):
    """Real workspace-shaped tables + a personal org, rolled back after."""
    engine = create_engine(DB_URL)
    conn = engine.connect()
    trans = conn.begin()
    try:
        conn.execute(
            text(
                """
                CREATE TABLE workspaces (
                    id uuid PRIMARY KEY,
                    slug text UNIQUE,
                    owner_id uuid,
                    is_personal boolean NOT NULL DEFAULT false,
                    deleted_at timestamptz
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE workspace_members (
                    workspace_id uuid NOT NULL REFERENCES workspaces(id),
                    member_id uuid NOT NULL,
                    is_active boolean NOT NULL DEFAULT true,
                    deleted_at timestamptz
                )
                """
            )
        )
        for workspace_id, slug, is_personal in (
            (KEY_WORKSPACE_ID, "union-team", False),
            (PERSONAL_WORKSPACE_ID, "union-personal", True),
        ):
            conn.execute(
                text(
                    "INSERT INTO workspaces (id, slug, owner_id, is_personal) "
                    "VALUES (:id, :slug, :owner, :p)"
                ),
                {
                    "id": workspace_id,
                    "slug": slug,
                    "owner": SUBJECT_USER,
                    "p": is_personal,
                },
            )
            conn.execute(
                text(
                    "INSERT INTO workspace_members (workspace_id, member_id) "
                    "VALUES (:w, :m)"
                ),
                {"w": workspace_id, "m": SUBJECT_USER},
            )
        for org_id, slug, workspace_id, owner in (
            (KEY_ORG, "union-team", KEY_WORKSPACE_ID, None),
            (PERSONAL_ORG, "union-personal", PERSONAL_WORKSPACE_ID, SUBJECT_USER),
        ):
            conn.execute(
                text(
                    "INSERT INTO organizations "
                    "(org_id, name, account_slug, workspace_id, "
                    " personal_owner_user_id) "
                    "VALUES (:org_id, :name, :slug, :workspace_id, :owner)"
                ),
                {
                    "org_id": org_id,
                    "name": org_id,
                    "slug": slug,
                    "workspace_id": workspace_id,
                    "owner": owner,
                },
            )

        monkeypatch.setattr(
            identity_module,
            "resolve_principal_plan",
            lambda request: (KEY_ORG, "agent-1", "free", None),
        )
        monkeypatch.setattr(
            identity_module, "resolve_request_user_id", lambda request: SUBJECT_USER
        )
        monkeypatch.setattr(
            mcp_app, "resolve_request_user_id", lambda request: SUBJECT_USER
        )

        @contextlib.contextmanager
        def _fixture_workspace_db(workspace_db=None):
            yield conn

        monkeypatch.setattr(
            identity_module, "_workspace_connection", _fixture_workspace_db
        )
        monkeypatch.setattr(
            mcp_app, "SessionLocal", MagicMock(return_value=_KeepOpen(conn))
        )

        yield conn
    finally:
        trans.rollback()
        conn.close()
        engine.dispose()


def _recall_patches(retrieve_mock):
    return (
        patch.object(mcp_app, "check_memory_quota", return_value=False),
        patch.object(mcp_app, "record_usage"),
        patch.object(mcp_app, "resolve_project_from_request", return_value=None),
        patch.object(mcp_app, "retrieve_memories_scoped", new=retrieve_mock),
        patch.object(mcp_app, "log_recall_event"),
        patch.object(mcp_app.claims, "live_claims", return_value=[]),
    )


def _record(org_id: str, content: str) -> MemoryRecord:
    return MemoryRecord(
        id=uuid.uuid4(),
        session_id="sess-1",
        org_id=org_id,
        agent_id="agent-1",
        memory_type="fact",
        content=content,
        scope="org",
    )


async def _run_recall(ctx, retrieve_mock):
    p = _recall_patches(retrieve_mock)
    with p[0], p[1], p[2], p[3], p[4], p[5]:
        return await mcp_app._perform_recall(
            ctx, query="q", scope="org", space=None, project=None
        )


needs_db = requires_db("the personal-union recall wiring tests")


@needs_db
@pytest.mark.asyncio
async def test_recall_passes_the_personal_org_as_extra_org_id(wired):
    retrieve_mock = AsyncMock(return_value=[])
    result = await _run_recall(_ctx(_Req({})), retrieve_mock)
    assert "error" not in result
    retrieve_mock.assert_awaited_once()
    assert retrieve_mock.await_args.args[1] == KEY_ORG
    assert retrieve_mock.await_args.kwargs["extra_org_id"] == PERSONAL_ORG


@needs_db
@pytest.mark.asyncio
async def test_memories_are_tagged_by_source(wired):
    team = _record(KEY_ORG, "team fact")
    personal = _record(PERSONAL_ORG, "personal fact")
    retrieve_mock = AsyncMock(return_value=[team, personal])
    result = await _run_recall(_ctx(_Req({})), retrieve_mock)
    assert "error" not in result
    # compress_payload hoists row-identical fields into `common`; source
    # differs across the two rows here, so it stays per-row.
    by_content = {m["content"]: m for m in result["memories"]}
    assert by_content["team fact"]["source"] == "active"
    assert by_content["personal fact"]["source"] == "personal"


@needs_db
@pytest.mark.asyncio
async def test_no_union_when_the_active_org_is_the_personal_org(wired):
    """Selecting the personal node makes it the ACTIVE org — stacking it under
    itself would double-read one tenant, so the union stands down."""
    retrieve_mock = AsyncMock(return_value=[])
    result = await _run_recall(_ctx(_Req({"X-Klio-Node": PERSONAL_ORG})), retrieve_mock)
    assert "error" not in result
    assert retrieve_mock.await_args.args[1] == PERSONAL_ORG
    assert retrieve_mock.await_args.kwargs.get("extra_org_id") is None


@needs_db
@pytest.mark.asyncio
async def test_no_union_for_an_unattributable_caller(wired, monkeypatch):
    """No user ⇒ no personal org to stack, and no source tags either."""
    monkeypatch.setattr(mcp_app, "resolve_request_user_id", lambda request: None)
    team = _record(KEY_ORG, "team fact")
    retrieve_mock = AsyncMock(return_value=[team])
    result = await _run_recall(_ctx(_Req({})), retrieve_mock)
    assert "error" not in result
    assert retrieve_mock.await_args.kwargs.get("extra_org_id") is None
    assert "source" not in result["memories"][0]


@needs_db
@pytest.mark.asyncio
async def test_no_union_when_no_personal_org_exists(wired, monkeypatch):
    """An attributable caller without a personal org recalls exactly as
    before — condition (b) of the three."""
    wired.execute(
        text(
            "UPDATE organizations SET personal_owner_user_id = NULL WHERE org_id = :o"
        ),
        {"o": PERSONAL_ORG},
    )
    retrieve_mock = AsyncMock(return_value=[])
    result = await _run_recall(_ctx(_Req({})), retrieve_mock)
    assert "error" not in result
    assert retrieve_mock.await_args.kwargs.get("extra_org_id") is None
