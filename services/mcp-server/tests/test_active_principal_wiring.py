"""The MCP tools act in the SELECTED node, not the key's org.

Task B1 of the context-network wiring: every tool body resolves its principal
through ``resolve_active_principal(request, db)``, so an ``X-Klio-Node``
header that names a reachable node redirects the whole call — writes land
there, reads look there — while the absence of a header keeps today's
behaviour byte-identical (the untouched rest of the suite is the proof of
that; the no-header test here is a belt on those braces).

The seams are the tool-layer conventions used across this directory: identity
is patched at ``app.identity`` (so the REAL membership check runs against real
workspace-shaped tables inside a rolled-back transaction, exactly as
``test_node_selection.py`` builds them), while storage is patched at
``app.mcp_app`` and asserted on the ``org_id`` it was handed — the binding
this task exists to move.
"""

import contextlib
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from shared.testing import database_url, requires_db

from app import identity as identity_module
from app import mcp_app

DB_URL = database_url()

SUBJECT_USER = "44444444-4444-4444-4444-444444444444"
OTHER_USER = "55555555-5555-5555-5555-555555555555"

KEY_ORG = "org_wiring_team"
OTHER_NODE_ORG = "org_wiring_personal"
STRANGER_ORG = "org_wiring_stranger"

KEY_WORKSPACE_ID = "cccccccc-0000-0000-0000-000000000001"
PERSONAL_WORKSPACE_ID = "cccccccc-0000-0000-0000-000000000002"
STRANGER_WORKSPACE_ID = "cccccccc-0000-0000-0000-000000000003"


class _State:
    pass


class _Req:
    """Just enough request: dict headers (``.get`` is all that is read)."""

    def __init__(self, headers=None):
        self.headers = headers or {}
        self.state = _State()


def _ctx(request):
    ctx = MagicMock()
    ctx.request_context.request = request
    return ctx


class _KeepOpen:
    """The fixture connection with ``close()`` neutered.

    The tool bodies close the session they open — correct per-request, fatal
    for a fixture-owned connection the rollback still needs.
    """

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):  # noqa: D102 - the whole point of the wrapper
        pass


@pytest.fixture()
def wired(monkeypatch):
    """Real workspace-shaped tables + patched identity, rolled back after.

    Identity is patched where :mod:`app.identity` looks it up, so the wrapper
    under test and the membership check inside ``resolve_active_org`` run for
    real; only who-holds-the-key and who-the-human-is are inputs here.
    """
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
        for workspace_id, slug, owner in (
            (KEY_WORKSPACE_ID, "wiring-team", SUBJECT_USER),
            (PERSONAL_WORKSPACE_ID, "wiring-personal", SUBJECT_USER),
            (STRANGER_WORKSPACE_ID, "wiring-stranger", OTHER_USER),
        ):
            conn.execute(
                text(
                    "INSERT INTO workspaces (id, slug, owner_id) "
                    "VALUES (:id, :slug, :owner)"
                ),
                {"id": workspace_id, "slug": slug, "owner": owner},
            )
        for workspace_id, member in (
            (KEY_WORKSPACE_ID, SUBJECT_USER),
            (PERSONAL_WORKSPACE_ID, SUBJECT_USER),
            (STRANGER_WORKSPACE_ID, OTHER_USER),
        ):
            conn.execute(
                text(
                    "INSERT INTO workspace_members (workspace_id, member_id) "
                    "VALUES (:w, :m)"
                ),
                {"w": workspace_id, "m": member},
            )
        for org_id, slug, workspace_id in (
            (KEY_ORG, "wiring-team", KEY_WORKSPACE_ID),
            (OTHER_NODE_ORG, "wiring-personal", PERSONAL_WORKSPACE_ID),
            (STRANGER_ORG, "wiring-stranger", STRANGER_WORKSPACE_ID),
        ):
            conn.execute(
                text(
                    "INSERT INTO organizations "
                    "(org_id, name, account_slug, workspace_id) "
                    "VALUES (:org_id, :name, :slug, :workspace_id)"
                ),
                {
                    "org_id": org_id,
                    "name": org_id,
                    "slug": slug,
                    "workspace_id": workspace_id,
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


def _write_patches(write_mock):
    return (
        patch.object(mcp_app, "check_memory_quota", return_value=False),
        patch.object(mcp_app, "record_usage"),
        patch.object(mcp_app, "resolve_project_from_request", return_value=None),
        patch.object(mcp_app, "_explicit_write_scope", return_value="org"),
        patch.object(mcp_app.brain, "write_scoped_memory", new=write_mock),
    )


def _recall_patches(retrieve_mock):
    return (
        patch.object(mcp_app, "check_memory_quota", return_value=False),
        patch.object(mcp_app, "record_usage"),
        patch.object(mcp_app, "resolve_project_from_request", return_value=None),
        patch.object(mcp_app, "retrieve_memories_scoped", new=retrieve_mock),
        patch.object(mcp_app, "log_recall_event"),
        patch.object(mcp_app.claims, "live_claims", return_value=[]),
    )


needs_db = requires_db("the active-principal wiring tests")


@needs_db
@pytest.mark.asyncio
async def test_remember_stores_in_the_selected_node(wired):
    write_mock = AsyncMock(return_value={"stored": True, "id": "m1"})
    ctx = _ctx(_Req({"X-Klio-Node": OTHER_NODE_ORG}))
    p = _write_patches(write_mock)
    with p[0], p[1], p[2], p[3], p[4]:
        result = await mcp_app._perform_write(ctx, content="x", memory_type="memory")
    assert result["stored"] is True
    write_mock.assert_awaited_once()
    assert write_mock.await_args.kwargs["org_id"] == OTHER_NODE_ORG


@needs_db
@pytest.mark.asyncio
async def test_recall_reads_the_selected_node(wired):
    retrieve_mock = AsyncMock(return_value=[])
    ctx = _ctx(_Req({"X-Klio-Node": OTHER_NODE_ORG}))
    p = _recall_patches(retrieve_mock)
    with p[0], p[1], p[2], p[3], p[4], p[5]:
        result = await mcp_app._perform_recall(
            ctx, query="q", scope="org", space=None, project=None
        )
    assert "error" not in result
    retrieve_mock.assert_awaited_once()
    # (db, org_id, query, ...): the tenancy binding is the second positional.
    assert retrieve_mock.await_args.args[1] == OTHER_NODE_ORG


@needs_db
@pytest.mark.asyncio
async def test_without_a_header_the_keys_org_is_used(wired):
    write_mock = AsyncMock(return_value={"stored": True, "id": "m1"})
    ctx = _ctx(_Req({}))
    p = _write_patches(write_mock)
    with p[0], p[1], p[2], p[3], p[4]:
        result = await mcp_app._perform_write(ctx, content="x", memory_type="memory")
    assert result["stored"] is True
    assert write_mock.await_args.kwargs["org_id"] == KEY_ORG


@needs_db
@pytest.mark.asyncio
async def test_an_unreachable_node_is_the_404_shape_not_a_500(wired):
    """A refused selection is the clean unknown-node refusal, not a crash.

    The resolver's HTTPException(404) must be caught by the tool body's
    existing auth seam and come back as an error dict — never propagate as an
    unhandled exception (a 500 to the transport) and never touch storage.
    """
    write_mock = AsyncMock()
    retrieve_mock = AsyncMock(return_value=[])

    ctx = _ctx(_Req({"X-Klio-Node": STRANGER_ORG}))
    p = _write_patches(write_mock)
    with p[0], p[1], p[2], p[3], p[4]:
        result = await mcp_app._perform_write(ctx, content="x", memory_type="memory")
    assert result["stored"] is False
    assert result["error"] == "unauthorized"
    assert "unknown node" in result["detail"]
    write_mock.assert_not_awaited()

    p = _recall_patches(retrieve_mock)
    with p[0], p[1], p[2], p[3], p[4], p[5]:
        result = await mcp_app._perform_recall(
            ctx, query="q", scope="org", space=None, project=None
        )
    assert result["error"] == "unauthorized"
    assert "unknown node" in result["detail"]
    retrieve_mock.assert_not_awaited()


_ = ANY  # keep the import honest if assertions above stop needing it
