"""The ``project_link`` tool, and the linked reach on recall and writes.

Task L1, tool-layer half. Three surfaces under test:

* ``_perform_project_link`` — gates in order (attributable caller;
  engine-side ``can_write`` OR creator; active plane membership at role ≥ 15;
  neither side already linked elsewhere), then the guarded write.
* ``_perform_recall`` — passes ``extra_project_ids`` (engine ids of org
  projects linked into the caller's plane memberships) to
  ``retrieve_memories_scoped`` when the caller is attributable and a
  workspace engine is configured; ``None`` otherwise, byte-identical.
* ``_explicit_write_scope`` — a non-engine-member lands a ``project`` write
  when the project is linked and their plane role ≥ 15.

Seams are the wiring conventions (``test_recall_personal_union_wiring.py``):
identity patched at ``app.identity``; ``SessionLocal`` patched to the fixture
connection; the workspace engine patched at ``app.mcp_app`` to a stub whose
connections are pinned to a committed Plane-shaped schema — engine and
workspace tables share names, so the schema pin is what keeps the two sides
resolving exactly as the two production servers would.
"""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from shared import project_link
from shared.testing import database_url, requires_db

from app import identity as identity_module
from app import mcp_app

DB_URL = database_url()

SUBJECT_USER = "55555555-5555-4555-8555-555555555555"

needs_db = requires_db("the project-link tool tests")


class _State:
    pass


class _Req:
    def __init__(self, headers=None):
        self.headers = headers or {}
        self.state = _State()


def _ctx(request=None):
    ctx = MagicMock()
    ctx.request_context.request = request if request is not None else _Req({})
    return ctx


class _KeepOpen:
    """The fixture connection with ``close()`` neutered (see B1)."""

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):  # noqa: D102 - the whole point of the wrapper
        pass


class _SchemaEngine:
    """A workspace-engine stub: real connections pinned to the Plane schema."""

    def __init__(self, engine, schema):
        self._engine = engine
        self._schema = schema

    def connect(self):
        conn = self._engine.connect()
        conn.execute(text(f'SET search_path TO "{self._schema}"'))
        return conn


class _Wired:
    def __init__(self, conn, ws_engine, org_id):
        self.db = conn
        self.ws_engine = ws_engine
        self.org = org_id
        self.orgs = {org_id}


@pytest.fixture()
def wired(monkeypatch):
    """Engine rows deleted by org id; Plane schema committed and dropped.

    No outer rollback for the engine side: the tool COMMITS on a successful
    link, so the union-wiring rolled-back-transaction shape would leak rows.
    """
    engine = create_engine(DB_URL)
    schema = f"plane_ws_{uuid.uuid4().hex[:8]}"
    conn = engine.connect()
    setup = engine.connect()
    org_id = f"org_linktool_{uuid.uuid4().hex[:8]}"
    try:
        setup.execute(text(f'CREATE SCHEMA "{schema}"'))
        setup.execute(text(f'SET search_path TO "{schema}"'))
        setup.execute(
            text(
                "CREATE TABLE workspaces (id uuid PRIMARY KEY, deleted_at timestamptz)"
            )
        )
        setup.execute(
            text(
                """
                CREATE TABLE projects (
                    id uuid PRIMARY KEY,
                    workspace_id uuid NOT NULL REFERENCES workspaces(id),
                    deleted_at timestamptz
                )
                """
            )
        )
        setup.execute(
            text(
                """
                CREATE TABLE project_members (
                    project_id uuid NOT NULL REFERENCES projects(id),
                    member_id uuid NOT NULL,
                    role smallint NOT NULL DEFAULT 5,
                    is_active boolean NOT NULL DEFAULT true,
                    deleted_at timestamptz
                )
                """
            )
        )
        setup.commit()

        ws_engine = _SchemaEngine(engine, schema)
        monkeypatch.setattr(
            identity_module,
            "resolve_principal_plan",
            lambda request: (org_id, "agent-1", "free", None),
        )
        monkeypatch.setattr(
            identity_module, "resolve_request_user_id", lambda request: SUBJECT_USER
        )
        monkeypatch.setattr(
            mcp_app, "resolve_request_user_id", lambda request: SUBJECT_USER
        )
        monkeypatch.setattr(
            mcp_app, "SessionLocal", MagicMock(return_value=_KeepOpen(conn))
        )
        monkeypatch.setattr(mcp_app, "workspace_engine", lambda: ws_engine)

        yield _Wired(conn, ws_engine, org_id)
    finally:
        try:
            conn.rollback()
            for org in getattr(conn, "_test_orgs", None) or {org_id}:
                conn.execute(
                    text(
                        """
                        DELETE FROM project_members WHERE project_id IN (
                            SELECT id FROM projects WHERE org_id = :o)
                        """
                    ),
                    {"o": org},
                )
                conn.execute(text("DELETE FROM projects WHERE org_id = :o"), {"o": org})
            conn.commit()
            setup.rollback()
            setup.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            setup.commit()
        finally:
            conn.close()
            setup.close()
            engine.dispose()


# --- helpers ----------------------------------------------------------------


def _track_org(wired, org: str) -> str:
    orgs = getattr(wired.db, "_test_orgs", None)
    if orgs is None:
        orgs = set()
        wired.db._test_orgs = orgs
    orgs.update(wired.orgs)
    orgs.add(org)
    return org


def _plane_project(wired, *, members=()) -> str:
    ws_conn = wired.ws_engine.connect()
    try:
        workspace_id = str(uuid.uuid4())
        plane_id = str(uuid.uuid4())
        ws_conn.execute(
            text("INSERT INTO workspaces (id) VALUES (:id)"), {"id": workspace_id}
        )
        ws_conn.execute(
            text("INSERT INTO projects (id, workspace_id) VALUES (:id, :w)"),
            {"id": plane_id, "w": workspace_id},
        )
        for member_id, role in members:
            ws_conn.execute(
                text(
                    "INSERT INTO project_members (project_id, member_id, role) "
                    "VALUES (:p, :m, :r)"
                ),
                {"p": plane_id, "m": member_id, "r": role},
            )
        ws_conn.commit()
        return plane_id
    finally:
        ws_conn.close()


def _engine_project(
    wired,
    *,
    org_id=None,
    name="linkable",
    plane_project_id=None,
    members=(),
    created_by=None,
) -> str:
    org = _track_org(wired, org_id or wired.org)
    project_id = str(uuid.uuid4())
    wired.db.execute(
        text(
            """
            INSERT INTO projects
                (id, org_id, git_remote, display_name, plane_project_id,
                 created_by)
            VALUES (:id, :org, :remote, :name, :plane, :created_by)
            """
        ),
        {
            "id": project_id,
            "org": org,
            "remote": f"git@example.com:{uuid.uuid4().hex[:8]}.git",
            "name": name,
            "plane": plane_project_id,
            "created_by": created_by,
        },
    )
    for member_id, role in members:
        wired.db.execute(
            text(
                "INSERT INTO project_members (project_id, user_id, role) "
                "VALUES (:p, :u, :r)"
            ),
            {"p": project_id, "u": member_id, "r": role},
        )
    wired.db.commit()
    return project_id


def _stored_plane_id(wired, project_id: str):
    row = wired.db.execute(
        text("SELECT plane_project_id FROM projects WHERE id = :p"),
        {"p": project_id},
    ).scalar()
    return str(row) if row is not None else None


async def _link(wired, project: str, workspace_project: str):
    return await mcp_app._perform_project_link(
        _ctx(), project=project, workspace_project=workspace_project
    )


# ---------------------------------------------------------------------------
# The tool: inventory
# ---------------------------------------------------------------------------


def test_project_link_is_a_registered_tool():
    assert "project_link" in mcp_app.EXPECTED_TOOL_NAMES


# ---------------------------------------------------------------------------
# The tool: happy path and idempotence
# ---------------------------------------------------------------------------


@needs_db
@pytest.mark.asyncio
async def test_an_engine_writer_and_plane_member_links(wired):
    plane_id = _plane_project(
        wired, members=[(SUBJECT_USER, project_link.PLANE_ROLE_MEMBER)]
    )
    project_id = _engine_project(wired, members=[(SUBJECT_USER, "write")])

    result = await _link(wired, "linkable", plane_id)

    assert result == {
        "linked": True,
        "project_id": project_id,
        "workspace_project": plane_id,
    }
    assert _stored_plane_id(wired, project_id) == plane_id


@needs_db
@pytest.mark.asyncio
async def test_the_creator_may_link_without_a_membership_row(wired):
    plane_id = _plane_project(
        wired, members=[(SUBJECT_USER, project_link.PLANE_ROLE_ADMIN)]
    )
    project_id = _engine_project(wired, created_by=SUBJECT_USER)

    result = await _link(wired, project_id, plane_id)
    assert result["linked"] is True
    assert _stored_plane_id(wired, project_id) == plane_id


@needs_db
@pytest.mark.asyncio
async def test_linking_the_same_pair_twice_is_idempotent(wired):
    plane_id = _plane_project(
        wired, members=[(SUBJECT_USER, project_link.PLANE_ROLE_MEMBER)]
    )
    project_id = _engine_project(
        wired, members=[(SUBJECT_USER, "write")], plane_project_id=plane_id
    )
    result = await _link(wired, project_id, plane_id)
    assert result["linked"] is True
    assert _stored_plane_id(wired, project_id) == plane_id


# ---------------------------------------------------------------------------
# The tool: gates (test 8)
# ---------------------------------------------------------------------------


@needs_db
@pytest.mark.asyncio
async def test_a_non_plane_member_gets_not_found(wired):
    plane_id = _plane_project(wired, members=[])  # nobody is a member
    project_id = _engine_project(wired, members=[(SUBJECT_USER, "write")])

    result = await _link(wired, project_id, plane_id)
    assert result["linked"] is False
    assert result["error"] == "not_found"
    assert _stored_plane_id(wired, project_id) is None


@needs_db
@pytest.mark.asyncio
async def test_a_plane_guest_may_not_link(wired):
    """Linking is member-gated at role ≥ 15; a guest (5) reads, never links."""
    plane_id = _plane_project(
        wired, members=[(SUBJECT_USER, project_link.PLANE_ROLE_GUEST)]
    )
    project_id = _engine_project(wired, members=[(SUBJECT_USER, "write")])

    result = await _link(wired, project_id, plane_id)
    assert result["linked"] is False
    assert result["error"] == "not_found"
    assert _stored_plane_id(wired, project_id) is None


@needs_db
@pytest.mark.asyncio
async def test_an_engine_reader_who_is_no_creator_gets_not_found(wired):
    plane_id = _plane_project(
        wired, members=[(SUBJECT_USER, project_link.PLANE_ROLE_ADMIN)]
    )
    project_id = _engine_project(wired, members=[(SUBJECT_USER, "read")])

    result = await _link(wired, project_id, plane_id)
    assert result["linked"] is False
    assert result["error"] == "not_found"
    assert _stored_plane_id(wired, project_id) is None


@needs_db
@pytest.mark.asyncio
async def test_an_unknown_project_gets_not_found(wired):
    plane_id = _plane_project(
        wired, members=[(SUBJECT_USER, project_link.PLANE_ROLE_MEMBER)]
    )
    result = await _link(wired, "no-such-project", plane_id)
    assert result["linked"] is False
    assert result["error"] == "not_found"


@needs_db
@pytest.mark.asyncio
async def test_a_malformed_workspace_project_gets_not_found(wired):
    _engine_project(wired, members=[(SUBJECT_USER, "write")])
    result = await _link(wired, "linkable", "not-a-uuid")
    assert result["linked"] is False
    assert result["error"] == "not_found"


@needs_db
@pytest.mark.asyncio
async def test_an_unattributed_caller_links_nothing(wired, monkeypatch):
    monkeypatch.setattr(mcp_app, "resolve_request_user_id", lambda request: None)
    plane_id = _plane_project(wired)
    project_id = _engine_project(wired, members=[(SUBJECT_USER, "write")])
    result = await _link(wired, project_id, plane_id)
    assert result["linked"] is False
    assert result["error"] == "unattributed_caller"
    assert _stored_plane_id(wired, project_id) is None


@needs_db
@pytest.mark.asyncio
async def test_no_workspace_engine_is_a_clear_error(wired, monkeypatch):
    monkeypatch.setattr(mcp_app, "workspace_engine", lambda: None)
    plane_id = str(uuid.uuid4())
    project_id = _engine_project(wired, members=[(SUBJECT_USER, "write")])
    result = await _link(wired, project_id, plane_id)
    assert result["linked"] is False
    assert result["error"] == "workspace_unavailable"
    assert _stored_plane_id(wired, project_id) is None


# ---------------------------------------------------------------------------
# The tool: never re-point, and the error names nothing about the other org
# ---------------------------------------------------------------------------


@needs_db
@pytest.mark.asyncio
async def test_an_already_linked_engine_project_refuses_a_new_plane_id(wired):
    old_plane = _plane_project(
        wired, members=[(SUBJECT_USER, project_link.PLANE_ROLE_MEMBER)]
    )
    new_plane = _plane_project(
        wired, members=[(SUBJECT_USER, project_link.PLANE_ROLE_MEMBER)]
    )
    project_id = _engine_project(
        wired, members=[(SUBJECT_USER, "write")], plane_project_id=old_plane
    )
    result = await _link(wired, project_id, new_plane)
    assert result["linked"] is False
    assert result["error"] == "already_linked"
    assert _stored_plane_id(wired, project_id) == old_plane


@needs_db
@pytest.mark.asyncio
async def test_a_plane_id_linked_in_another_org_is_refused_namelessly(wired):
    """The refusal must not confirm WHERE the plane project is linked: no
    other org id, no other project id, nothing an outsider can harvest."""
    plane_id = _plane_project(
        wired, members=[(SUBJECT_USER, project_link.PLANE_ROLE_MEMBER)]
    )
    other_org = f"org_secret_{uuid.uuid4().hex[:8]}"
    other_project = _engine_project(
        wired, org_id=other_org, name="secret-project", plane_project_id=plane_id
    )
    project_id = _engine_project(wired, members=[(SUBJECT_USER, "write")])

    result = await _link(wired, project_id, plane_id)
    assert result["linked"] is False
    assert result["error"] == "already_linked"
    assert _stored_plane_id(wired, project_id) is None

    serialized = json.dumps(result)
    assert other_org not in serialized
    assert other_project not in serialized
    assert "secret-project" not in serialized


# ---------------------------------------------------------------------------
# Recall wiring: extra_project_ids reaches retrieve_memories_scoped
# ---------------------------------------------------------------------------


def _recall_patches(retrieve_mock):
    return (
        patch.object(mcp_app, "check_memory_quota", return_value=False),
        patch.object(mcp_app, "record_usage"),
        patch.object(mcp_app, "resolve_project_from_request", return_value=None),
        patch.object(mcp_app, "retrieve_memories_scoped", new=retrieve_mock),
        patch.object(mcp_app, "log_recall_event"),
        patch.object(mcp_app.claims, "live_claims", return_value=[]),
    )


async def _run_recall(retrieve_mock):
    p = _recall_patches(retrieve_mock)
    with p[0], p[1], p[2], p[3], p[4], p[5]:
        return await mcp_app._perform_recall(
            _ctx(), query="q", scope="org", space=None, project=None
        )


@needs_db
@pytest.mark.asyncio
async def test_recall_passes_the_linked_reach_as_extra_project_ids(wired):
    plane_id = _plane_project(
        wired, members=[(SUBJECT_USER, project_link.PLANE_ROLE_MEMBER)]
    )
    project_id = _engine_project(wired, plane_project_id=plane_id)

    retrieve_mock = AsyncMock(return_value=[])
    result = await _run_recall(retrieve_mock)
    assert "error" not in result
    retrieve_mock.assert_awaited_once()
    assert retrieve_mock.await_args.kwargs["extra_project_ids"] == [project_id]


@needs_db
@pytest.mark.asyncio
async def test_a_guest_membership_still_grants_read_reach(wired):
    """Read reach begins at role 5: guests recall, they just cannot write."""
    plane_id = _plane_project(
        wired, members=[(SUBJECT_USER, project_link.PLANE_ROLE_GUEST)]
    )
    project_id = _engine_project(wired, plane_project_id=plane_id)
    retrieve_mock = AsyncMock(return_value=[])
    result = await _run_recall(retrieve_mock)
    assert "error" not in result
    assert retrieve_mock.await_args.kwargs["extra_project_ids"] == [project_id]


@needs_db
@pytest.mark.asyncio
async def test_recall_without_a_workspace_engine_is_byte_identical(wired, monkeypatch):
    monkeypatch.setattr(mcp_app, "workspace_engine", lambda: None)
    plane_id = _plane_project(
        wired, members=[(SUBJECT_USER, project_link.PLANE_ROLE_MEMBER)]
    )
    _engine_project(wired, plane_project_id=plane_id)
    retrieve_mock = AsyncMock(return_value=[])
    result = await _run_recall(retrieve_mock)
    assert "error" not in result
    assert retrieve_mock.await_args.kwargs.get("extra_project_ids") is None


@needs_db
@pytest.mark.asyncio
async def test_recall_without_plane_memberships_passes_no_reach(wired):
    """The arm is built from the CALLER's memberships, never from the org's
    links: a linked project with no membership for this caller adds nothing."""
    plane_id = _plane_project(wired, members=[])
    _engine_project(wired, plane_project_id=plane_id)
    retrieve_mock = AsyncMock(return_value=[])
    result = await _run_recall(retrieve_mock)
    assert "error" not in result
    assert retrieve_mock.await_args.kwargs.get("extra_project_ids") is None


@needs_db
@pytest.mark.asyncio
async def test_recall_for_an_unattributed_caller_passes_no_reach(wired, monkeypatch):
    monkeypatch.setattr(mcp_app, "resolve_request_user_id", lambda request: None)
    plane_id = _plane_project(
        wired, members=[(SUBJECT_USER, project_link.PLANE_ROLE_MEMBER)]
    )
    _engine_project(wired, plane_project_id=plane_id)
    retrieve_mock = AsyncMock(return_value=[])
    result = await _run_recall(retrieve_mock)
    assert "error" not in result
    assert retrieve_mock.await_args.kwargs.get("extra_project_ids") is None


@needs_db
@pytest.mark.asyncio
async def test_a_workspace_lookup_failure_degrades_to_no_reach(wired, monkeypatch):
    """Best-effort: a workspace DB failure must never fail the recall."""

    class _Broken:
        def connect(self):
            raise RuntimeError("workspace db is down")

    monkeypatch.setattr(mcp_app, "workspace_engine", lambda: _Broken())
    retrieve_mock = AsyncMock(return_value=[])
    result = await _run_recall(retrieve_mock)
    assert "error" not in result
    assert retrieve_mock.await_args.kwargs.get("extra_project_ids") is None


# ---------------------------------------------------------------------------
# Explicit writes: a plane member ≥ 15 lands a project write on a linked
# project; a guest does not
# ---------------------------------------------------------------------------


@needs_db
def test_explicit_write_scope_reaches_through_the_link(wired):
    plane_id = _plane_project(
        wired,
        members=[
            (SUBJECT_USER, project_link.PLANE_ROLE_MEMBER),
        ],
    )
    project_id = _engine_project(wired, plane_project_id=plane_id)

    scope = mcp_app._explicit_write_scope(
        wired.db,
        requested_scope="org",
        org_id=wired.org,
        user_id=SUBJECT_USER,
        project_id=project_id,
    )
    assert scope == "project"


@needs_db
def test_explicit_write_scope_denies_a_plane_guest(wired):
    plane_id = _plane_project(
        wired, members=[(SUBJECT_USER, project_link.PLANE_ROLE_GUEST)]
    )
    project_id = _engine_project(wired, plane_project_id=plane_id)

    scope = mcp_app._explicit_write_scope(
        wired.db,
        requested_scope="org",
        org_id=wired.org,
        user_id=SUBJECT_USER,
        project_id=project_id,
    )
    assert scope == "org"


@needs_db
def test_explicit_write_scope_without_a_workspace_engine_is_unchanged(
    wired, monkeypatch
):
    monkeypatch.setattr(mcp_app, "workspace_engine", lambda: None)
    plane_id = _plane_project(
        wired, members=[(SUBJECT_USER, project_link.PLANE_ROLE_MEMBER)]
    )
    linked = _engine_project(wired, plane_project_id=plane_id)
    membered = _engine_project(
        wired, name="membered", members=[(SUBJECT_USER, "write")]
    )

    kwargs = {"requested_scope": "org", "org_id": wired.org, "user_id": SUBJECT_USER}
    # The linked-only path closes without a workspace engine…
    assert mcp_app._explicit_write_scope(wired.db, project_id=linked, **kwargs) == "org"
    # …while engine membership keeps working exactly as today.
    assert (
        mcp_app._explicit_write_scope(wired.db, project_id=membered, **kwargs)
        == "project"
    )
