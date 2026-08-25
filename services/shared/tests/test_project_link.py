"""Workspace ⇄ engine project links — membership reaches through. Real Postgres.

Task L1 of the project-link track: an engine project joined to a workspace
(Plane) project by ``projects.plane_project_id`` lets the WORKSPACE project's
membership grant read reach (and, at role ≥ 15, explicit-write permission)
over the engine project's ``scope='project'`` memories.

Two databases, joined in Python — the ``shared/membership.py`` pattern. The
workspace side answers "which plane projects is this user in, at what role";
the engine side answers "which of those are linked to my org's projects"; the
recall ladder gains one precomputed ``ANY(:extra_project_ids)`` arm.

The fixture creates Plane-shaped ``workspaces`` / ``projects`` /
``project_members`` tables FOR REAL — but in a dedicated schema, because the
engine's own database also has tables named ``projects`` and
``project_members``. The workspace-side connection pins ``search_path`` to
that schema, so bare table names resolve exactly as they do in production:
the workspace connection sees Plane's tables, the engine connection sees the
engine's. The schema is dropped on teardown.

``embed_text`` is mocked so every row shares one vector and ranking cannot
mask an admission bug — under test is which rows the WHERE clause admits.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

from shared.brain import find_project_id
from shared.memory import retrieve_memories_scoped
from shared.project_link import (
    PLANE_ROLE_ADMIN,
    PLANE_ROLE_GUEST,
    PLANE_ROLE_MEMBER,
    can_write_linked,
    link_engine_project,
    linked_engine_projects,
    plane_project_memberships,
)
from shared.testing import database_url, requires_db

DB_URL = database_url()

needs_db = requires_db("the project-link tests")

EMBED_DIM = 1024
FIXED_EMBEDDING = [0.1] * EMBED_DIM
FIXED_EMBEDDING_LITERAL = "[" + ",".join(str(x) for x in FIXED_EMBEDDING) + "]"

ME = "aaaaaaaa-1111-2222-3333-555555555555"
SOMEONE_ELSE = "bbbbbbbb-1111-2222-3333-555555555555"

QUERY = "the shared query every fixed-vector row matches"


class _Link:
    """The fixture's return value: the two connections plus bookkeeping."""

    def __init__(self, ws, eng, orgs):
        self.ws = ws
        self.eng = eng
        self.orgs = orgs


@pytest.fixture()
def link():
    """Plane-shaped tables in a dedicated schema; engine rows cleaned after.

    The schema is COMMITTED (a second connection must see it) and dropped in
    teardown; engine-side rows are deleted by org id, the
    ``test_recall_personal_union.py`` convention.
    """
    engine = create_engine(DB_URL)
    schema = f"plane_ws_{uuid.uuid4().hex[:8]}"
    ws = engine.connect()
    eng = engine.connect()
    orgs: set[str] = set()
    try:
        ws.execute(text(f'CREATE SCHEMA "{schema}"'))
        # Pin the workspace connection to the Plane schema ALONE: if the
        # workspace-side SQL ever names an engine-only table, it errors here
        # instead of silently resolving against public.
        ws.execute(text(f'SET search_path TO "{schema}"'))
        ws.execute(
            text(
                """
                CREATE TABLE workspaces (
                    id uuid PRIMARY KEY,
                    deleted_at timestamptz
                )
                """
            )
        )
        ws.execute(
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
        ws.execute(
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
        ws.commit()
        yield _Link(ws, eng, orgs)
    finally:
        try:
            eng.rollback()
            for org_id in orgs:
                eng.execute(
                    text(
                        """
                        DELETE FROM project_members WHERE project_id IN (
                            SELECT id FROM projects WHERE org_id = :o)
                        """
                    ),
                    {"o": org_id},
                )
                eng.execute(
                    text("DELETE FROM session_memories WHERE org_id = :o"),
                    {"o": org_id},
                )
                eng.execute(
                    text("DELETE FROM projects WHERE org_id = :o"), {"o": org_id}
                )
            eng.commit()
            ws.rollback()
            ws.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            ws.commit()
        finally:
            ws.close()
            eng.close()
            engine.dispose()


# --- fixture helpers ---------------------------------------------------------


def _plane_workspace(link) -> str:
    workspace_id = str(uuid.uuid4())
    link.ws.execute(
        text("INSERT INTO workspaces (id) VALUES (:id)"), {"id": workspace_id}
    )
    link.ws.commit()
    return workspace_id


def _plane_project(link, workspace_id: str) -> str:
    project_id = str(uuid.uuid4())
    link.ws.execute(
        text("INSERT INTO projects (id, workspace_id) VALUES (:id, :w)"),
        {"id": project_id, "w": workspace_id},
    )
    link.ws.commit()
    return project_id


def _plane_member(
    link, project_id: str, member_id: str, *, role: int = PLANE_ROLE_MEMBER
) -> None:
    link.ws.execute(
        text(
            "INSERT INTO project_members (project_id, member_id, role) "
            "VALUES (:p, :m, :r)"
        ),
        {"p": project_id, "m": member_id, "r": role},
    )
    link.ws.commit()


def _org(link, prefix: str = "org_link") -> str:
    org_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
    link.orgs.add(org_id)
    return org_id


def _engine_project(
    link,
    *,
    org_id: str,
    name: str = "linked",
    plane_project_id: str | None = None,
    members: list[str] | None = None,
    created_by: str | None = None,
) -> str:
    project_id = str(uuid.uuid4())
    link.eng.execute(
        text(
            """
            INSERT INTO projects
                (id, org_id, git_remote, display_name, plane_project_id,
                 created_by)
            VALUES (:id, :org_id, :remote, :name, :plane, :created_by)
            """
        ),
        {
            "id": project_id,
            "org_id": org_id,
            "remote": f"git@example.com:{uuid.uuid4().hex[:8]}.git",
            "name": name,
            "plane": plane_project_id,
            "created_by": created_by,
        },
    )
    for member in members or []:
        link.eng.execute(
            text(
                "INSERT INTO project_members (project_id, user_id, role) "
                "VALUES (:p, :u, 'write')"
            ),
            {"p": project_id, "u": member},
        )
    link.eng.commit()
    return project_id


def _seed(
    link,
    *,
    org_id: str,
    content: str,
    scope: str = "project",
    project_id: str | None = None,
    user_id: str | None = None,
) -> str:
    memory_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    link.eng.execute(
        text(
            """
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, execution_id,
                memory_type, content, embedding, scope, status,
                user_id, provenance, project_id, metadata,
                created_at, updated_at
            ) VALUES (
                :id, :session_id, :org_id, 'agent-1', NULL,
                'fact', :content, :embedding, :scope, 'active',
                :user_id, 'EXTRACTED', :project_id, '{}'::jsonb,
                :created_at, :updated_at
            )
            """
        ),
        {
            "id": memory_id,
            "session_id": f"sess-{uuid.uuid4().hex[:6]}",
            "org_id": org_id,
            "content": content,
            "embedding": FIXED_EMBEDDING_LITERAL,
            "scope": scope,
            "user_id": user_id,
            "project_id": project_id,
            "created_at": now,
            "updated_at": now,
        },
    )
    link.eng.commit()
    return memory_id


def _mock_embed():
    return patch(
        "shared.memory.embed_text",
        new_callable=AsyncMock,
        return_value=FIXED_EMBEDDING,
    )


def _reach(link, *, org_id: str, user_id: str) -> list[str] | None:
    """Compose the two halves exactly as the recall path does."""
    memberships = plane_project_memberships(link.ws, user_id=user_id)
    plane_ids = [pid for pid, role in memberships.items() if role >= PLANE_ROLE_GUEST]
    if not plane_ids:
        return None
    linked = linked_engine_projects(link.eng, org_id=org_id, plane_ids=plane_ids)
    return sorted(linked) or None


async def _recall(link, org_id: str, **kwargs):
    with _mock_embed():
        return await retrieve_memories_scoped(
            link.eng, org_id, QUERY, limit=50, **kwargs
        )


# ---------------------------------------------------------------------------
# 1. Linked + plane-member recalls via the extra arm; unlinked gets nothing
# ---------------------------------------------------------------------------


@needs_db
@pytest.mark.asyncio
async def test_linked_plane_member_recalls_and_the_link_is_load_bearing(link):
    """A plane member (role 15) of a LINKED project reads its memories; the
    same user with the link removed gets nothing. Mutation check: dropping
    the extra arm from ``retrieve_memories_scoped`` fails this test — ME is
    not an engine ``project_members`` row, so no other arm admits the row."""
    org = _org(link)
    plane_ws = _plane_workspace(link)
    plane_project = _plane_project(link, plane_ws)
    _plane_member(link, plane_project, ME, role=PLANE_ROLE_MEMBER)
    engine_project = _engine_project(link, org_id=org, plane_project_id=plane_project)
    row = _seed(link, org_id=org, content="linked truth", project_id=engine_project)

    records = await _recall(
        link, org, user_id=ME, extra_project_ids=_reach(link, org_id=org, user_id=ME)
    )
    assert row in {str(r.id) for r in records}

    # Remove the link: the same user's reach recomputes to nothing.
    link.eng.execute(
        text("UPDATE projects SET plane_project_id = NULL WHERE id = :p"),
        {"p": engine_project},
    )
    link.eng.commit()
    records = await _recall(
        link, org, user_id=ME, extra_project_ids=_reach(link, org_id=org, user_id=ME)
    )
    assert row not in {str(r.id) for r in records}


@needs_db
@pytest.mark.asyncio
async def test_empty_or_none_extra_project_ids_change_nothing(link):
    """``None`` and ``[]`` are both "no extra arm" — byte-identical recall."""
    org = _org(link)
    engine_project = _engine_project(link, org_id=org)
    row = _seed(link, org_id=org, content="fenced", project_id=engine_project)
    for absent in (None, []):
        records = await _recall(link, org, user_id=ME, extra_project_ids=absent)
        assert row not in {str(r.id) for r in records}


# ---------------------------------------------------------------------------
# 2. The arm is built from THEIR memberships, not from all org links
# ---------------------------------------------------------------------------


@needs_db
@pytest.mark.asyncio
async def test_membership_not_linkage_is_the_boundary(link):
    """A plane member of an UNLINKED project gets nothing; a plane NON-member
    of a linked project gets nothing. Mutation check: building the arm from
    ALL org links (instead of the caller's memberships) fails the second
    half — SOMEONE_ELSE would read a project they were never added to."""
    org = _org(link)
    plane_ws = _plane_workspace(link)

    # ME is a member of an UNLINKED plane project...
    unlinked_plane = _plane_project(link, plane_ws)
    _plane_member(link, unlinked_plane, ME)
    # ...while a DIFFERENT plane project is linked, with a different member.
    linked_plane = _plane_project(link, plane_ws)
    _plane_member(link, linked_plane, SOMEONE_ELSE)
    engine_project = _engine_project(link, org_id=org, plane_project_id=linked_plane)
    row = _seed(link, org_id=org, content="members only", project_id=engine_project)

    # ME's membership is in an unlinked project: no reach.
    records = await _recall(
        link, org, user_id=ME, extra_project_ids=_reach(link, org_id=org, user_id=ME)
    )
    assert row not in {str(r.id) for r in records}

    # A stranger to every plane project: no reach either.
    stranger = str(uuid.uuid4())
    records = await _recall(
        link,
        org,
        user_id=stranger,
        extra_project_ids=_reach(link, org_id=org, user_id=stranger),
    )
    assert row not in {str(r.id) for r in records}

    # And the actual member DOES reach it — the fixture is not vacuous.
    records = await _recall(
        link,
        org,
        user_id=SOMEONE_ELSE,
        extra_project_ids=_reach(link, org_id=org, user_id=SOMEONE_ELSE),
    )
    assert row in {str(r.id) for r in records}


@needs_db
def test_revoked_and_soft_deleted_memberships_grant_no_reach(link):
    """Plane revocation is ``is_active=false`` / soft-delete; each kills the
    reach on the next call, as does a soft-deleted project or workspace."""
    org = _org(link)
    plane_ws = _plane_workspace(link)
    plane_project = _plane_project(link, plane_ws)
    _plane_member(link, plane_project, ME)
    _engine_project(link, org_id=org, plane_project_id=plane_project)
    assert _reach(link, org_id=org, user_id=ME)

    link.ws.execute(
        text("UPDATE project_members SET is_active = false WHERE member_id = :m"),
        {"m": ME},
    )
    link.ws.commit()
    assert _reach(link, org_id=org, user_id=ME) is None

    link.ws.execute(
        text(
            "UPDATE project_members SET is_active = true, deleted_at = now() "
            "WHERE member_id = :m"
        ),
        {"m": ME},
    )
    link.ws.commit()
    assert _reach(link, org_id=org, user_id=ME) is None

    link.ws.execute(
        text("UPDATE project_members SET deleted_at = NULL WHERE member_id = :m"),
        {"m": ME},
    )
    link.ws.execute(
        text("UPDATE projects SET deleted_at = now() WHERE id = :p"),
        {"p": plane_project},
    )
    link.ws.commit()
    assert _reach(link, org_id=org, user_id=ME) is None

    link.ws.execute(
        text("UPDATE projects SET deleted_at = NULL WHERE id = :p"),
        {"p": plane_project},
    )
    link.ws.execute(
        text("UPDATE workspaces SET deleted_at = now() WHERE id = :w"),
        {"w": plane_ws},
    )
    link.ws.commit()
    assert _reach(link, org_id=org, user_id=ME) is None


@needs_db
def test_blank_user_reaches_nothing(link):
    for blank in (None, "", "   "):
        assert plane_project_memberships(link.ws, user_id=blank) == {}


# ---------------------------------------------------------------------------
# 3. Cross-org: a linked project in org B never enters org A's arm
# ---------------------------------------------------------------------------


@needs_db
@pytest.mark.asyncio
async def test_a_link_in_another_org_never_enters_this_orgs_arm(link):
    org_a = _org(link, "org_link_a")
    org_b = _org(link, "org_link_b")
    plane_ws = _plane_workspace(link)
    plane_project = _plane_project(link, plane_ws)
    _plane_member(link, plane_project, ME)
    b_project = _engine_project(link, org_id=org_b, plane_project_id=plane_project)
    b_row = _seed(link, org_id=org_b, content="org b secret", project_id=b_project)

    # The engine half is tenant-scoped: org A resolves no linked projects.
    assert (
        linked_engine_projects(link.eng, org_id=org_a, plane_ids=[plane_project]) == {}
    )
    assert _reach(link, org_id=org_a, user_id=ME) is None

    # And end to end, org A's recall never surfaces org B's row.
    records = await _recall(
        link,
        org_a,
        user_id=ME,
        extra_project_ids=_reach(link, org_id=org_a, user_id=ME),
    )
    assert b_row not in {str(r.id) for r in records}


@needs_db
def test_linked_engine_projects_empty_plane_ids_query_nothing(link):
    assert linked_engine_projects(link.eng, org_id=_org(link), plane_ids=[]) == {}


# ---------------------------------------------------------------------------
# 4. Never re-point, both directions
# ---------------------------------------------------------------------------


@needs_db
def test_a_link_is_never_repointed_in_either_direction(link):
    """Mutation check: dropping ``plane_project_id IS NULL`` from the guarded
    UPDATE (or the unique index on the plane id) fails this test."""
    org = _org(link)
    plane_a = str(uuid.uuid4())
    plane_b = str(uuid.uuid4())
    project = _engine_project(link, org_id=org, plane_project_id=plane_a)

    # Engine side: an already-linked project refuses a DIFFERENT plane id.
    result = link_engine_project(
        link.eng, org_id=org, project_id=project, plane_project_id=plane_b
    )
    assert result["linked"] is False
    assert result["error"] == "already_linked"
    still = link.eng.execute(
        text("SELECT plane_project_id FROM projects WHERE id = :p"), {"p": project}
    ).scalar()
    assert str(still) == plane_a

    # Same plane id again is idempotent, not a refusal.
    result = link_engine_project(
        link.eng, org_id=org, project_id=project, plane_project_id=plane_a
    )
    assert result["linked"] is True

    # Plane side: a plane id already linked to a different engine project is
    # refused for a second engine project (the partial unique index holds).
    second = _engine_project(link, org_id=org, name="second")
    result = link_engine_project(
        link.eng, org_id=org, project_id=second, plane_project_id=plane_a
    )
    assert result["linked"] is False
    assert result["error"] == "already_linked"
    still = link.eng.execute(
        text("SELECT plane_project_id FROM projects WHERE id = :p"), {"p": second}
    ).scalar()
    assert still is None


@needs_db
def test_link_engine_project_links_an_unlinked_project(link):
    org = _org(link)
    plane_id = str(uuid.uuid4())
    project = _engine_project(link, org_id=org)
    result = link_engine_project(
        link.eng, org_id=org, project_id=project, plane_project_id=plane_id
    )
    link.eng.commit()
    assert result["linked"] is True
    assert result["project_id"] == project
    now = link.eng.execute(
        text("SELECT plane_project_id FROM projects WHERE id = :p"), {"p": project}
    ).scalar()
    assert str(now) == plane_id


@needs_db
def test_link_engine_project_is_tenant_scoped(link):
    """Another org's project id is ``not_found`` here, not linkable."""
    org_a = _org(link, "org_link_a")
    org_b = _org(link, "org_link_b")
    b_project = _engine_project(link, org_id=org_b)
    result = link_engine_project(
        link.eng,
        org_id=org_a,
        project_id=b_project,
        plane_project_id=str(uuid.uuid4()),
    )
    assert result["linked"] is False
    assert result["error"] == "not_found"


# ---------------------------------------------------------------------------
# 5. find_project_id resolves a plane uuid to the linked engine project
# ---------------------------------------------------------------------------


@needs_db
def test_find_project_id_resolves_a_linked_plane_uuid(link):
    org = _org(link)
    plane_id = str(uuid.uuid4())
    project = _engine_project(link, org_id=org, plane_project_id=plane_id)
    assert find_project_id(link.eng, org_id=org, identifier=plane_id) == project


@needs_db
def test_find_project_id_returns_none_for_an_unlinked_plane_uuid(link):
    org = _org(link)
    _engine_project(link, org_id=org)
    assert find_project_id(link.eng, org_id=org, identifier=str(uuid.uuid4())) is None


@needs_db
def test_find_project_id_plane_fallback_is_tenant_scoped(link):
    org_a = _org(link, "org_link_a")
    org_b = _org(link, "org_link_b")
    plane_id = str(uuid.uuid4())
    _engine_project(link, org_id=org_b, plane_project_id=plane_id)
    assert find_project_id(link.eng, org_id=org_a, identifier=plane_id) is None


# ---------------------------------------------------------------------------
# 6. Recording-connection guards: the two databases are queried separately
# ---------------------------------------------------------------------------


class _Row:
    def __init__(self, *values, **kw):
        self._values = values
        self.__dict__.update(kw)

    def __getitem__(self, index):
        return self._values[index]


class _RecordingConn:
    """Records the SQL it is handed; pops canned row batches FIFO."""

    def __init__(self, batches):
        self._batches = list(batches)
        self.seen: list[str] = []

    def execute(self, sql, params=None):
        self.seen.append(str(sql))
        rows = self._batches.pop(0) if self._batches else []

        class _Result:
            def fetchall(self_inner):
                return rows

            def fetchone(self_inner):
                return rows[0] if rows else None

            def first(self_inner):
                return rows[0] if rows else None

        return _Result()


def _assert_sides_stay_apart(workspace_conn, engine_conn):
    workspace_sql = " ".join(workspace_conn.seen).lower()
    engine_sql = " ".join(engine_conn.seen).lower()
    assert workspace_conn.seen, "the workspace DB side was never queried"
    assert engine_conn.seen, "the engine side was never queried"
    # Both sides legitimately name projects/project_members — the tables
    # exist under both servers — so the guard keys on each side's EXCLUSIVE
    # tables instead.
    for engine_only in ("organizations", "session_memories"):
        assert engine_only not in workspace_sql, (
            f"the workspace DB connection was asked about {engine_only}, "
            "which does not exist there"
        )
    assert "workspaces" not in engine_sql, (
        "the engine connection was asked about workspaces, which does not exist there"
    )


def test_the_recall_reach_queries_the_two_databases_separately():
    plane_id = str(uuid.uuid4())
    engine_id = str(uuid.uuid4())
    workspace_conn = _RecordingConn(
        [[_Row(plane_project_id=plane_id, role=PLANE_ROLE_MEMBER)]]
    )
    engine_conn = _RecordingConn([[_Row(id=engine_id, plane_project_id=plane_id)]])

    memberships = plane_project_memberships(workspace_conn, user_id="user-1")
    assert memberships == {plane_id: PLANE_ROLE_MEMBER}
    linked = linked_engine_projects(
        engine_conn, org_id="org-1", plane_ids=list(memberships)
    )
    assert linked == {engine_id: plane_id}

    _assert_sides_stay_apart(workspace_conn, engine_conn)


def test_can_write_linked_queries_the_two_databases_separately():
    plane_id = str(uuid.uuid4())
    project_id = str(uuid.uuid4())
    engine_conn = _RecordingConn(
        [
            [_Row(project_id, id=project_id)],  # _owned_project_id
            [],  # member_role: not an engine member
            [_Row(plane_id, plane_project_id=plane_id)],  # the link lookup
        ]
    )
    workspace_conn = _RecordingConn(
        [[_Row(plane_project_id=plane_id, role=PLANE_ROLE_MEMBER)]]
    )

    assert (
        can_write_linked(
            workspace_conn,
            engine_conn,
            org_id="org-1",
            project_id=project_id,
            user_id="user-1",
        )
        is True
    )
    _assert_sides_stay_apart(workspace_conn, engine_conn)


# ---------------------------------------------------------------------------
# 7. can_write_linked: role 15 writes, role 5 does not, None keeps engine-only
# ---------------------------------------------------------------------------


@needs_db
def test_can_write_linked_roles(link):
    org = _org(link)
    plane_ws = _plane_workspace(link)
    plane_project = _plane_project(link, plane_ws)
    project = _engine_project(link, org_id=org, plane_project_id=plane_project)

    writer = str(uuid.uuid4())
    guest = str(uuid.uuid4())
    admin = str(uuid.uuid4())
    _plane_member(link, plane_project, writer, role=PLANE_ROLE_MEMBER)
    _plane_member(link, plane_project, guest, role=PLANE_ROLE_GUEST)
    _plane_member(link, plane_project, admin, role=PLANE_ROLE_ADMIN)

    kwargs = {"org_id": org, "project_id": project}
    assert can_write_linked(link.ws, link.eng, user_id=writer, **kwargs) is True
    assert can_write_linked(link.ws, link.eng, user_id=admin, **kwargs) is True
    # Role 5 (guest) reads but never writes.
    assert can_write_linked(link.ws, link.eng, user_id=guest, **kwargs) is False
    # A stranger to the plane project writes nothing.
    assert (
        can_write_linked(link.ws, link.eng, user_id=str(uuid.uuid4()), **kwargs)
        is False
    )


@needs_db
def test_can_write_linked_without_a_workspace_db_keeps_engine_behaviour(link):
    org = _org(link)
    plane_ws = _plane_workspace(link)
    plane_project = _plane_project(link, plane_ws)
    _plane_member(link, plane_project, ME, role=PLANE_ROLE_MEMBER)
    project = _engine_project(
        link,
        org_id=org,
        plane_project_id=plane_project,
        members=[SOMEONE_ELSE],
    )

    kwargs = {"org_id": org, "project_id": project}
    # No workspace DB: the plane role cannot be consulted, so ME cannot write…
    assert can_write_linked(None, link.eng, user_id=ME, **kwargs) is False
    # …while an engine member's answer is unchanged.
    assert can_write_linked(None, link.eng, user_id=SOMEONE_ELSE, **kwargs) is True


@needs_db
def test_can_write_linked_unlinked_project_is_engine_only(link):
    org = _org(link)
    plane_ws = _plane_workspace(link)
    plane_project = _plane_project(link, plane_ws)
    _plane_member(link, plane_project, ME, role=PLANE_ROLE_ADMIN)
    project = _engine_project(link, org_id=org)  # no link

    assert (
        can_write_linked(link.ws, link.eng, org_id=org, project_id=project, user_id=ME)
        is False
    )
