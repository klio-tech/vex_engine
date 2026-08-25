"""``project_list`` must not echo local filesystem paths to the whole org.

Real Postgres. ``repo_root_path`` is the local machine path a project was
auto-created from (see ``brain.ensure_project``'s upsert-on-write — a row
exists for every directory anyone has ever run an agent in). Live, one org's
``project_list`` returned ``repo_root_path`` for all 118 of its projects,
including paths like ``/Users/<name>/Me/Job Application`` and
``/Users/<name>/Personal/<person-name>`` — none of it memory *content*, all
of it disclosure to any org member or key holder.

The fix is output-only: ``repo_root_path`` stays a valid *input* for project
resolution (``project_create``, and lookup by path) but is never echoed back.
These tests pin that boundary and guard the two things it must not break:
resolution by path, and the rest of the response shape (``git_remote``,
``display_name``, ``member_count``).

Projects are members-only (migration 039, 2026-08-12 ruling): ``project_list``
now requires the caller to be a member of a project before it appears at all.
These tests are about serialization, not that separate boundary — see
``test_project_list_membership.py`` for the membership boundary itself — so
every seeded project below explicitly adds PRIYA (the caller throughout this
file) as a member. Without that, every one of these tests would see an empty
list and assert nothing about the thing they exist to check.
"""

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from app import mcp_app
from shared import brain, projects
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the project_list privacy tests")

PRIYA = "11111111-2222-3333-4444-555555555555"
DEV = "99999999-8888-7777-6666-555555555555"


def _principal(user_id, org_id):
    from app.oauth import OAuthPrincipal

    return OAuthPrincipal(
        org_id=org_id,
        account_slug=org_id,
        user_id=user_id,
        client_id="cli",
        plan="free",
        plan_overrides=None,
    )


def _ctx(user_id, org_id):
    request = MagicMock()
    request.state = SimpleNamespace(oauth_principal=_principal(user_id, org_id))
    request.headers = {}
    ctx = MagicMock()
    ctx.request_context.request = request
    return ctx


@pytest.fixture()
def db():
    engine = create_engine(DB_URL)
    conn = engine.connect()
    orgs: set[str] = set()
    conn.created_org_ids = orgs  # type: ignore[attr-defined]
    try:
        yield conn
    finally:
        try:
            for org_id in orgs:
                conn.execute(
                    text(
                        "DELETE FROM project_members WHERE project_id IN ("
                        "SELECT id FROM projects WHERE org_id = :o)"
                    ),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM project_grant_audit WHERE org_id = :o"),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM projects WHERE org_id = :o"), {"o": org_id}
                )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _org(db) -> str:
    org_id = f"org_{uuid.uuid4().hex[:8]}"
    db.created_org_ids.add(org_id)
    return org_id


class _KeepOpen:
    """The real connection, with ``close()`` neutered.

    ``_perform_project_list`` closes the session it opened, which is correct
    for a per-request session and fatal for a fixture-owned connection the
    assertions still need to read through. Everything else is forwarded
    untouched, so the tool runs against real Postgres exactly as it would in
    production.
    """

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):  # noqa: D102 - the whole point of the wrapper
        pass


def _live(db, org_id):
    """Patch only the transport seams; the DB is real."""
    return patch.multiple(
        mcp_app,
        resolve_active_principal=MagicMock(
            return_value=(org_id, "agent-x", "free", None)
        ),
        SessionLocal=MagicMock(return_value=_KeepOpen(db)),
    )


async def _list(db, org_id):
    with _live(db, org_id):
        return await mcp_app._perform_project_list(_ctx(PRIYA, org_id))


def _seed_project(
    db,
    *,
    org_id: str,
    display_name: str,
    git_remote: str | None,
    repo_root_path: str | None,
) -> str:
    project_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO projects (
                id, org_id, git_remote, repo_root_path, display_name,
                last_seen_at
            ) VALUES (
                :id, :org_id, :remote, :path, :name, NOW()
            )
            """
        ),
        {
            "id": project_id,
            "org_id": org_id,
            "remote": git_remote,
            "path": repo_root_path,
            "name": display_name,
        },
    )
    db.commit()
    return project_id


# ---------------------------------------------------------------------------
# the leak itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_project_list_omits_repo_root_path_for_every_project(db):
    """The live disclosure: local filesystem paths must never reach output.

    Mutation caught: restoring ``"repo_root_path": p["repo_root_path"]`` to
    ``_serialize_project`` makes this fail — the key reappears in every
    project's dict.
    """
    org = _org(db)
    p1 = _seed_project(
        db,
        org_id=org,
        display_name="astrology",
        git_remote=None,
        repo_root_path="/Users/dev/Me/Astrology",
    )
    p2 = _seed_project(
        db,
        org_id=org,
        display_name="klio",
        git_remote="git@example.com:acme/klio.git",
        repo_root_path="/Users/dev/Hive/klio",
    )
    projects.add_member(db, org_id=org, project_id=p1, user_id=PRIYA, granted_by=None)
    projects.add_member(db, org_id=org, project_id=p2, user_id=PRIYA, granted_by=None)

    result = await _list(db, org)

    assert result["count"] == 2
    for project in result["projects"]:
        assert "repo_root_path" not in project


@pytest.mark.asyncio
async def test_project_list_omits_the_key_entirely_rather_than_nulling_it(db):
    """Omission, not ``null``: a client checking ``"repo_root_path" in project``
    must see it is genuinely absent, not merely empty — and a project that
    never had a path (git-remote-only) must serialize identically to one that
    does, so the key's presence can't itself leak which projects have local
    paths on file.
    """
    org = _org(db)
    p1 = _seed_project(
        db,
        org_id=org,
        display_name="path-only",
        git_remote=None,
        repo_root_path="/Users/dev/Personal/notes",
    )
    p2 = _seed_project(
        db,
        org_id=org,
        display_name="remote-only",
        git_remote="git@example.com:acme/remote-only.git",
        repo_root_path=None,
    )
    projects.add_member(db, org_id=org, project_id=p1, user_id=PRIYA, granted_by=None)
    projects.add_member(db, org_id=org, project_id=p2, user_id=PRIYA, granted_by=None)

    result = await _list(db, org)

    keys_per_project = {p["name"]: set(p.keys()) for p in result["projects"]}
    assert keys_per_project["path-only"] == keys_per_project["remote-only"]
    assert "repo_root_path" not in keys_per_project["path-only"]


# ---------------------------------------------------------------------------
# what must keep working
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_git_remote_and_name_still_present(db):
    """``git_remote`` is a deliberately shared identifier and stays.

    Mutation caught: deleting the ``"git_remote": p["git_remote"]`` line from
    ``_serialize_project`` (an over-eager sibling edit to the
    ``repo_root_path`` removal) makes this fail.
    """
    org = _org(db)
    remote = "git@example.com:acme/klio.git"
    project_id = _seed_project(
        db,
        org_id=org,
        display_name="Klio",
        git_remote=remote,
        repo_root_path="/Users/dev/Hive/klio",
    )
    projects.add_member(
        db, org_id=org, project_id=project_id, user_id=PRIYA, granted_by=None
    )

    result = await _list(db, org)

    [project] = result["projects"]
    assert project["name"] == "Klio"
    assert project["git_remote"] == remote


@pytest.mark.asyncio
async def test_member_counts_are_still_correct(db):
    """Counts, not membership rows themselves, is what ``project_list`` may
    disclose (see the module docstring on ``shared.projects.list_projects``).

    Mutation caught: hardcoding or dropping ``member_count`` in
    ``_serialize_project`` makes this fail — it asserts the actual number of
    rows added, not just that the key exists.

    Both projects must have PRIYA (the caller throughout this file) as a
    member, since listing is now members-only — a project she does not
    belong to would not appear in ``result["projects"]`` at all, which is a
    different thing this test is not about (see
    ``test_project_list_membership.py``). "staffed" vs "solo" gives two
    different counts so a hardcoded return value still fails the assertion.
    """
    org = _org(db)
    staffed_id = _seed_project(
        db,
        org_id=org,
        display_name="staffed",
        git_remote="git@example.com:acme/staffed.git",
        repo_root_path=None,
    )
    solo_id = _seed_project(
        db,
        org_id=org,
        display_name="solo",
        git_remote="git@example.com:acme/solo.git",
        repo_root_path=None,
    )
    projects.add_member(
        db, org_id=org, project_id=staffed_id, user_id=PRIYA, granted_by=None
    )
    projects.add_member(
        db, org_id=org, project_id=staffed_id, user_id=DEV, granted_by=None
    )
    projects.add_member(
        db, org_id=org, project_id=solo_id, user_id=PRIYA, granted_by=None
    )

    result = await _list(db, org)

    counts = {p["id"]: p["member_count"] for p in result["projects"]}
    assert counts[staffed_id] == 2
    assert counts[solo_id] == 1


@pytest.mark.asyncio
async def test_a_project_is_still_resolvable_by_its_repo_root_path(db):
    """Regression guard: the change is output-only. ``repo_root_path``
    disappearing from ``project_list``'s response must not touch lookup —
    ``find_project_id`` (used by ``project_members``, ``artifact_put``,
    ``project_create``'s adoption path, ...) still resolves a project by the
    exact path it was seeded with.

    Mutation caught: this would fail if the fix had instead (wrongly) blanked
    ``repo_root_path`` at the query/column level in ``list_projects`` or in
    the table itself, rather than only in the tool's output projection —
    e.g. dropping the ``OR repo_root_path = :ident`` clause from
    ``find_project_id``'s WHERE, or the fix reaching into storage.
    """
    org = _org(db)
    path = "/Users/dev/Me/Job Application"
    project_id = _seed_project(
        db,
        org_id=org,
        display_name="job-hunt",
        git_remote=None,
        repo_root_path=path,
    )

    resolved = brain.find_project_id(db, org_id=org, identifier=path)

    assert resolved == project_id
