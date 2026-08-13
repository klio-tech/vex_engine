"""Projects are visible only to their members (ruling 2026-08-12).

Real Postgres, matching ``test_project_list_privacy.py``'s conventions in
this same directory (which now covers *serialization* — this file covers the
membership *boundary* itself: who gets a project in their list at all).

Mutations this file is built to catch:

* ``shared.projects.list_projects`` drops its membership ``EXISTS`` clause
  and goes back to filtering on ``org_id`` alone
  -> ``test_a_non_member_in_the_same_org_does_not_see_the_project``
* An unattributed caller (``user_id=None``) falls back to org-wide listing
  instead of failing closed to an empty list — the exact pre-ruling
  behaviour this migration exists to retire
  -> ``test_an_unattributed_caller_sees_nothing``
  (note: merely deleting the ``if not user_id: return []`` line is NOT, by
  itself, an observable mutation here — binding ``user_id=NULL`` into
  ``pm.user_id = :user_id`` already evaluates to unknown/false in SQL, so the
  membership ``EXISTS`` clause independently returns nothing. The guard is
  correct defense-in-depth and documents the intent explicitly, but the
  mutation this test is built to catch is a REGRESSION to the old org-wide
  query, not the guard's absence in isolation.)
* ``ensure_project`` stops seating the creator as admin, so a members-only
  listing would make a brand-new project invisible to the person who just
  created it
  -> ``test_the_creator_of_a_fresh_project_can_see_and_administer_it``
* ``project_grant`` regresses to leaking existence via ``forbidden`` for a
  true non-member (covered more directly with mocks in
  ``test_cross_user_attack.py``; this is the real-DB end-to-end version)
  -> ``test_a_non_member_gets_not_found_not_forbidden_from_project_grant``
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

pytestmark = requires_db("the project membership boundary tests")

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


def _unattributed_ctx(org_id):
    """An OAuth principal that resolves an org but no person — the
    'unattributed key' case, matching ``resolve_request_user_id``'s contract
    for a principal that names no user."""
    request = MagicMock()
    request.state = SimpleNamespace(oauth_principal=_principal(None, org_id))
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
    """The real connection, with ``close()`` neutered — see
    ``test_project_list_privacy.py`` for why this is needed."""

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):  # noqa: D102
        pass


def _live(db, org_id):
    return patch.multiple(
        mcp_app,
        resolve_principal=MagicMock(return_value=(org_id, "agent-x")),
        SessionLocal=MagicMock(return_value=_KeepOpen(db)),
    )


async def _list(db, org_id, ctx):
    with _live(db, org_id):
        return await mcp_app._perform_project_list(ctx)


def _seed_project(db, *, org_id: str, display_name: str) -> str:
    project_id = str(uuid.uuid4())
    db.execute(
        text(
            "INSERT INTO projects (id, org_id, git_remote, display_name, last_seen_at) "
            "VALUES (:id, :org, :remote, :name, NOW())"
        ),
        {
            "id": project_id,
            "org": org_id,
            "remote": f"git@example.com:acme/{uuid.uuid4().hex[:8]}.git",
            "name": display_name,
        },
    )
    db.commit()
    return project_id


# ---------------------------------------------------------------------------
# the boundary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_member_sees_their_project(db):
    org = _org(db)
    project_id = _seed_project(db, org_id=org, display_name="apollo")
    projects.add_member(
        db, org_id=org, project_id=project_id, user_id=PRIYA, granted_by=None
    )

    result = await _list(db, org, _ctx(PRIYA, org))

    assert result["count"] == 1
    assert result["projects"][0]["id"] == project_id


@pytest.mark.asyncio
async def test_a_non_member_in_the_same_org_does_not_see_the_project(db):
    """The core of the ruling: DEV is in the same org as PRIYA, is a fully
    valid caller, and still sees nothing — not a redacted entry, no entry."""
    org = _org(db)
    project_id = _seed_project(db, org_id=org, display_name="apollo")
    projects.add_member(
        db, org_id=org, project_id=project_id, user_id=PRIYA, granted_by=None
    )

    result = await _list(db, org, _ctx(DEV, org))

    assert result["count"] == 0
    assert result["projects"] == []


@pytest.mark.asyncio
async def test_a_non_member_sees_nothing_even_when_other_org_projects_exist(db):
    """Multiple projects in the org, membership on none of them for the
    caller: the list is empty, not filtered-but-nonempty."""
    org = _org(db)
    p1 = _seed_project(db, org_id=org, display_name="apollo")
    p2 = _seed_project(db, org_id=org, display_name="artemis")
    projects.add_member(db, org_id=org, project_id=p1, user_id=PRIYA, granted_by=None)
    projects.add_member(db, org_id=org, project_id=p2, user_id=PRIYA, granted_by=None)

    result = await _list(db, org, _ctx(DEV, org))

    assert result == {"projects": [], "count": 0}


@pytest.mark.asyncio
async def test_an_unattributed_caller_sees_nothing(db):
    """No resolvable user id -> empty list, never the org's full project set.
    Fails CLOSED rather than falling back to org-wide."""
    org = _org(db)
    project_id = _seed_project(db, org_id=org, display_name="apollo")
    # Membership exists for someone; an unattributed caller must not inherit it.
    projects.add_member(
        db, org_id=org, project_id=project_id, user_id=PRIYA, granted_by=None
    )

    result = await _list(db, org, _unattributed_ctx(org))

    assert result == {"projects": [], "count": 0}


# ---------------------------------------------------------------------------
# the creator's seat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_creator_of_a_fresh_project_can_see_and_administer_it(db):
    """End to end: a project auto-created via ``ensure_project`` (the write
    path every hook capture goes through) is immediately visible to its
    creator via ``project_list``, and the creator can use ``project_grant``
    to add a teammate — proving the auto-seated admin role is REAL
    membership, not a cosmetic label."""
    org = _org(db)
    remote = f"git@example.com:acme/{uuid.uuid4().hex[:8]}.git"

    project_id = brain.ensure_project(
        db, org_id=org, display_name="fresh", git_remote=remote, created_by=PRIYA
    )

    listing = await _list(db, org, _ctx(PRIYA, org))
    assert listing["count"] == 1
    assert listing["projects"][0]["id"] == project_id

    with (
        _live(db, org),
        patch("app.mcp_app.resolve_org_member_by_email", return_value=DEV),
    ):
        grant = await mcp_app._perform_project_grant(
            _ctx(PRIYA, org), project=remote, user="dev@example.com", role="member"
        )

    assert grant["granted"] is True
    assert projects.is_member(db, org_id=org, project_id=project_id, user_id=DEV)

    # DEV is now a member and sees the project too.
    dev_listing = await _list(db, org, _ctx(DEV, org))
    assert dev_listing["count"] == 1
    assert dev_listing["projects"][0]["id"] == project_id


# ---------------------------------------------------------------------------
# existence must not leak
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_non_member_gets_not_found_not_forbidden_from_project_grant(db):
    """A real project, a real (non-member) caller, real Postgres: the answer
    must be ``not_found``, identical to what a genuinely missing project
    name gets — never ``forbidden``, which would confirm the project is
    real to someone standing entirely outside it."""
    org = _org(db)
    remote = f"git@example.com:acme/{uuid.uuid4().hex[:8]}.git"
    project_id = _seed_project(db, org_id=org, display_name="apollo")
    db.execute(
        text("UPDATE projects SET git_remote = :r WHERE id = :p"),
        {"r": remote, "p": project_id},
    )
    db.commit()
    projects.add_member(
        db, org_id=org, project_id=project_id, user_id=PRIYA, granted_by=None
    )

    with (
        _live(db, org),
        patch("app.mcp_app.resolve_org_member_by_email", return_value=DEV),
    ):
        result = await mcp_app._perform_project_grant(
            _ctx(DEV, org), project=remote, user="dev@example.com", role="member"
        )
        missing = await mcp_app._perform_project_grant(
            _ctx(DEV, org),
            project="totally-unknown-project-xyz",
            user="dev@example.com",
            role="member",
        )

    assert result["error"] == "not_found"
    assert missing["error"] == "not_found"
    assert result["detail"] == f"unknown project: {remote}"
