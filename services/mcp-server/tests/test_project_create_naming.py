"""``project_create`` must persist the name it reports back. Real Postgres.

The tool's own description says to "use it to give a project a proper name
ahead of time". Live, ``project_create(name="Klio", git_remote=...)`` returned
``{"created": true, "name": "Klio", ...}`` while the stored ``display_name``
stayed ``klio``: ``ensure_project``'s ``ON CONFLICT DO UPDATE SET
last_seen_at = NOW()`` never writes the name on the hit path, and its docstring
says as much. So the tool reported a name it had not stored, and the caller's
next ``project_members("Klio")`` or ``artifact_put(project="Klio")`` failed.

Whether the rename actually landed is a fact about a row, so these run against
a real engine rather than asserting a call was made.
"""

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from app import mcp_app
from shared import projects
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the project_create naming tests")

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

    ``_perform_project_create`` closes the session it opened, which is correct
    for a per-request session and fatal for a fixture-owned connection several
    assertions still have to read through. Everything else is forwarded
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
        resolve_principal=MagicMock(return_value=(org_id, "agent-x")),
        SessionLocal=MagicMock(return_value=_KeepOpen(db)),
    )


def _stored_name(db, *, org_id: str, project_id: str) -> str:
    return db.execute(
        text("SELECT display_name FROM projects WHERE id = :p AND org_id = :o"),
        {"p": project_id, "o": org_id},
    ).scalar()


async def _create(db, *, org_id, user_id, name, git_remote):
    with _live(db, org_id):
        return await mcp_app._perform_project_create(
            _ctx(user_id, org_id),
            name=name,
            git_remote=git_remote,
            repo_root_path=None,
        )


@pytest.mark.asyncio
async def test_naming_an_auto_created_project_actually_persists_the_name(db):
    """The live failure, end to end.

    An auto-created project is named ``klio`` (the last path segment). Its
    author calls project_create(name="Klio") to give it a proper name, is told
    ``created: true, name: "Klio"`` — and the row must now say ``Klio``.
    """
    org = _org(db)
    remote = f"git@example.com:acme/{uuid.uuid4().hex[:8]}.git"

    # How the project comes to exist in production: auto-created by a write.
    project_id = await _create_auto(db, org_id=org, name="klio", git_remote=remote)
    assert _stored_name(db, org_id=org, project_id=project_id) == "klio"

    result = await _create(
        db, org_id=org, user_id=PRIYA, name="Klio", git_remote=remote
    )

    assert result["created"] is True
    assert result["id"] == project_id
    assert result["name"] == "Klio"
    # The claim the tool made about itself.
    assert _stored_name(db, org_id=org, project_id=project_id) == "Klio"


async def _create_auto(db, *, org_id, name, git_remote) -> str:
    """Auto-creation as an ordinary write does it: no members, no admin."""
    from shared import brain

    return brain.ensure_project(
        db, org_id=org_id, display_name=name, git_remote=git_remote
    )


@pytest.mark.asyncio
async def test_a_brand_new_project_is_stored_under_the_name_given(db):
    """The control: creation from nothing must still name the row correctly.

    Without it, a rename applied on the wrong branch could pass the test above
    while breaking the ordinary create path.
    """
    org = _org(db)
    remote = f"git@example.com:acme/{uuid.uuid4().hex[:8]}.git"

    result = await _create(
        db, org_id=org, user_id=PRIYA, name="Apollo", git_remote=remote
    )

    assert result["created"] is True
    assert _stored_name(db, org_id=org, project_id=result["id"]) == "Apollo"


@pytest.mark.asyncio
async def test_an_admin_can_rename_their_own_project_later(db):
    """Renaming is the tool's documented purpose, not a one-shot at creation."""
    org = _org(db)
    remote = f"git@example.com:acme/{uuid.uuid4().hex[:8]}.git"

    first = await _create(
        db, org_id=org, user_id=PRIYA, name="apollo", git_remote=remote
    )
    again = await _create(
        db, org_id=org, user_id=PRIYA, name="Apollo Program", git_remote=remote
    )

    assert again["id"] == first["id"]
    assert _stored_name(db, org_id=org, project_id=first["id"]) == "Apollo Program"


@pytest.mark.asyncio
async def test_a_project_someone_else_administers_cannot_be_renamed(db):
    """The rename must inherit the adoption rule, not sidestep it.

    A rename is a write to a project's shared identity. If it happened before
    (or regardless of) the ``forbidden`` check, ``project_create`` would become
    an unauthenticated way to rename another team's project out from under
    them — vandalism that every later lookup by the old name would then blame
    on the tool.
    """
    org = _org(db)
    remote = f"git@example.com:acme/{uuid.uuid4().hex[:8]}.git"

    owned = await _create(
        db, org_id=org, user_id=PRIYA, name="apollo", git_remote=remote
    )
    assert owned["created"] is True  # the positive control for the fixture

    intruder = await _create(
        db, org_id=org, user_id=DEV, name="pwned", git_remote=remote
    )

    assert intruder["created"] is False
    assert intruder["error"] == "forbidden"
    assert _stored_name(db, org_id=org, project_id=owned["id"]) == "apollo"
    assert not projects.is_member(db, org_id=org, project_id=owned["id"], user_id=DEV)
