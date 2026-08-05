"""A swallowed DB error must not leave the request's session unusable. Real Postgres.

The mcp-server half of ``shared/tests/test_rollback_on_swallowed_error.py``;
that module's docstring explains the defect and why a ``MagicMock`` cannot see
it. What is specific here is *whose* session gets poisoned.

All four helpers below run on a **reused request session** — the one
``SessionLocal()`` the tool body or route opened and will go on using for the
actual work. Three of them are the same call, ``brain.find_project_id`` /
``brain.ensure_project``, wrapped three times for three surfaces:

* ``app.capture._ensure_project_id`` — resolves the project a capture is filed
  under, then the capture is written on the same session;
* ``app.capture._find_project_id`` — the read-side twin, before recall;
* ``app.onboarding._resolve_project_id`` — the same lookup during ``init``,
  before the memories are stored.

And the fourth is the widest of all:

* ``app.auth.resolve_project_from_request`` — the ``X-Vex-Project`` header
  resolver, on **every** MCP request path. It is documented to fail open to
  org-wide scope, and it does; what it also did was hand the tool body a dead
  session, so the tool failed for a reason with nothing to do with projects.

Project resolution is a *relevance* feature, explicitly not a security
boundary, which is exactly why it is entitled to swallow. Nothing entitles it
to take the request with it.
"""

import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from shared import brain
from shared.testing import database_url, requires_db

from app import auth, capture, onboarding

DB_URL = database_url()

pytestmark = requires_db("the swallowed-error rollback tests")

GIT_REMOTE = "https://github.com/klio-tech/rollback-fixture.git"


def _unique_org() -> str:
    return f"org_{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def db():
    """A real Connection, with every row this test created removed afterwards."""
    engine = create_engine(DB_URL)
    conn = engine.connect()
    orgs: set[str] = set()
    conn.created_org_ids = orgs  # type: ignore[attr-defined]
    try:
        yield conn
    finally:
        try:
            conn.rollback()
            for org_id in orgs:
                conn.execute(
                    text(
                        """
                        DELETE FROM project_members WHERE project_id IN (
                            SELECT id FROM projects WHERE org_id = :o)
                        """
                    ),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM projects WHERE org_id = :o"), {"o": org_id}
                )
                conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _poison(db) -> None:
    """Leave ``db`` in a genuinely aborted transaction, as production does."""
    try:
        db.execute(text("SELECT no_such_column_at_all"))
    except Exception:
        pass


def _session_is_usable(db) -> bool:
    """True when the connection will answer an ordinary statement."""
    try:
        return db.execute(text("SELECT 1")).scalar() == 1
    except Exception:
        return False


def _project_request(identifier: str | None = GIT_REMOTE):
    """A request carrying (or not carrying) the X-Vex-Project header."""
    headers = {} if identifier is None else {auth.PROJECT_HEADER: identifier}
    return SimpleNamespace(
        headers=SimpleNamespace(get=lambda k, default=None: headers.get(k, default))
    )


def test_the_poison_actually_poisons(db):
    """The fixture's own control: without it every test here is vacuous."""
    assert _session_is_usable(db)
    _poison(db)
    assert not _session_is_usable(db)
    db.rollback()
    assert _session_is_usable(db)


# ---------------------------------------------------------------------------
# app/capture.py — the write side and the read side
# ---------------------------------------------------------------------------


def test_ensure_project_id_leaves_the_session_usable(db):
    org_id = _unique_org()
    db.created_org_ids.add(org_id)
    _poison(db)

    project_id = capture._ensure_project_id(
        db, org_id=org_id, git_remote=GIT_REMOTE, repo_root=None
    )

    assert project_id is None, "documented: an unresolved project writes unfiled"
    assert _session_is_usable(db), (
        "the project lookup swallowed its error and left the transaction "
        "aborted; the capture written on this same session dies next"
    )


def test_ensure_project_id_really_creates_a_project(db):
    """Positive control: it is not a stub that always returns None."""
    org_id = _unique_org()
    db.created_org_ids.add(org_id)

    project_id = capture._ensure_project_id(
        db, org_id=org_id, git_remote=GIT_REMOTE, repo_root=None
    )

    assert project_id is not None
    assert _session_is_usable(db)


def test_find_project_id_leaves_the_session_usable(db):
    org_id = _unique_org()
    db.created_org_ids.add(org_id)
    _poison(db)

    project_id = capture._find_project_id(db, org_id=org_id, git_remote=GIT_REMOTE)

    assert project_id is None, "documented: an unresolved project recalls org-wide"
    assert _session_is_usable(db), (
        "the read-side project lookup left the transaction aborted; the recall "
        "it was scoping dies on the same session"
    )


def test_find_project_id_really_resolves_a_project(db):
    """Positive control: a seeded project IS found on a healthy session."""
    org_id = _unique_org()
    db.created_org_ids.add(org_id)
    seeded = brain.ensure_project(
        db, org_id=org_id, display_name="fixture", git_remote=GIT_REMOTE
    )

    found = capture._find_project_id(db, org_id=org_id, git_remote=GIT_REMOTE)

    assert found == seeded


# ---------------------------------------------------------------------------
# app/onboarding.py — the same lookup during init
# ---------------------------------------------------------------------------


def test_resolve_project_id_leaves_the_session_usable(db):
    org_id = _unique_org()
    db.created_org_ids.add(org_id)
    _poison(db)

    project_id = onboarding._resolve_project_id(
        db, org_id=org_id, git_remote=GIT_REMOTE
    )

    assert project_id is None, "documented: a miss files the memories unfiled"
    assert _session_is_usable(db), (
        "init's project lookup left the transaction aborted; every memory the "
        "run was about to store goes with it"
    )


def test_resolve_project_id_really_resolves_a_project(db):
    """Positive control."""
    org_id = _unique_org()
    db.created_org_ids.add(org_id)
    seeded = brain.ensure_project(
        db, org_id=org_id, display_name="fixture", git_remote=GIT_REMOTE
    )

    found = onboarding._resolve_project_id(db, org_id=org_id, git_remote=GIT_REMOTE)

    assert found == seeded


# ---------------------------------------------------------------------------
# app/auth.py — the X-Vex-Project header, on every MCP request path
# ---------------------------------------------------------------------------


def test_resolve_project_from_request_leaves_the_session_usable(db):
    org_id = _unique_org()
    db.created_org_ids.add(org_id)
    _poison(db)

    project_id = auth.resolve_project_from_request(
        _project_request(), db, org_id=org_id
    )

    assert project_id is None, "documented: fails open to org-wide"
    assert _session_is_usable(db), (
        "the X-Vex-Project resolver runs on every MCP request; leaving the "
        "transaction aborted fails the tool for a reason unrelated to projects"
    )


def test_resolve_project_from_request_really_resolves_the_header(db):
    """Positive control: the header IS honoured on a healthy session."""
    org_id = _unique_org()
    db.created_org_ids.add(org_id)
    seeded = brain.ensure_project(
        db, org_id=org_id, display_name="fixture", git_remote=GIT_REMOTE
    )

    found = auth.resolve_project_from_request(_project_request(), db, org_id=org_id)

    assert found == seeded
    # And a request with no header short-circuits before touching the DB at all.
    assert (
        auth.resolve_project_from_request(_project_request(None), db, org_id=org_id)
        is None
    )
