"""Migration 039: ``created_by`` + the backfill that makes members-only
listing safe to ship (ruling 2026-08-12, "projects are visible only to their
members").

Own scratch database, matching ``test_migration_downgrade_safety.py``'s
convention: this migration moves the schema up and down and needs to seed
data in a specific pre-migration shape (``session_memories`` rows with known
authors and timestamps, and pre-existing ``project_members`` rows) that a
shared dev database cannot safely host without colliding with every other
test file that also creates projects there.

Each test starts by downgrading to ``038`` (the revision immediately before
this one — ``session_memories``, ``projects`` and ``project_members`` already
exist there, this migration adds nothing they depend on), seeds rows with
plain SQL, then runs ``alembic upgrade head`` and asserts what 039 did.

Mutations this file is built to catch, and the test that catches each:

* Backfill picks the LATEST author instead of the earliest
  -> ``test_backfill_seats_the_earliest_attributed_author_as_admin``
* Backfill's "ignore NULL user_id" filter is dropped, so an unattributed
  write can win "earliest" over a real person
  -> ``test_backfill_ignores_unattributed_memories_when_picking_earliest``
* The membership INSERT's ``WHERE created_by IS NOT NULL`` guard is dropped
  (attempts a NULL-user_id membership row, which violates
  ``project_members.user_id NOT NULL`` and aborts the whole migration)
  -> ``test_a_project_with_no_attributed_memory_gets_no_membership_row``
* ``ON CONFLICT DO NOTHING`` on the membership insert is weakened to
  ``DO UPDATE`` (or dropped, which would raise ``IntegrityError`` outright)
  -> ``test_backfill_never_touches_a_pre_existing_membership_row``
* The downgrade drops more than the column (e.g. also deletes the backfilled
  ``project_members`` rows), or upgrading twice duplicates a membership row
  -> ``test_round_trip_drops_only_the_column_and_re_upgrade_is_idempotent``
"""

import os
import subprocess
import sys
import uuid

import pytest
from sqlalchemy import create_engine, text
from urllib.parse import urlsplit, urlunsplit

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",
)

SCRATCH_DB = "klio_mig_039_project_membership"

MIGRATIONS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORG = "t039-org"


def _with_database(url: str, database: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(
        (parts.scheme, parts.netloc, f"/{database}", parts.query, parts.fragment)
    )


def _alembic(url: str, *args: str):
    env = {**os.environ, "DATABASE_URL": url}
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=MIGRATIONS_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _require(proc, what: str) -> None:
    assert proc.returncode == 0, f"{what} failed:\n{proc.stdout}\n{proc.stderr}"


@pytest.fixture(scope="module")
def scratch_url():
    admin_url = _with_database(DB_URL, "postgres")
    try:
        admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        with admin.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception:  # pragma: no cover - environment guard
        pytest.skip(
            "Postgres not reachable; migration 039 tests require the live test DB"
        )

    url = _with_database(DB_URL, SCRATCH_DB)
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)'))
        c.execute(text(f'CREATE DATABASE "{SCRATCH_DB}"'))
    try:
        _require(_alembic(url, "upgrade", "038"), "initial upgrade to 038")
        yield url
    finally:
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)'))


@pytest.fixture
def db(scratch_url):
    """Reset to revision 038 before each test, with a real Connection."""
    _require(_alembic(scratch_url, "downgrade", "038"), "reset to 038")
    engine = create_engine(scratch_url, isolation_level="AUTOCOMMIT")
    with engine.connect() as c:
        c.execute(text("DELETE FROM project_members"))
        c.execute(text("DELETE FROM session_memories"))
        c.execute(text("DELETE FROM projects"))
        yield c
    engine.dispose()


# ---------------------------------------------------------------------------
# seeding — 038-era shapes only (no created_by column exists yet)
# ---------------------------------------------------------------------------


def _seed_project(
    conn, *, git_remote: str | None = None, repo_root_path: str | None = None
) -> str:
    project_id = str(uuid.uuid4())
    conn.execute(
        text(
            "INSERT INTO projects (id, org_id, git_remote, repo_root_path, display_name) "
            "VALUES (:id, :org, :remote, :path, 'proj')"
        ),
        {
            "id": project_id,
            "org": ORG,
            "remote": git_remote or f"git@example.com:acme/{uuid.uuid4().hex[:8]}.git",
            "path": repo_root_path,
        },
    )
    return project_id


def _seed_memory(
    conn, *, project_id: str, user_id: str | None, minutes_ago: int
) -> str:
    memory_id = str(uuid.uuid4())
    conn.execute(
        text(
            "INSERT INTO session_memories "
            "(id, org_id, agent_id, session_id, scope, status, content, "
            " memory_type, project_id, user_id, created_at) "
            "VALUES (:id, :org, 'a', 's', 'org', 'active', 'c', 'fact', "
            "        :project_id, :user_id, NOW() - (:mins || ' minutes')::interval)"
        ),
        {
            "id": memory_id,
            "org": ORG,
            "project_id": project_id,
            "user_id": user_id,
            "mins": minutes_ago,
        },
    )
    return memory_id


def _seed_membership(conn, *, project_id: str, user_id: str, role: str) -> None:
    conn.execute(
        text(
            "INSERT INTO project_members (project_id, user_id, role) "
            "VALUES (:p, :u, :r)"
        ),
        {"p": project_id, "u": user_id, "r": role},
    )


def _created_by(conn, project_id: str) -> str | None:
    return conn.execute(
        text("SELECT created_by FROM projects WHERE id = :p"), {"p": project_id}
    ).scalar()


def _members(conn, project_id: str) -> list[tuple[str, str]]:
    rows = conn.execute(
        text(
            "SELECT user_id, role FROM project_members WHERE project_id = :p "
            "ORDER BY user_id"
        ),
        {"p": project_id},
    ).fetchall()
    return [(r.user_id, r.role) for r in rows]


# ---------------------------------------------------------------------------
# the backfill
# ---------------------------------------------------------------------------


def test_backfill_seats_the_earliest_attributed_author_as_admin(scratch_url, db):
    """The earliest (by created_at) author of a project's memories becomes its
    admin — not the latest, not an arbitrary one."""
    project_id = _seed_project(db)
    _seed_memory(db, project_id=project_id, user_id="user-later", minutes_ago=5)
    _seed_memory(db, project_id=project_id, user_id="user-earliest", minutes_ago=50)
    _seed_memory(db, project_id=project_id, user_id="user-middle", minutes_ago=20)

    _require(_alembic(scratch_url, "upgrade", "head"), "upgrade to 039")

    assert _created_by(db, project_id) == "user-earliest"
    assert _members(db, project_id) == [("user-earliest", "admin")]


def test_backfill_ignores_unattributed_memories_when_picking_earliest(scratch_url, db):
    """A NULL user_id must never win 'earliest', however early it was written.

    Without the ``user_id IS NOT NULL`` filter, the unattributed row (written
    100 minutes ago) would be picked over the real, attributed author — and
    an unattributed write cannot honestly become anyone's admin seat.
    """
    project_id = _seed_project(db)
    _seed_memory(db, project_id=project_id, user_id=None, minutes_ago=100)
    _seed_memory(db, project_id=project_id, user_id="real-user", minutes_ago=10)

    _require(_alembic(scratch_url, "upgrade", "head"), "upgrade to 039")

    assert _created_by(db, project_id) == "real-user"
    assert _members(db, project_id) == [("real-user", "admin")]


def test_a_project_with_no_attributed_memory_gets_no_membership_row(scratch_url, db):
    """No session_memories at all, and a project written to only by an
    unattributed caller: both must leave created_by NULL and add no
    membership row — an orphan the ruling says must stay invisible, not a
    row this migration invents an owner for."""
    silent_project = _seed_project(db)
    unattributed_project = _seed_project(db)
    _seed_memory(db, project_id=unattributed_project, user_id=None, minutes_ago=5)

    _require(_alembic(scratch_url, "upgrade", "head"), "upgrade to 039")

    for project_id in (silent_project, unattributed_project):
        assert _created_by(db, project_id) is None
        assert _members(db, project_id) == []


def test_backfill_never_touches_a_pre_existing_membership_row(scratch_url, db):
    """The 3-rows-in-production guarantee: ``ON CONFLICT DO NOTHING`` means a
    row that already exists in ``project_members`` keeps its EXACT role,
    even when the backfill's own heuristic would have derived the SAME user
    as the project's admin.

    If the migration used ``DO UPDATE SET role = EXCLUDED.role`` instead, a
    pre-existing 'member' would be silently promoted to 'admin' here — this
    assertion is exactly what catches that.
    """
    project_id = _seed_project(db)
    _seed_membership(
        db, project_id=project_id, user_id="pre-existing-user", role="member"
    )
    # This user is also the earliest attributed author — the heuristic would
    # otherwise seat them as admin.
    _seed_memory(db, project_id=project_id, user_id="pre-existing-user", minutes_ago=30)

    _require(_alembic(scratch_url, "upgrade", "head"), "upgrade to 039")

    assert _created_by(db, project_id) == "pre-existing-user"
    assert _members(db, project_id) == [("pre-existing-user", "member")], (
        "a pre-existing membership row must never be clobbered or promoted "
        "by the backfill's heuristic"
    )


def test_backfill_adds_a_second_admin_alongside_an_unrelated_pre_existing_row(
    scratch_url, db
):
    """A pre-existing row for a DIFFERENT user does not block the backfill
    from seating the actual derived creator — ``ON CONFLICT`` only fires on
    the exact ``(project_id, user_id)`` pair, not on the project alone."""
    project_id = _seed_project(db)
    _seed_membership(
        db, project_id=project_id, user_id="already-a-member", role="member"
    )
    _seed_memory(db, project_id=project_id, user_id="the-real-creator", minutes_ago=10)

    _require(_alembic(scratch_url, "upgrade", "head"), "upgrade to 039")

    assert _created_by(db, project_id) == "the-real-creator"
    assert set(_members(db, project_id)) == {
        ("already-a-member", "member"),
        ("the-real-creator", "admin"),
    }


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------


def test_round_trip_drops_only_the_column_and_re_upgrade_is_idempotent(scratch_url, db):
    """Forward, back, forward again — against a real database, as the task
    requires. The column disappears and reappears; the backfilled membership
    row is never duplicated by running the backfill twice.
    """
    project_id = _seed_project(db)
    _seed_memory(db, project_id=project_id, user_id="repeat-user", minutes_ago=15)

    _require(_alembic(scratch_url, "upgrade", "head"), "first upgrade to 039")
    assert _created_by(db, project_id) == "repeat-user"
    assert _members(db, project_id) == [("repeat-user", "admin")]

    _require(_alembic(scratch_url, "downgrade", "038"), "downgrade 039 -> 038")
    has_column = db.execute(
        text(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name = 'projects' AND column_name = 'created_by'"
        )
    ).scalar()
    assert has_column == 0, "downgrade must drop created_by"
    # The membership row the first upgrade backfilled is NOT undone by the
    # downgrade — see the migration's docstring on why revoking access as a
    # side effect of an unrelated rollback would be its own surprise.
    assert _members(db, project_id) == [("repeat-user", "admin")], (
        "downgrade must not touch project_members — only the column is reversed"
    )

    _require(_alembic(scratch_url, "upgrade", "head"), "second upgrade to 039")
    assert _created_by(db, project_id) == "repeat-user"
    assert _members(db, project_id) == [("repeat-user", "admin")], (
        "re-running the backfill must not duplicate the membership row "
        "(ON CONFLICT DO NOTHING)"
    )


def test_downgrade_of_a_clean_database_succeeds(scratch_url, db):
    """No projects at all: the simplest possible downgrade must still work
    cleanly — this is the literal 'runs the migration forward AND back'
    check the task asks for, with nothing to backfill either direction."""
    _require(_alembic(scratch_url, "upgrade", "head"), "upgrade to 039 (empty)")
    _require(_alembic(scratch_url, "downgrade", "038"), "downgrade to 038 (empty)")
    _require(_alembic(scratch_url, "upgrade", "head"), "re-upgrade to 039 (empty)")


def test_created_by_column_is_nullable(scratch_url, db):
    """A pre-existing row with no honest answer must remain representable —
    the column must not be NOT NULL."""
    _require(_alembic(scratch_url, "upgrade", "head"), "upgrade to 039")
    nullable = db.execute(
        text(
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_name = 'projects' AND column_name = 'created_by'"
        )
    ).scalar()
    assert nullable == "YES"
