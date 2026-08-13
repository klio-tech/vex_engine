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

Amendment (same day): the workspace-owner fallback for projects step 1 left
unattributed. Mutations this half is built to catch, and the test that
catches each:

* The fallback fires even when step 1 already resolved a ``created_by``
  (stealing a correctly-attributed project for the org owner instead of
  leaving it alone)
  -> ``test_fallback_never_overrides_an_attributable_author``
* The fallback is skipped entirely, or picks the wrong user, when no
  attributed memory exists
  -> ``test_fallback_seats_the_workspace_owner_when_no_attributed_author``
* The ``is_personal_account = false`` guard is dropped from the account
  lookup (a personal account's owner leaks in as a workspace's fallback
  owner)
  -> ``test_fallback_ignores_a_matching_personal_account``
* The ``primary_owner_user_id`` NULL case silently resolves to nobody
  instead of falling back to the earliest membership
  -> ``test_fallback_uses_earliest_membership_when_no_primary_owner``
* An org with no resolvable Supabase owner (no ``SUPABASE_DATABASE_URL``, or
  no matching account) is left with an invented owner instead of staying
  NULL
  -> ``test_org_with_no_identifiable_owner_leaves_project_ownerless``
* The fallback's membership INSERT clobbers or promotes a pre-existing
  ``project_members`` row for the same user
  -> ``test_fallback_never_touches_a_pre_existing_membership_row``
* Re-running the migration after the fallback already ran duplicates the
  membership row or re-derives a different owner
  -> ``test_fallback_round_trip_is_idempotent``
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
SUPABASE_SHADOW_DB = "klio_mig_039_supabase_shadow"

MIGRATIONS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORG = "t039-org"


def _with_database(url: str, database: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(
        (parts.scheme, parts.netloc, f"/{database}", parts.query, parts.fragment)
    )


def _alembic(url: str, *args: str, supabase_url: str | None = None):
    env = {**os.environ, "DATABASE_URL": url}
    # Never let an ambient SUPABASE_DATABASE_URL (e.g. from a developer's
    # shell profile) leak into a test that means to exercise "not
    # configured" — every caller either states the shadow URL explicitly or
    # gets an explicit absence.
    if supabase_url is None:
        env.pop("SUPABASE_DATABASE_URL", None)
    else:
        env["SUPABASE_DATABASE_URL"] = supabase_url
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
        c.execute(text("DELETE FROM organizations"))
        yield c
    engine.dispose()


@pytest.fixture(scope="module")
def supabase_shadow_url():
    """A real, separate Postgres database standing in for Supabase.

    The fallback migration code doesn't know or care that it's talking to
    "Supabase" — it just opens ``SUPABASE_DATABASE_URL`` and queries
    ``accounts`` / ``accounts_memberships``, exactly as
    ``app/directory.py`` and ``shared/auth.py`` do in production. A second
    real database with those two tables is a faithful double for this
    migration's purposes, and keeps the test on real Postgres as the task
    requires rather than mocking the SQL layer.
    """
    admin_url = _with_database(DB_URL, "postgres")
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text("SELECT 1"))
    url = _with_database(DB_URL, SUPABASE_SHADOW_DB)
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{SUPABASE_SHADOW_DB}" WITH (FORCE)'))
        c.execute(text(f'CREATE DATABASE "{SUPABASE_SHADOW_DB}"'))
    engine = create_engine(url, isolation_level="AUTOCOMMIT")
    with engine.connect() as c:
        c.execute(
            text(
                """
                CREATE TABLE accounts (
                    id uuid PRIMARY KEY,
                    slug text UNIQUE,
                    is_personal_account boolean NOT NULL DEFAULT false,
                    primary_owner_user_id text,
                    created_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
        )
        c.execute(
            text(
                """
                CREATE TABLE accounts_memberships (
                    account_id uuid NOT NULL REFERENCES accounts(id),
                    user_id text NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
        )
    engine.dispose()
    try:
        yield url
    finally:
        with admin.connect() as c:
            c.execute(
                text(f'DROP DATABASE IF EXISTS "{SUPABASE_SHADOW_DB}" WITH (FORCE)')
            )


@pytest.fixture
def shadow(supabase_shadow_url):
    """A clean ``accounts`` / ``accounts_memberships`` per test."""
    engine = create_engine(supabase_shadow_url, isolation_level="AUTOCOMMIT")
    with engine.connect() as c:
        c.execute(text("DELETE FROM accounts_memberships"))
        c.execute(text("DELETE FROM accounts"))
        yield c
    engine.dispose()


# ---------------------------------------------------------------------------
# seeding — 038-era shapes only (no created_by column exists yet)
# ---------------------------------------------------------------------------


def _seed_project(
    conn,
    *,
    org: str = ORG,
    git_remote: str | None = None,
    repo_root_path: str | None = None,
) -> str:
    project_id = str(uuid.uuid4())
    conn.execute(
        text(
            "INSERT INTO projects (id, org_id, git_remote, repo_root_path, display_name) "
            "VALUES (:id, :org, :remote, :path, 'proj')"
        ),
        {
            "id": project_id,
            "org": org,
            "remote": git_remote or f"git@example.com:acme/{uuid.uuid4().hex[:8]}.git",
            "path": repo_root_path,
        },
    )
    return project_id


def _seed_org(conn, *, org_id: str, account_slug: str) -> None:
    """A row in ``organizations`` — the local join the fallback needs to turn
    an ``org_id`` into the ``account_slug`` it looks up in the shadow DB."""
    conn.execute(
        text(
            "INSERT INTO organizations (org_id, name, account_slug) "
            "VALUES (:org_id, :org_id, :slug) "
            "ON CONFLICT (org_id) DO UPDATE SET account_slug = EXCLUDED.account_slug"
        ),
        {"org_id": org_id, "slug": account_slug},
    )


def _seed_account(
    shadow_conn,
    *,
    slug: str,
    primary_owner_user_id: str | None,
    is_personal_account: bool = False,
) -> str:
    account_id = str(uuid.uuid4())
    shadow_conn.execute(
        text(
            "INSERT INTO accounts (id, slug, is_personal_account, primary_owner_user_id) "
            "VALUES (:id, :slug, :personal, :owner)"
        ),
        {
            "id": account_id,
            "slug": slug,
            "personal": is_personal_account,
            "owner": primary_owner_user_id,
        },
    )
    return account_id


def _seed_account_membership(
    shadow_conn, *, account_id: str, user_id: str, minutes_ago: int
) -> None:
    shadow_conn.execute(
        text(
            "INSERT INTO accounts_memberships (account_id, user_id, created_at) "
            "VALUES (:account_id, :user_id, NOW() - (:mins || ' minutes')::interval)"
        ),
        {"account_id": account_id, "user_id": user_id, "mins": minutes_ago},
    )


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


# ---------------------------------------------------------------------------
# the workspace-owner fallback (amendment, same day)
# ---------------------------------------------------------------------------


def test_fallback_seats_the_workspace_owner_when_no_attributed_author(
    scratch_url, db, shadow, supabase_shadow_url
):
    """No session_memories at all for this project: the fallback assigns the
    org's Supabase account owner and seats them as admin — the orphan does
    NOT stay ownerless when an owner is honestly identifiable."""
    org_id = "org-with-owner"
    _seed_org(db, org_id=org_id, account_slug="acme")
    _seed_account(shadow, slug="acme", primary_owner_user_id="owner-user")
    project_id = _seed_project(db, org=org_id)

    _require(
        _alembic(scratch_url, "upgrade", "head", supabase_url=supabase_shadow_url),
        "upgrade to 039 with fallback",
    )

    assert _created_by(db, project_id) == "owner-user"
    assert _members(db, project_id) == [("owner-user", "admin")]


def test_fallback_never_overrides_an_attributable_author(
    scratch_url, db, shadow, supabase_shadow_url
):
    """The regression that matters most: a project WITH an attributable
    author keeps that author, even when the org also has a resolvable
    owner the fallback would otherwise have assigned. Step 1 must win, and
    step 3 must never reconsider a project it already resolved."""
    org_id = "org-with-owner-2"
    _seed_org(db, org_id=org_id, account_slug="acme2")
    _seed_account(shadow, slug="acme2", primary_owner_user_id="the-owner")
    project_id = _seed_project(db, org=org_id)
    _seed_memory(db, project_id=project_id, user_id="real-author", minutes_ago=10)

    _require(
        _alembic(scratch_url, "upgrade", "head", supabase_url=supabase_shadow_url),
        "upgrade to 039 with fallback",
    )

    assert _created_by(db, project_id) == "real-author"
    assert _members(db, project_id) == [("real-author", "admin")]


def test_fallback_ignores_a_matching_personal_account(
    scratch_url, db, shadow, supabase_shadow_url
):
    """A personal account (``is_personal_account = true``) sharing the same
    slug must never be picked as the workspace owner — only a workspace
    (non-personal) account counts. Dropping that filter would let a
    stranger's personal-account owner become a project's admin."""
    org_id = "org-personal-collision"
    _seed_org(db, org_id=org_id, account_slug="shared-slug")
    _seed_account(
        shadow,
        slug="shared-slug",
        primary_owner_user_id="wrong-personal-owner",
        is_personal_account=True,
    )
    project_id = _seed_project(db, org=org_id)

    _require(
        _alembic(scratch_url, "upgrade", "head", supabase_url=supabase_shadow_url),
        "upgrade to 039 with fallback",
    )

    assert _created_by(db, project_id) is None
    assert _members(db, project_id) == []


def test_fallback_uses_earliest_membership_when_no_primary_owner(
    scratch_url, db, shadow, supabase_shadow_url
):
    """When the account has no ``primary_owner_user_id`` (e.g. a deleted
    owner user), the deterministic tie-break is the EARLIEST
    ``accounts_memberships`` row for that account — not the latest, not an
    arbitrary one."""
    org_id = "org-no-primary-owner"
    _seed_org(db, org_id=org_id, account_slug="no-owner-acme")
    account_id = _seed_account(
        shadow, slug="no-owner-acme", primary_owner_user_id=None
    )
    _seed_account_membership(
        shadow, account_id=account_id, user_id="later-member", minutes_ago=5
    )
    _seed_account_membership(
        shadow, account_id=account_id, user_id="earliest-member", minutes_ago=90
    )
    project_id = _seed_project(db, org=org_id)

    _require(
        _alembic(scratch_url, "upgrade", "head", supabase_url=supabase_shadow_url),
        "upgrade to 039 with fallback",
    )

    assert _created_by(db, project_id) == "earliest-member"
    assert _members(db, project_id) == [("earliest-member", "admin")]


def test_org_with_no_identifiable_owner_leaves_project_ownerless(scratch_url, db):
    """No ``SUPABASE_DATABASE_URL`` configured at all: the fallback must
    fail closed, exactly like every other cross-database identity lookup in
    this codebase (``app/directory.py``, ``shared/auth.py``). The project
    stays an orphan — invisible once members-only listing ships — rather
    than getting an invented owner."""
    org_id = "org-unreachable"
    _seed_org(db, org_id=org_id, account_slug="whatever")
    project_id = _seed_project(db, org=org_id)

    _require(
        _alembic(scratch_url, "upgrade", "head"),  # no supabase_url -> unset
        "upgrade to 039 without SUPABASE_DATABASE_URL",
    )

    assert _created_by(db, project_id) is None
    assert _members(db, project_id) == []


def test_org_with_no_matching_account_leaves_project_ownerless(
    scratch_url, db, shadow, supabase_shadow_url
):
    """``SUPABASE_DATABASE_URL`` IS configured and reachable, but no
    ``accounts`` row matches this org's slug: still no invented owner."""
    org_id = "org-no-account-row"
    _seed_org(db, org_id=org_id, account_slug="does-not-exist-in-shadow")
    project_id = _seed_project(db, org=org_id)

    _require(
        _alembic(scratch_url, "upgrade", "head", supabase_url=supabase_shadow_url),
        "upgrade to 039 with fallback, no matching account",
    )

    assert _created_by(db, project_id) is None
    assert _members(db, project_id) == []


def test_fallback_never_touches_a_pre_existing_membership_row(
    scratch_url, db, shadow, supabase_shadow_url
):
    """The org owner the fallback would derive already holds a
    ``project_members`` row with a DIFFERENT role: ``ON CONFLICT DO
    NOTHING`` must leave that role exactly as it was, the same guarantee
    step 2's insert already gives the earliest-author path."""
    org_id = "org-pre-existing-member"
    _seed_org(db, org_id=org_id, account_slug="preexist-acme")
    _seed_account(shadow, slug="preexist-acme", primary_owner_user_id="owner-user")
    project_id = _seed_project(db, org=org_id)
    _seed_membership(db, project_id=project_id, user_id="owner-user", role="member")

    _require(
        _alembic(scratch_url, "upgrade", "head", supabase_url=supabase_shadow_url),
        "upgrade to 039 with fallback",
    )

    assert _created_by(db, project_id) == "owner-user"
    assert _members(db, project_id) == [("owner-user", "member")], (
        "a pre-existing membership row must never be clobbered or promoted "
        "by the fallback's heuristic either"
    )


def test_fallback_round_trip_is_idempotent(
    scratch_url, db, shadow, supabase_shadow_url
):
    """Forward, back, forward again, with the fallback path exercised both
    times: the derived owner must be re-derived identically and the
    membership row must not be duplicated."""
    org_id = "org-fallback-roundtrip"
    _seed_org(db, org_id=org_id, account_slug="roundtrip-acme")
    _seed_account(shadow, slug="roundtrip-acme", primary_owner_user_id="rt-owner")
    project_id = _seed_project(db, org=org_id)

    _require(
        _alembic(scratch_url, "upgrade", "head", supabase_url=supabase_shadow_url),
        "first upgrade to 039 with fallback",
    )
    assert _created_by(db, project_id) == "rt-owner"
    assert _members(db, project_id) == [("rt-owner", "admin")]

    _require(_alembic(scratch_url, "downgrade", "038"), "downgrade 039 -> 038")
    assert _members(db, project_id) == [("rt-owner", "admin")], (
        "downgrade must not touch project_members for the fallback path either"
    )

    _require(
        _alembic(scratch_url, "upgrade", "head", supabase_url=supabase_shadow_url),
        "second upgrade to 039 with fallback",
    )
    assert _created_by(db, project_id) == "rt-owner"
    assert _members(db, project_id) == [("rt-owner", "admin")], (
        "re-running the fallback backfill must not duplicate the membership row"
    )
