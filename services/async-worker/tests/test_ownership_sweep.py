"""The project-ownership-transfer sweep (spec: docs/superpowers/specs/
2026-08-13-project-ownership-transfer-design.md).

Real Postgres for the main database (``DATABASE_URL``) plus a second, real
Postgres database standing in for Supabase (``accounts`` /
``accounts_memberships``), matching
``services/migrations/tests/test_migration_039_project_membership.py``'s
"shadow database" convention exactly — the sweep code doesn't know or care
that it's talking to "Supabase", it just opens an engine and queries those
two tables, same as ``app/directory.py`` and ``shared/auth.py`` do in
production.

Mutations this file is built to catch, and the test that catches each:

* A current member gets treated as departed (the query drops its
  ``user_id NOT IN membership_set`` filter, or inverts it)
  -> ``test_a_current_member_is_a_complete_noop`` — asserts the row's exact
  ``granted_at`` timestamp AND the full audit-table row count are both
  unchanged, not just "the member is still listed" (which a buggy
  seat-then-re-add could satisfy vacuously).
* The multi-member gate is dropped, so a healthy multi-member project gets
  an unwanted owner seat on every departure
  -> ``test_departed_member_of_multi_member_project_is_removed_without_transfer``
* The sole-owner transfer doesn't seat the workspace owner at all, or seats
  the wrong user
  -> ``test_departed_sole_owner_transfers_to_workspace_owner``
* A Supabase query failure is treated as "nobody is a member" instead of
  "skip this org"
  -> ``test_supabase_query_failure_skips_org_no_rows_changed``
* An empty Supabase membership result is treated as "everyone departed"
  -> ``test_supabase_empty_membership_set_skips_org_and_logs``
* An unresolvable owner still gets the departed row removed (leaving the
  project one step from memberless with no seat to show for it)
  -> ``test_owner_unresolvable_leaves_project_and_departed_row_untouched``
* The owner-seat INSERT loses its ``ON CONFLICT DO NOTHING`` and clobbers or
  promotes a pre-existing role
  -> ``test_existing_owner_role_is_never_downgraded_by_the_transfer``
* Running the sweep twice does something different the second time (the
  departed-row query isn't re-evaluated against live state)
  -> ``test_running_the_sweep_twice_is_idempotent``
* Dry-run performs a write
  -> ``test_dry_run_changes_nothing_but_reports_the_same_actions``
* The transfer cap doesn't stop a cascade
  -> ``test_transfer_cap_trips_and_stops_the_pass``
"""

import uuid

import pytest
from sqlalchemy import create_engine, text

from app.ownership_sweep import run_ownership_sweep
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the project-ownership-transfer sweep tests")

SUPABASE_SHADOW_DB = "klio_test_ownership_sweep_supabase_shadow"

# A dead endpoint: connecting always raises, standing in for "the Supabase
# query itself failed" without needing to simulate a real outage.
UNREACHABLE_SUPABASE_URL = "postgresql://baduser:badpass@127.0.0.1:1/nonexistent"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(DB_URL)
    yield eng
    eng.dispose()


@pytest.fixture
def db(engine):
    """A connection with per-org cleanup, matching
    ``mcp-server/tests/test_project_list_membership.py``'s ``db`` fixture."""
    conn = engine.connect()
    orgs: set[str] = set()
    conn.created_org_ids = orgs  # type: ignore[attr-defined]
    try:
        yield conn
    finally:
        try:
            for org_id in orgs:
                conn.execute(
                    text("DELETE FROM project_grant_audit WHERE org_id = :o"),
                    {"o": org_id},
                )
                conn.execute(
                    text(
                        "DELETE FROM project_members WHERE project_id IN "
                        "(SELECT id FROM projects WHERE org_id = :o)"
                    ),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM projects WHERE org_id = :o"), {"o": org_id}
                )
                conn.execute(
                    text("DELETE FROM organizations WHERE org_id = :o"), {"o": org_id}
                )
            conn.commit()
        finally:
            conn.close()


def _org(db) -> str:
    org_id = f"sweep-org-{uuid.uuid4().hex[:8]}"
    db.created_org_ids.add(org_id)
    return org_id


@pytest.fixture(scope="module")
def supabase_shadow_url():
    admin_url = database_url()
    # Reuse the same host/creds as the main test DB, swapping only the
    # database name — matches migration 039's ``_with_database`` helper.
    from urllib.parse import urlsplit, urlunsplit

    def _with_database(url: str, database: str) -> str:
        parts = urlsplit(url)
        return urlunsplit(
            (parts.scheme, parts.netloc, f"/{database}", parts.query, parts.fragment)
        )

    admin = create_engine(
        _with_database(admin_url, "postgres"), isolation_level="AUTOCOMMIT"
    )
    url = _with_database(admin_url, SUPABASE_SHADOW_DB)
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{SUPABASE_SHADOW_DB}" WITH (FORCE)'))
        c.execute(text(f'CREATE DATABASE "{SUPABASE_SHADOW_DB}"'))
    shadow_engine = create_engine(url, isolation_level="AUTOCOMMIT")
    with shadow_engine.connect() as c:
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
    shadow_engine.dispose()
    try:
        yield url
    finally:
        with admin.connect() as c:
            c.execute(
                text(f'DROP DATABASE IF EXISTS "{SUPABASE_SHADOW_DB}" WITH (FORCE)')
            )
        admin.dispose()


@pytest.fixture
def supabase_engine(supabase_shadow_url):
    """The Engine passed to ``run_ownership_sweep`` — production shape."""
    eng = create_engine(supabase_shadow_url)
    yield eng
    eng.dispose()


@pytest.fixture
def shadow(supabase_engine):
    """A clean connection for seeding ``accounts`` / ``accounts_memberships``,
    matching migration 039's ``shadow`` fixture."""
    conn = supabase_engine.connect()
    conn.execute(text("DELETE FROM accounts_memberships"))
    conn.execute(text("DELETE FROM accounts"))
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------


def _seed_org(db, *, org_id: str, account_slug: str | None) -> None:
    db.execute(
        text(
            "INSERT INTO organizations (org_id, name, account_slug) "
            "VALUES (:org_id, :org_id, :slug) "
            "ON CONFLICT (org_id) DO UPDATE SET account_slug = EXCLUDED.account_slug"
        ),
        {"org_id": org_id, "slug": account_slug},
    )
    db.commit()


def _seed_project(db, *, org_id: str) -> str:
    project_id = str(uuid.uuid4())
    db.execute(
        text(
            "INSERT INTO projects (id, org_id, git_remote, display_name) "
            "VALUES (:id, :org, :remote, 'proj')"
        ),
        {
            "id": project_id,
            "org": org_id,
            "remote": f"git@example.com:acme/{uuid.uuid4().hex[:8]}.git",
        },
    )
    db.commit()
    return project_id


def _seed_membership(
    db, *, project_id: str, user_id: str, role: str = "member"
) -> None:
    db.execute(
        text(
            "INSERT INTO project_members (project_id, user_id, role) "
            "VALUES (:p, :u, :r)"
        ),
        {"p": project_id, "u": user_id, "r": role},
    )
    db.commit()


def _seed_account(
    shadow,
    *,
    account_id: str | None = None,
    slug: str | None,
    primary_owner_user_id: str | None,
    is_personal_account: bool = False,
) -> str:
    account_id = account_id or str(uuid.uuid4())
    shadow.execute(
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
    shadow.commit()
    return account_id


def _seed_account_membership(shadow, *, account_id: str, user_id: str) -> None:
    shadow.execute(
        text(
            "INSERT INTO accounts_memberships (account_id, user_id) "
            "VALUES (:account_id, :user_id)"
        ),
        {"account_id": account_id, "user_id": user_id},
    )
    shadow.commit()


def _members(db, project_id: str) -> list[tuple[str, str]]:
    rows = db.execute(
        text(
            "SELECT user_id, role FROM project_members WHERE project_id = :p "
            "ORDER BY user_id"
        ),
        {"p": project_id},
    ).fetchall()
    return [(r.user_id, r.role) for r in rows]


def _member_row(db, project_id: str, user_id: str):
    return db.execute(
        text(
            "SELECT user_id, role, granted_at FROM project_members "
            "WHERE project_id = :p AND user_id = :u"
        ),
        {"p": project_id, "u": user_id},
    ).fetchone()


def _audit_rows(db, project_id: str) -> list[tuple[str, str, str, str]]:
    rows = db.execute(
        text(
            "SELECT granted_to, role, surface, action FROM project_grant_audit "
            "WHERE project_id = :p ORDER BY granted_at, granted_to"
        ),
        {"p": project_id},
    ).fetchall()
    return [(r.granted_to, r.role, r.surface, r.action) for r in rows]


def _run(engine, supabase_engine, *, dry_run: bool = False, max_transfers: int = 25):
    return run_ownership_sweep(
        engine,
        supabase_engine=supabase_engine,
        dry_run=dry_run,
        max_transfers=max_transfers,
    )


# ---------------------------------------------------------------------------
# 1. sole owner departs -> transfer
# ---------------------------------------------------------------------------


def test_departed_sole_owner_transfers_to_workspace_owner(
    engine, db, supabase_engine, shadow
):
    org = _org(db)
    _seed_org(db, org_id=org, account_slug="acme-1")
    account_id = _seed_account(
        shadow, slug="acme-1", primary_owner_user_id="workspace-owner"
    )
    _seed_account_membership(shadow, account_id=account_id, user_id="workspace-owner")
    project_id = _seed_project(db, org_id=org)
    _seed_membership(db, project_id=project_id, user_id="departed-solo-owner")

    result = _run(engine, supabase_engine)

    assert result.transfer_count == 1
    assert result.transfers[0].project_id == project_id
    assert result.transfers[0].new_owner_user_id == "workspace-owner"
    assert _members(db, project_id) == [("workspace-owner", "admin")]
    # Grant is written before revoke (see _execute_transfer), so it commits
    # with the earlier granted_at and sorts first.
    assert _audit_rows(db, project_id) == [
        ("workspace-owner", "admin", "sweep", "grant"),
        ("departed-solo-owner", "member", "sweep", "revoke"),
    ]


# ---------------------------------------------------------------------------
# 2. departed member of a multi-member project -> no transfer
# ---------------------------------------------------------------------------


def test_departed_member_of_multi_member_project_is_removed_without_transfer(
    engine, db, supabase_engine, shadow
):
    org = _org(db)
    _seed_org(db, org_id=org, account_slug="acme-2")
    account_id = _seed_account(shadow, slug="acme-2", primary_owner_user_id="owner-2")
    _seed_account_membership(shadow, account_id=account_id, user_id="owner-2")
    _seed_account_membership(shadow, account_id=account_id, user_id="still-here")
    project_id = _seed_project(db, org_id=org)
    _seed_membership(db, project_id=project_id, user_id="still-here", role="admin")
    _seed_membership(db, project_id=project_id, user_id="departed-teammate")

    result = _run(engine, supabase_engine)

    assert result.transfer_count == 0
    assert len(result.removals) == 1
    assert result.removals[0].project_id == project_id
    # "no transfer needed, project untouched for the others": the owner
    # never appears, and the remaining member's row is exactly as it was.
    assert _members(db, project_id) == [("still-here", "admin")]
    assert _audit_rows(db, project_id) == [
        ("departed-teammate", "member", "sweep", "revoke")
    ]


# ---------------------------------------------------------------------------
# 3. current member -> complete no-op (the regression that matters most)
# ---------------------------------------------------------------------------


def test_a_current_member_is_a_complete_noop(engine, db, supabase_engine, shadow):
    org = _org(db)
    _seed_org(db, org_id=org, account_slug="acme-3")
    account_id = _seed_account(shadow, slug="acme-3", primary_owner_user_id="owner-3")
    _seed_account_membership(shadow, account_id=account_id, user_id="owner-3")
    _seed_account_membership(shadow, account_id=account_id, user_id="still-a-member")
    project_id = _seed_project(db, org_id=org)
    _seed_membership(db, project_id=project_id, user_id="still-a-member")

    before = _member_row(db, project_id, "still-a-member")
    assert before is not None

    result = _run(engine, supabase_engine)

    assert result.transfer_count == 0
    assert result.removals == []
    after = _member_row(db, project_id, "still-a-member")
    # Not just "still present" — the EXACT same row, timestamp included. A
    # buggy implementation that removed and immediately re-seated this user
    # would still pass a weaker "is a member" assertion; this one would not.
    assert after == before
    assert _audit_rows(db, project_id) == []


# ---------------------------------------------------------------------------
# 4 & 5. Supabase failure modes
# ---------------------------------------------------------------------------


def test_supabase_query_failure_skips_org_no_rows_changed(engine, db):
    org = _org(db)
    _seed_org(db, org_id=org, account_slug="acme-4")
    project_id = _seed_project(db, org_id=org)
    _seed_membership(db, project_id=project_id, user_id="whoever")

    unreachable = create_engine(UNREACHABLE_SUPABASE_URL)
    try:
        result = _run(engine, unreachable)
    finally:
        unreachable.dispose()

    assert result.orgs_skipped_lookup_failed == [org]
    assert result.transfer_count == 0
    assert result.removals == []
    assert _members(db, project_id) == [("whoever", "member")]
    assert _audit_rows(db, project_id) == []


def test_supabase_empty_membership_set_skips_org_and_logs(
    engine, db, supabase_engine, shadow
):
    org = _org(db)
    _seed_org(db, org_id=org, account_slug="acme-5")
    # Account exists, but has ZERO accounts_memberships rows.
    _seed_account(shadow, slug="acme-5", primary_owner_user_id="owner-5")
    project_id = _seed_project(db, org_id=org)
    _seed_membership(db, project_id=project_id, user_id="whoever")

    result = _run(engine, supabase_engine)

    assert result.orgs_skipped_empty_membership == [org]
    assert result.transfer_count == 0
    assert result.removals == []
    assert _members(db, project_id) == [("whoever", "member")]
    assert _audit_rows(db, project_id) == []


# ---------------------------------------------------------------------------
# 6. owner unresolvable -> project left EXACTLY alone
# ---------------------------------------------------------------------------


def test_owner_unresolvable_leaves_project_and_departed_row_untouched(
    engine, db, supabase_engine, shadow
):
    """An org whose ``organizations.account_slug`` was never backfilled, but
    whose ``org_id`` happens to be a UUID matching a Supabase account by
    ``id``: membership CAN be read (the UUID join needs no slug), but
    ``resolve_workspace_owners`` requires ``account_slug`` and therefore
    never even attempts to resolve an owner for this org — a real,
    reachable "owner unresolvable" state, not a contrived one.
    """
    org = str(uuid.uuid4())
    db.created_org_ids.add(org)
    # Deliberately NOT seeding an `organizations` row for `org` at all: no
    # account_slug exists anywhere for this org_id.
    account_id = _seed_account(
        shadow, account_id=org, slug=None, primary_owner_user_id=None
    )
    _seed_account_membership(shadow, account_id=account_id, user_id="someone-else")
    project_id = str(uuid.uuid4())
    db.execute(
        text(
            "INSERT INTO projects (id, org_id, git_remote, display_name) "
            "VALUES (:id, :org, :remote, 'proj')"
        ),
        {
            "id": project_id,
            "org": org,
            "remote": f"git@example.com:acme/{uuid.uuid4().hex[:8]}.git",
        },
    )
    db.commit()
    _seed_membership(db, project_id=project_id, user_id="departed-solo-owner")

    result = _run(engine, supabase_engine)

    assert result.transfer_count == 0
    assert result.projects_left_alone_owner_unresolved == [project_id]
    # The departed row must NOT be removed: a dark project is recoverable,
    # a memberless one is not.
    assert _members(db, project_id) == [("departed-solo-owner", "member")]
    assert _audit_rows(db, project_id) == []


# ---------------------------------------------------------------------------
# 7. existing owner role is never downgraded
# ---------------------------------------------------------------------------


def test_existing_owner_role_is_never_downgraded_by_the_transfer(
    engine, db, supabase_engine, shadow
):
    """The owner already holds a project row with role ``write``. The
    Supabase membership set does NOT include the owner directly (only a
    different, unrelated current member does — a real, if unusual, gap
    between ``primary_owner_user_id`` and ``accounts_memberships`` that
    migration 039's own docstring flags as possible), so the owner's OWN
    project row would otherwise be classified "departed" the same as the
    real departure. The transfer must still: (a) never upgrade the owner's
    existing write role to 'admin', and (b) never delete the owner's row
    as a side effect of seating them — the seat and the removal must not
    fight each other on the same user.
    """
    org = _org(db)
    _seed_org(db, org_id=org, account_slug="acme-7")
    account_id = _seed_account(shadow, slug="acme-7", primary_owner_user_id="owner-7")
    _seed_account_membership(
        shadow, account_id=account_id, user_id="unrelated-current-member"
    )
    project_id = _seed_project(db, org_id=org)
    _seed_membership(db, project_id=project_id, user_id="owner-7", role="write")
    _seed_membership(db, project_id=project_id, user_id="departed-solo-owner")

    result = _run(engine, supabase_engine)

    assert result.transfer_count == 1
    assert result.transfers[0].new_owner_user_id == "owner-7"
    # The real departure was removed; the owner's own row must never be
    # counted among the rows this transfer removes.
    assert result.transfers[0].departed_user_ids == ("departed-solo-owner",)
    assert "owner-7" not in result.transfers[0].departed_user_ids
    assert _members(db, project_id) == [("owner-7", "write")]


# ---------------------------------------------------------------------------
# 8. idempotent
# ---------------------------------------------------------------------------


def test_running_the_sweep_twice_is_idempotent(engine, db, supabase_engine, shadow):
    org = _org(db)
    _seed_org(db, org_id=org, account_slug="acme-8")
    account_id = _seed_account(shadow, slug="acme-8", primary_owner_user_id="owner-8")
    _seed_account_membership(shadow, account_id=account_id, user_id="owner-8")
    project_id = _seed_project(db, org_id=org)
    _seed_membership(db, project_id=project_id, user_id="departed-solo-owner")

    first = _run(engine, supabase_engine)
    assert first.transfer_count == 1
    after_first = _members(db, project_id)
    assert after_first == [("owner-8", "admin")]

    second = _run(engine, supabase_engine)

    assert second.transfer_count == 0
    assert second.removals == []
    assert _members(db, project_id) == after_first
    # Only the first pass's grant+revoke pair, never duplicated.
    assert _audit_rows(db, project_id) == [
        ("owner-8", "admin", "sweep", "grant"),
        ("departed-solo-owner", "member", "sweep", "revoke"),
    ]


# ---------------------------------------------------------------------------
# 9. dry-run
# ---------------------------------------------------------------------------


def test_dry_run_changes_nothing_but_reports_the_same_actions(
    engine, db, supabase_engine, shadow
):
    org = _org(db)
    _seed_org(db, org_id=org, account_slug="acme-9")
    account_id = _seed_account(shadow, slug="acme-9", primary_owner_user_id="owner-9")
    _seed_account_membership(shadow, account_id=account_id, user_id="owner-9")
    project_id = _seed_project(db, org_id=org)
    _seed_membership(db, project_id=project_id, user_id="departed-solo-owner")

    before = _members(db, project_id)

    dry = _run(engine, supabase_engine, dry_run=True)
    assert dry.dry_run is True
    assert dry.transfer_count == 1
    assert dry.transfers[0].project_id == project_id
    assert dry.transfers[0].new_owner_user_id == "owner-9"
    assert _members(db, project_id) == before  # nothing changed
    assert _audit_rows(db, project_id) == []  # nothing logged either

    real = _run(engine, supabase_engine, dry_run=False)

    # The real run reports the exact same set the dry run predicted.
    assert (real.transfers[0].project_id, real.transfers[0].new_owner_user_id) == (
        dry.transfers[0].project_id,
        dry.transfers[0].new_owner_user_id,
    )
    assert _members(db, project_id) == [("owner-9", "admin")]


# ---------------------------------------------------------------------------
# the transfer cap
# ---------------------------------------------------------------------------


def test_transfer_cap_trips_and_stops_the_pass(engine, db, supabase_engine, shadow):
    org = _org(db)
    _seed_org(db, org_id=org, account_slug="acme-cap")
    account_id = _seed_account(
        shadow, slug="acme-cap", primary_owner_user_id="owner-cap"
    )
    _seed_account_membership(shadow, account_id=account_id, user_id="owner-cap")

    project_ids = []
    for _ in range(3):
        pid = _seed_project(db, org_id=org)
        _seed_membership(db, project_id=pid, user_id=f"departed-{uuid.uuid4().hex[:6]}")
        project_ids.append(pid)

    result = _run(engine, supabase_engine, max_transfers=2)

    assert result.cap_tripped is True
    assert result.transfer_count == 2
    # Exactly one of the three projects was left untouched by the tripped
    # cap — its departed row must still be there, not removed.
    transferred_projects = {t.project_id for t in result.transfers}
    untouched = [p for p in project_ids if p not in transferred_projects]
    assert len(untouched) == 1
    role_rows = _members(db, untouched[0])
    assert len(role_rows) == 1 and role_rows[0][1] == "member"  # still the departed row
