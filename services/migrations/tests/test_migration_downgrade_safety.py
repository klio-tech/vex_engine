"""Rolling back must leave a database the upgrade can re-apply.

These are DB tests against a real Postgres, matching the convention in
``test_migration_031.py``: they read ``DATABASE_URL`` and skip only when the
server is genuinely unreachable — a skip here is not a pass.

Unlike the other migration tests, these cannot run against the shared
development database: they move the schema up and down. Each module run
creates its own scratch database, migrates it to head, exercises the
round-trip, and drops it again.

Two families of failure are covered.

**C3 — the 031 round trip.** Three independently fatal states:

1. ``scope = 'private'`` rows outlive the downgrade that drops ``user_id``,
   so the re-upgrade trips ``ck_session_memories_private_has_owner``.
2. Rows whose ``scope`` is a typo predate 031 in production, and 031's own
   docstring says so, yet the upgrade aborts on them instead of setting them
   aside.
3. Once the curator writes per-user rows, restoring ``brain_curator_state``'s
   single-column primary key raises ``UniqueViolation``.

**H6 — audit-trail downgrades.** 032 drops the whole ACL history and 033
drops the column that distinguishes a revoke from a grant. Neither may happen
silently while the information is still there.
"""

import os
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit

import pytest
from sqlalchemy import create_engine, text

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",
)

SCRATCH_DB = "klio_mig_downgrade_safety"

MIGRATIONS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _with_database(url: str, database: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(
        (parts.scheme, parts.netloc, f"/{database}", parts.query, parts.fragment)
    )


def _alembic(url: str, *args: str, env_extra: dict | None = None):
    """Run the alembic CLI against ``url`` and return the completed process."""
    env = {**os.environ, "DATABASE_URL": url}
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=MIGRATIONS_DIR,
        env=env,
        capture_output=True,
        text=True,
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
            "Postgres not reachable; downgrade-safety tests require the live test DB"
        )

    url = _with_database(DB_URL, SCRATCH_DB)
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)'))
        c.execute(text(f'CREATE DATABASE "{SCRATCH_DB}"'))
    try:
        _require(_alembic(url, "upgrade", "head"), "initial upgrade to head")
        yield url
    finally:
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)'))


@pytest.fixture
def db(scratch_url):
    engine = create_engine(scratch_url, isolation_level="AUTOCOMMIT")
    with engine.connect() as c:
        yield c
    engine.dispose()


def _current_revision(conn) -> str:
    return conn.execute(text("SELECT version_num FROM alembic_version")).scalar()


def _reset_to_head(scratch_url, db):
    """Bring the scratch database back to head and empty the seeded tables."""
    _require(_alembic(scratch_url, "upgrade", "head"), "restore to head")
    db.execute(text("DELETE FROM session_memories"))
    db.execute(text("DELETE FROM brain_curator_state"))
    db.execute(text("DELETE FROM project_grant_audit"))
    db.execute(text("DROP TABLE IF EXISTS session_memories_quarantine"))
    db.execute(text("DROP TABLE IF EXISTS session_memories_user_backup"))
    db.execute(text("DROP TABLE IF EXISTS brain_curator_state_user_backup"))


# --------------------------------------------------------------------------
# seeding
# --------------------------------------------------------------------------


def _seed_private_memories(conn, count: int = 3) -> None:
    for i in range(count):
        conn.execute(
            text(
                "INSERT INTO session_memories "
                "(org_id, agent_id, session_id, scope, status, content, memory_type, user_id) "
                "VALUES ('orgA', 'a', 's', 'private', 'active', :c, 'fact', :u)"
            ),
            {"c": f"private memory {i}", "u": f"user-{i}"},
        )


def _seed_per_user_curator_rows(conn) -> None:
    conn.execute(
        text(
            "INSERT INTO brain_curator_state "
            "(org_id, user_id, processed_count, last_processed_at) VALUES "
            "('orgA', 'user-0', 10, NOW() - INTERVAL '2 hours'), "
            "('orgA', 'user-1', 99, NOW() - INTERVAL '1 hour'), "
            "('orgB', 'user-7', 5, NOW())"
        )
    )


def _seed_typo_scopes(conn, count: int = 2) -> None:
    """Only insertable below 031 — the check constraint is what rejects them."""
    for i in range(count):
        conn.execute(
            text(
                "INSERT INTO session_memories "
                "(org_id, agent_id, session_id, scope, status, content, memory_type) "
                "VALUES ('orgA', 'a', 's', :s, 'active', :c, 'fact')"
            ),
            {"s": "privte", "c": f"typo scope memory {i}"},
        )


def _seed_audit_rows(conn, *, grants: int = 1, revokes: int = 0) -> None:
    for action, count in (("grant", grants), ("revoke", revokes)):
        for _ in range(count):
            conn.execute(
                text(
                    "INSERT INTO project_grant_audit "
                    "(id, org_id, project_id, granted_to, role, surface, action) "
                    "VALUES (gen_random_uuid(), 'orgA', gen_random_uuid(), "
                    "'user-0', 'member', 'mcp', :a)"
                ),
                {"a": action},
            )


# --------------------------------------------------------------------------
# C3 — the 031 round trip
# --------------------------------------------------------------------------


def test_downgrade_survives_per_user_curator_rows(scratch_url, db):
    """Failure 3: restoring the (org_id) primary key over per-user rows."""
    _reset_to_head(scratch_url, db)
    _seed_per_user_curator_rows(db)

    down = _alembic(scratch_url, "downgrade", "030")
    _require(down, "downgrade 035 -> 030")

    assert _current_revision(db) == "030"
    orgs = db.execute(
        text("SELECT org_id, processed_count FROM brain_curator_state ORDER BY org_id")
    ).fetchall()
    assert [r[0] for r in orgs] == ["orgA", "orgB"], (
        "downgrade must collapse to exactly one cursor row per org"
    )
    assert orgs[0][1] == 99, (
        "the surviving orgA cursor must be the furthest-advanced one, deterministically"
    )


def test_reupgrade_survives_private_memories(scratch_url, db):
    """Failure 1: private rows outlive the column that owns them."""
    _reset_to_head(scratch_url, db)
    _seed_private_memories(db)

    _require(_alembic(scratch_url, "downgrade", "030"), "downgrade 035 -> 030")
    _require(_alembic(scratch_url, "upgrade", "035"), "re-upgrade 030 -> 035")

    assert _current_revision(db) == "035"
    ownerless = db.execute(
        text(
            "SELECT count(*) FROM session_memories "
            "WHERE scope = 'private' AND user_id IS NULL"
        )
    ).scalar()
    assert ownerless == 0
    restored = db.execute(
        text(
            "SELECT count(*) FROM session_memories "
            "WHERE scope = 'private' AND user_id IS NOT NULL"
        )
    ).scalar()
    assert restored == 3, "the downgrade must not lose who owned a private memory"


def test_upgrade_quarantines_typoed_scopes(scratch_url, db):
    """Failure 2: 031 aborts on the rows its own docstring says exist."""
    _reset_to_head(scratch_url, db)
    _require(_alembic(scratch_url, "downgrade", "030"), "downgrade 035 -> 030")
    _seed_typo_scopes(db)

    _require(_alembic(scratch_url, "upgrade", "035"), "re-upgrade over typo'd scopes")

    live = db.execute(
        text("SELECT count(*) FROM session_memories WHERE scope = 'privte'")
    ).scalar()
    assert live == 0, "a typo'd scope must not survive into the constrained table"
    quarantined = db.execute(
        text(
            "SELECT count(*) FROM session_memories_quarantine "
            "WHERE payload->>'scope' = 'privte'"
        )
    ).scalar()
    assert quarantined == 2, "the rows must be set aside, not destroyed"


def test_round_trip_is_repeatable(scratch_url, db):
    """All three states at once, twice in a row, with nothing left over."""
    _reset_to_head(scratch_url, db)
    _seed_private_memories(db)
    _seed_per_user_curator_rows(db)

    for cycle in range(2):
        _require(_alembic(scratch_url, "downgrade", "030"), f"cycle {cycle} downgrade")
        _seed_typo_scopes(db, count=1)
        _require(_alembic(scratch_url, "upgrade", "035"), f"cycle {cycle} upgrade")

        assert _current_revision(db) == "035"
        assert (
            db.execute(
                text(
                    "SELECT count(*) FROM session_memories "
                    "WHERE scope = 'private' AND user_id IS NOT NULL"
                )
            ).scalar()
            == 3
        ), f"cycle {cycle}: private ownership must survive every round trip"
        assert (
            db.execute(
                text("SELECT count(*) FROM session_memories WHERE scope = 'privte'")
            ).scalar()
            == 0
        ), f"cycle {cycle}: typo'd scopes must be quarantined every time"

    assert (
        db.execute(text("SELECT count(*) FROM session_memories_quarantine")).scalar()
        == 2
    ), "one typo'd row quarantined per cycle, none duplicated"


# --------------------------------------------------------------------------
# H6 — audit-trail downgrades
# --------------------------------------------------------------------------


def test_032_downgrade_refuses_to_destroy_audit_history(scratch_url, db):
    # Grant-only: 033's guard has nothing to object to, so 032's guard is what
    # this asserts on.
    _reset_to_head(scratch_url, db)
    _seed_audit_rows(db, grants=2)

    down = _alembic(scratch_url, "downgrade", "031")
    assert down.returncode != 0, "dropping a populated audit trail must not be silent"
    combined = down.stdout + down.stderr
    assert "project_grant_audit" in combined
    assert "KLIO_ALLOW_AUDIT_LOSS" in combined
    assert "2" in combined, "the refusal must say how many rows would be lost"
    assert _current_revision(db) == "035", "a refused downgrade must change nothing"


def test_032_downgrade_proceeds_with_explicit_override(scratch_url, db):
    _reset_to_head(scratch_url, db)
    _seed_audit_rows(db, grants=2)

    down = _alembic(
        scratch_url, "downgrade", "031", env_extra={"KLIO_ALLOW_AUDIT_LOSS": "1"}
    )
    _require(down, "overridden downgrade 035 -> 031")
    assert _current_revision(db) == "031"
    _require(_alembic(scratch_url, "upgrade", "head"), "restore to head")


def test_032_downgrade_on_empty_audit_table_is_free(scratch_url, db):
    _reset_to_head(scratch_url, db)

    _require(_alembic(scratch_url, "downgrade", "031"), "downgrade with empty audit")
    assert _current_revision(db) == "031"
    _require(_alembic(scratch_url, "upgrade", "head"), "restore to head")


def test_033_downgrade_refuses_when_revokes_would_become_grants(scratch_url, db):
    _reset_to_head(scratch_url, db)
    _seed_audit_rows(db, grants=1, revokes=1)

    down = _alembic(scratch_url, "downgrade", "032")
    assert down.returncode != 0, (
        "dropping `action` turns a revoke into a grant — worse than a missing row"
    )
    combined = down.stdout + down.stderr
    assert "revoke" in combined
    assert "KLIO_ALLOW_AUDIT_LOSS" in combined
    assert _current_revision(db) == "035"


def test_033_downgrade_proceeds_with_explicit_override(scratch_url, db):
    _reset_to_head(scratch_url, db)
    _seed_audit_rows(db, grants=1, revokes=1)

    down = _alembic(
        scratch_url, "downgrade", "032", env_extra={"KLIO_ALLOW_AUDIT_LOSS": "1"}
    )
    _require(down, "overridden downgrade 035 -> 032")
    assert _current_revision(db) == "032"
    _require(_alembic(scratch_url, "upgrade", "head"), "restore to head")


def test_033_downgrade_on_empty_audit_table_is_free(scratch_url, db):
    _reset_to_head(scratch_url, db)

    _require(_alembic(scratch_url, "downgrade", "032"), "downgrade with empty audit")
    assert _current_revision(db) == "032"
    _require(_alembic(scratch_url, "upgrade", "head"), "restore to head")


def test_033_downgrade_is_free_when_only_grants_exist(scratch_url, db):
    """Nothing is lost: re-upgrade restores `action='grant'` by default."""
    _reset_to_head(scratch_url, db)
    _seed_audit_rows(db, grants=1)

    _require(_alembic(scratch_url, "downgrade", "032"), "grant-only downgrade")
    _require(_alembic(scratch_url, "upgrade", "head"), "restore to head")
    grants = db.execute(
        text("SELECT count(*) FROM project_grant_audit WHERE action = 'grant'")
    ).scalar()
    assert grants == 1
