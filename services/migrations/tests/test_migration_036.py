"""Migration 036: artifact cards filed under a project stop at that project.

A data migration, not a schema one, so the assertions have to be about ROWS:
which ones moved, which ones were left alone, and whether the downgrade puts
back exactly what the upgrade took.

Like ``test_migration_downgrade_safety.py`` this cannot run against the shared
development database — it moves the schema up and down — so it builds its own
scratch database, seeds it, and drops it again.
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

SCRATCH_DB = "klio_mig_036"

MIGRATIONS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORG = "org_036"
PROJECT = "00000000-0000-0000-0000-0000000000aa"


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
        pytest.skip("Postgres not reachable; migration 036 tests require the test DB")

    url = _with_database(DB_URL, SCRATCH_DB)
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)'))
        c.execute(text(f'CREATE DATABASE "{SCRATCH_DB}"'))
    try:
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


def _seed(conn) -> None:
    """Five rows that between them cover every branch of the predicate."""
    conn.execute(text("DELETE FROM session_memories"))
    conn.execute(text("DELETE FROM projects"))
    conn.execute(
        text(
            "INSERT INTO projects (id, org_id, git_remote, display_name) "
            "VALUES (:p, :o, 'git@example.com:apollo.git', 'apollo')"
        ),
        {"p": PROJECT, "o": ORG},
    )
    rows = [
        # (label, memory_type, project_id, scope)  -> expected after upgrade
        ("moves", "artifact", PROJECT, "org"),  # THE target set
        ("moves_too", "artifact", PROJECT, "org"),  # ...and it is a SET, not one row
        ("org_wide_artifact", "artifact", None, "org"),  # no project: untouched
        ("plain_fact", "fact", PROJECT, "org"),  # not an artifact: untouched
        ("already_project", "artifact", PROJECT, "project"),  # already narrow
    ]
    for label, memory_type, project_id, scope in rows:
        conn.execute(
            text(
                "INSERT INTO session_memories "
                "(org_id, agent_id, session_id, scope, status, content, "
                " memory_type, project_id) "
                "VALUES (:o, 'a', 's', :scope, 'active', :c, :mt, :p)"
            ),
            {
                "o": ORG,
                "scope": scope,
                "c": label,
                "mt": memory_type,
                "p": project_id,
            },
        )


def _scopes(conn) -> dict[str, str]:
    rows = conn.execute(
        text("SELECT content, scope FROM session_memories WHERE org_id = :o"),
        {"o": ORG},
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def _count_to_narrow(conn) -> int:
    return conn.execute(
        text(
            "SELECT count(*) FROM session_memories "
            "WHERE memory_type = 'artifact' AND project_id IS NOT NULL "
            "AND scope = 'org'"
        )
    ).scalar()


def test_upgrade_narrows_only_project_filed_artifact_cards(scratch_url, db):
    _require(_alembic(scratch_url, "upgrade", "035"), "upgrade to 035")
    _seed(db)

    before = _count_to_narrow(db)
    assert before == 2, "the fixture must actually contain rows to narrow"
    assert _scopes(db)["moves"] == "org"  # the pre-state the fix is about

    _require(_alembic(scratch_url, "upgrade", "036"), "upgrade 035 -> 036")

    after = _scopes(db)
    # Narrowed.
    assert after["moves"] == "project"
    assert after["moves_too"] == "project"
    # Untouched — each of these is a way the predicate could over-reach.
    assert after["org_wide_artifact"] == "org", "no project means no boundary to narrow"
    assert after["plain_fact"] == "org", "project_id on a fact is a relevance hint"
    assert after["already_project"] == "project"
    # Nothing was deleted and nothing lost its project.
    assert _count_to_narrow(db) == 0
    assert (
        db.execute(
            text("SELECT count(*) FROM session_memories WHERE org_id = :o"), {"o": ORG}
        ).scalar()
        == 5
    )


def test_downgrade_restores_exactly_the_rows_the_upgrade_narrowed(scratch_url, db):
    _require(_alembic(scratch_url, "upgrade", "035"), "upgrade to 035")
    _seed(db)
    _require(_alembic(scratch_url, "upgrade", "036"), "upgrade 035 -> 036")

    _require(_alembic(scratch_url, "downgrade", "035"), "downgrade 036 -> 035")

    after = _scopes(db)
    assert after["moves"] == "org"
    assert after["moves_too"] == "org"
    # The row that was ALREADY project-scoped before 036 ran must not be
    # widened by rolling 036 back — the downgrade reverses this migration, it
    # does not publish someone else's project memory to the org.
    assert after["already_project"] == "project"
    assert after["org_wide_artifact"] == "org"
    assert after["plain_fact"] == "org"


def test_a_card_written_at_project_scope_after_036_survives_a_rollback(scratch_url, db):
    """The rollback must not leak what the fixed code deliberately narrowed."""
    _require(_alembic(scratch_url, "upgrade", "035"), "upgrade to 035")
    _seed(db)
    _require(_alembic(scratch_url, "upgrade", "036"), "upgrade 035 -> 036")

    # The fixed application writes this one AFTER the migration ran.
    db.execute(
        text(
            "INSERT INTO session_memories "
            "(org_id, agent_id, session_id, scope, status, content, "
            " memory_type, project_id) "
            "VALUES (:o, 'a', 's', 'project', 'active', 'written_after', "
            " 'artifact', :p)"
        ),
        {"o": ORG, "p": PROJECT},
    )

    _require(_alembic(scratch_url, "downgrade", "035"), "downgrade 036 -> 035")

    assert _scopes(db)["written_after"] == "project"


def test_upgrade_is_re_appliable_after_a_downgrade(scratch_url, db):
    _require(_alembic(scratch_url, "upgrade", "035"), "upgrade to 035")
    _seed(db)
    for _ in range(2):
        _require(_alembic(scratch_url, "upgrade", "036"), "upgrade 035 -> 036")
        assert _scopes(db)["moves"] == "project"
        _require(_alembic(scratch_url, "downgrade", "035"), "downgrade 036 -> 035")
        assert _scopes(db)["moves"] == "org"
