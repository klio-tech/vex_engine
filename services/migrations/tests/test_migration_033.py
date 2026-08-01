"""Migration 033 makes a revoke recordable without weakening 032.

DB tests: they need the live test Postgres and skip without it, matching the
convention in test_migration_031.py.

The three things worth asserting are the three things that could go wrong:
the column exists and is constrained to the two verbs; the default keeps the
engine's existing column-list INSERT writing correct rows; and 032's role
constraint is still exactly as it was, because "add a verb" must not quietly
become "let anything into role".
"""

import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",
)


@pytest.fixture(scope="module")
def conn():
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception:
        pytest.skip(
            "Postgres not reachable; migration 033 tests require the live test DB"
        )
    with create_engine(DB_URL).connect() as c:
        yield c


ORG = "test-org-033"


def _insert(conn, **overrides):
    params = {
        "id": str(uuid.uuid4()),
        "org_id": ORG,
        "project_id": str(uuid.uuid4()),
        "granted_to": "user-a",
        "granted_to_email": "a@example.com",
        "granted_by": "user-b",
        "role": "member",
        "surface": "mcp",
    }
    params.update(overrides)
    columns = ", ".join(params)
    values = ", ".join(f":{name}" for name in params)
    conn.execute(
        text(f"INSERT INTO project_grant_audit ({columns}) VALUES ({values})"),
        params,
    )


@pytest.fixture(autouse=True)
def cleanup(conn):
    yield
    conn.execute(
        text("DELETE FROM project_grant_audit WHERE org_id = :o"), {"o": ORG}
    )
    conn.commit()


def test_action_column_exists_and_is_not_null(conn):
    row = conn.execute(
        text(
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_name = 'project_grant_audit' AND column_name = 'action'"
        )
    ).fetchone()
    assert row is not None, "migration 033 did not add project_grant_audit.action"
    assert row[0] == "NO", "action must be NOT NULL — every row describes some verb"


def test_existing_insert_without_action_still_records_a_grant(conn):
    """The engine's record_grant_audit names its columns and omits `action`.

    Without the server default that statement would fail outright, so an ACL
    change over MCP would be refused because its audit row would not insert.
    """
    _insert(conn)
    action = conn.execute(
        text("SELECT action FROM project_grant_audit WHERE org_id = :o"), {"o": ORG}
    ).scalar()
    assert action == "grant"


def test_a_revoke_is_recordable(conn):
    _insert(conn, action="revoke", role="admin")
    row = conn.execute(
        text("SELECT action, role FROM project_grant_audit WHERE org_id = :o"),
        {"o": ORG},
    ).fetchone()
    # The role a revoke records is the role the member actually held.
    assert row == ("revoke", "admin")


def test_an_unknown_verb_is_rejected(conn):
    with pytest.raises(IntegrityError):
        _insert(conn, action="deleted")
    conn.rollback()


def test_032_role_constraint_is_untouched(conn):
    """033 adds a verb; it does not widen what `role` accepts."""
    with pytest.raises(IntegrityError):
        _insert(conn, role="owner")
    conn.rollback()

    definition = conn.execute(
        text(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conname = 'ck_project_grant_audit_role'"
        )
    ).scalar()
    assert definition is not None, "032's role CHECK must still exist"
    assert "'member'" in definition and "'admin'" in definition
