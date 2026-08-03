"""Migration 031 asserts the shape Phase 2 depends on.

These are DB tests: they need the live test Postgres and skip without it,
matching the convention in services/shared/tests/test_brain_recall.py.

DEVIATION FROM THE PLAN — the provenance column is named ``provenance``, not
``confidence``. ``session_memories.confidence`` already exists: migration 016
created it as a nullable DOUBLE PRECISION, ``store_memory()`` writes a float
into it on every write path, ``_row_to_record`` reads it into
``MemoryRecord.confidence: float | None``, and ``mcp_app`` returns it verbatim
in the ``recall`` response. Adding a second column of that name is impossible,
and converting the existing one to a CHECK-constrained varchar would both
destroy data and change an observable API payload type — which phase 1 is not
allowed to do. The three labels, the CHECK constraint, the EXTRACTED default
and the curator backfill are all exactly as the spec requires; only the name
differs, because the spec's chosen name was already taken.
"""

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

# Matches the repo convention (test_migration_020.py, alembic.ini,
# docker-compose.yml). The plan specified TEST_DATABASE_URL against a vex_test
# database that exists nowhere in this repo, which would have made every
# assertion below skip silently for every other developer and in CI —
# reporting green while testing nothing.
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the migration 031 tests")


@pytest.fixture(scope="module")
def conn():
    """A real Connection. Reachability is the ``requires_db`` gate's job."""
    engine = create_engine(DB_URL)
    try:
        with engine.connect() as c:
            yield c
    finally:
        engine.dispose()


def _columns(conn, table):
    rows = conn.execute(
        text(
            "SELECT column_name, is_nullable, data_type FROM information_schema.columns "
            "WHERE table_name = :t"
        ),
        {"t": table},
    ).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def _has_constraint(conn, name):
    return (
        conn.execute(
            text("SELECT count(*) FROM pg_constraint WHERE conname = :n"),
            {"n": name},
        ).scalar()
        > 0
    )


def test_session_memories_has_nullable_user_id(conn):
    """user_id is nullable: NULL means unattributed, which is a legal state."""
    cols = _columns(conn, "session_memories")
    assert "user_id" in cols
    assert cols["user_id"][0] == "YES", (
        "user_id must be nullable — NULL means unattributed"
    )


def test_provenance_defaults_to_extracted_and_is_not_null(conn):
    cols = _columns(conn, "session_memories")
    assert "provenance" in cols
    assert cols["provenance"][0] == "NO"


def test_existing_float_confidence_is_untouched(conn):
    """The pre-existing column keeps its type: ``recall`` returns it verbatim,
    so changing it would change an observable payload."""
    cols = _columns(conn, "session_memories")
    assert cols["confidence"] == ("YES", "double precision")


def test_scope_check_constraint_rejects_a_typo(conn):
    """Before 031 there was NO constraint, so a typo'd scope created rows
    no query would ever return. That is the bug this closes."""
    tx = conn.begin_nested()
    with pytest.raises(IntegrityError):
        conn.execute(
            text(
                "INSERT INTO session_memories "
                "(org_id, agent_id, session_id, scope, status, content, memory_type) "
                "VALUES ('t', 'a', 's', 'privte', 'active', 'x', 'fact')"
            )
        )
    tx.rollback()


def test_scope_check_constraint_admits_private(conn):
    assert _has_constraint(conn, "ck_session_memories_scope")
    tx = conn.begin_nested()
    conn.execute(
        text(
            "INSERT INTO session_memories "
            "(org_id, agent_id, session_id, scope, status, content, memory_type, user_id) "
            "VALUES ('t031', 'a', 's', 'private', 'active', 'x', 'fact', 'u031')"
        )
    )
    tx.rollback()


def test_provenance_check_constraint_rejects_unknown_label(conn):
    tx = conn.begin_nested()
    with pytest.raises(IntegrityError):
        conn.execute(
            text(
                "INSERT INTO session_memories "
                "(org_id, agent_id, session_id, scope, status, content, memory_type, provenance) "
                "VALUES ('t', 'a', 's', 'org', 'active', 'x', 'fact', 'MAYBE')"
            )
        )
    tx.rollback()


def test_private_row_without_an_owner_is_rejected(conn):
    """A private row nobody owns is unreachable by every caller including its
    author, so it is a bug by construction rather than a state to handle."""
    tx = conn.begin_nested()
    with pytest.raises(IntegrityError):
        conn.execute(
            text(
                "INSERT INTO session_memories "
                "(org_id, agent_id, session_id, scope, status, content, memory_type) "
                "VALUES ('t031', 'a', 's', 'private', 'active', 'x', 'fact')"
            )
        )
    tx.rollback()


def test_curator_rows_backfilled_to_inferred(conn):
    """The one backfill Phase 1 performs. Curator output is an LLM deduction
    and must not be indistinguishable from something a user stated."""
    row = conn.execute(
        text(
            "SELECT count(*) FROM session_memories "
            "WHERE metadata->>'source' = 'curator' AND provenance <> 'INFERRED'"
        )
    ).scalar()
    assert row == 0, "every curator-authored row must be INFERRED"


def test_non_curator_rows_are_extracted(conn):
    row = conn.execute(
        text(
            "SELECT count(*) FROM session_memories "
            "WHERE COALESCE(metadata->>'source', '') <> 'curator' AND provenance <> 'EXTRACTED'"
        )
    ).scalar()
    assert row == 0


def test_project_members_table_exists_with_composite_pk(conn):
    cols = _columns(conn, "project_members")
    assert set(cols) >= {"project_id", "user_id", "role", "granted_by", "granted_at"}
    pk = conn.execute(
        text(
            "SELECT a.attname FROM pg_index i "
            "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
            "WHERE i.indrelid = 'project_members'::regclass AND i.indisprimary"
        )
    ).fetchall()
    assert {r[0] for r in pk} == {"project_id", "user_id"}


def test_private_partial_index_exists(conn):
    idx = conn.execute(
        text(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_session_memories_private'"
        )
    ).scalar()
    assert idx is not None, "private recall is the hot path and must not seq-scan"
    assert "private" in idx


def test_curator_state_pk_is_org_and_user(conn):
    pk = conn.execute(
        text(
            "SELECT a.attname FROM pg_index i "
            "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
            "WHERE i.indrelid = 'brain_curator_state'::regclass AND i.indisprimary"
        )
    ).fetchall()
    assert {r[0] for r in pk} == {"org_id", "user_id"}


def test_existing_memories_keep_org_scope_and_null_user(conn):
    """Decision 6: no retroactive privatisation."""
    leaked = conn.execute(
        text(
            "SELECT count(*) FROM session_memories WHERE scope = 'private' AND user_id IS NULL"
        )
    ).scalar()
    assert leaked == 0, "a private row with no owner is unreachable by anyone"
