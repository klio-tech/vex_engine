"""Migration 050's branch registry: schema shapes and the branch lifecycle.

``context_branches`` is the registry a context branch lives in — a row, not a
naming convention — and ``session_memories.branch_id`` is the nullable column
that puts a memory on one. ``branch_id IS NULL`` IS main, so both columns are
nullable by construction and every pre-branch row is legal.

These tests run against the live dev database, like
``test_workspace_handshake_keys.py``, because the two load-bearing shapes are
Postgres behaviour, not Python behaviour: the partial unique index
(one OPEN branch per name; a merged/discarded name is reusable, like git) and
the ``ON CONFLICT`` partial-index inference ``ensure_branch`` upserts through.
Every write is cleaned up afterwards, so the database is left as found.
"""

import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from shared import branches
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the context-branch registry tests")


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
                    text("DELETE FROM session_memories WHERE org_id = :o"),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM context_branches WHERE org_id = :o"),
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
    org_id = f"org_br_{uuid.uuid4().hex[:8]}"
    db.created_org_ids.add(org_id)
    return org_id


def _insert_branch(db, *, org_id: str, project_id: str, name: str, status: str):
    db.execute(
        text(
            """
            INSERT INTO context_branches (org_id, project_id, name, status)
            VALUES (:org_id, :project_id, :name, :status)
            """
        ),
        {"org_id": org_id, "project_id": project_id, "name": name, "status": status},
    )


def _project(db, *, org_id: str) -> str:
    """A real projects row — session_memories.project_id is FK-constrained.

    ``context_branches.project_id`` deliberately is NOT (migration 050 matches
    the repo's unconstrained-cross-ref convention), so registry-only tests may
    use bare UUIDs; only tests that seed memories need this.
    """
    project_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO projects (id, org_id, git_remote, display_name)
            VALUES (:id, :org_id, :remote, 'branch-schema-test')
            """
        ),
        {
            "id": project_id,
            "org_id": org_id,
            "remote": f"git@example.com:{uuid.uuid4().hex[:8]}.git",
        },
    )
    db.commit()
    return project_id


def _seed_memory(
    db, *, org_id: str, project_id: str, branch_id: str, status: str = "active"
) -> str:
    memory_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, memory_type, content,
                scope, status, project_id, branch_id, metadata
            ) VALUES (
                :id, :session_id, :org_id, 'agent-1', 'note', :content,
                'project', :status, :project_id, :branch_id, '{}'::jsonb
            )
            """
        ),
        {
            "id": memory_id,
            "session_id": f"sess-{uuid.uuid4().hex[:6]}",
            "org_id": org_id,
            "content": f"branch memory {memory_id}",
            "status": status,
            "project_id": project_id,
            "branch_id": branch_id,
        },
    )
    return memory_id


# ---------------------------------------------------------------------------
# Schema shapes
# ---------------------------------------------------------------------------


def _column(db, table: str, column: str):
    return db.execute(
        text(
            """
            SELECT udt_name, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = :table
              AND column_name = :column
            """
        ),
        {"table": table, "column": column},
    ).first()


def test_context_branches_table_shape(db):
    expectations = {
        "id": ("uuid", "NO"),
        "org_id": ("varchar", "NO"),
        "project_id": ("uuid", "NO"),
        "name": ("varchar", "NO"),
        "status": ("varchar", "NO"),
        "created_by": ("varchar", "YES"),
        "created_at": ("timestamptz", "NO"),
        "closed_at": ("timestamptz", "YES"),
    }
    for column, (udt, nullable) in expectations.items():
        row = _column(db, "context_branches", column)
        assert row is not None, f"context_branches.{column} is missing"
        assert row.udt_name == udt, f"context_branches.{column} is {row.udt_name}"
        assert row.is_nullable == nullable


def test_session_memories_branch_id_is_a_nullable_uuid(db):
    row = _column(db, "session_memories", "branch_id")
    assert row is not None, "session_memories.branch_id is missing"
    assert row.udt_name == "uuid"
    assert row.is_nullable == "YES"


def test_status_check_constraint_rejects_unknown_status(db):
    org = _org(db)
    txn = db.begin()
    try:
        with pytest.raises(IntegrityError) as excinfo:
            _insert_branch(
                db,
                org_id=org,
                project_id=str(uuid.uuid4()),
                name="feat/x",
                status="closed",
            )
        assert "ck_context_branches_status" in str(excinfo.value)
    finally:
        txn.rollback()


def test_one_open_branch_per_name_but_closed_names_are_reusable(db):
    """The partial unique index both ways: a second OPEN row with the same
    (org, project, name) raises; the same name over a merged row is fine."""
    org = _org(db)
    project = str(uuid.uuid4())
    txn = db.begin()
    try:
        _insert_branch(db, org_id=org, project_id=project, name="feat/x", status="open")
        with pytest.raises(IntegrityError) as excinfo:
            _insert_branch(
                db, org_id=org, project_id=project, name="feat/x", status="open"
            )
        assert "uq_context_branches_open_name" in str(excinfo.value)
    finally:
        txn.rollback()

    txn = db.begin()
    try:
        _insert_branch(
            db, org_id=org, project_id=project, name="feat/x", status="merged"
        )
        _insert_branch(db, org_id=org, project_id=project, name="feat/x", status="open")
    finally:
        txn.rollback()


# ---------------------------------------------------------------------------
# ensure_branch: upsert through the partial index, and name validation
# ---------------------------------------------------------------------------


def test_ensure_branch_creates_then_returns_the_same_row(db):
    """The ON CONFLICT partial-index inference is the point: a second ensure
    for the same open name must land on the existing row, not raise."""
    org = _org(db)
    project = str(uuid.uuid4())

    first = branches.ensure_branch(
        db, org_id=org, project_id=project, name="feat/x", created_by="priya"
    )
    assert first["name"] == "feat/x"
    assert first["status"] == "open"
    assert first["created_by"] == "priya"

    again = branches.ensure_branch(
        db, org_id=org, project_id=project, name="feat/x", created_by="dev"
    )
    assert again["id"] == first["id"]
    assert again["status"] == "open"


def test_ensure_branch_strips_whitespace(db):
    org = _org(db)
    project = str(uuid.uuid4())
    created = branches.ensure_branch(
        db, org_id=org, project_id=project, name="  feat/x  ", created_by=None
    )
    assert created["name"] == "feat/x"
    same = branches.ensure_branch(
        db, org_id=org, project_id=project, name="feat/x", created_by=None
    )
    assert same["id"] == created["id"]


@pytest.mark.parametrize("bad_name", ["", "   ", None])
def test_ensure_branch_refuses_blank_names(db, bad_name):
    with pytest.raises(ValueError):
        branches.ensure_branch(
            db,
            org_id=_org(db),
            project_id=str(uuid.uuid4()),
            name=bad_name,
            created_by=None,
        )


@pytest.mark.parametrize("reserved", ["main", "MAIN", "Main", "  main  "])
def test_ensure_branch_refuses_main(db, reserved):
    """``main`` is the ABSENCE of a branch (branch_id IS NULL). A registry row
    named main would fork the meaning of NULL, so the name is reserved."""
    with pytest.raises(ValueError):
        branches.ensure_branch(
            db,
            org_id=_org(db),
            project_id=str(uuid.uuid4()),
            name=reserved,
            created_by=None,
        )


def test_ensure_branch_refuses_overlong_names(db):
    with pytest.raises(ValueError):
        branches.ensure_branch(
            db,
            org_id=_org(db),
            project_id=str(uuid.uuid4()),
            name="x" * 201,
            created_by=None,
        )


def test_ensure_branch_reuses_a_name_after_merge(db):
    """Exactly like git: a merged branch's name is free for a new branch."""
    org = _org(db)
    project = str(uuid.uuid4())
    first = branches.ensure_branch(
        db, org_id=org, project_id=project, name="feat/x", created_by=None
    )
    db.execute(
        text("UPDATE context_branches SET status = 'merged' WHERE id = :id"),
        {"id": first["id"]},
    )
    db.commit()

    second = branches.ensure_branch(
        db, org_id=org, project_id=project, name="feat/x", created_by=None
    )
    assert second["id"] != first["id"]
    assert second["status"] == "open"


# ---------------------------------------------------------------------------
# get_branch / get_branch_by_id / list_branches
# ---------------------------------------------------------------------------


def test_get_branch_returns_open_branches_only(db):
    org = _org(db)
    project = str(uuid.uuid4())
    created = branches.ensure_branch(
        db, org_id=org, project_id=project, name="feat/x", created_by=None
    )

    found = branches.get_branch(db, org_id=org, project_id=project, name="feat/x")
    assert found is not None
    assert found["id"] == created["id"]

    db.execute(
        text("UPDATE context_branches SET status = 'discarded' WHERE id = :id"),
        {"id": created["id"]},
    )
    db.commit()
    assert (
        branches.get_branch(db, org_id=org, project_id=project, name="feat/x") is None
    )


def test_get_branch_by_id_is_org_scoped(db):
    """A branch id from org A resolves in org A and NOWHERE else — the id is
    a UUID a caller may have obtained from anywhere."""
    org_a = _org(db)
    org_b = _org(db)
    project = str(uuid.uuid4())
    created = branches.ensure_branch(
        db, org_id=org_a, project_id=project, name="feat/x", created_by=None
    )

    assert (
        branches.get_branch_by_id(db, org_id=org_a, branch_id=created["id"]) is not None
    )
    assert branches.get_branch_by_id(db, org_id=org_b, branch_id=created["id"]) is None


def test_get_branch_by_id_tolerates_a_malformed_id(db):
    assert (
        branches.get_branch_by_id(db, org_id=_org(db), branch_id="not-a-uuid") is None
    )


def test_list_branches_reports_unmerged_count_and_last_activity(db):
    org = _org(db)
    project = _project(db, org_id=org)
    branch = branches.ensure_branch(
        db, org_id=org, project_id=project, name="feat/x", created_by=None
    )
    _seed_memory(db, org_id=org, project_id=project, branch_id=branch["id"])
    _seed_memory(db, org_id=org, project_id=project, branch_id=branch["id"])
    # A retracted row is not "unmerged work" and must not count.
    _seed_memory(
        db, org_id=org, project_id=project, branch_id=branch["id"], status="retracted"
    )
    db.commit()

    listed = branches.list_branches(db, org_id=org, project_id=project)
    assert len(listed) == 1
    assert listed[0]["id"] == branch["id"]
    assert listed[0]["unmerged_count"] == 2
    assert listed[0]["last_activity"] is not None


def test_list_branches_empty_branch_has_zero_count_and_no_activity(db):
    org = _org(db)
    project = str(uuid.uuid4())
    branches.ensure_branch(
        db, org_id=org, project_id=project, name="feat/idle", created_by=None
    )
    listed = branches.list_branches(db, org_id=org, project_id=project)
    assert len(listed) == 1
    assert listed[0]["unmerged_count"] == 0
    assert listed[0]["last_activity"] is None


def test_branches_are_invisible_across_orgs(db):
    """The tenancy boundary: org B never sees org A's branch, by name or list."""
    org_a = _org(db)
    org_b = _org(db)
    project = str(uuid.uuid4())
    branches.ensure_branch(
        db, org_id=org_a, project_id=project, name="feat/x", created_by=None
    )

    assert (
        branches.get_branch(db, org_id=org_b, project_id=project, name="feat/x") is None
    )
    assert branches.list_branches(db, org_id=org_b, project_id=project) == []
