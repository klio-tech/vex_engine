"""Migration 049's join keys exist and their partial uniqueness holds.

``organizations.workspace_id`` and ``projects.plane_project_id`` are the
immutable Plane <-> engine join keys. Both are nullable (every pre-handshake
row is legal) and unique where present (the match is one-to-one). These tests
run against the live dev database, like ``test_scopes_single_definition.py``,
and prove the partial indexes both ways: duplicate non-NULL values raise,
while multiple NULLs coexist. Every write happens inside a transaction that
is rolled back, so the database is left exactly as found.
"""

import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from shared.testing import database_url, requires_db

DB_URL = database_url()


@pytest.fixture()
def db_conn():
    """A connection to the dev database. Callers manage their own txns."""
    engine = create_engine(DB_URL)
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()
        engine.dispose()


def _column_type(conn, table: str, column: str):
    """The udt_name of ``table.column``, or None if the column is absent."""
    row = conn.execute(
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
    return row


def _insert_org(conn, *, workspace_id):
    conn.execute(
        text(
            """
            INSERT INTO organizations (org_id, name, workspace_id)
            VALUES (:org_id, :name, :workspace_id)
            """
        ),
        {
            "org_id": f"test-hs-{uuid.uuid4().hex[:12]}",
            "name": "handshake-test",
            "workspace_id": workspace_id,
        },
    )


def _insert_project(conn, *, org_id, plane_project_id):
    conn.execute(
        text(
            """
            INSERT INTO projects
                (org_id, display_name, repo_root_path, plane_project_id)
            VALUES
                (:org_id, :display_name, :repo_root_path, :plane_project_id)
            """
        ),
        {
            "org_id": org_id,
            "display_name": "handshake-test",
            # ck_projects_has_identity needs one of git_remote/repo_root_path;
            # a unique path keeps ix_projects_org_path out of the way.
            "repo_root_path": f"/tmp/handshake-test/{uuid.uuid4().hex}",
            "plane_project_id": plane_project_id,
        },
    )


@requires_db("the workspace handshake key columns test")
def test_organizations_workspace_id_is_a_nullable_uuid(db_conn):
    row = _column_type(db_conn, "organizations", "workspace_id")
    assert row is not None, "organizations.workspace_id is missing"
    assert row.udt_name == "uuid"
    assert row.is_nullable == "YES"


@requires_db("the workspace handshake key columns test")
def test_projects_plane_project_id_is_a_nullable_uuid(db_conn):
    row = _column_type(db_conn, "projects", "plane_project_id")
    assert row is not None, "projects.plane_project_id is missing"
    assert row.udt_name == "uuid"
    assert row.is_nullable == "YES"


@requires_db("the workspace_id uniqueness test")
def test_duplicate_non_null_workspace_id_raises(db_conn):
    ws = str(uuid.uuid4())
    txn = db_conn.begin()
    try:
        _insert_org(db_conn, workspace_id=ws)
        with pytest.raises(IntegrityError) as excinfo:
            _insert_org(db_conn, workspace_id=ws)
        assert "uq_organizations_workspace_id" in str(excinfo.value)
    finally:
        txn.rollback()


@requires_db("the workspace_id uniqueness test")
def test_null_workspace_ids_coexist(db_conn):
    txn = db_conn.begin()
    try:
        # Pre-handshake orgs all have NULL workspace_id; the index must be
        # partial so any number of them are legal side by side.
        _insert_org(db_conn, workspace_id=None)
        _insert_org(db_conn, workspace_id=None)
    finally:
        txn.rollback()


@requires_db("the plane_project_id uniqueness test")
def test_duplicate_non_null_plane_project_id_raises(db_conn):
    plane_id = str(uuid.uuid4())
    org_id = f"test-hs-{uuid.uuid4().hex[:12]}"
    txn = db_conn.begin()
    try:
        _insert_org(db_conn, workspace_id=None)
        db_conn.execute(
            text("INSERT INTO organizations (org_id, name) VALUES (:o, 'hs')"),
            {"o": org_id},
        )
        _insert_project(db_conn, org_id=org_id, plane_project_id=plane_id)
        with pytest.raises(IntegrityError) as excinfo:
            _insert_project(db_conn, org_id=org_id, plane_project_id=plane_id)
        assert "uq_projects_plane_project_id" in str(excinfo.value)
    finally:
        txn.rollback()


@requires_db("the plane_project_id uniqueness test")
def test_null_plane_project_ids_coexist(db_conn):
    org_id = f"test-hs-{uuid.uuid4().hex[:12]}"
    txn = db_conn.begin()
    try:
        db_conn.execute(
            text("INSERT INTO organizations (org_id, name) VALUES (:o, 'hs')"),
            {"o": org_id},
        )
        _insert_project(db_conn, org_id=org_id, plane_project_id=None)
        _insert_project(db_conn, org_id=org_id, plane_project_id=None)
    finally:
        txn.rollback()
