"""Migration 040 remaps member→write and widens the role CHECK.

Own scratch database: this file moves the schema and seeds a pre-040
``member`` row that the shared dev database must not host.
"""

import os
import subprocess
import sys
import uuid
from urllib.parse import urlsplit, urlunsplit

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",
)

SCRATCH_DB = "klio_mig_040_project_access_roles"
MIGRATIONS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORG = "org-040"
PROJECT = str(uuid.uuid4())
USER = "user-040"


def _with_database(url: str, database: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(
        (parts.scheme, parts.netloc, f"/{database}", parts.query, parts.fragment)
    )


def _alembic(url: str, *args: str):
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=MIGRATIONS_DIR,
        env={**os.environ, "DATABASE_URL": url},
        capture_output=True,
        text=True,
    )


def _require(proc, what: str) -> None:
    assert proc.returncode == 0, f"{what} failed:\n{proc.stdout}\n{proc.stderr}"


@pytest.fixture(scope="module")
def scratch():
    admin = create_engine(
        _with_database(DB_URL, "postgres"), isolation_level="AUTOCOMMIT"
    )
    try:
        with admin.connect() as conn:
            conn.execute(text(f"DROP DATABASE IF EXISTS {SCRATCH_DB}"))
            conn.execute(text(f"CREATE DATABASE {SCRATCH_DB}"))
    finally:
        admin.dispose()

    url = _with_database(DB_URL, SCRATCH_DB)
    _require(_alembic(url, "upgrade", "039"), "upgrade to 039")
    try:
        yield url
    finally:
        admin = create_engine(
            _with_database(DB_URL, "postgres"), isolation_level="AUTOCOMMIT"
        )
        try:
            with admin.connect() as conn:
                conn.execute(text(f"DROP DATABASE IF EXISTS {SCRATCH_DB}"))
        finally:
            admin.dispose()


def test_member_rows_become_write_and_read_is_legal(scratch):
    engine = create_engine(scratch)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO projects (id, org_id, display_name, repo_root_path) "
                    "VALUES (:id, :org, 'p', '/tmp/p')"
                ),
                {"id": PROJECT, "org": ORG},
            )
            conn.execute(
                text(
                    "INSERT INTO project_members (project_id, user_id, role) "
                    "VALUES (:p, :u, 'member')"
                ),
                {"p": PROJECT, "u": USER},
            )
            conn.execute(
                text(
                    "INSERT INTO project_grant_audit "
                    "(id, org_id, project_id, granted_to, role, surface) "
                    "VALUES (:id, :org, :p, :u, 'member', 'dashboard')"
                ),
                {"id": str(uuid.uuid4()), "org": ORG, "p": PROJECT, "u": USER},
            )
    finally:
        engine.dispose()

    _require(_alembic(scratch, "upgrade", "head"), "upgrade 040")

    engine = create_engine(scratch)
    try:
        with engine.connect() as conn:
            role = conn.execute(
                text("SELECT role FROM project_members WHERE user_id = :u"),
                {"u": USER},
            ).scalar()
            assert role == "write"
            audit = conn.execute(
                text("SELECT role FROM project_grant_audit WHERE granted_to = :u"),
                {"u": USER},
            ).scalar()
            assert audit == "write"

            conn.execute(
                text(
                    "INSERT INTO project_members (project_id, user_id, role) "
                    "VALUES (:p, 'reader', 'read')"
                ),
                {"p": PROJECT},
            )
            conn.commit()

            with pytest.raises(IntegrityError):
                conn.execute(
                    text(
                        "INSERT INTO project_members (project_id, user_id, role) "
                        "VALUES (:p, 'owner', 'owner')"
                    ),
                    {"p": PROJECT},
                )
            conn.rollback()
    finally:
        engine.dispose()
