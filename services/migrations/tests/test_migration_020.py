import os
import subprocess
import sys
import uuid

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",
)


def _alembic_upgrade_head():
    env = os.environ.copy()
    env["DATABASE_URL"] = DB_URL
    r = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        capture_output=True,
        text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        env=env,
    )
    assert r.returncode == 0, f"alembic upgrade failed:\n{r.stdout}\n{r.stderr}"


def test_spaces_and_projects_tables_exist():
    _alembic_upgrade_head()
    eng = create_engine(DB_URL)
    with eng.connect() as c:
        tables = set(inspect(c).get_table_names(schema="public"))
    assert "spaces" in tables
    assert "projects" in tables


def test_session_memories_gains_nullable_space_project():
    _alembic_upgrade_head()
    eng = create_engine(DB_URL)
    with eng.connect() as c:
        cols = {col["name"]: col for col in inspect(c).get_columns("session_memories")}
    assert "space_id" in cols and cols["space_id"]["nullable"] is True
    assert "project_id" in cols and cols["project_id"]["nullable"] is True


def test_projects_partial_unique_indexes():
    _alembic_upgrade_head()
    eng = create_engine(DB_URL)
    with eng.connect() as c:
        idx = {i["name"]: i for i in inspect(c).get_indexes("projects")}
    assert (
        "ix_projects_org_remote" in idx
        and idx["ix_projects_org_remote"]["unique"] is True
    )
    assert (
        "ix_projects_org_path" in idx and idx["ix_projects_org_path"]["unique"] is True
    )


def test_existing_session_memory_row_undisturbed():
    """Additive contract: a row without space/project stays valid."""
    _alembic_upgrade_head()
    eng = create_engine(DB_URL)
    with eng.begin() as c:
        sid = str(uuid.uuid4())
        c.execute(
            text("""
            INSERT INTO session_memories
              (id, session_id, org_id, agent_id, memory_type, content, scope, status)
            VALUES (gen_random_uuid(), :sid, 'org_test', 'agent_test',
                    'fact', 'legacy row', 'session', 'active')
        """),
            {"sid": sid},
        )
        row = c.execute(
            text(
                "SELECT space_id, project_id FROM session_memories WHERE session_id=:sid"
            ),
            {"sid": sid},
        ).one()
        # cleanup
        c.execute(
            text("DELETE FROM session_memories WHERE session_id=:sid"), {"sid": sid}
        )
    assert row.space_id is None and row.project_id is None


def test_projects_rejects_both_identity_null():
    """A project with neither git_remote nor repo_root_path is unrepresentable
    (ck_projects_has_identity) — prevents unlimited duplicate anonymous rows."""
    _alembic_upgrade_head()
    eng = create_engine(DB_URL)
    with pytest.raises(IntegrityError):
        with eng.begin() as c:
            c.execute(
                text("""
                INSERT INTO projects (id, org_id, display_name)
                VALUES (gen_random_uuid(), 'org_test', 'anon')
            """)
            )
