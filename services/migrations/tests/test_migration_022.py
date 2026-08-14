import importlib.util
import os
import subprocess
import sys
import uuid

from sqlalchemy import create_engine, inspect, text
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the migration 022 tests")

_MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "..")
_VERSION_022 = os.path.join(
    _MIGRATIONS_DIR, "alembic", "versions", "022_brain_recall_events.py"
)


def _alembic(*args: str) -> None:
    env = os.environ.copy()
    env["DATABASE_URL"] = DB_URL
    r = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        capture_output=True,
        text=True,
        cwd=_MIGRATIONS_DIR,
        env=env,
    )
    assert r.returncode == 0, (
        f"alembic {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}"
    )


def _alembic_upgrade_head():
    _alembic("upgrade", "head")


def _load_migration_022_module():
    spec = importlib.util.spec_from_file_location("migration_022", _VERSION_022)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_down_revision_points_at_021():
    module = _load_migration_022_module()
    assert module.revision == "022"
    assert module.down_revision == "021"


def test_brain_recall_events_table_exists():
    _alembic_upgrade_head()
    eng = create_engine(DB_URL)
    with eng.connect() as c:
        tables = set(inspect(c).get_table_names(schema="public"))
    assert "brain_recall_events" in tables


def test_brain_recall_events_columns():
    _alembic_upgrade_head()
    eng = create_engine(DB_URL)
    with eng.connect() as c:
        cols = {
            col["name"]: col for col in inspect(c).get_columns("brain_recall_events")
        }
    assert set(cols) == {
        "id",
        "org_id",
        "agent_id",
        "project_id",
        "query_text",
        "result_count",
        "source",
        "created_at",
        # Added by migration 042 (recall evidence, Build 2): memory_ids links
        # a recall event to the memories it surfaced, session_id links it to
        # the brain session it happened in. Both nullable, both additive.
        "memory_ids",
        "session_id",
    }
    assert cols["org_id"]["nullable"] is False
    assert cols["agent_id"]["nullable"] is True
    assert cols["project_id"]["nullable"] is True
    assert cols["query_text"]["nullable"] is True
    assert cols["result_count"]["nullable"] is False
    assert cols["source"]["nullable"] is True
    assert cols["created_at"]["nullable"] is False


def test_brain_recall_events_indexes():
    _alembic_upgrade_head()
    eng = create_engine(DB_URL)
    with eng.connect() as c:
        idx = {i["name"]: i for i in inspect(c).get_indexes("brain_recall_events")}
    assert "ix_brain_recall_events_org_created" in idx
    assert idx["ix_brain_recall_events_org_created"]["column_names"] == [
        "org_id",
        "created_at",
    ]
    assert "ix_brain_recall_events_org_agent" in idx
    assert idx["ix_brain_recall_events_org_agent"]["column_names"] == [
        "org_id",
        "agent_id",
    ]


def test_insert_minimal_row_uses_server_defaults():
    """Inserting only the required fields lets the server defaults fire:
    ``result_count`` defaults to 0 and ``created_at`` defaults to now()."""
    _alembic_upgrade_head()
    eng = create_engine(DB_URL)
    row_id = uuid.uuid4()
    with eng.begin() as c:
        c.execute(
            text("""
            INSERT INTO brain_recall_events (id, org_id)
            VALUES (:id, :org_id)
        """),
            {"id": str(row_id), "org_id": "org_test"},
        )
        row = c.execute(
            text(
                "SELECT result_count, created_at FROM brain_recall_events WHERE id=:id"
            ),
            {"id": str(row_id)},
        ).one()
        # cleanup
        c.execute(
            text("DELETE FROM brain_recall_events WHERE id=:id"),
            {"id": str(row_id)},
        )
    assert row.result_count == 0
    assert row.created_at is not None


def test_downgrade_removes_table_and_indexes():
    _alembic_upgrade_head()
    _alembic("downgrade", "021")
    eng = create_engine(DB_URL)
    with eng.connect() as c:
        tables = set(inspect(c).get_table_names(schema="public"))
    assert "brain_recall_events" not in tables
    # Restore head so the test suite leaves the DB at the latest revision.
    _alembic_upgrade_head()
