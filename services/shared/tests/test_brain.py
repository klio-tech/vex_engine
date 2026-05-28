"""Tests for the shared Klio brain module (spaces + projects ensure helpers).

These are DB-logic functions, so the tests run against real Postgres (mirroring
``services/migrations/tests/test_migration_020.py``) rather than mocks. Each
test uses a unique ``org_id`` so runs are isolated on the shared dev DB, and a
session-scoped fixture cleans up any rows created for those org ids.
"""

import os
import time
import uuid

import pytest
from sqlalchemy import create_engine, text

from shared.brain import ensure_project, ensure_space

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",
)


def _unique_org() -> str:
    """Generate a unique org id so tests don't collide on the shared dev DB."""
    return f"org_{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def db_session():
    """Yield a real SQLAlchemy Connection and clean up its rows afterward.

    A ``Connection`` in SQLAlchemy 2.x exposes ``.execute()`` and ``.commit()``,
    matching the synchronous interface used by the brain helpers and by
    ``shared.memory``. Tracked org ids are deleted on teardown to keep the
    shared dev database clean.
    """
    engine = create_engine(DB_URL)
    conn = engine.connect()
    created_org_ids: set[str] = set()
    conn.created_org_ids = created_org_ids  # type: ignore[attr-defined]
    try:
        yield conn
    finally:
        # Clean up in a fresh transaction; projects first (FK to spaces).
        try:
            for org_id in created_org_ids:
                conn.execute(
                    text("DELETE FROM projects WHERE org_id = :o"), {"o": org_id}
                )
                conn.execute(
                    text("DELETE FROM spaces WHERE org_id = :o"), {"o": org_id}
                )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _track(db_session, org_id: str) -> str:
    """Register an org id for teardown cleanup and return it."""
    db_session.created_org_ids.add(org_id)
    return org_id


# ---------------------------------------------------------------------------
# ensure_space
# ---------------------------------------------------------------------------


def test_ensure_space_creates_then_dedupes(db_session):
    """Two calls with the same (org_id, slug) return the same space id."""
    org_id = _track(db_session, _unique_org())

    first = ensure_space(
        db_session, org_id=org_id, name="My Space", slug="my-space"
    )
    second = ensure_space(
        db_session, org_id=org_id, name="My Space (renamed)", slug="my-space"
    )

    assert isinstance(first, str)
    assert first == second

    count = db_session.execute(
        text("SELECT COUNT(*) FROM spaces WHERE org_id = :o AND slug = :s"),
        {"o": org_id, "s": "my-space"},
    ).scalar()
    assert count == 1


# ---------------------------------------------------------------------------
# ensure_project
# ---------------------------------------------------------------------------


def test_ensure_project_creates_on_first(db_session):
    """ensure_project returns an id and persists a row with the right fields."""
    org_id = _track(db_session, _unique_org())

    project_id = ensure_project(
        db_session,
        org_id=org_id,
        display_name="Klio Engine",
        git_remote="git@github.com:klio-tech/engine.git",
        repo_root_path="/home/dev/engine",
    )

    assert isinstance(project_id, str)

    row = db_session.execute(
        text(
            """
            SELECT org_id, display_name, git_remote, repo_root_path
            FROM projects WHERE id = :id
            """
        ),
        {"id": project_id},
    ).one()
    assert row.org_id == org_id
    assert row.display_name == "Klio Engine"
    assert row.git_remote == "git@github.com:klio-tech/engine.git"
    assert row.repo_root_path == "/home/dev/engine"


def test_ensure_project_dedupes_by_remote(db_session):
    """Same git_remote dedupes even when repo_root_path differs."""
    org_id = _track(db_session, _unique_org())
    remote = "git@github.com:klio-tech/engine.git"

    first = ensure_project(
        db_session,
        org_id=org_id,
        display_name="Engine A",
        git_remote=remote,
        repo_root_path="/home/dev/checkout-a",
    )
    second = ensure_project(
        db_session,
        org_id=org_id,
        display_name="Engine B",
        git_remote=remote,
        repo_root_path="/home/dev/checkout-b",
    )

    assert first == second

    count = db_session.execute(
        text("SELECT COUNT(*) FROM projects WHERE org_id = :o AND git_remote = :r"),
        {"o": org_id, "r": remote},
    ).scalar()
    assert count == 1


def test_ensure_project_dedupes_by_path_when_no_remote(db_session):
    """With git_remote=None, the same repo_root_path dedupes."""
    org_id = _track(db_session, _unique_org())
    path = "/home/dev/no-remote-repo"

    first = ensure_project(
        db_session,
        org_id=org_id,
        display_name="Local Repo",
        git_remote=None,
        repo_root_path=path,
    )
    second = ensure_project(
        db_session,
        org_id=org_id,
        display_name="Local Repo Again",
        git_remote=None,
        repo_root_path=path,
    )

    assert first == second

    count = db_session.execute(
        text(
            """
            SELECT COUNT(*) FROM projects
            WHERE org_id = :o AND git_remote IS NULL AND repo_root_path = :p
            """
        ),
        {"o": org_id, "p": path},
    ).scalar()
    assert count == 1


def test_ensure_project_bumps_last_seen_at(db_session):
    """A second ensure on an existing project advances last_seen_at."""
    org_id = _track(db_session, _unique_org())
    remote = "git@github.com:klio-tech/bump.git"

    project_id = ensure_project(
        db_session,
        org_id=org_id,
        display_name="Bump Repo",
        git_remote=remote,
    )
    first_seen = db_session.execute(
        text("SELECT last_seen_at FROM projects WHERE id = :id"),
        {"id": project_id},
    ).scalar()

    # Each ensure commits, so the second runs in a new transaction and
    # transaction_timestamp() (NOW()) advances. A tiny sleep guarantees the
    # clock moves on fast machines.
    time.sleep(0.01)

    ensure_project(
        db_session,
        org_id=org_id,
        display_name="Bump Repo",
        git_remote=remote,
    )
    second_seen = db_session.execute(
        text("SELECT last_seen_at FROM projects WHERE id = :id"),
        {"id": project_id},
    ).scalar()

    assert second_seen > first_seen


def test_ensure_project_raises_when_both_identity_none(db_session):
    """ensure_project raises ValueError when both identity fields are None."""
    org_id = _track(db_session, _unique_org())

    with pytest.raises(ValueError):
        ensure_project(
            db_session,
            org_id=org_id,
            display_name="No Identity",
            git_remote=None,
            repo_root_path=None,
        )


def test_ensure_project_tenant_isolation(db_session):
    """The same git_remote under two orgs yields two distinct projects."""
    org_a = _track(db_session, _unique_org())
    org_b = _track(db_session, _unique_org())
    remote = "git@github.com:klio-tech/shared-name.git"

    id_a = ensure_project(
        db_session, org_id=org_a, display_name="A", git_remote=remote
    )
    id_b = ensure_project(
        db_session, org_id=org_b, display_name="B", git_remote=remote
    )

    assert id_a != id_b

    # Re-ensuring org A's project must never return org B's id.
    id_a_again = ensure_project(
        db_session, org_id=org_a, display_name="A", git_remote=remote
    )
    assert id_a_again == id_a
    assert id_a_again != id_b
