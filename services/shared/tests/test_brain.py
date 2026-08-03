"""Tests for the shared Klio brain module (spaces + projects ensure helpers).

These are DB-logic functions, so the tests run against real Postgres (mirroring
``services/migrations/tests/test_migration_020.py``) rather than mocks. Each
test uses a unique ``org_id`` so runs are isolated on the shared dev DB, and a
session-scoped fixture cleans up any rows created for those org ids.
"""

import threading
import time
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from shared.brain import ensure_project, ensure_space
from shared.testing import database_url, requires_db

DB_URL = database_url()


pytestmark = requires_db("the brain storage tests")


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

    first = ensure_space(db_session, org_id=org_id, name="My Space", slug="my-space")
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

    id_a = ensure_project(db_session, org_id=org_a, display_name="A", git_remote=remote)
    id_b = ensure_project(db_session, org_id=org_b, display_name="B", git_remote=remote)

    assert id_a != id_b

    # Re-ensuring org A's project must never return org B's id.
    id_a_again = ensure_project(
        db_session, org_id=org_a, display_name="A", git_remote=remote
    )
    assert id_a_again == id_a
    assert id_a_again != id_b


def test_ensure_project_concurrent_calls_converge():
    """Two concurrent ensure_project calls for the same (org_id, git_remote)
    converge: no exception, both return the same id, exactly one row exists.

    Each worker uses its own ``create_engine`` + connection so the two calls
    genuinely race on separate DB sessions. This exercises the ON CONFLICT
    upsert path that replaced the old SELECT-then-INSERT (which would raise an
    unhandled IntegrityError under this race).
    """
    org = f"org_{uuid.uuid4().hex[:8]}"
    remote = "git@github.com:acme/repo.git"
    results: dict[str, str] = {}
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def worker(key: str) -> None:
        try:
            eng = create_engine(DB_URL)
            # The brain helper commits internally, so hand it a plain
            # Connection (matching the db_session fixture's interface) rather
            # than a begin() block that would commit twice.
            conn = eng.connect()
            try:
                # Maximize the race window: both threads attempt the insert
                # at the same instant.
                barrier.wait(timeout=10)
                pid = ensure_project(
                    conn, org_id=org, display_name="repo", git_remote=remote
                )
                results[key] = pid
            finally:
                conn.close()
                eng.dispose()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    try:
        assert not errors, f"concurrent ensure raised: {errors}"
        assert results["a"] == results["b"], (
            "concurrent ensures must converge to one id"
        )

        verify_eng = create_engine(DB_URL)
        with verify_eng.connect() as conn:
            n = conn.execute(
                text(
                    "SELECT count(*) FROM projects "
                    "WHERE org_id = :o AND git_remote = :r"
                ),
                {"o": org, "r": remote},
            ).scalar()
        verify_eng.dispose()
        assert n == 1, f"expected exactly one project row, found {n}"
    finally:
        # Teardown: remove the org's project rows.
        cleanup_eng = create_engine(DB_URL)
        with cleanup_eng.connect() as conn:
            conn.execute(text("DELETE FROM projects WHERE org_id = :o"), {"o": org})
            conn.commit()
        cleanup_eng.dispose()


# ---------------------------------------------------------------------------
# ensure_project: cross-tenant space_id guard (mock-based)
#
# These two tests exercise the defensive ownership check in isolation, without
# a live DB. The check runs a `SELECT 1 FROM spaces WHERE id=:space_id AND
# org_id=:org_id` before the INSERT; a missing row means the space is not owned
# by this org (or does not exist) and the call must be rejected. A real DB is
# unnecessary — we only need to assert the control flow around that SELECT — so
# a tiny fake session dispatches on the SQL text to simulate the two outcomes.
# ---------------------------------------------------------------------------


class _FakeResult:
    """Stand-in for a SQLAlchemy ``Result``, returning canned rows."""

    def __init__(self, *, fetchone_value=None, one_value=None):
        self._fetchone_value = fetchone_value
        self._one_value = one_value

    def fetchone(self):
        return self._fetchone_value

    def one(self):
        return self._one_value


class _FakeSession:
    """Minimal db_session double for ``ensure_project``'s SQL.

    Dispatches on the statement text: the ownership ``SELECT`` returns a row
    only when ``space_owned`` is True; the project ``INSERT ... RETURNING``
    yields a fixed id. ``commit`` is recorded so a test can assert the insert
    path actually committed.
    """

    def __init__(self, *, space_owned: bool):
        self.space_owned = space_owned
        self.committed = False
        self.executed_sql: list[str] = []

    def execute(self, clause, params=None):
        sql = str(clause)
        self.executed_sql.append(sql)
        if "FROM spaces" in sql:
            return _FakeResult(fetchone_value=(1,) if self.space_owned else None)
        # The only other statement ensure_project issues is the INSERT.
        return _FakeResult(one_value=SimpleNamespace(id="proj-fixed-id"))

    def commit(self):
        self.committed = True


def test_ensure_project_rejects_cross_org_space():
    """A space_id not owned by org_id is rejected before any INSERT/commit."""
    session = _FakeSession(space_owned=False)

    with pytest.raises(ValueError, match="space_id does not belong to org_id"):
        ensure_project(
            session,
            org_id="org_a",
            display_name="X",
            git_remote="git@github.com:acme/repo.git",
            space_id="space-owned-by-org-b",
        )

    # The ownership SELECT ran, but the INSERT and commit must not have.
    assert any("FROM spaces" in s for s in session.executed_sql)
    assert not any("INSERT INTO projects" in s for s in session.executed_sql)
    assert session.committed is False


def test_ensure_project_accepts_same_org_space():
    """A space_id owned by org_id passes the guard and inserts + commits."""
    session = _FakeSession(space_owned=True)

    project_id = ensure_project(
        session,
        org_id="org_a",
        display_name="X",
        git_remote="git@github.com:acme/repo.git",
        space_id="space-owned-by-org-a",
    )

    assert project_id == "proj-fixed-id"
    assert any("FROM spaces" in s for s in session.executed_sql)
    assert any("INSERT INTO projects" in s for s in session.executed_sql)
    assert session.committed is True
