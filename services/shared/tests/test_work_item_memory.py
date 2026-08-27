"""Migration 051: memories and branches filed under a work item.

Two things are pinned here, and they are the two that would rot silently.

The **schema** half: both columns are nullable, so every pre-051 row stays
legal, and the partial indexes exist so a work-item read is not a seq scan.

The **resolution** half: :func:`shared.work_items.resolve_work_item` is the
only thing standing between a caller-supplied uuid and a memory filed under
it, so its refusals matter more than its successes. A work item whose project
is not linked into the caller's org, a soft-deleted one, a draft, and a
work item belonging to ANOTHER org must all come back as ``None`` — the same
shape as "no such work item", so this is never an existence oracle.

Against the live dev database, like ``test_context_branches_schema.py``,
because these are Postgres behaviours rather than Python ones. The workspace
side (``issues``/``projects``) is stubbed in-process: those tables live on a
DIFFERENT server in production, so a test that put them in this one would be
asserting a join that cannot run in prod.
"""

import uuid

import pytest
from sqlalchemy import create_engine, text

from shared import work_items
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the work-item memory tests")


@pytest.fixture()
def db():
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
                for table in ("session_memories", "context_branches", "projects"):
                    conn.execute(
                        text(f"DELETE FROM {table} WHERE org_id = :o"), {"o": org_id}
                    )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _org(db) -> str:
    org_id = f"org_wi_{uuid.uuid4().hex[:8]}"
    db.created_org_ids.add(org_id)
    return org_id


def _project(db, *, org_id: str, plane_project_id=None) -> str:
    project_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO projects (id, org_id, git_remote, display_name,
                                  plane_project_id)
            VALUES (:id, :org_id, :remote, 'work-item-test', :plane)
            """
        ),
        {
            "id": project_id,
            "org_id": org_id,
            "remote": f"git@example.com:{uuid.uuid4().hex[:8]}.git",
            "plane": plane_project_id,
        },
    )
    return project_id


class _StubRow:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _StubWorkspace:
    """Stands in for the workspace DB connection.

    Returns whatever row it was seeded with, for any query. The point of the
    tests below is the ENGINE-side half of the decision (is this work item's
    project linked into MY org?), which is where the security lives.
    """

    def __init__(self, row=None, raises=False):
        self._row = row
        self._raises = raises
        self.queries = 0

    def execute(self, *_args, **_kwargs):
        self.queries += 1
        if self._raises:
            raise RuntimeError("workspace database is down")
        return self

    def fetchone(self):
        return self._row


def _issue_row(plane_project_id, *, name="Fix the thing", seq=42, ident="KLIO"):
    return _StubRow(
        id=uuid.uuid4(),
        name=name,
        sequence_id=seq,
        project_id=plane_project_id,
        workspace_id=uuid.uuid4(),
        project_identifier=ident,
    )


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


class TestSchema:
    def test_both_columns_are_nullable(self, db):
        rows = db.execute(
            text(
                """
                SELECT table_name, is_nullable
                FROM information_schema.columns
                WHERE column_name = 'work_item_id'
                  AND table_name IN ('session_memories', 'context_branches')
                """
            )
        ).fetchall()
        found = {r.table_name: r.is_nullable for r in rows}
        assert found == {"session_memories": "YES", "context_branches": "YES"}

    def test_partial_indexes_exist(self, db):
        names = {
            r.indexname
            for r in db.execute(
                text(
                    """
                    SELECT indexname FROM pg_indexes
                    WHERE indexname IN ('ix_session_memories_work_item',
                                        'ix_context_branches_work_item')
                    """
                )
            ).fetchall()
        }
        assert names == {
            "ix_session_memories_work_item",
            "ix_context_branches_work_item",
        }

    def test_a_branch_may_carry_a_work_item(self, db):
        org_id = _org(db)
        project_id = _project(db, org_id=org_id)
        item_id = str(uuid.uuid4())
        db.execute(
            text(
                """
                INSERT INTO context_branches
                    (org_id, project_id, name, work_item_id)
                VALUES (:o, :p, 'feat/x', :w)
                """
            ),
            {"o": org_id, "p": project_id, "w": item_id},
        )
        got = db.execute(
            text(
                "SELECT work_item_id FROM context_branches "
                "WHERE org_id = :o AND name = 'feat/x'"
            ),
            {"o": org_id},
        ).fetchone()
        assert str(got.work_item_id) == item_id

    def test_one_open_branch_name_per_project_still_holds(self, db):
        """051 must not have loosened 050's uniqueness by adding a column.

        Two work items in one project cannot both open 'feat/x'. That is git's
        rule (one repo, one branch name) and merge depends on it: a merge
        promotes to PROJECT main, so two same-named open branches would race
        for the same destination.
        """
        from sqlalchemy.exc import IntegrityError

        org_id = _org(db)
        project_id = _project(db, org_id=org_id)
        for work_item in (str(uuid.uuid4()), str(uuid.uuid4())):
            try:
                db.execute(
                    text(
                        """
                        INSERT INTO context_branches
                            (org_id, project_id, name, work_item_id)
                        VALUES (:o, :p, 'feat/dupe', :w)
                        """
                    ),
                    {"o": org_id, "p": project_id, "w": work_item},
                )
            except IntegrityError:
                db.rollback()
                return
        pytest.fail("a second open branch of the same name was accepted")


# --------------------------------------------------------------------------
# Resolution — the refusals are the contract
# --------------------------------------------------------------------------


class TestResolution:
    def test_resolves_a_linked_work_item(self, db):
        org_id = _org(db)
        plane_project_id = str(uuid.uuid4())
        engine_project_id = _project(
            db, org_id=org_id, plane_project_id=plane_project_id
        )
        ws = _StubWorkspace(_issue_row(plane_project_id))

        got = work_items.resolve_work_item(
            db, ws, org_id=org_id, identifier=str(uuid.uuid4())
        )
        assert got is not None
        assert got["project_id"] == engine_project_id
        assert got["plane_project_id"] == plane_project_id
        assert got["reference"] == "KLIO-42"

    def test_a_human_reference_resolves(self, db):
        org_id = _org(db)
        plane_project_id = str(uuid.uuid4())
        _project(db, org_id=org_id, plane_project_id=plane_project_id)
        ws = _StubWorkspace(_issue_row(plane_project_id, seq=7, ident="LAUNCH"))

        got = work_items.resolve_work_item(db, ws, org_id=org_id, identifier="LAUNCH-7")
        assert got is not None and got["reference"] == "LAUNCH-7"

    def test_an_unlinked_project_is_refused(self, db):
        """The work item is real; nothing in this org can hold its memories."""
        org_id = _org(db)
        _project(db, org_id=org_id, plane_project_id=None)
        ws = _StubWorkspace(_issue_row(str(uuid.uuid4())))

        assert (
            work_items.resolve_work_item(
                db, ws, org_id=org_id, identifier=str(uuid.uuid4())
            )
            is None
        )

    def test_another_orgs_work_item_is_refused(self, db):
        """The linked project exists — under someone else's org."""
        owner_org = _org(db)
        caller_org = _org(db)
        plane_project_id = str(uuid.uuid4())
        _project(db, org_id=owner_org, plane_project_id=plane_project_id)
        ws = _StubWorkspace(_issue_row(plane_project_id))

        assert (
            work_items.resolve_work_item(
                db, ws, org_id=caller_org, identifier=str(uuid.uuid4())
            )
            is None
        )

    def test_an_unknown_work_item_is_refused(self, db):
        org_id = _org(db)
        assert (
            work_items.resolve_work_item(
                db, _StubWorkspace(None), org_id=org_id, identifier=str(uuid.uuid4())
            )
            is None
        )

    def test_no_workspace_connection_is_refused_not_skipped(self, db):
        """``None`` is never 'proceed unchecked'."""
        org_id = _org(db)
        assert (
            work_items.resolve_work_item(
                db, None, org_id=org_id, identifier=str(uuid.uuid4())
            )
            is None
        )

    def test_a_workspace_outage_refuses_rather_than_raising(self, db):
        """A write must never fail because filing could not be established."""
        org_id = _org(db)
        ws = _StubWorkspace(raises=True)
        assert (
            work_items.resolve_work_item(
                db, ws, org_id=org_id, identifier=str(uuid.uuid4())
            )
            is None
        )

    def test_garbage_never_reaches_the_database(self, db):
        """Not a uuid and not PROJ-123: refused before any query runs."""
        org_id = _org(db)
        ws = _StubWorkspace(None)
        assert (
            work_items.resolve_work_item(
                db, ws, org_id=org_id, identifier="not a work item"
            )
            is None
        )
        assert ws.queries == 0
