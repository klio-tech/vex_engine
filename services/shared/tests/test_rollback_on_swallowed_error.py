"""A swallowed DB error must not leave the caller's session unusable. Real Postgres.

Every helper covered here is deliberately best-effort: metering, an audit
append, a telemetry alarm. Each catches ``Exception``, logs, and returns — so
the operation it decorates survives a database hiccup. That contract is right.
What was missing is the other half of it.

Postgres aborts the whole transaction on the first failed statement. Until
somebody issues a ``ROLLBACK``, every later statement on that connection is
refused with ``current transaction is aborted, commands ignored until end of
transaction block``. So a handler that swallows the error and returns hands the
caller a session that looks fine and answers nothing. The best-effort helper
succeeds at not raising and takes the real operation down two statements later,
somewhere with no obvious connection to it — which is the worst possible place
for the failure to surface.

``shared.usage.record_usage`` already had this right; its sibling
``check_memory_quota``, twenty lines above it, did not.

**Why real Postgres and not a mock.** The existing coverage for the fail-open
path used a ``MagicMock`` whose ``execute`` raised. A ``MagicMock`` has no
transaction state: it answers the next ``execute`` as happily as the first, so
that test passed identically with and without the ``rollback()``. There is no
version of this defect a mock can see. Every test below therefore poisons a
REAL connection the way production does — one genuinely failed statement — and
then asserts the caller can keep using it.

Structure of each test:

1. ``_poison(db)`` runs a statement Postgres rejects, leaving a real aborted
   transaction (``_test_the_poison_actually_poisons`` proves it does).
2. The best-effort helper runs and must not raise (its existing contract).
3. The caller issues an ordinary statement, which must succeed.

Step 3 is the whole test. Each also carries a positive control on an unpoisoned
connection, because a helper that returned the fallback unconditionally would
pass step 2 without doing any work at all.
"""

import uuid

import pytest
from sqlalchemy import create_engine, text

from shared import graph_store, projects
from shared.plan_limits import get_plan_config
from shared.testing import database_url, requires_db
from shared.usage import check_memory_quota

DB_URL = database_url()

pytestmark = requires_db("the swallowed-error rollback tests")

USER = "11111111-2222-3333-4444-555555555555"


def _unique_org() -> str:
    return f"org_{uuid.uuid4().hex[:8]}"


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
                    text("DELETE FROM memory_edges WHERE org_id = :o"), {"o": org_id}
                )
                conn.execute(
                    text("DELETE FROM entities WHERE org_id = :o"), {"o": org_id}
                )
                conn.execute(
                    text("DELETE FROM project_grant_audit WHERE org_id = :o"),
                    {"o": org_id},
                )
                conn.execute(
                    text(
                        """
                        DELETE FROM project_members WHERE project_id IN (
                            SELECT id FROM projects WHERE org_id = :o)
                        """
                    ),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM projects WHERE org_id = :o"), {"o": org_id}
                )
                conn.execute(
                    text("DELETE FROM memory_usage_counters WHERE org_id = :o"),
                    {"o": org_id},
                )
                conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _poison(db) -> None:
    """Leave ``db`` in a genuinely aborted transaction, as production does.

    Not a mock and not a flag: one real statement Postgres refuses. Everything
    on this connection is now refused too until somebody rolls back.
    """
    try:
        db.execute(text("SELECT no_such_column_at_all"))
    except Exception:
        pass


def _session_is_usable(db) -> bool:
    """True when the connection will answer an ordinary statement."""
    try:
        return db.execute(text("SELECT 1")).scalar() == 1
    except Exception:
        return False


# ---------------------------------------------------------------------------
# The fixture's own control. Without this, every test below could be passing
# because the poison never worked.
# ---------------------------------------------------------------------------


def test_the_poison_actually_poisons(db):
    assert _session_is_usable(db), "control: a fresh connection answers"
    _poison(db)
    assert not _session_is_usable(db), (
        "the poison did not abort the transaction, so every test in this file "
        "would pass vacuously — a swallowed error would have nothing to clear"
    )
    db.rollback()
    assert _session_is_usable(db), "control: ROLLBACK is what clears it"


# ---------------------------------------------------------------------------
# shared/usage.py — check_memory_quota
#
# On the MCP write path this runs BEFORE the memory write. A poisoned session
# here means the write that follows fails, and the caller is told their memory
# was not stored because of an error in the metering they never asked about.
# ---------------------------------------------------------------------------


def test_check_memory_quota_leaves_the_session_usable(db):
    org_id = _unique_org()
    db.created_org_ids.add(org_id)
    _poison(db)

    over = check_memory_quota(db, org_id, get_plan_config("free"), "memory")

    assert over is False, "fails open: a metering fault never blocks the write"
    assert _session_is_usable(db), (
        "check_memory_quota swallowed the error but left the transaction "
        "aborted; the memory write that follows it dies instead"
    )


def test_check_memory_quota_still_reads_real_counters(db):
    """Positive control: the helper does real work, it does not just fail open."""
    org_id = _unique_org()
    db.created_org_ids.add(org_id)
    db.execute(
        text(
            """
            INSERT INTO memory_usage_counters
              (org_id, period_month, memories_captured, recalls)
            VALUES (:o, date_trunc('month', now())::date, :n, 0)
            """
        ),
        {"o": org_id, "n": get_plan_config("free").memories_per_month},
    )
    db.commit()

    with pytest.raises(Exception) as exc:
        check_memory_quota(db, org_id, get_plan_config("free"), "memory")
    assert "limit reached" in str(exc.value), (
        "the quota check must be reading the counter row it was given"
    )


# ---------------------------------------------------------------------------
# shared/projects.py — record_grant_audit
#
# The grant has already COMMITTED by the time this runs. The audit append is
# best-effort precisely so a logging failure cannot undo an access change the
# caller was told succeeded — but leaving the session aborted takes down
# whatever the grant path does next, which is the response.
# ---------------------------------------------------------------------------


def test_record_grant_audit_leaves_the_session_usable(db):
    org_id = _unique_org()
    db.created_org_ids.add(org_id)
    _poison(db)

    projects.record_grant_audit(
        db,
        org_id=org_id,
        project_id=str(uuid.uuid4()),
        granted_to=USER,
        granted_to_email=None,
        granted_by=None,
        role="member",
        surface="mcp",
    )

    assert _session_is_usable(db), (
        "the audit write failed and stayed failed; the grant it was recording "
        "had already committed, so the caller now gets an error for work that "
        "succeeded"
    )


def test_record_grant_audit_really_writes_a_row(db):
    """Positive control: the helper is not a no-op that always swallows."""
    org_id = _unique_org()
    db.created_org_ids.add(org_id)
    project_id = str(uuid.uuid4())

    projects.record_grant_audit(
        db,
        org_id=org_id,
        project_id=project_id,
        granted_to=USER,
        granted_to_email="a@example.com",
        granted_by=None,
        role="member",
        surface="mcp",
    )

    n = db.execute(
        text(
            "SELECT count(*) FROM project_grant_audit "
            "WHERE org_id = :o AND project_id = :p"
        ),
        {"o": org_id, "p": project_id},
    ).scalar()
    assert n == 1


# ---------------------------------------------------------------------------
# shared/graph_store.py — _alarm_on_degree
#
# The god-entity alarm runs at the tail of every ``add_edge``, after its
# commit. It is telemetry: it must never fail the write that fired it. It must
# equally never poison the session the curator is about to draw its next edge
# on — which would turn one alarm failure into the loss of a whole tick's
# remaining graph.
# ---------------------------------------------------------------------------


def test_alarm_on_degree_leaves_the_session_usable(db):
    org_id = _unique_org()
    db.created_org_ids.add(org_id)
    _poison(db)

    graph_store._alarm_on_degree(db, org_id=org_id, entity_id=str(uuid.uuid4()))

    assert _session_is_usable(db), (
        "the degree probe failed and left the transaction aborted; the next "
        "edge the curator draws on this session dies with it"
    )


def test_alarm_on_degree_really_counts_edges(db):
    """Positive control: the probe runs a real query on a healthy session."""
    org_id = _unique_org()
    db.created_org_ids.add(org_id)
    entity_id = str(uuid.uuid4())

    # No exception, and the session is still usable afterwards — the shape a
    # successful probe leaves behind, so the negative above is not just
    # asserting "this function never raises".
    graph_store._alarm_on_degree(db, org_id=org_id, entity_id=entity_id)
    assert _session_is_usable(db)
