"""Persisted dreamer trigger state (``app.dream_state``, migration 046).

REAL POSTGRES, NOT MOCKS. Every function under test is a single SQL statement
whose correctness lives entirely in its WHERE clause — a mocked session would
assert that a string was passed to ``execute`` and prove nothing about whether
the predicate selects the right rows. That distinction is not academic here:
the bug this module exists to fix (the dreamer never firing, zero proposals for
the life of the feature) was a trigger condition that was never true.

Mutations this file is built to catch, and the test that catches each:

* The count trigger fires early or late
  -> ``test_claim_fires_at_exactly_the_threshold``
* The interval trigger is dropped, so a low-traffic org never dreams
  -> ``test_claim_fires_on_the_interval_with_no_new_sessions``
* Two replicas both claim the same pass, so one org is dreamed for twice and
  ``dream_passes`` double-counts
  -> ``test_a_claim_loses_to_a_replica_that_won_first``
* The claim forgets to reset, so every subsequent graded session re-fires
  -> ``test_claim_resets_the_counter_and_the_clock``
* The claim reports post-reset zeros instead of the values that justified it
  -> ``test_claim_reports_the_pre_reset_snapshot``
* An org with traces but no state row is invisible to the interval timer —
  the lazy path production actually takes first, since no backfill is run
  -> ``test_org_with_traces_and_no_state_row_is_due``
* ``ensure_state_row`` seeds the clock to now(), making a long-idle org wait a
  further full interval instead of being immediately due
  -> ``test_ensure_state_row_seeds_the_clock_from_the_oldest_trace``
"""

from __future__ import annotations

import threading
import time

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from shared.testing import requires_db, database_url

from app import dream_state

pytestmark = requires_db("the dream_state trigger tests")

ORG = "org-dream-test"

#: A dedicated Postgres schema for this file's tables.
#:
#: CI runs `alembic upgrade head` and then the suites against that migrated
#: database. These fixtures need to CREATE tables, and creating them in
#: `public` means dropping the real ones first — which would destroy the
#: migrated schema for every test that runs afterwards. Namespacing keeps the
#: tables real (the predicates under test are the point) while leaving `public`
#: untouched.
TEST_SCHEMA = "test_dream_state"

#: The columns are restated here rather than imported from migration 046: a
#: test that builds its fixture by running the code under test cannot detect
#: the migration disagreeing with the queries.
SCHEMA = f"""
DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE;
CREATE SCHEMA {TEST_SCHEMA};
SET search_path TO {TEST_SCHEMA};

CREATE TABLE dream_state (
  org_id varchar(64) PRIMARY KEY,
  sessions_since integer NOT NULL DEFAULT 0,
  last_run_at timestamptz NOT NULL DEFAULT now(),
  last_dreamed_at timestamptz,
  last_proposals_written integer,
  dream_passes integer NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE session_traces (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id varchar(64) NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
"""


def _engine():
    """An engine whose connections resolve unqualified names to TEST_SCHEMA."""
    return sa.create_engine(
        database_url(),
        connect_args={"options": f"-csearch_path={TEST_SCHEMA}"},
    )


def _reset(engine, *, org=ORG, sessions_since=0, last_run_hours_ago=0.0):
    with engine.begin() as conn:
        conn.execute(sa.text(SCHEMA))
        conn.execute(
            sa.text(
                """
                INSERT INTO dream_state (org_id, sessions_since, last_run_at)
                VALUES (:org, :n, now() - make_interval(secs => :secs))
                """
            ),
            {"org": org, "n": sessions_since, "secs": last_run_hours_ago * 3600.0},
        )


def _session(engine):
    return sessionmaker(bind=engine)()


def _state(engine, org=ORG):
    with engine.begin() as conn:
        return (
            conn.execute(
                sa.text(
                    "SELECT sessions_since, dream_passes, last_run_at "
                    "FROM dream_state WHERE org_id = :o"
                ),
                {"o": org},
            )
            .mappings()
            .first()
        )


def test_record_graded_session_increments_and_creates_the_row():
    engine = _engine()
    with engine.begin() as conn:
        conn.execute(sa.text(SCHEMA))

    db = _session(engine)
    for _ in range(3):
        dream_state.record_graded_session(db, org_id=ORG)

    assert _state(engine)["sessions_since"] == 3


def test_claim_fires_at_exactly_the_threshold():
    engine = _engine()

    # One short: no claim.
    _reset(engine, sessions_since=9)
    assert (
        dream_state.claim_dream_pass(
            _session(engine), org_id=ORG, min_graded_sessions=10, interval_hours=24
        )
        is None
    )

    # Exactly at it: claim.
    _reset(engine, sessions_since=10)
    claim = dream_state.claim_dream_pass(
        _session(engine), org_id=ORG, min_graded_sessions=10, interval_hours=24
    )
    assert claim is not None
    assert claim.reason == "graded_sessions"


def test_claim_fires_on_the_interval_with_no_new_sessions():
    # The trigger that matters for a low-traffic org: zero graded sessions,
    # clock elapsed. This is the one production will hit first.
    engine = _engine()
    _reset(engine, sessions_since=0, last_run_hours_ago=25)

    claim = dream_state.claim_dream_pass(
        _session(engine), org_id=ORG, min_graded_sessions=10, interval_hours=24
    )

    assert claim is not None
    assert claim.reason == "interval"
    assert claim.elapsed_hours >= 24


def test_claim_does_not_fire_before_either_trigger():
    engine = _engine()
    _reset(engine, sessions_since=2, last_run_hours_ago=1)

    assert (
        dream_state.claim_dream_pass(
            _session(engine), org_id=ORG, min_graded_sessions=10, interval_hours=24
        )
        is None
    )


def test_claim_resets_the_counter_and_the_clock():
    engine = _engine()
    _reset(engine, sessions_since=12, last_run_hours_ago=30)

    dream_state.claim_dream_pass(
        _session(engine), org_id=ORG, min_graded_sessions=10, interval_hours=24
    )

    after = _state(engine)
    assert after["sessions_since"] == 0
    assert after["dream_passes"] == 1

    # And the reset must actually close the trigger: an immediate second
    # attempt finds nothing, which is what stops every later graded session
    # from re-firing a pass.
    assert (
        dream_state.claim_dream_pass(
            _session(engine), org_id=ORG, min_graded_sessions=10, interval_hours=24
        )
        is None
    )


def test_claim_reports_the_pre_reset_snapshot():
    # The claim resets as it wins, so the numbers that justified it exist only
    # in the RETURNING clause. Reporting post-reset zeros would make every log
    # line read "dreamed because 0 sessions".
    engine = _engine()
    _reset(engine, sessions_since=17, last_run_hours_ago=2)

    claim = dream_state.claim_dream_pass(
        _session(engine), org_id=ORG, min_graded_sessions=10, interval_hours=24
    )

    assert claim is not None
    assert claim.sessions_since == 17


def test_a_claim_loses_to_a_replica_that_won_first():
    """THE RACE, deterministically interleaved.

    Thread timing alone does not reproduce this — two threads calling the
    claim back to back simply serialise, and a READ-THEN-WRITE implementation
    passes that test just as happily as the atomic one. (Verified: mutating
    ``claim_dream_pass`` into a SELECT-then-UPDATE left a thread-race test
    green.) A concurrency test that cannot fail is worse than no test, so the
    interleave here is forced rather than hoped for.

    The sequence, which is exactly what two async-worker replicas do:

      1. Replica A takes the row lock and resets the row, as a winning claim
         does, but has not committed yet.
      2. Replica B calls ``claim_dream_pass``. Its UPDATE blocks on A's lock.
      3. A commits.
      4. B unblocks. Postgres re-checks B's WHERE against A's committed row,
         which now has ``sessions_since = 0`` and a fresh ``last_run_at``, so
         B must match nothing and return None.

    Under a read-then-write implementation, B's SELECT does not block on a row
    lock — it reads the stale pre-reset snapshot, decides it is due, and then
    performs its UPDATE once A commits. ``dream_passes`` reaches 2 and this
    test fails, which is the whole point of it.
    """
    engine = _engine()
    _reset(engine, sessions_since=50, last_run_hours_ago=100)

    result: dict[str, object] = {}
    b_started = threading.Event()

    def replica_b():
        db = _session(engine)
        try:
            b_started.set()
            result["claim"] = dream_state.claim_dream_pass(
                db, org_id=ORG, min_graded_sessions=10, interval_hours=24
            )
        finally:
            db.close()

    conn_a = engine.connect()
    trans_a = conn_a.begin()
    conn_a.execute(
        sa.text(
            """
            UPDATE dream_state
            SET sessions_since = 0, last_run_at = now(),
                dream_passes = dream_passes + 1, updated_at = now()
            WHERE org_id = :o
            """
        ),
        {"o": ORG},
    )

    thread = threading.Thread(target=replica_b)
    thread.start()
    b_started.wait(timeout=5)
    # Give B time to reach (and block on) its UPDATE while A holds the lock.
    time.sleep(1.0)
    trans_a.commit()
    conn_a.close()

    thread.join(timeout=30)

    assert not thread.is_alive(), "the losing claim must not deadlock"
    assert result["claim"] is None, "a replica that lost the race must not claim"
    assert _state(engine)["dream_passes"] == 1, (
        "exactly one pass may be counted; 2 means both replicas dreamed"
    )


def test_org_with_traces_and_no_state_row_is_due():
    # No backfill is run, so on the first tick after deploy EVERY org is in
    # this state. If the timer cannot see a row-less org, the dreamer stays
    # exactly as dead as it was.
    engine = _engine()
    with engine.begin() as conn:
        conn.execute(sa.text(SCHEMA))
        conn.execute(
            sa.text(
                "INSERT INTO session_traces (org_id, created_at) "
                "VALUES (:o, now() - interval '10 days')"
            ),
            {"o": ORG},
        )

    due = dream_state.orgs_due_for_dream(_session(engine), interval_hours=24, limit=5)

    assert ORG in due


def test_a_recently_dreamed_org_is_not_due():
    engine = _engine()
    _reset(engine, last_run_hours_ago=1)
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO session_traces (org_id, created_at) "
                "VALUES (:o, now() - interval '10 days')"
            ),
            {"o": ORG},
        )

    due = dream_state.orgs_due_for_dream(_session(engine), interval_hours=24, limit=5)

    assert ORG not in due


def test_orgs_due_respects_the_per_tick_limit():
    engine = _engine()
    with engine.begin() as conn:
        conn.execute(sa.text(SCHEMA))
        for i in range(7):
            conn.execute(
                sa.text(
                    "INSERT INTO session_traces (org_id, created_at) "
                    "VALUES (:o, now() - interval '10 days')"
                ),
                {"o": f"{ORG}-{i}"},
            )

    due = dream_state.orgs_due_for_dream(_session(engine), interval_hours=24, limit=5)

    assert len(due) == 5


def test_ensure_state_row_seeds_the_clock_from_the_oldest_trace():
    # Seeding to now() would make a long-idle org wait a further full interval
    # before its first pass — the difference between "dreams on the next tick"
    # and "dreams tomorrow" for every existing org, since nothing is backfilled.
    engine = _engine()
    with engine.begin() as conn:
        conn.execute(sa.text(SCHEMA))
        conn.execute(
            sa.text(
                "INSERT INTO session_traces (org_id, created_at) "
                "VALUES (:o, now() - interval '30 days')"
            ),
            {"o": ORG},
        )

    dream_state.ensure_state_row(_session(engine), org_id=ORG)

    claim = dream_state.claim_dream_pass(
        _session(engine), org_id=ORG, min_graded_sessions=10, interval_hours=24
    )

    assert claim is not None, "a row seeded from a 30-day-old trace is due now"
    assert claim.reason == "interval"


def test_ensure_state_row_never_clobbers_an_existing_row():
    engine = _engine()
    _reset(engine, sessions_since=6, last_run_hours_ago=1)

    dream_state.ensure_state_row(_session(engine), org_id=ORG)

    assert _state(engine)["sessions_since"] == 6


def test_record_dream_result_records_completion_separately_from_the_claim():
    # last_run_at moves on CLAIM; last_dreamed_at moves on COMPLETION. The gap
    # between them is how an operator tells "ran and found nothing" from
    # "claimed and then crashed" — the exact ambiguity that made zero
    # proposals undiagnosable for weeks.
    engine = _engine()
    _reset(engine, sessions_since=12)

    dream_state.claim_dream_pass(
        _session(engine), org_id=ORG, min_graded_sessions=10, interval_hours=24
    )
    dream_state.record_dream_result(_session(engine), org_id=ORG, written=0)

    with engine.begin() as conn:
        row = (
            conn.execute(
                sa.text(
                    "SELECT last_dreamed_at, last_proposals_written "
                    "FROM dream_state WHERE org_id = :o"
                ),
                {"o": ORG},
            )
            .mappings()
            .first()
        )

    assert row["last_dreamed_at"] is not None
    assert row["last_proposals_written"] == 0
