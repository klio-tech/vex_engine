"""Migration 038: a non-positive retention keeps everything, against real SQL.

The behaviour under test is a plpgsql expression, so a Python-level assertion
about it would prove nothing. These run against Postgres and check what the
function actually deletes.

The failure being prevented is not subtle: with the pre-038 body,
``NOW() - ('-1 days')::interval`` evaluates to ``NOW() + 1 day``, every
``created_at`` is below it, and the DELETE takes the org's whole history. -1 is
also the value the codebase uses everywhere else to mean "unlimited", so the
obvious way to express "keep forever" was the most destructive one available.
"""

import os
import uuid

import psycopg2
import pytest

DSN = os.environ.get("DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not DSN, reason="DATABASE_URL is required for the retention SQL tests"
)


def _seed(cur, org_id: str, *, age_days: int) -> str:
    """One memory, `age_days` old. Returns its id."""
    memory_id = str(uuid.uuid4())
    cur.execute(
        """
        INSERT INTO session_memories
            (id, session_id, org_id, agent_id, memory_type, content,
             confidence, sequence_number, scope, status, created_at, updated_at)
        VALUES
            (%s, %s, %s, 'a', 'fact', 'retention probe',
             1.0, 1, 'org', 'active',
             NOW() - (%s || ' days')::interval,
             NOW() - (%s || ' days')::interval)
        """,
        (memory_id, f"s-{memory_id}", org_id, age_days, age_days),
    )
    return memory_id


def _count(cur, org_id: str) -> int:
    cur.execute("SELECT count(*) FROM session_memories WHERE org_id = %s", (org_id,))
    return cur.fetchone()[0]


@pytest.mark.parametrize("days", [-1, 0])
def test_a_non_positive_retention_deletes_nothing(days):
    """The whole point: "forever" must mean forever, for old rows too.

    The seeded row is 400 days old, so any positive retention would remove it.
    That is what makes this a test of the guard rather than of the age filter.
    """
    org_id = f"ret-guard-{uuid.uuid4()}"
    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            _seed(cur, org_id, age_days=400)
            assert _count(cur, org_id) == 1

            cur.execute("SELECT enforce_memory_retention(%s, %s)", (org_id, days))

            assert _count(cur, org_id) == 1, (
                f"retention={days} deleted a 400-day-old row; a non-positive "
                "value must mean keep forever, not delete everything"
            )
            cur.execute("DELETE FROM session_memories WHERE org_id = %s", (org_id,))


def test_a_positive_retention_still_deletes_what_it_should():
    """Positive control. Without this, the test above passes if the function is
    broken into doing nothing at all."""
    org_id = f"ret-pos-{uuid.uuid4()}"
    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            old = _seed(cur, org_id, age_days=400)
            fresh = _seed(cur, org_id, age_days=0)

            cur.execute("SELECT enforce_memory_retention(%s, %s)", (org_id, 30))

            cur.execute(
                "SELECT id::text FROM session_memories WHERE org_id = %s", (org_id,)
            )
            remaining = {r[0] for r in cur.fetchall()}
            assert old not in remaining, "a 400-day-old row survived 30-day retention"
            assert fresh in remaining, "a fresh row was deleted by 30-day retention"
            cur.execute("DELETE FROM session_memories WHERE org_id = %s", (org_id,))


def test_null_retention_deletes_nothing():
    """NULL is not a day count. Left unguarded the comparison is NULL, which is
    not TRUE, so nothing matches — but relying on that is relying on an accident
    rather than on an intention that is written down."""
    org_id = f"ret-null-{uuid.uuid4()}"
    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            _seed(cur, org_id, age_days=400)

            cur.execute("SELECT enforce_memory_retention(%s, NULL)", (org_id,))

            assert _count(cur, org_id) == 1
            cur.execute("DELETE FROM session_memories WHERE org_id = %s", (org_id,))


@pytest.mark.parametrize("days", [-1, 0, None])
def test_execution_retention_is_guarded_too(days):
    """``enforce_plan_retention`` is Vex's sweep and carries the same expression.

    Guarding only the memory function would have left the identical hazard on
    the product this change was told not to disturb.
    """
    org_id = f"ret-exec-{uuid.uuid4()}"
    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            # No seeding: the assertion is that the call is a no-op that does
            # not raise. Seeding executions would couple this to Vex's schema
            # for no extra signal.
            cur.execute("SELECT enforce_plan_retention(%s, %s)", (org_id, days))
