"""The observation sweep must not delete fuel the curator has not read.

REAL POSTGRES. The guard is a correlated EXISTS; a mocked connection proves a
string was passed to execute and nothing about which rows survive.

WHY THIS EXISTS. `RAW_OBSERVATION_TTL_DAYS` sweeps recall_hidden observations
on age alone. Those rows are curator fuel — until the curator has distilled a
row into a fact/decision/plan, that row is the only record of what happened.
The curator runs periodically over a per-(org, user) cursor and CAN fall behind
the TTL. Measured against production on 2026-08-18, it was six days behind for
the fleet's largest org, and an age-only sweep would have deleted 10,580
observations nobody had ever distilled — 9,848 from that one org.

Mutations this file is built to catch:

* The EXISTS guard is dropped, restoring the age-only sweep
  -> ``test_an_unread_observation_survives_the_sweep``
* The guard is inverted, so only unread rows are deleted
  -> ``test_a_curated_observation_is_swept``
* The per-user cursor is collapsed to per-org, so one user's progress licenses
  deleting another user's fuel
  -> ``test_one_users_cursor_does_not_license_deleting_anothers``
* A row whose owner has no cursor at all is swept
  -> ``test_an_observation_with_no_cursor_at_all_survives``
* The sweep starts deleting rows that are still visible to recall
  -> ``test_a_visible_memory_is_never_swept``
"""

from __future__ import annotations

import sqlalchemy as sa

from shared.testing import requires_db, database_url

pytestmark = requires_db("the retention curator-guard tests")

ORG = "org-guard-test"
ALICE = "11111111-1111-1111-1111-111111111111"
BOB = "22222222-2222-2222-2222-222222222222"

#: A dedicated Postgres schema for this file's tables.
#:
#: CI runs `alembic upgrade head` and then the suites against that migrated
#: database. These fixtures need to CREATE tables, and creating them in
#: `public` means dropping the real ones first — which would destroy the
#: migrated schema for every test that runs afterwards. Namespacing keeps the
#: tables real (the predicates under test are the point) while leaving `public`
#: untouched.
TEST_SCHEMA = "test_retention_guard"

SCHEMA = f"""
DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE;
CREATE SCHEMA {TEST_SCHEMA};
SET search_path TO {TEST_SCHEMA};

CREATE TABLE session_memories (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id varchar(64) NOT NULL,
  user_id varchar,
  memory_type varchar NOT NULL DEFAULT 'observation',
  content text NOT NULL,
  recall_hidden boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE brain_curator_state (
  org_id varchar(64) NOT NULL,
  user_id varchar,
  last_processed_at timestamptz
);
"""

# The exact predicate shipped in scripts/run_retention.py. Restated rather than
# imported: the script is a __main__ with env-var side effects at import time,
# and a test that ran the real module would need a whole fleet fixture. What
# must not drift is the PREDICATE, so that is what is pinned here — if the
# script's guard changes, this file must change with it in the same commit.
SWEEP = """
DELETE FROM session_memories m
WHERE m.recall_hidden = TRUE
  AND m.created_at < NOW() - (:days || ' days')::interval
  AND EXISTS (
        SELECT 1 FROM brain_curator_state c
        WHERE c.org_id = m.org_id
          AND c.user_id IS NOT DISTINCT FROM m.user_id
          AND c.last_processed_at IS NOT NULL
          AND c.last_processed_at >= m.created_at
  )
"""


def _engine():
    """An engine whose connections resolve unqualified names to TEST_SCHEMA."""
    return sa.create_engine(
        database_url(),
        connect_args={"options": f"-csearch_path={TEST_SCHEMA}"},
    )


def _seed(conn, *, user, days_old, content, hidden=True):
    conn.execute(
        sa.text(
            "INSERT INTO session_memories (org_id, user_id, content, "
            "recall_hidden, created_at) VALUES (:o,:u,:c,:h, "
            "now() - (:d || ' days')::interval)"
        ),
        {"o": ORG, "u": user, "c": content, "h": hidden, "d": days_old},
    )


def _cursor(conn, *, user, days_ago):
    conn.execute(
        sa.text(
            "INSERT INTO brain_curator_state (org_id, user_id, last_processed_at) "
            "VALUES (:o,:u, now() - (:d || ' days')::interval)"
        ),
        {"o": ORG, "u": user, "d": days_ago},
    )


def _sweep(conn, days=7):
    return conn.execute(sa.text(SWEEP), {"days": days}).rowcount


def _remaining(conn):
    return [r[0] for r in conn.execute(sa.text("SELECT content FROM session_memories"))]


def test_an_unread_observation_survives_the_sweep():
    # THE PRODUCTION BUG. 30 days old, well past the TTL — but the curator only
    # reached 40 days ago, so this row has never been distilled.
    engine = _engine()
    with engine.begin() as conn:
        conn.execute(sa.text(SCHEMA))
        _seed(conn, user=ALICE, days_old=30, content="UNREAD_FUEL")
        _cursor(conn, user=ALICE, days_ago=40)

        assert _sweep(conn) == 0
        assert "UNREAD_FUEL" in _remaining(conn)


def test_a_curated_observation_is_swept():
    # Positive control. Without it, a guard that deletes nothing at all would
    # pass every other test in this file while the sweep did no work.
    engine = _engine()
    with engine.begin() as conn:
        conn.execute(sa.text(SCHEMA))
        _seed(conn, user=ALICE, days_old=30, content="CURATED")
        _cursor(conn, user=ALICE, days_ago=10)

        assert _sweep(conn) == 1
        assert _remaining(conn) == []


def test_a_row_inside_the_ttl_survives_even_if_curated():
    engine = _engine()
    with engine.begin() as conn:
        conn.execute(sa.text(SCHEMA))
        _seed(conn, user=ALICE, days_old=2, content="FRESH")
        _cursor(conn, user=ALICE, days_ago=0)

        assert _sweep(conn) == 0
        assert "FRESH" in _remaining(conn)


def test_one_users_cursor_does_not_license_deleting_anothers():
    # Alice's curator is caught up; Bob's is far behind. Bob's fuel must
    # survive. A guard written per-org rather than per-user deletes it.
    engine = _engine()
    with engine.begin() as conn:
        conn.execute(sa.text(SCHEMA))
        _seed(conn, user=ALICE, days_old=30, content="ALICE_CURATED")
        _seed(conn, user=BOB, days_old=30, content="BOB_UNREAD")
        _cursor(conn, user=ALICE, days_ago=1)
        _cursor(conn, user=BOB, days_ago=60)

        assert _sweep(conn) == 1
        assert _remaining(conn) == ["BOB_UNREAD"]


def test_an_observation_with_no_cursor_at_all_survives():
    # The curator has never run for this owner, so every row is still fuel.
    engine = _engine()
    with engine.begin() as conn:
        conn.execute(sa.text(SCHEMA))
        _seed(conn, user=BOB, days_old=90, content="NEVER_CURATED")

        assert _sweep(conn) == 0
        assert "NEVER_CURATED" in _remaining(conn)


def test_a_null_owner_row_matches_a_null_owner_cursor():
    # Pre-silo rows carry user_id IS NULL and are curated by the NULL-user
    # cursor. `IS NOT DISTINCT FROM` is what makes NULL match NULL; plain `=`
    # would never match and those rows would accumulate forever.
    engine = _engine()
    with engine.begin() as conn:
        conn.execute(sa.text(SCHEMA))
        _seed(conn, user=None, days_old=30, content="PRE_SILO")
        _cursor(conn, user=None, days_ago=1)

        assert _sweep(conn) == 1


def test_a_visible_memory_is_never_swept():
    # recall_hidden = FALSE is a distilled memory, not fuel. Age and cursor are
    # both satisfied here; only the hidden flag protects it.
    engine = _engine()
    with engine.begin() as conn:
        conn.execute(sa.text(SCHEMA))
        _seed(conn, user=ALICE, days_old=300, content="A_REAL_FACT", hidden=False)
        _cursor(conn, user=ALICE, days_ago=0)

        assert _sweep(conn) == 0
        assert "A_REAL_FACT" in _remaining(conn)
