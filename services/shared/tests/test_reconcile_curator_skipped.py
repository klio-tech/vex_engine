"""Measuring what the curator's abandonment bug left behind.

Before WP-D, ``curator_tick`` partitioned a batch by ``(space_id, project_id)``,
``continue``d past any partition below ``min_inputs`` — and then advanced the
cursor unconditionally to the end of the whole batch. The read query only ever
looks *ahead* of the cursor, so those rows were never reconsidered. They were
not merely undistilled: ``run_retention.py`` deletes ``recall_hidden`` RAW
observations on a 7-day TTL, and nearly every RAW row is a passive capture, so
the skip put them on a one-week fuse.

That fuse is why this script reports two populations and never one. Rows still
present can be re-queued. Rows the cursor advanced past that no longer exist
cannot be, by anyone, ever — reporting them as work to do would be a lie, and
folding them into a single number would hide the only fact that matters.

Detection is deliberately conservative. A distilled row leaves a curator output
(``metadata->>'source' = 'curator'``) in its own partition at or after its
timestamp, because the curator writes at tick time. So a RAW row behind the
cursor whose partition holds no such output is a *candidate*. Where this
detector errs it errs low — a later tick's output in the same partition masks
an earlier skipped row — which is the right direction for a number that
justifies re-running an LLM over private memories.
"""

import importlib
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text

from shared.testing import database_url, requires_db

pytestmark = requires_db("the curator-skipped reconciliation tests")


def _load():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
    return importlib.import_module("reconcile_curator_skipped")


script = _load()
common = importlib.import_module("reconcile_common")

NIL = "00000000-0000-0000-0000-000000000000"


@pytest.fixture()
def db():
    engine = create_engine(database_url(), pool_pre_ping=True)
    with engine.connect() as conn:
        yield conn
    engine.dispose()


@pytest.fixture()
def org(db):
    """A unique org per test; every assertion is scoped to it."""
    org_id = f"skip-{uuid.uuid4().hex[:12]}"
    yield org_id
    db.execute(text("DELETE FROM session_memories WHERE org_id = :o"), {"o": org_id})
    db.execute(text("DELETE FROM brain_curator_state WHERE org_id = :o"), {"o": org_id})
    db.commit()


def _raw(db, org_id, user_id, *, age_days, content="ordinary observation", **kw):
    """Insert one RAW private capture and return (id, created_at)."""
    row_id = str(uuid.uuid4())
    created = datetime.now(timezone.utc) - timedelta(days=age_days)
    db.execute(
        text(
            """
            INSERT INTO session_memories
                (id, session_id, org_id, agent_id, user_id, memory_type,
                 content, scope, status, recall_hidden, space_id, project_id,
                 created_at)
            VALUES
                (:id, 'sess', :org, 'agent-1', :user, :mtype,
                 :content, 'private', :status, :hidden, :space, NULL,
                 :created)
            """
        ),
        {
            "id": row_id,
            "org": org_id,
            "user": user_id,
            "mtype": kw.get("memory_type", "observation"),
            "content": content,
            "status": kw.get("status", "active"),
            "hidden": kw.get("recall_hidden", True),
            "space": kw.get("space_id"),
            "created": created,
        },
    )
    db.commit()
    return row_id, created


def _curator_output(db, org_id, user_id, *, age_days, space_id=None):
    """Insert one row that looks like curator distillate for a partition."""
    db.execute(
        text(
            """
            INSERT INTO session_memories
                (id, session_id, org_id, agent_id, user_id, memory_type,
                 content, scope, status, metadata, space_id, project_id,
                 provenance, created_at)
            VALUES
                (:id, 'sess', :org, 'agent-1', :user, 'summary',
                 'distilled', 'private', 'active',
                 '{"source": "curator", "input_count": 5}'::jsonb,
                 :space, NULL, 'INFERRED', :created)
            """
        ),
        {
            "id": str(uuid.uuid4()),
            "org": org_id,
            "user": user_id,
            "space": space_id,
            "created": datetime.now(timezone.utc) - timedelta(days=age_days),
        },
    )
    db.commit()


def _cursor(db, org_id, user_id, *, at, row_id, processed_count=0):
    db.execute(
        text(
            """
            INSERT INTO brain_curator_state
                (org_id, user_id, last_processed_at, last_processed_id,
                 processed_count)
            VALUES (:org, :user, :at, :id, :n)
            ON CONFLICT (org_id, user_id) DO UPDATE
              SET last_processed_at = EXCLUDED.last_processed_at,
                  last_processed_id = EXCLUDED.last_processed_id,
                  processed_count = EXCLUDED.processed_count
            """
        ),
        {"org": org_id, "user": user_id, "at": at, "id": row_id, "n": processed_count},
    )
    db.commit()


def _only(populations):
    assert len(populations) == 1, populations
    return populations[0]


# ---------------------------------------------------------------------------
# The two populations
# ---------------------------------------------------------------------------


def test_a_row_behind_the_cursor_with_no_distillate_is_recoverable(db, org):
    """The ordinary abandoned row: still present, so it can be re-queued."""
    skipped_id, _ = _raw(db, org, "user-1", age_days=1)
    _cursor(db, org, "user-1", at=datetime.now(timezone.utc), row_id=NIL)

    found = _only(script.find_skipped(db, org_id=org))
    assert found.recoverable_ids == [skipped_id]
    assert found.expiring_ids == []


def test_a_hidden_row_past_the_ttl_is_reported_as_expiring_not_recoverable(db, org):
    """run_retention.py deletes recall_hidden RAW rows older than 7 days.

    A row already past that horizon is still readable right now but is on the
    next sweep's list. Counting it as ordinary work to do would promise a
    re-queue that the retention cron may win the race against.
    """
    doomed_id, _ = _raw(db, org, "user-1", age_days=30, recall_hidden=True)
    _cursor(db, org, "user-1", at=datetime.now(timezone.utc), row_id=NIL)

    found = _only(script.find_skipped(db, org_id=org))
    assert found.expiring_ids == [doomed_id]
    assert found.recoverable_ids == []


def test_a_visible_row_past_the_ttl_is_still_recoverable(db, org):
    """The TTL sweep is predicated on ``recall_hidden = TRUE``.

    Positive control against the obvious wrong implementation — bucketing on
    age alone — which would file this deliberate write under 'expiring' and
    tell an operator their notes were about to be deleted.
    """
    old_visible, _ = _raw(
        db, org, "user-1", age_days=90, recall_hidden=False, memory_type="note"
    )
    _cursor(db, org, "user-1", at=datetime.now(timezone.utc), row_id=NIL)

    found = _only(script.find_skipped(db, org_id=org))
    assert found.recoverable_ids == [old_visible]
    assert found.expiring_ids == []


def test_rows_the_cursor_passed_that_no_longer_exist_are_counted_as_gone(db, org):
    """The unrecoverable population, which cannot be read back directly.

    ``processed_count`` records how many rows the cursor advanced OVER. Rows
    still behind it are what survived. The difference is what the TTL took —
    an upper bound on the permanently lost, and the only measurement of it
    available, because the rows themselves are gone.
    """
    survivor, _ = _raw(db, org, "user-1", age_days=1)
    _cursor(
        db,
        org,
        "user-1",
        at=datetime.now(timezone.utc),
        row_id=NIL,
        processed_count=40,
    )

    found = _only(script.find_skipped(db, org_id=org))
    assert found.surviving_behind_cursor == 1
    assert found.unrecoverable_upper_bound == 39
    assert survivor in found.recoverable_ids


def test_gone_count_never_goes_negative(db, org):
    """processed_count is bookkeeping, not a constraint; it can lag reality.

    A cursor rewound by an earlier run of this very script leaves more
    surviving rows behind the cursor than processed_count claims. Reporting
    "-6 rows unrecoverable" would be worse than reporting nothing.
    """
    for _ in range(6):
        _raw(db, org, "user-1", age_days=1)
    _cursor(
        db, org, "user-1", at=datetime.now(timezone.utc), row_id=NIL, processed_count=0
    )

    found = _only(script.find_skipped(db, org_id=org))
    assert found.unrecoverable_upper_bound == 0


# ---------------------------------------------------------------------------
# What is NOT a candidate
# ---------------------------------------------------------------------------


def test_a_row_ahead_of_the_cursor_is_not_a_candidate(db, org):
    """Undistilled but never abandoned: the next tick will still read it."""
    _raw(db, org, "user-1", age_days=1)
    _cursor(
        db,
        org,
        "user-1",
        at=datetime.now(timezone.utc) - timedelta(days=5),
        row_id=NIL,
    )

    assert script.find_skipped(db, org_id=org) == []


def test_a_distilled_partition_is_not_a_candidate(db, org):
    """Positive control for the detector: a partition that produced curator
    output at or after the row's timestamp is exactly what a healthy tick
    leaves behind, and must not be reported as residue."""
    _raw(db, org, "user-1", age_days=3)
    _curator_output(db, org, "user-1", age_days=2)
    _cursor(db, org, "user-1", at=datetime.now(timezone.utc), row_id=NIL)

    assert script.find_skipped(db, org_id=org) == []


def test_distillate_in_another_partition_does_not_clear_this_one(db, org):
    """The bug was per-partition. So is the evidence of the bug."""
    space_a, space_b = str(uuid.uuid4()), str(uuid.uuid4())
    db.execute(
        text(
            "INSERT INTO spaces (id, org_id, name, slug) "
            "VALUES (:a, :o, 'a', 'a'), (:b, :o, 'b', 'b')"
        ),
        {"a": space_a, "b": space_b, "o": org},
    )
    db.commit()

    skipped_id, _ = _raw(db, org, "user-1", age_days=3, space_id=space_a)
    _raw(db, org, "user-1", age_days=3, space_id=space_b)
    _curator_output(db, org, "user-1", age_days=2, space_id=space_b)
    _cursor(db, org, "user-1", at=datetime.now(timezone.utc), row_id=NIL)

    found = _only(script.find_skipped(db, org_id=org))
    assert found.recoverable_ids == [skipped_id]

    db.execute(text("DELETE FROM spaces WHERE org_id = :o"), {"o": org})
    db.commit()


def test_the_curators_own_output_is_never_a_candidate(db, org):
    """summary/fact are not RAW input types; treating them as skipped input
    would make the script report the curator's successes as its failures."""
    _curator_output(db, org, "user-1", age_days=3)
    _cursor(db, org, "user-1", at=datetime.now(timezone.utc), row_id=NIL)

    assert script.find_skipped(db, org_id=org) == []


def test_session_scope_rows_are_never_candidates(db, org):
    """The curator only ever reads scope='private'. Verification's session
    rows live in the same table and are none of its business."""
    db.execute(
        text(
            """
            INSERT INTO session_memories
                (id, session_id, org_id, agent_id, memory_type, content,
                 scope, status, created_at)
            VALUES (:id, 'sess', :org, 'agent-1', 'observation', 'x',
                    'session', 'active', now() - interval '1 day')
            """
        ),
        {"id": str(uuid.uuid4()), "org": org},
    )
    db.commit()
    _cursor(db, org, "user-1", at=datetime.now(timezone.utc), row_id=NIL)

    assert script.find_skipped(db, org_id=org) == []


def test_a_user_with_no_cursor_row_has_nothing_abandoned(db, org):
    """No cursor means nothing was ever advanced past."""
    _raw(db, org, "user-1", age_days=1)
    assert script.find_skipped(db, org_id=org) == []


# ---------------------------------------------------------------------------
# Tenancy
# ---------------------------------------------------------------------------


def test_another_orgs_residue_is_never_reported(db, org):
    """Every query is org-scoped in SQL. A reconciliation script that reads
    wide and filters in Python is a cross-tenant read with extra steps."""
    other = f"skip-other-{uuid.uuid4().hex[:8]}"
    _raw(db, other, "user-1", age_days=1)
    _cursor(db, other, "user-1", at=datetime.now(timezone.utc), row_id=NIL)
    mine, _ = _raw(db, org, "user-1", age_days=1)
    _cursor(db, org, "user-1", at=datetime.now(timezone.utc), row_id=NIL)

    found = _only(script.find_skipped(db, org_id=org))
    assert found.recoverable_ids == [mine]

    db.execute(text("DELETE FROM session_memories WHERE org_id = :o"), {"o": other})
    db.execute(text("DELETE FROM brain_curator_state WHERE org_id = :o"), {"o": other})
    db.commit()


def test_two_users_in_one_org_are_reported_separately(db, org):
    """The cursor is per (org, user); so is the residue."""
    a, _ = _raw(db, org, "user-a", age_days=1)
    b, _ = _raw(db, org, "user-b", age_days=1)
    _cursor(db, org, "user-a", at=datetime.now(timezone.utc), row_id=NIL)
    _cursor(db, org, "user-b", at=datetime.now(timezone.utc), row_id=NIL)

    found = {p.user_id: p for p in script.find_skipped(db, org_id=org)}
    assert found["user-a"].recoverable_ids == [a]
    assert found["user-b"].recoverable_ids == [b]


# ---------------------------------------------------------------------------
# The guard — absent --execute, nothing is written
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing(db, org, capsys):
    """The whole package's reason for existing. Asserted against the cursor
    row itself, not against a mock."""
    _raw(db, org, "user-1", age_days=1)
    before = datetime.now(timezone.utc)
    _cursor(db, org, "user-1", at=before, row_id=NIL, processed_count=10)

    parser = common.build_parser("t")
    script.run(db, parser.parse_args(["--org", org]))

    after = db.execute(
        text(
            "SELECT last_processed_at, processed_count FROM brain_curator_state "
            "WHERE org_id = :o AND user_id = 'user-1'"
        ),
        {"o": org},
    ).fetchone()
    assert after.last_processed_at == before
    assert after.processed_count == 10


def test_dry_run_prints_no_memory_content(db, org, caplog):
    """These are private-scope rows. Ids, counts, timestamps and scopes only.

    Asserted through ``caplog`` and NOT ``capsys``: this script reports via the
    ``logging`` module, whose handler holds the real stderr from import time,
    so a ``capsys``-based version of this test captures an empty string and
    passes against a script that logs every row's content verbatim. It was
    written that way first and a mutation that logged ``m.content`` for every
    candidate did not fail it.
    """
    secret = "the deploy key is AKIAIOSFODNN7EXAMPLE"
    row_id, _ = _raw(db, org, "user-1", age_days=1, content=secret)
    _cursor(db, org, "user-1", at=datetime.now(timezone.utc), row_id=NIL)

    parser = common.build_parser("t")
    with caplog.at_level(logging.INFO, logger="reconcile.curator_skipped"):
        script.run(db, parser.parse_args(["--org", org]))

    # Positive control: the log really was captured, so the absences below
    # mean something.
    assert row_id in caplog.text
    assert "AKIA" not in caplog.text
    assert "deploy key" not in caplog.text


def test_execute_without_confirm_leaves_the_cursor_alone(db, org):
    _raw(db, org, "user-1", age_days=1)
    before = datetime.now(timezone.utc)
    _cursor(db, org, "user-1", at=before, row_id=NIL)

    parser = common.build_parser("t")
    with pytest.raises(common.ConfirmationError):
        script.run(db, parser.parse_args(["--org", org, "--execute"]))

    after = db.execute(
        text(
            "SELECT last_processed_at FROM brain_curator_state "
            "WHERE org_id = :o AND user_id = 'user-1'"
        ),
        {"o": org},
    ).fetchone()
    assert after.last_processed_at == before


# ---------------------------------------------------------------------------
# The execute path, against fixtures only
# ---------------------------------------------------------------------------


def test_confirmed_execute_rewinds_the_cursor_behind_the_earliest_candidate(db, org):
    """Positive control for the whole guard: with both factors present the
    remediation must actually happen, or every refusal test above is vacuous.

    Re-queueing is a cursor rewind and nothing else. It writes one row of
    ``brain_curator_state`` and never touches ``session_memories``, so the
    worst case is the curator re-reading rows it already distilled — absorbed
    by dedup, as ``curator._cursor_stop`` already documents.
    """
    earliest, earliest_at = _raw(db, org, "user-1", age_days=3)
    _cursor(db, org, "user-1", at=datetime.now(timezone.utc), row_id=NIL)

    populations = script.find_skipped(db, org_id=org)
    token = common.confirmation_token(
        script.SCRIPT_NAME, org, script.candidate_ids(populations)
    )
    parser = common.build_parser("t")
    script.run(db, parser.parse_args(["--org", org, "--execute", "--confirm", token]))

    after = db.execute(
        text(
            "SELECT last_processed_at FROM brain_curator_state "
            "WHERE org_id = :o AND user_id = 'user-1'"
        ),
        {"o": org},
    ).fetchone()
    assert after.last_processed_at < earliest_at
    assert earliest is not None


def test_execute_does_not_delete_or_modify_any_memory_row(db, org):
    """The remediation is a re-queue. Nothing about it removes evidence."""
    row_id, _ = _raw(db, org, "user-1", age_days=2)
    _cursor(db, org, "user-1", at=datetime.now(timezone.utc), row_id=NIL)

    populations = script.find_skipped(db, org_id=org)
    token = common.confirmation_token(
        script.SCRIPT_NAME, org, script.candidate_ids(populations)
    )
    parser = common.build_parser("t")
    script.run(db, parser.parse_args(["--org", org, "--execute", "--confirm", token]))

    still = db.execute(
        text("SELECT content, status, updated_at FROM session_memories WHERE id = :i"),
        {"i": row_id},
    ).fetchone()
    assert still is not None
    assert still.status == "active"


def test_execute_never_touches_another_org(db, org):
    other = f"skip-other-{uuid.uuid4().hex[:8]}"
    _raw(db, other, "user-1", age_days=1)
    other_at = datetime.now(timezone.utc)
    _cursor(db, other, "user-1", at=other_at, row_id=NIL)
    _raw(db, org, "user-1", age_days=1)
    _cursor(db, org, "user-1", at=datetime.now(timezone.utc), row_id=NIL)

    populations = script.find_skipped(db, org_id=org)
    token = common.confirmation_token(
        script.SCRIPT_NAME, org, script.candidate_ids(populations)
    )
    parser = common.build_parser("t")
    script.run(db, parser.parse_args(["--org", org, "--execute", "--confirm", token]))

    untouched = db.execute(
        text(
            "SELECT last_processed_at FROM brain_curator_state "
            "WHERE org_id = :o AND user_id = 'user-1'"
        ),
        {"o": other},
    ).fetchone()
    assert untouched.last_processed_at == other_at

    db.execute(text("DELETE FROM session_memories WHERE org_id = :o"), {"o": other})
    db.execute(text("DELETE FROM brain_curator_state WHERE org_id = :o"), {"o": other})
    db.commit()
