"""sweep_evidence_for_org: session_traces / session_outcomes / recall_outcomes
expire with plan retention. Real Postgres — the boundary is the SQL (org_id
predicate, the cutoff arithmetic, the <= 0 guard), so a mocked connection
would pass against a DELETE missing any one of those.

``memory_proposals`` (migration 045) is a decision made FROM this evidence,
not evidence itself, and is not swept here — see the docstring on
``sweep_evidence_for_org`` in ``run_retention.py``. ``claims`` self-expire on
their own TTL, also out of scope for this sweep.
"""

import importlib
import os
import sys
import uuid

import pytest
from sqlalchemy import create_engine, text

from shared.testing import database_url, requires_db


def _load_run_retention():
    sys.path.insert(
        0,
        os.path.join(os.path.dirname(__file__), "..", "..", "scripts"),
    )
    return importlib.import_module("run_retention")


@pytest.fixture()
def evidence_db():
    engine = create_engine(database_url())
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
                    text("DELETE FROM recall_outcomes WHERE org_id = :o"),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM session_outcomes WHERE org_id = :o"),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM session_traces WHERE org_id = :o"),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM brain_recall_events WHERE org_id = :o"),
                    {"o": org_id},
                )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _org(db) -> str:
    org_id = f"org_{uuid.uuid4().hex[:8]}"
    db.created_org_ids.add(org_id)
    return org_id


def _store_trace(db, *, org_id: str, age_days: int) -> str:
    """Insert one session_traces row, backdated ``age_days`` into the past."""
    trace_id = uuid.uuid4()
    db.execute(
        text(
            """
            INSERT INTO session_traces (
                id, org_id, session_id, transcript_ref,
                message_count, tool_calls_count, redaction_version,
                created_at, updated_at
            ) VALUES (
                :id, :org, :session, :ref, 3, 1, '1',
                NOW() - make_interval(days => :age),
                NOW() - make_interval(days => :age)
            )
            """
        ),
        {
            "id": trace_id,
            "org": org_id,
            "session": f"session-{trace_id}",
            "ref": f"s3://bucket/{trace_id}",
            "age": age_days,
        },
    )
    db.commit()
    return str(trace_id)


def _store_session_outcome(db, *, org_id: str, age_days: int) -> str:
    """Insert one session_outcomes row, backdated via graded_at."""
    outcome_id = uuid.uuid4()
    db.execute(
        text(
            """
            INSERT INTO session_outcomes (
                id, org_id, session_id, verdict, grader_version, graded_at
            ) VALUES (
                :id, :org, :session, 'pass', 'v1',
                NOW() - make_interval(days => :age)
            )
            """
        ),
        {
            "id": outcome_id,
            "org": org_id,
            "session": f"session-{outcome_id}",
            "age": age_days,
        },
    )
    db.commit()
    return str(outcome_id)


def _store_recall_event(db, *, org_id: str) -> str:
    event_id = uuid.uuid4()
    db.execute(
        text(
            """
            INSERT INTO brain_recall_events (
                id, org_id, agent_id, query_text, created_at
            ) VALUES (:id, :org, 'agent-1', 'q', NOW())
            """
        ),
        {"id": event_id, "org": org_id},
    )
    db.commit()
    return str(event_id)


def _store_recall_outcome(
    db, *, org_id: str, recall_event_id: str, session_outcome_id: str, age_days: int
) -> str:
    """Insert one recall_outcomes row, backdated via created_at."""
    row_id = uuid.uuid4()
    db.execute(
        text(
            """
            INSERT INTO recall_outcomes (
                id, org_id, recall_event_id, memory_id, session_outcome_id,
                used, created_at
            ) VALUES (
                :id, :org, :event, :memory, :outcome, true,
                NOW() - make_interval(days => :age)
            )
            """
        ),
        {
            "id": row_id,
            "org": org_id,
            "event": recall_event_id,
            "memory": uuid.uuid4(),
            "outcome": session_outcome_id,
            "age": age_days,
        },
    )
    db.commit()
    return str(row_id)


def _traces(db, *, org_id: str) -> set:
    rows = db.execute(
        text("SELECT id FROM session_traces WHERE org_id = :o"), {"o": org_id}
    ).fetchall()
    return {str(r[0]) for r in rows}


def _outcomes(db, *, org_id: str) -> set:
    rows = db.execute(
        text("SELECT id FROM session_outcomes WHERE org_id = :o"), {"o": org_id}
    ).fetchall()
    return {str(r[0]) for r in rows}


def _recall_outcomes(db, *, org_id: str) -> set:
    rows = db.execute(
        text("SELECT id FROM recall_outcomes WHERE org_id = :o"), {"o": org_id}
    ).fetchall()
    return {str(r[0]) for r in rows}


# ---------------------------------------------------------------------------
# session_traces
# ---------------------------------------------------------------------------


@requires_db("the evidence retention sweep — session_traces")
def test_expired_trace_is_deleted_fresh_trace_is_kept(evidence_db):
    mod = _load_run_retention()
    org = _org(evidence_db)

    expired = _store_trace(evidence_db, org_id=org, age_days=40)
    fresh = _store_trace(evidence_db, org_id=org, age_days=1)

    counts = mod.sweep_evidence_for_org(evidence_db, org_id=org, retention_days=30)

    assert counts["session_traces"] == 1
    remaining = _traces(evidence_db, org_id=org)
    assert remaining == {fresh}
    assert expired not in remaining


@requires_db("the evidence retention sweep — session_traces")
def test_non_positive_retention_skips_the_trace_sweep(evidence_db):
    mod = _load_run_retention()
    org = _org(evidence_db)

    expired = _store_trace(evidence_db, org_id=org, age_days=400)

    counts = mod.sweep_evidence_for_org(evidence_db, org_id=org, retention_days=0)
    assert counts == {
        "session_traces": 0,
        "session_outcomes": 0,
        "recall_outcomes": 0,
    }
    assert expired in _traces(evidence_db, org_id=org)

    counts = mod.sweep_evidence_for_org(evidence_db, org_id=org, retention_days=-1)
    assert counts["session_traces"] == 0
    assert expired in _traces(evidence_db, org_id=org)


# ---------------------------------------------------------------------------
# session_outcomes
# ---------------------------------------------------------------------------


@requires_db("the evidence retention sweep — session_outcomes")
def test_expired_session_outcome_is_deleted_fresh_is_kept(evidence_db):
    mod = _load_run_retention()
    org = _org(evidence_db)

    expired = _store_session_outcome(evidence_db, org_id=org, age_days=40)
    fresh = _store_session_outcome(evidence_db, org_id=org, age_days=1)

    counts = mod.sweep_evidence_for_org(evidence_db, org_id=org, retention_days=30)

    assert counts["session_outcomes"] == 1
    remaining = _outcomes(evidence_db, org_id=org)
    assert remaining == {fresh}
    assert expired not in remaining


@requires_db("the evidence retention sweep — session_outcomes")
def test_non_positive_retention_skips_the_session_outcome_sweep(evidence_db):
    mod = _load_run_retention()
    org = _org(evidence_db)

    expired = _store_session_outcome(evidence_db, org_id=org, age_days=400)

    counts = mod.sweep_evidence_for_org(evidence_db, org_id=org, retention_days=0)
    assert counts["session_outcomes"] == 0
    assert expired in _outcomes(evidence_db, org_id=org)


# ---------------------------------------------------------------------------
# recall_outcomes
# ---------------------------------------------------------------------------


@requires_db("the evidence retention sweep — recall_outcomes")
def test_expired_recall_outcome_is_deleted_fresh_is_kept(evidence_db):
    mod = _load_run_retention()
    org = _org(evidence_db)

    event = _store_recall_event(evidence_db, org_id=org)
    # A fresh session_outcome parent so the row is NOT also removed by the
    # ON DELETE CASCADE from session_outcomes — this test is isolating the
    # recall_outcomes cutoff itself, not the cascade.
    outcome = _store_session_outcome(evidence_db, org_id=org, age_days=1)

    expired = _store_recall_outcome(
        evidence_db,
        org_id=org,
        recall_event_id=event,
        session_outcome_id=outcome,
        age_days=40,
    )
    fresh = _store_recall_outcome(
        evidence_db,
        org_id=org,
        recall_event_id=event,
        session_outcome_id=outcome,
        age_days=1,
    )

    counts = mod.sweep_evidence_for_org(evidence_db, org_id=org, retention_days=30)

    assert counts["recall_outcomes"] == 1
    remaining = _recall_outcomes(evidence_db, org_id=org)
    assert remaining == {fresh}
    assert expired not in remaining


@requires_db("the evidence retention sweep — recall_outcomes")
def test_non_positive_retention_skips_the_recall_outcome_sweep(evidence_db):
    mod = _load_run_retention()
    org = _org(evidence_db)

    event = _store_recall_event(evidence_db, org_id=org)
    outcome = _store_session_outcome(evidence_db, org_id=org, age_days=1)
    expired = _store_recall_outcome(
        evidence_db,
        org_id=org,
        recall_event_id=event,
        session_outcome_id=outcome,
        age_days=400,
    )

    counts = mod.sweep_evidence_for_org(evidence_db, org_id=org, retention_days=0)
    assert counts["recall_outcomes"] == 0
    assert expired in _recall_outcomes(evidence_db, org_id=org)


@requires_db("the evidence retention sweep — recall_outcomes cascade")
def test_recall_outcome_also_disappears_when_its_session_outcome_expires(
    evidence_db,
):
    """The cascade and the direct delete agree: a recall_outcomes row whose
    OWN created_at is fresh but whose parent session_outcome has expired is
    still gone after the sweep, via ON DELETE CASCADE."""
    mod = _load_run_retention()
    org = _org(evidence_db)

    event = _store_recall_event(evidence_db, org_id=org)
    expired_outcome = _store_session_outcome(evidence_db, org_id=org, age_days=40)
    row = _store_recall_outcome(
        evidence_db,
        org_id=org,
        recall_event_id=event,
        session_outcome_id=expired_outcome,
        age_days=1,
    )

    mod.sweep_evidence_for_org(evidence_db, org_id=org, retention_days=30)

    assert row not in _recall_outcomes(evidence_db, org_id=org)
    assert expired_outcome not in _outcomes(evidence_db, org_id=org)


# ---------------------------------------------------------------------------
# cross-org isolation
# ---------------------------------------------------------------------------


@requires_db("the evidence retention sweep — org isolation")
def test_the_sweep_never_reaches_into_another_org(evidence_db):
    mod = _load_run_retention()
    org_a = _org(evidence_db)
    org_b = _org(evidence_db)

    a_expired = _store_trace(evidence_db, org_id=org_a, age_days=40)
    b_expired = _store_trace(evidence_db, org_id=org_b, age_days=40)

    counts = mod.sweep_evidence_for_org(evidence_db, org_id=org_a, retention_days=30)

    assert counts["session_traces"] == 1
    assert a_expired not in _traces(evidence_db, org_id=org_a)
    assert b_expired in _traces(evidence_db, org_id=org_b), (
        "one org's retention run deleted another org's trace"
    )
