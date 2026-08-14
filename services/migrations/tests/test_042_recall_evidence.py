import sqlalchemy as sa

from shared.testing import requires_db, database_url

pytestmark = requires_db("the recall_evidence migration tests")


def _insp():
    return sa.inspect(sa.create_engine(database_url()))


def test_recall_events_columns():
    cols = {c["name"] for c in _insp().get_columns("brain_recall_events")}
    assert {"memory_ids", "session_id"} <= cols


def test_outcome_tables():
    so = {c["name"] for c in _insp().get_columns("session_outcomes")}
    assert {
        "org_id",
        "session_id",
        "verdict",
        "confidence",
        "memory_consistency",
        "grader_version",
    } <= so
    ro = {c["name"] for c in _insp().get_columns("recall_outcomes")}
    assert {
        "org_id",
        "recall_event_id",
        "memory_id",
        "session_outcome_id",
        "used",
        "usage_score",
        "served_stale",
    } <= ro
    fks = {fk["referred_table"] for fk in _insp().get_foreign_keys("recall_outcomes")}
    assert {"brain_recall_events", "session_outcomes"} <= fks


def test_session_outcomes_unique_constraint():
    constraints = _insp().get_unique_constraints("session_outcomes")
    u = [
        c for c in constraints if c["name"] == "uq_session_outcomes_org_session_grader"
    ]
    assert u and set(u[0]["column_names"]) == {"org_id", "session_id", "grader_version"}
