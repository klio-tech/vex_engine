"""Tests for shared.traces (session_traces row upsert + fetch).

Mocked db/s3: the pool mock never evaluates SQL — these assert SQL shape and
bound params, not real Postgres behaviour (mirrors the mocked-pool policy
used elsewhere, e.g. test_evidence_stream.py).
"""

from unittest.mock import MagicMock

from shared import traces


def test_upsert_richer_wins_sql():
    db = MagicMock()
    result = traces.upsert_trace(
        db, org_id="o", session_id="s", transcript_ref="s3://b/k",
        message_count=5, tool_calls_count=2, redaction_version="1",
    )
    sql = " ".join(str(db.execute.call_args.args[0]).split())
    assert "INSERT INTO session_traces" in sql
    assert "ON CONFLICT" in sql and "DO UPDATE" in sql
    assert "EXCLUDED.message_count > session_traces.message_count" in sql
    assert "RETURNING id" in sql
    # MagicMock's auto-generated .first() is truthy, so the mocked "row was
    # written" path returns a dict, not the bare id string the old contract
    # returned.
    assert isinstance(result, dict) and "id" in result and "previous_ref" in result


def test_upsert_lost_race_returns_none():
    """ON CONFLICT ... WHERE matching zero rows means RETURNING yields none —
    upsert_trace must report that as "nothing written", not a fabricated id."""
    db = MagicMock()
    db.execute.return_value.first.return_value = None
    db.execute.return_value.mappings.return_value.first.return_value = {
        "transcript_ref": "s3://b/existing.json"
    }
    assert traces.upsert_trace(
        db, org_id="o", session_id="s", transcript_ref="s3://b/new.json",
        message_count=1, tool_calls_count=0, redaction_version="1",
    ) is None


def test_upsert_never_raises():
    db = MagicMock()
    db.execute.side_effect = RuntimeError("down")
    assert traces.upsert_trace(
        db, org_id="o", session_id="s", transcript_ref="r",
        message_count=1, tool_calls_count=0, redaction_version="1",
    ) is None


def test_fetch_predicates_on_org():
    db = MagicMock()
    db.execute.return_value.mappings.return_value.first.return_value = None
    traces.fetch_trace(db, MagicMock(), org_id="o", session_id="s")
    assert "org_id = :org_id" in str(db.execute.call_args.args[0])
