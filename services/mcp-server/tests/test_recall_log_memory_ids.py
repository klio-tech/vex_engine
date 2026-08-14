"""Tests for the evidence-loop additions to ``app.recall_log`` (Task 5).

Migration 042 added ``memory_ids`` (``UUID[]``) and ``session_id`` (``TEXT``)
to ``brain_recall_events``. These tests cover the two new optional kwargs on
``log_recall_event``: that they land in the INSERT, and that omitting them
(every pre-existing call site) still works and inserts ``NULL`` for both.
"""

from unittest.mock import MagicMock

from app.recall_log import log_recall_event


def test_log_recall_event_includes_memory_ids_and_session_id():
    db = MagicMock()

    log_recall_event(
        db,
        org_id="o",
        agent_id="a",
        project_id=None,
        query_text="q",
        result_count=1,
        source="mcp",
        memory_ids=["11111111-1111-1111-1111-111111111111"],
        session_id="s9",
    )

    sql = str(db.execute.call_args.args[0])
    params = db.execute.call_args.args[1]
    assert "memory_ids" in sql and "session_id" in sql
    assert params["memory_ids"] == ["11111111-1111-1111-1111-111111111111"]
    assert params["session_id"] == "s9"


def test_log_recall_event_backward_compatible_without_new_kwargs():
    """Every pre-existing call site omits the new kwargs; both must default
    to ``None`` so the INSERT still runs (and lands NULL for both columns)."""
    db = MagicMock()

    log_recall_event(
        db,
        org_id="o",
        agent_id=None,
        project_id=None,
        query_text=None,
        result_count=0,
        source="mcp",
    )

    params = db.execute.call_args.args[1]
    assert params["memory_ids"] is None
    assert params["session_id"] is None
