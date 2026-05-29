"""Tests for the best-effort recall-event logger (``app.recall_log``).

Mock-based, mirroring ``tests/test_capture.py``: no live Postgres.  The single
``log_recall_event`` helper takes the DB session as a plain argument, so a bare
``unittest.mock.MagicMock`` stands in for the session — we assert on the
``execute``/``commit`` calls and the bound INSERT params, never touching a real
table.  The contract under test is twofold: (1) a happy-path INSERT + commit,
and (2) best-effort swallowing — a failing ``execute`` must NOT propagate, so a
logging failure can never break the recall that triggered it.
"""

from unittest.mock import MagicMock

from app.recall_log import log_recall_event


def test_log_recall_event_inserts_row():
    """Happy path: one INSERT with the expected params, then a single commit."""
    db = MagicMock()

    log_recall_event(
        db,
        org_id="test-org",
        agent_id="agent-1",
        project_id="proj-9",
        query_text="auth flow",
        result_count=3,
        source="mcp",
    )

    db.execute.assert_called_once()
    # Bound params are the second positional arg to ``Session.execute``.
    params = db.execute.call_args.args[1]
    assert params["org_id"] == "test-org"
    assert params["agent_id"] == "agent-1"
    assert params["project_id"] == "proj-9"
    assert params["query_text"] == "auth flow"
    assert params["result_count"] == 3
    assert params["source"] == "mcp"
    assert params["id"]  # an app-minted UUID is present
    db.commit.assert_called_once()


def test_log_recall_event_swallows_errors():
    """A failing execute is caught and swallowed: never raises, never commits."""
    db = MagicMock()
    db.execute.side_effect = RuntimeError("boom")

    # Must not raise — logging a recall can never break the recall.
    log_recall_event(
        db,
        org_id="test-org",
        agent_id=None,
        project_id=None,
        query_text=None,
        result_count=0,
        source="capture",
    )

    db.commit.assert_not_called()
