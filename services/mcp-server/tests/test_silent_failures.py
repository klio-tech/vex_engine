"""The swallowed failures that used to leave no trace at all.

Two sites swallow an exception and continue. Both are *correct* to swallow —
neither may break the request it sits inside — but both were silent, so the
degradation they cause was undiagnosable from logs.

**Control flow is not changed by any of this.** ``recall_log``'s swallow-and-
continue, and its commit behaviour in particular, is deliberate, commented, and
pinned by a live-Postgres regression test; the design spec lists it as
explicitly out of scope. These tests assert a log line is emitted and that the
outcome is *exactly* what it was before.
"""

import logging

from unittest.mock import MagicMock

from app.recall_log import log_recall_event


# ── recall_log: the redaction failure ─────────────────────────────────


def _failing_redact(_text):
    raise RuntimeError("redactor exploded")


def test_redaction_failure_is_logged_with_the_org(monkeypatch, caplog):
    """A redactor that raises drops the query text. That must leave a trace.

    Without it, ``query_text`` silently becomes NULL for every row and the
    audit table looks merely empty rather than broken.
    """
    monkeypatch.setattr("app.recall_log.redact", _failing_redact)
    db = MagicMock()

    with caplog.at_level(logging.WARNING, logger="agentguard.mcp.recall_log"):
        log_recall_event(
            db,
            org_id="org-42",
            agent_id="agent-1",
            project_id=None,
            query_text="something secret",
            result_count=1,
            source="mcp",
        )

    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert records, "the redaction failure was swallowed with no log line"
    assert "org-42" in caplog.text
    assert "redact" in caplog.text.lower()


def test_redaction_failure_log_does_not_echo_the_unredacted_query(monkeypatch, caplog):
    """The log line must not become the leak the redactor was there to prevent."""
    monkeypatch.setattr("app.recall_log.redact", _failing_redact)

    with caplog.at_level(logging.DEBUG):
        log_recall_event(
            MagicMock(),
            org_id="org-42",
            agent_id=None,
            project_id=None,
            query_text="sk-live-supersecret-token",
            result_count=0,
            source="mcp",
        )

    assert "sk-live-supersecret-token" not in caplog.text


def test_redaction_failure_still_stores_null_and_still_commits(monkeypatch):
    """Control flow is unchanged: NULL query_text, INSERT, commit, no raise.

    This is the guard on the whole change. WP-F adds a log line and nothing
    else; if the swallow, the NULL, or the commit moved, this goes red.
    """
    monkeypatch.setattr("app.recall_log.redact", _failing_redact)
    db = MagicMock()

    log_recall_event(
        db,
        org_id="org-42",
        agent_id="agent-1",
        project_id="proj-9",
        query_text="something secret",
        result_count=2,
        source="mcp",
    )

    db.execute.assert_called_once()
    params = db.execute.call_args.args[1]
    assert params["query_text"] is None
    assert params["result_count"] == 2
    db.commit.assert_called_once()
    db.rollback.assert_not_called()


def test_the_redaction_log_assertion_is_not_vacuous(caplog):
    """Positive control: with a working redactor, no warning is emitted.

    Proves the test above detects the failure branch specifically, rather than
    catching some warning this module logs unconditionally.
    """
    with caplog.at_level(logging.WARNING, logger="agentguard.mcp.recall_log"):
        log_recall_event(
            MagicMock(),
            org_id="org-42",
            agent_id=None,
            project_id=None,
            query_text="a perfectly ordinary query",
            result_count=1,
            source="mcp",
        )

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


# ── identity: the OAuthPrincipal import guard ─────────────────────────


def test_oauth_principal_import_failure_is_logged(monkeypatch, caplog):
    """The last logless swallow in ``app.identity``.

    If ``app.oauth`` cannot be imported, every OAuth caller silently degrades
    to unattributed. That is the right outcome and the wrong silence.
    """
    import builtins

    from app import identity

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "app.oauth":
            raise ImportError("simulated import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    request = MagicMock()
    request.state.oauth_principal = object()

    with caplog.at_level(logging.DEBUG, logger="app.identity"):
        assert identity._oauth_principal(request) is None

    assert "app.oauth" in caplog.text
