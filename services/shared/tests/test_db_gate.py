"""The gate that decides whether a database-backed test runs, skips or fails.

This is the test for the fix to the "green CI that tested nothing" class of
bug: every tenancy-boundary suite in this repo gated itself on a module-level
``_db_reachable()`` that returned ``False`` on any exception, so a CI job with
no Postgres turned 112 adversarial tests into skips and still exited 0.

The gate keeps the local ergonomics (no Postgres → skip, so a developer can
run the suite on a laptop) but makes the same situation a hard failure when
``KLIO_REQUIRE_DB`` is set, which CI does. These tests pin that decision
table, because the decision table IS the fix.
"""

import pytest

from shared.testing import db_gate


class _FakeMarker:
    def __init__(self, *args):
        self.args = args


class _FakeItem:
    """The slice of a pytest ``Item`` the hook actually touches."""

    def __init__(self, marker=None):
        self._marker = marker

    def get_closest_marker(self, name):
        return self._marker if name == db_gate.MARKER_NAME else None


# ── the decision table ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reachable", "required", "expected"),
    [
        (True, False, db_gate.RUN),
        (True, True, db_gate.RUN),
        (False, False, db_gate.SKIP),
        (False, True, db_gate.FAIL),
    ],
)
def test_gate_action_decision_table(reachable, required, expected):
    assert db_gate.gate_action(reachable=reachable, required=required) == expected


def test_unreachable_and_required_is_never_a_skip():
    """The whole point. A skip here is the bug being fixed."""
    assert db_gate.gate_action(reachable=False, required=True) != db_gate.SKIP


# ── KLIO_REQUIRE_DB parsing ───────────────────────────────────────────


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " 1 "])
def test_db_required_accepts_truthy_values(monkeypatch, value):
    monkeypatch.setenv(db_gate.REQUIRE_DB_ENV, value)
    assert db_gate.db_required() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_db_required_rejects_everything_else(monkeypatch, value):
    monkeypatch.setenv(db_gate.REQUIRE_DB_ENV, value)
    assert db_gate.db_required() is False


def test_db_required_defaults_to_false_when_unset(monkeypatch):
    monkeypatch.delenv(db_gate.REQUIRE_DB_ENV, raising=False)
    assert db_gate.db_required() is False


# ── the hook ──────────────────────────────────────────────────────────


def test_unmarked_test_is_untouched(monkeypatch):
    monkeypatch.setattr(db_gate, "db_reachable", lambda: False)
    monkeypatch.setenv(db_gate.REQUIRE_DB_ENV, "1")
    db_gate.pytest_runtest_setup(_FakeItem(marker=None))


def test_marked_test_runs_when_db_is_reachable(monkeypatch):
    monkeypatch.setattr(db_gate, "db_reachable", lambda: True)
    db_gate.pytest_runtest_setup(_FakeItem(_FakeMarker("boundary tests")))


def test_marked_test_skips_locally_when_db_is_unreachable(monkeypatch):
    monkeypatch.setattr(db_gate, "db_reachable", lambda: False)
    monkeypatch.delenv(db_gate.REQUIRE_DB_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception):
        db_gate.pytest_runtest_setup(_FakeItem(_FakeMarker("boundary tests")))


def test_marked_test_fails_in_ci_when_db_is_unreachable(monkeypatch):
    monkeypatch.setattr(db_gate, "db_reachable", lambda: False)
    monkeypatch.setenv(db_gate.REQUIRE_DB_ENV, "1")
    with pytest.raises(pytest.fail.Exception) as excinfo:
        db_gate.pytest_runtest_setup(_FakeItem(_FakeMarker("boundary tests")))
    assert db_gate.REQUIRE_DB_ENV in str(excinfo.value)


def test_failure_message_names_the_suite(monkeypatch):
    monkeypatch.setattr(db_gate, "db_reachable", lambda: False)
    monkeypatch.setenv(db_gate.REQUIRE_DB_ENV, "1")
    with pytest.raises(pytest.fail.Exception) as excinfo:
        db_gate.pytest_runtest_setup(_FakeItem(_FakeMarker("the user-silo suite")))
    assert "the user-silo suite" in str(excinfo.value)


# ── credentials must not reach the log ────────────────────────────────


def test_redaction_removes_the_password(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://agentguard:sup3rs3cret@db.internal:5432/klio"
    )
    redacted = db_gate.redacted_database_url()
    assert "sup3rs3cret" not in redacted
    assert "db.internal:5432/klio" in redacted
    assert "agentguard" in redacted


def test_redaction_leaves_a_passwordless_url_alone(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost:5432/klio")
    assert db_gate.redacted_database_url() == "postgresql://localhost:5432/klio"


def test_skip_and_failure_messages_never_leak_the_password(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://agentguard:sup3rs3cret@db.internal:5432/klio"
    )
    monkeypatch.setattr(db_gate, "db_reachable", lambda: False)

    monkeypatch.delenv(db_gate.REQUIRE_DB_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception) as skipped:
        db_gate.pytest_runtest_setup(_FakeItem(_FakeMarker("boundary tests")))
    assert "sup3rs3cret" not in str(skipped.value)

    monkeypatch.setenv(db_gate.REQUIRE_DB_ENV, "1")
    with pytest.raises(pytest.fail.Exception) as failed:
        db_gate.pytest_runtest_setup(_FakeItem(_FakeMarker("boundary tests")))
    assert "sup3rs3cret" not in str(failed.value)


# ── the marker factory ────────────────────────────────────────────────


def test_requires_db_produces_a_marker_carrying_the_reason():
    mark = db_gate.requires_db("the graph boundary suite")
    assert mark.name == db_gate.MARKER_NAME
    assert mark.args == ("the graph boundary suite",)


def test_requires_db_rejects_an_empty_reason():
    with pytest.raises(ValueError):
        db_gate.requires_db("   ")


def test_database_url_falls_back_to_the_dev_default(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert db_gate.database_url() == db_gate.DEFAULT_DATABASE_URL


def test_unreachable_url_probes_false():
    assert db_gate.probe("postgresql://nobody:nobody@127.0.0.1:1/nope") is False
