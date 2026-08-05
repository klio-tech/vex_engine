"""The curator and the capture path report their own progress.

The curator is a background loop with no caller to notice when it stops making
progress. Its characteristic failure is not an exception — it is a pass that
reads rows, distils none of them, and holds its cursor, forever, quietly. Rows
read against rows distilled, and cursor advanced against cursor held, are the
two ratios that make that legible.
"""

import pytest
from app import curator
from shared import metrics


@pytest.fixture(autouse=True)
def _clean():
    metrics.reset_for_test()
    yield
    metrics.reset_for_test()


# ── the counters the curator must move ────────────────────────────────


def test_rows_read_and_distilled_are_separate_counters():
    """The gap between them IS the skip rate — one number cannot express it."""
    metrics.CURATOR_ROWS_READ.inc(10)
    metrics.CURATOR_ROWS_DISTILLED.inc(4)
    assert metrics.CURATOR_ROWS_READ.value() == 10.0
    assert metrics.CURATOR_ROWS_DISTILLED.value() == 4.0


def test_the_curator_records_a_tick_outcome():
    curator.record_tick("ok", duration_s=0.5)
    assert metrics.CURATOR_TICKS.value(outcome="ok") == 1.0
    assert metrics.CURATOR_TICK_SECONDS.snapshot().count == 1


def test_a_failed_tick_is_recorded_as_an_error():
    curator.record_tick("error", duration_s=0.1)
    assert metrics.CURATOR_TICKS.value(outcome="error") == 1.0
    assert metrics.CURATOR_TICKS.value(outcome="ok") == 0.0


def test_record_progress_splits_read_from_distilled():
    curator.record_progress(rows_read=10, rows_distilled=4)
    assert metrics.CURATOR_ROWS_READ.value() == 10.0
    assert metrics.CURATOR_ROWS_DISTILLED.value() == 4.0


def test_record_progress_reports_a_fully_skipped_pass():
    """The silent failure: rows were read and nothing came of them."""
    curator.record_progress(rows_read=50, rows_distilled=0)
    assert metrics.CURATOR_ROWS_READ.value() == 50.0
    assert metrics.CURATOR_ROWS_DISTILLED.value() == 0.0


def test_a_held_cursor_and_an_advanced_cursor_are_distinguishable():
    curator.record_cursor(advanced=False)
    curator.record_cursor(advanced=True)
    assert metrics.CURATOR_CURSOR.value(action="held") == 1.0
    assert metrics.CURATOR_CURSOR.value(action="advanced") == 1.0


def test_entities_created_and_referenced_are_distinguishable():
    curator.record_entity(created=True)
    curator.record_entity(created=False)
    curator.record_entity(created=False)
    assert metrics.CURATOR_ENTITIES.value(disposition="created") == 1.0
    assert metrics.CURATOR_ENTITIES.value(disposition="referenced") == 2.0


@pytest.mark.parametrize("budget", ["entity", "reconcile"])
def test_budget_exhaustion_is_attributed_to_the_budget_that_ran_out(budget):
    curator.record_budget_exhausted(budget)
    assert metrics.CURATOR_BUDGET_EXHAUSTED.value(budget=budget) == 1.0


def test_an_unknown_budget_name_does_not_raise(caplog):
    """Instrumentation must never be the thing that breaks the curator."""
    curator.record_budget_exhausted("not-a-real-budget")
    assert metrics.CURATOR_BUDGET_EXHAUSTED.value(budget="entity") == 0.0


def test_broken_instrumentation_does_not_break_a_tick(monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("metrics on fire")

    monkeypatch.setattr(metrics.CURATOR_TICKS, "inc", explode)
    curator.record_tick("ok", duration_s=0.1)  # must not raise


# ── the counters are actually wired into the curator ──────────────────


def test_the_tick_helpers_are_called_by_the_curator():
    """Positive control against a set of helpers nothing invokes.

    Declaring ``record_progress`` and never calling it would leave every test
    above passing while the dashboard stayed empty.
    """
    import inspect

    source = inspect.getsource(curator)
    for helper in (
        "record_tick(",
        "record_progress(",
        "record_cursor(",
        "record_entity(",
        "record_budget_exhausted(",
    ):
        # Once for the definition, at least once more for a call site.
        assert source.count(helper) >= 2, f"{helper} is declared but never called"


# ── capture ───────────────────────────────────────────────────────────


def test_capture_counts_an_accepted_event():
    from app import capture

    capture.record_capture("accepted")
    assert metrics.CAPTURE_EVENTS.value(outcome="accepted") == 1.0


def test_capture_counts_a_rejected_event_separately():
    from app import capture

    capture.record_capture("rejected")
    assert metrics.CAPTURE_EVENTS.value(outcome="rejected") == 1.0
    assert metrics.CAPTURE_EVENTS.value(outcome="accepted") == 0.0


def test_a_redaction_hit_is_counted():
    from app import capture

    capture.record_redaction(hit=True)
    assert metrics.CAPTURE_REDACTIONS.value() == 1.0


def test_a_clean_payload_is_not_counted_as_a_redaction():
    """Positive control: the counter measures hits, not calls."""
    from app import capture

    capture.record_redaction(hit=False)
    assert metrics.CAPTURE_REDACTIONS.value() == 0.0


def test_the_capture_helpers_are_called_by_the_capture_routes():
    import inspect

    from app import capture

    source = inspect.getsource(capture)
    assert source.count("record_capture(") >= 2
    assert source.count("record_redaction(") >= 2
