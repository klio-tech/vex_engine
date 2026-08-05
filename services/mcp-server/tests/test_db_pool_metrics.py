"""The connection pool reports its own pressure.

Pool exhaustion is the failure that presents as "everything is slow" and is
invisible in every application-level metric: requests are not erroring, they are
queueing. Three signals make it legible — how many checkouts are in use, how
long callers wait for one, and how many gave up.

``klio_db_connect_failures_total`` is the fourth, and it is the one that would
have named the 2026-08-04 outage in a dashboard instead of in a log grep: a
pooler refusing every connection is a connect failure, not a checkout timeout.
"""

import pytest
from app import db as app_db
from shared import metrics
from sqlalchemy import create_engine, text
from sqlalchemy.exc import TimeoutError as SATimeoutError
from sqlalchemy.pool import QueuePool


@pytest.fixture(autouse=True)
def _clean():
    metrics.reset_for_test()
    yield
    metrics.reset_for_test()


@pytest.fixture
def instrumented_engine():
    """A tiny in-memory engine carrying the same instrumentation as the real one.

    SQLite rather than Postgres because these are assertions about SQLAlchemy's
    pool events, which are engine-agnostic — the point is that the listeners are
    attached and do the right arithmetic, not that Postgres behaves.

    ``QueuePool`` explicitly: in-memory SQLite defaults to ``SingletonThreadPool``,
    which hands the same connection to every caller, so concurrent checkouts
    would not stack and the gauge test would be measuring the pool class rather
    than the listener.
    """
    engine = create_engine("sqlite://", poolclass=QueuePool)
    app_db.instrument_pool(engine)
    return engine


def test_a_checkout_raises_the_in_use_gauge(instrumented_engine):
    with instrumented_engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        assert metrics.DB_POOL_IN_USE.value() == 1.0


def test_returning_a_connection_lowers_the_gauge(instrumented_engine):
    with instrumented_engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    assert metrics.DB_POOL_IN_USE.value() == 0.0


def test_concurrent_checkouts_are_counted_together(instrumented_engine):
    first = instrumented_engine.connect()
    second = instrumented_engine.connect()
    try:
        assert metrics.DB_POOL_IN_USE.value() == 2.0
    finally:
        first.close()
        second.close()
    assert metrics.DB_POOL_IN_USE.value() == 0.0


def test_checkout_wait_time_is_observed(instrumented_engine):
    with instrumented_engine.connect():
        pass
    assert metrics.DB_POOL_WAIT_SECONDS.snapshot().count == 1


def test_the_wait_histogram_is_not_incremented_without_a_checkout(
    instrumented_engine,
):
    """Positive control: the observation follows a checkout, it is not free."""
    assert metrics.DB_POOL_WAIT_SECONDS.snapshot().count == 0


def test_a_pool_timeout_is_counted(instrumented_engine, monkeypatch):
    """A checkout that times out is the pool-exhaustion signal."""

    def timeout(*_args, **_kwargs):
        raise SATimeoutError("QueuePool limit reached")

    monkeypatch.setattr(app_db, "_raw_connect", timeout, raising=False)
    with pytest.raises(SATimeoutError):
        app_db.checkout_connection(instrumented_engine)

    assert metrics.DB_POOL_TIMEOUTS.value() == 1.0
    assert metrics.DB_CONNECT_FAILURES.value() == 0.0


def test_a_refused_connection_is_counted_separately_from_a_pool_timeout(
    instrumented_engine, monkeypatch
):
    """The 2026-08-04 outage. A pooler refusing connections is not exhaustion.

    Folding the two together would have shown "pool pressure" during an
    incident whose cause was that no connection could be opened at all.
    """
    from sqlalchemy.exc import OperationalError

    def refuse(*_args, **_kwargs):
        raise OperationalError(
            "connect", {}, Exception("unsupported startup parameter in options")
        )

    monkeypatch.setattr(app_db, "_raw_connect", refuse, raising=False)
    with pytest.raises(OperationalError):
        app_db.checkout_connection(instrumented_engine)

    assert metrics.DB_CONNECT_FAILURES.value() == 1.0
    assert metrics.DB_POOL_TIMEOUTS.value() == 0.0


def test_a_successful_checkout_counts_neither_failure(instrumented_engine):
    """Positive control for both counters above."""
    app_db.checkout_connection(instrumented_engine).close()
    assert metrics.DB_POOL_TIMEOUTS.value() == 0.0
    assert metrics.DB_CONNECT_FAILURES.value() == 0.0


def test_the_real_engine_is_instrumented():
    """The listeners must be attached to the engine the service actually uses."""
    from sqlalchemy import event

    assert event.contains(app_db.engine, "checkout", app_db._on_checkout)
    assert event.contains(app_db.engine, "checkin", app_db._on_checkin)


def test_broken_instrumentation_does_not_break_a_checkout(
    instrumented_engine, monkeypatch
):
    def explode(*_args, **_kwargs):
        raise RuntimeError("metrics on fire")

    monkeypatch.setattr(metrics.DB_POOL_IN_USE, "inc", explode)
    with instrumented_engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1
