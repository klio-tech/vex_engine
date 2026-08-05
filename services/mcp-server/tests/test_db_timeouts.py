"""Every wait on the database must be bounded.

``pool_size`` and ``max_overflow`` *are* set — the "missing pool sizing"
reading of ``db.py`` is wrong. What was genuinely missing is every **timeout**:

* no ``pool_timeout``, so SQLAlchemy's 30 s default was the only thing between
  a saturated pool and an unbounded queue;
* no ``connect_timeout``, so opening a connection to an unreachable host waited
  on the OS TCP timeout — minutes, not seconds;
* no ``statement_timeout``, so a single pathological query could hold its
  connection and its worker indefinitely. ``pool_pre_ping`` does not help with
  any of these: it detects a *dead* connection, not a slow one.

Each is configurable, because the right value differs between a request path
and a background sweep, and none of them should need a code change to tune.
"""

import pytest
from app import db
from shared.testing import requires_db
from sqlalchemy import text as sa_text


def test_the_pool_is_still_sized_the_same_way():
    """Positive control: this fix must not change the pool's dimensions."""
    pool = db.engine.pool
    assert pool.size() == 5
    assert pool._max_overflow == 10


def test_the_pool_checkout_wait_is_bounded():
    assert db.engine.pool._timeout == db.POOL_TIMEOUT_S
    assert 0 < db.POOL_TIMEOUT_S <= 60


def test_connecting_to_an_unreachable_host_is_bounded():
    assert db.CONNECT_ARGS["connect_timeout"] == db.CONNECT_TIMEOUT_S
    assert 0 < db.CONNECT_TIMEOUT_S <= 60


def test_a_pathological_query_is_bounded():
    assert 0 < db.STATEMENT_TIMEOUT_MS <= 300_000


def test_the_statement_timeout_is_never_a_startup_parameter():
    """Regression guard for a production outage.

    Passing ``options: -c statement_timeout=...`` makes libpq send it as a
    *startup* parameter. A PgBouncer transaction pooler — which is what Neon's
    pooled endpoint is, and what production connects to — refuses the whole
    connection:

        ERROR: unsupported startup parameter in options: statement_timeout

    Every request failed while every test passed, because the tests run against
    a direct local Postgres that accepts it. The timeout is applied as a plain
    ``SET`` on connect instead; this test pins the form, not just the value, so
    the convenient-looking version cannot come back.
    """
    assert "options" not in db.CONNECT_ARGS, (
        "statement_timeout must not be a libpq startup parameter: a pooled "
        "endpoint rejects the connection outright"
    )


@requires_db("the statement timeout is applied by the server, not just declared")
def test_the_server_actually_applies_the_statement_timeout():
    """The declaration is worthless if libpq silently drops it."""
    with db.engine.connect() as conn:
        # Postgres normalises the unit on the way out ("30000ms" -> "30s"), so
        # compare in milliseconds rather than on the literal it echoes back.
        applied_ms = conn.execute(
            sa_text(
                "SELECT setting::bigint FROM pg_settings "
                "WHERE name = 'statement_timeout'"
            )
        ).scalar()
    assert applied_ms == db.STATEMENT_TIMEOUT_MS, (
        f"the backend applied statement_timeout={applied_ms}ms, "
        f"expected {db.STATEMENT_TIMEOUT_MS}ms"
    )


def test_pre_ping_and_recycle_are_still_on():
    """Positive control: the existing liveness settings survive."""
    assert db.engine.pool._pre_ping is True
    assert db.engine.pool._recycle == 3600


@pytest.mark.parametrize(
    "env,attr,default",
    [
        ("KLIO_DB_POOL_TIMEOUT_S", "POOL_TIMEOUT_S", 10.0),
        ("KLIO_DB_CONNECT_TIMEOUT_S", "CONNECT_TIMEOUT_S", 10),
        ("KLIO_DB_STATEMENT_TIMEOUT_MS", "STATEMENT_TIMEOUT_MS", 30_000),
    ],
)
def test_each_bound_has_a_documented_default(env, attr, default):
    assert getattr(db, attr) == default, (
        f"{attr} changed; {env} documentation and this test must agree"
    )


def test_a_malformed_knob_does_not_stop_the_process_importing(monkeypatch):
    """A typo in a timeout must not be the reason the service will not boot."""
    monkeypatch.setenv("KLIO_DB_POOL_TIMEOUT_S", "10s")
    monkeypatch.setenv("KLIO_DB_CONNECT_TIMEOUT_S", "soon")
    monkeypatch.setenv("KLIO_DB_STATEMENT_TIMEOUT_MS", "30 seconds")
    assert db.pool_timeout_s() == 10.0
    assert db.connect_timeout_s() == 10
    assert db.statement_timeout_ms() == 30_000


def test_a_blank_knob_is_an_absence_not_an_instruction(monkeypatch):
    """Railway and docker-compose emit "" for a declared-but-unset variable."""
    for name in (
        "KLIO_DB_POOL_TIMEOUT_S",
        "KLIO_DB_CONNECT_TIMEOUT_S",
        "KLIO_DB_STATEMENT_TIMEOUT_MS",
    ):
        monkeypatch.setenv(name, "   ")
    assert db.pool_timeout_s() == 10.0
    assert db.connect_timeout_s() == 10
    assert db.statement_timeout_ms() == 30_000


def test_an_out_of_range_knob_falls_back(monkeypatch):
    """A bound of zero or an hour is not a bound anyone meant to set."""
    monkeypatch.setenv("KLIO_DB_POOL_TIMEOUT_S", "0")
    monkeypatch.setenv("KLIO_DB_CONNECT_TIMEOUT_S", "9999")
    monkeypatch.setenv("KLIO_DB_STATEMENT_TIMEOUT_MS", "-1")
    assert db.pool_timeout_s() == 10.0
    assert db.connect_timeout_s() == 10
    assert db.statement_timeout_ms() == 30_000


def test_the_knobs_are_honoured_when_valid(monkeypatch):
    """Positive control: the guards must not swallow a real value."""
    monkeypatch.setenv("KLIO_DB_POOL_TIMEOUT_S", "2.5")
    monkeypatch.setenv("KLIO_DB_CONNECT_TIMEOUT_S", "3")
    monkeypatch.setenv("KLIO_DB_STATEMENT_TIMEOUT_MS", "45000")
    assert db.pool_timeout_s() == 2.5
    assert db.connect_timeout_s() == 3
    assert db.statement_timeout_ms() == 45_000
