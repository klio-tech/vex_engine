"""Every long-lived engine bounds how long it will wait.

The defect these tests pin is an *absence*, which is why they assert on the
engine objects rather than on behaviour: an unbounded connect attempt cannot be
demonstrated in a unit test without an unroutable host and a multi-minute wait.
Pinning the setting is the only cheap, deterministic proof available, so each
test states what the missing setting costs in production.
"""

import pytest
from shared import engine_bounds
from sqlalchemy import create_engine


def test_the_bounds_include_a_connect_timeout():
    """Without it, an unreachable host is bounded only by the OS TCP timeout."""
    bounds = engine_bounds.engine_bounds()
    assert (
        bounds["connect_args"]["connect_timeout"] == engine_bounds.connect_timeout_s()
    )
    assert 0 < bounds["connect_args"]["connect_timeout"] <= 60


def test_the_bounds_include_a_pool_recycle():
    """Neon closes idle connections; a stale pool discovers this the hard way."""
    bounds = engine_bounds.engine_bounds()
    assert bounds["pool_recycle"] == engine_bounds.pool_recycle_s()
    assert bounds["pool_recycle"] > 0


def test_the_bounds_never_carry_a_startup_parameter():
    """Regression guard for the 2026-08-04 outage.

    ``connect_args["options"]`` becomes a libpq *startup* parameter, and Neon's
    pooled endpoint (a PgBouncer transaction pooler) refuses the whole
    connection when it does not recognise one. Production connects to that
    endpoint; the tests connect to a direct local Postgres that accepts it. So
    this must be pinned as a *form*, because no local test can catch it.
    """
    assert "options" not in engine_bounds.engine_bounds()["connect_args"], (
        "a pooled endpoint rejects connections carrying startup parameters; "
        "apply session settings with a SET on connect instead"
    )


def test_the_bounds_are_accepted_by_create_engine():
    """Positive control: the kwargs are real, not merely a dict of hopes."""
    engine = create_engine(
        "postgresql://u:p@localhost:1/does_not_matter", **engine_bounds.engine_bounds()
    )
    assert engine.pool._recycle == engine_bounds.pool_recycle_s()


@pytest.mark.parametrize(
    "env,fn,default",
    [
        ("KLIO_DB_CONNECT_TIMEOUT_S", "connect_timeout_s", 10),
        ("KLIO_DB_POOL_RECYCLE_S", "pool_recycle_s", 3600),
    ],
)
def test_a_malformed_knob_falls_back_instead_of_raising(monkeypatch, env, fn, default):
    """A bad timeout must not be the reason a service cannot import."""
    for bad in ("ten", "", "   ", "-1", "99999999"):
        monkeypatch.setenv(env, bad)
        assert getattr(engine_bounds, fn)() == default, f"{env}={bad!r} was honoured"


@pytest.mark.parametrize(
    "env,fn,value",
    [
        ("KLIO_DB_CONNECT_TIMEOUT_S", "connect_timeout_s", 5),
        ("KLIO_DB_POOL_RECYCLE_S", "pool_recycle_s", 900),
    ],
)
def test_a_valid_knob_is_honoured(monkeypatch, env, fn, value):
    """Positive control: the fallback above is a fallback, not a hardcode."""
    monkeypatch.setenv(env, str(value))
    assert getattr(engine_bounds, fn)() == value


def test_the_api_key_resolver_bounds_both_of_its_engines(monkeypatch):
    """The bounds are worthless if a call site forgets them.

    ``shared.auth`` runs on every authenticated request and opens two engines —
    the memory database and Supabase. Asserted by capturing the kwargs rather
    than by inspecting the built engine, because SQLAlchemy does not expose an
    engine's ``connect_args`` for reading; the call is the only observable.
    """
    import shared.auth as auth_module

    captured = []

    def fake_create_engine(url, **kwargs):
        captured.append(kwargs)
        return create_engine("postgresql://u:p@localhost:1/x")

    monkeypatch.setattr(auth_module, "create_engine", fake_create_engine)
    auth_module.KeyValidator(
        database_url="postgresql://u:p@localhost:1/x",
        required_scope="memory",
        supabase_database_url="postgresql://u:p@localhost:1/y",
    )

    assert len(captured) == 2, "expected the memory engine and the Supabase engine"
    for kwargs in captured:
        assert kwargs["pool_recycle"] == engine_bounds.pool_recycle_s()
        assert kwargs["connect_args"]["connect_timeout"] > 0
        # Sizing stays a per-call-site decision; the bounds must not erase it.
        assert "pool_size" in kwargs
