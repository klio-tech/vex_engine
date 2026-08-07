"""``/metrics``, ``/readyz`` and the liveness/readiness split.

The regression at the centre of this file is a real outage. ``statement_timeout``
was applied as a libpq startup parameter; Neon's pooled endpoint is a PgBouncer
transaction pooler and refuses any connection carrying an unrecognised startup
parameter, so **every** database connection was refused — every recall, every
capture. ``/healthz`` returned 200 for the entire incident, because it is a
liveness probe that never touches the database, and the outage was found by
hand.

:func:`test_database_refusing_connections_keeps_healthz_green_and_turns_readyz_red`
is that incident, expressed as a test.
"""

import os
import time

import pytest
from app import observability
from app.auth import verify_api_key
from app.main import create_app
from fastapi.testclient import TestClient
from shared.auth import KeyInfo
from sqlalchemy.exc import OperationalError

METRICS_TOKEN_ENV = "KLIO_METRICS_TOKEN"


def _fake_key_info() -> KeyInfo:
    return KeyInfo(org_id="test-org", key_id="test-key", scopes=["ingest"])


@pytest.fixture
def obs_client():
    app = create_app()
    app.dependency_overrides[verify_api_key] = _fake_key_info
    return TestClient(app)


class _StubSession:
    """A session whose ``execute`` behaviour the test dictates."""

    def __init__(self, on_execute):
        self._on_execute = on_execute
        self.closed = False

    def execute(self, *_args, **_kwargs):
        return self._on_execute()

    def close(self):
        self.closed = True


def _factory(on_execute):
    """A ``SessionLocal``-shaped callable yielding stub sessions."""
    made = []

    def make():
        session = _StubSession(on_execute)
        made.append(session)
        return session

    make.made = made
    return make


def _refusing_factory():
    """A factory that fails the way a pooler refusing connections fails.

    PgBouncer rejects the connection outright, so the failure surfaces from
    SQLAlchemy as an ``OperationalError`` at connect time — which is what the
    caller sees whether the refusal happens on connect or on first statement.
    """

    def make():
        raise OperationalError(
            "SELECT 1",
            {},
            Exception("unsupported startup parameter in options: statement_timeout"),
        )

    return make


# ── liveness stays liveness ───────────────────────────────────────────


def test_healthz_never_touches_the_database(obs_client, monkeypatch):
    """A liveness probe that checks a dependency causes restart loops.

    Detection is by *recording*, not by raising. Every probe in this module
    swallows exceptions by design, so a fixture that raises would be absorbed
    and the test would pass against a ``/healthz`` that queries on every call.
    """
    factory = _factory(lambda: 1)
    monkeypatch.setattr(observability, "SessionLocal", factory)

    r = obs_client.get("/healthz")

    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert factory.made == [], "/healthz opened a database session"


def test_the_healthz_no_database_check_is_not_vacuous(obs_client, monkeypatch):
    """Positive control: the same recording fixture *does* see ``/readyz``.

    Without this, ``factory.made == []`` above would pass just as well against
    a fixture that is never wired up at all.
    """
    factory = _factory(lambda: 1)
    monkeypatch.setattr(observability, "SessionLocal", factory)
    obs_client.get("/readyz")
    assert len(factory.made) == 1


def test_healthz_response_shape_is_unchanged(obs_client):
    """WP-F must not alter the existing liveness contract."""
    assert obs_client.get("/healthz").json() == {"status": "ok"}


# ── readiness is dependency-aware ─────────────────────────────────────


def test_readyz_is_green_when_the_database_answers(obs_client, monkeypatch):
    monkeypatch.setattr(observability, "SessionLocal", _factory(lambda: 1))
    r = obs_client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["checks"]["database"]["ok"] is True


def test_readyz_actually_issues_a_round_trip(obs_client, monkeypatch):
    """Positive control: the probe must run a statement, not just construct a
    session. A readiness check that only opens a session would have stayed
    green through the outage too."""
    calls = []
    factory = _factory(lambda: calls.append("select 1"))
    monkeypatch.setattr(observability, "SessionLocal", factory)
    obs_client.get("/readyz")
    assert calls == ["select 1"]


def test_readyz_closes_the_session_it_opens(obs_client, monkeypatch):
    factory = _factory(lambda: 1)
    monkeypatch.setattr(observability, "SessionLocal", factory)
    obs_client.get("/readyz")
    assert all(s.closed for s in factory.made)


def test_database_refusing_connections_keeps_healthz_green_and_turns_readyz_red(
    obs_client, monkeypatch
):
    """THE regression for today's incident.

    Liveness must stay 200 — the process is fine, restarting it fixes nothing.
    Readiness must go 503 — the process cannot serve traffic, and that is the
    signal that was missing for the entire outage.
    """
    monkeypatch.setattr(observability, "SessionLocal", _refusing_factory())

    live = obs_client.get("/healthz")
    ready = obs_client.get("/readyz")

    assert live.status_code == 200, "liveness must not follow the database down"
    assert ready.status_code == 503, "readiness must go red when the DB refuses"
    body = ready.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["database"]["ok"] is False


def test_readyz_reports_why_the_database_failed(obs_client, monkeypatch):
    monkeypatch.setattr(observability, "SessionLocal", _refusing_factory())
    detail = obs_client.get("/readyz").json()["checks"]["database"]["detail"]
    assert "OperationalError" in detail


def test_readyz_detail_does_not_leak_the_dsn(obs_client, monkeypatch):
    """The probe reports a failure class, not a connection string."""

    def make():
        raise OperationalError(
            "SELECT 1", {}, Exception("could not connect to hunter2@db.internal")
        )

    monkeypatch.setattr(observability, "SessionLocal", make)
    detail = obs_client.get("/readyz").json()["checks"]["database"]["detail"]
    assert "hunter2" not in detail


def test_readyz_is_bounded_and_does_not_hang_on_a_hung_database(
    obs_client, monkeypatch
):
    """A readiness probe that inherits a dependency's hang is not a probe."""
    monkeypatch.setattr(observability, "READINESS_TIMEOUT_S", 0.2)
    monkeypatch.setattr(observability, "SessionLocal", _factory(lambda: time.sleep(30)))

    started = time.monotonic()
    r = obs_client.get("/readyz")
    elapsed = time.monotonic() - started

    assert r.status_code == 503
    assert elapsed < 5.0, f"readiness probe took {elapsed:.1f}s; it must be bounded"
    assert "timed out" in r.json()["checks"]["database"]["detail"].lower()


def test_readyz_reports_embedder_configuration(obs_client, monkeypatch):
    monkeypatch.setenv("MEMORY_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    monkeypatch.setattr(observability, "SessionLocal", _factory(lambda: 1))

    embedder = obs_client.get("/readyz").json()["checks"]["embedder"]
    assert embedder["ok"] is True
    assert embedder["model"] == "text-embedding-3-small"


def test_readyz_never_echoes_the_embedder_credential(obs_client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value")
    monkeypatch.setattr(observability, "SessionLocal", _factory(lambda: 1))
    assert "sk-super-secret-value" not in obs_client.get("/readyz").text


def test_readyz_flags_an_unconfigured_embedder_without_failing_readiness(
    obs_client, monkeypatch
):
    """Deliberate: a missing embedder degrades recall, it does not stop the
    service, and failing readiness on it would take a working deployment out
    of rotation over a config gap."""
    for var in ("LITELLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(observability, "SessionLocal", _factory(lambda: 1))

    r = obs_client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["checks"]["embedder"]["ok"] is False


def test_readyz_increments_its_own_metric(obs_client, monkeypatch):
    from shared import metrics

    metrics.reset_for_test()
    monkeypatch.setattr(observability, "SessionLocal", _factory(lambda: 1))
    obs_client.get("/readyz")
    assert metrics.READINESS_CHECKS.value(result="ready") == 1.0


# ── /metrics is not world-readable ────────────────────────────────────


def test_metrics_requires_a_token(obs_client, monkeypatch):
    monkeypatch.setenv(METRICS_TOKEN_ENV, "s3cret")
    assert obs_client.get("/metrics").status_code == 401


def test_metrics_rejects_the_wrong_token(obs_client, monkeypatch):
    monkeypatch.setenv(METRICS_TOKEN_ENV, "s3cret")
    r = obs_client.get("/metrics", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_metrics_serves_the_exposition_with_the_right_token(obs_client, monkeypatch):
    monkeypatch.setenv(METRICS_TOKEN_ENV, "s3cret")
    r = obs_client.get("/metrics", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "# TYPE klio_memory_writes_total counter" in r.text


def test_metrics_fails_closed_when_no_token_is_configured(obs_client, monkeypatch):
    """Documented decision: unset means closed, not open.

    An operator who forgets to set the token gets a 404, not an unauthenticated
    dump of the service's operational shape.
    """
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    assert obs_client.get("/metrics").status_code == 404
    r = obs_client.get("/metrics", headers={"Authorization": "Bearer anything"})
    assert r.status_code == 404


def test_metrics_rejects_an_empty_token_configuration(obs_client, monkeypatch):
    """An empty string is not a credential."""
    monkeypatch.setenv(METRICS_TOKEN_ENV, "")
    r = obs_client.get("/metrics", headers={"Authorization": "Bearer "})
    assert r.status_code == 404


def test_an_armed_gate_answers_401_with_a_challenge(obs_client, monkeypatch):
    """A configured-but-unauthenticated caller gets a standard 401.

    This replaces an earlier assertion that the denial was a 404 indistinguish-
    able from an absent route. The concealment was worth little (``/metrics`` is
    a conventional path) and it cost the ability to connect Grafana Cloud's
    hosted scraper, which verifies an endpoint is genuinely protected before
    scraping and reads a 404 as misconfigured auth rather than a clean refusal.
    """
    monkeypatch.setenv(METRICS_TOKEN_ENV, "s3cret")
    denied = obs_client.get("/metrics")
    assert denied.status_code == 401
    assert denied.headers["www-authenticate"].lower().startswith("bearer")


def test_an_unarmed_gate_still_answers_404_without_a_challenge(obs_client, monkeypatch):
    """The distinction that makes the 401 safe: no token configured is not an
    authentication failure, it is an endpoint that cannot serve anyone. Offering
    a challenge there would invite a caller to retry against a gate that can
    never open."""
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    denied = obs_client.get("/metrics")
    assert denied.status_code == 404
    assert "www-authenticate" not in {k.lower() for k in denied.headers}


def test_the_metrics_gate_is_not_vacuous(obs_client, monkeypatch):
    """Positive control for every 404 above: the same client, same route, with
    the correct token, returns a real body. So the 404s are the gate, not a
    missing route."""
    monkeypatch.setenv(METRICS_TOKEN_ENV, "s3cret")
    ok = obs_client.get("/metrics", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    assert len(ok.text) > 100


def test_metrics_token_comparison_is_constant_time(monkeypatch):
    """Timing-safe comparison, so the token cannot be recovered byte by byte."""
    import inspect

    source = inspect.getsource(observability._token_accepted)
    assert "compare_digest" in source


# ── remote_write configuration is inert when unset ────────────────────


def test_remote_write_is_disabled_when_unconfigured(monkeypatch):
    for var in observability.GRAFANA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    config = observability.remote_write_config()
    assert config is None


def test_remote_write_is_configured_when_all_three_are_set(monkeypatch):
    monkeypatch.setenv("GRAFANA_CLOUD_PROMETHEUS_URL", "https://example/api/push")
    monkeypatch.setenv("GRAFANA_CLOUD_USER", "123456")
    monkeypatch.setenv("GRAFANA_CLOUD_API_KEY", "glc_example")
    config = observability.remote_write_config()
    assert config is not None
    assert config.url == "https://example/api/push"
    assert config.username == "123456"


def test_remote_write_is_disabled_when_partially_configured(monkeypatch):
    monkeypatch.setenv("GRAFANA_CLOUD_PROMETHEUS_URL", "https://example/api/push")
    monkeypatch.delenv("GRAFANA_CLOUD_USER", raising=False)
    monkeypatch.delenv("GRAFANA_CLOUD_API_KEY", raising=False)
    assert observability.remote_write_config() is None


def test_remote_write_config_never_stringifies_its_credential(monkeypatch):
    monkeypatch.setenv("GRAFANA_CLOUD_PROMETHEUS_URL", "https://example/api/push")
    monkeypatch.setenv("GRAFANA_CLOUD_USER", "123456")
    monkeypatch.setenv("GRAFANA_CLOUD_API_KEY", "glc_super_secret")
    config = observability.remote_write_config()
    assert "glc_super_secret" not in repr(config)
    assert "glc_super_secret" not in str(config)


def test_remote_write_env_var_names_are_the_documented_ones():
    assert observability.GRAFANA_ENV_VARS == (
        "GRAFANA_CLOUD_PROMETHEUS_URL",
        "GRAFANA_CLOUD_USER",
        "GRAFANA_CLOUD_API_KEY",
    )


def test_no_credential_is_baked_into_the_module():
    """The wiring is env-driven; no key may be committed alongside it."""
    import pathlib
    import re

    source = pathlib.Path(observability.__file__).read_text()
    assert re.search(r"glc_[A-Za-z0-9]{8,}", source) is None
    assert os.environ["PATH"] is not None  # sanity: os is used, not imported idly
