"""Operational surface: ``/metrics``, ``/readyz``, and remote_write wiring.

Liveness is not readiness
-------------------------
``/healthz`` (in :mod:`app.main`) answers "is this process alive?" and touches
nothing. ``/readyz`` answers "can this process serve traffic?" and therefore
must actually reach its dependencies. Conflating them is not a style question:
a liveness probe that fails on a database outage causes the orchestrator to
restart every replica, repeatedly, while the database is the thing that is
broken — a restart loop layered on top of an outage.

The incident this module exists for
-----------------------------------
``statement_timeout`` was once applied as a libpq startup parameter. Neon's
pooled endpoint is a PgBouncer transaction pooler, and a transaction pooler
refuses any connection carrying an unrecognised startup parameter. Every
database connection was refused; every recall and every capture failed. For the
entire incident ``/healthz`` returned 200, because it never touched the
database, so nothing went red and the outage was found by hand.

:func:`probe_database` issues a real ``SELECT 1``. That failure mode now turns
``/readyz`` 503 while ``/healthz`` stays 200, which is exactly the pair of
signals that was missing.

Bounding the probe
------------------
The probe runs in a worker thread under :data:`READINESS_TIMEOUT_S`. A readiness
check that inherits a hung dependency's hang is not a check — the orchestrator's
own probe timeout fires and the result is indistinguishable from the process
being wedged. On timeout the probe is abandoned (a blocked thread cannot be
cancelled; it will unblock when its own socket timeout fires and exit) and
readiness reports red immediately.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import anyio.to_thread
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from shared import metrics
from sqlalchemy import text

from app.db import SessionLocal

logger = logging.getLogger(__name__)

router = APIRouter()

#: Seconds the readiness probe may spend on its database round-trip.
#:
#: Short on purpose. This is not a query budget; it is the answer to "is the
#: database reachable right now?", and a database that has not answered
#: ``SELECT 1`` in three seconds is not one this service can serve from.
READINESS_TIMEOUT_S = 3.0

#: Env var holding the bearer token that guards ``/metrics``.
METRICS_TOKEN_ENV = "KLIO_METRICS_TOKEN"

#: The three env vars that configure Prometheus remote_write to Grafana Cloud.
GRAFANA_ENV_VARS = (
    "GRAFANA_CLOUD_PROMETHEUS_URL",
    "GRAFANA_CLOUD_USER",
    "GRAFANA_CLOUD_API_KEY",
)


# ── readiness ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CheckResult:
    """One dependency's verdict. Immutable; probes return, they do not mutate."""

    ok: bool
    detail: str
    #: Extra non-sensitive facts for the response body (e.g. the model name).
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Render for the JSON body."""
        return {"ok": self.ok, "detail": self.detail, **self.extra}


@dataclass(frozen=True)
class Readiness:
    """The whole verdict: overall state plus each dependency's result."""

    ready: bool
    checks: dict[str, CheckResult]

    def as_dict(self) -> dict[str, Any]:
        """Render for the JSON body."""
        return {
            "status": "ready" if self.ready else "not_ready",
            "checks": {name: c.as_dict() for name, c in self.checks.items()},
        }


def _failure_detail(exc: BaseException) -> str:
    """Describe a failure by class, never by message.

    A driver's exception message routinely carries the DSN, and the DSN carries
    the password. The exception *type* is what an operator needs to distinguish
    "refused" from "timed out" from "authentication failed"; the message is a
    credential-leak waiting for someone to paste a probe response into a ticket.
    The full exception, message and all, goes to the log instead.
    """
    return type(exc).__name__


def probe_database(session_factory: Optional[Callable[[], Any]] = None) -> CheckResult:
    """Issue a real ``SELECT 1``. Synchronous; call from a worker thread.

    Opening a session is not enough — SQLAlchemy pools lazily, so a factory call
    can succeed against a database that refuses every connection. Only a
    statement proves the round-trip, and only the round-trip would have caught
    the pooler outage.
    """
    factory = session_factory or SessionLocal
    session = None
    try:
        session = factory()
        session.execute(text("SELECT 1"))
        return CheckResult(ok=True, detail="round-trip ok")
    except Exception as exc:
        logger.warning("readiness: database round-trip failed", exc_info=True)
        return CheckResult(ok=False, detail=f"{_failure_detail(exc)}")
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                logger.debug(
                    "readiness: closing the probe session failed", exc_info=True
                )


def probe_embedder() -> CheckResult:
    """Report the embedder's *configuration* state. No network call.

    Deliberately non-fatal for readiness. A missing embedding credential
    degrades recall quality; it does not stop the service, and failing
    readiness on it would pull a working deployment out of rotation over a
    config gap. It is reported so the gap is visible rather than silent.

    The credential is never read into the response — only whether one exists,
    and the model name, which is configuration rather than a secret.
    """
    model = os.environ.get("MEMORY_EMBEDDING_MODEL") or "default"
    has_key = bool(
        os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )
    return CheckResult(
        ok=has_key,
        detail="credential present"
        if has_key
        else "no embedding credential configured",
        extra={"model": model},
    )


async def check_readiness(
    session_factory: Optional[Callable[[], Any]] = None,
    timeout_s: Optional[float] = None,
) -> Readiness:
    """Run every readiness probe, bounded, and combine them.

    Only the database is fatal. See :func:`probe_embedder` for why the embedder
    is reported but does not fail the probe.
    """
    budget = READINESS_TIMEOUT_S if timeout_s is None else timeout_s
    try:
        database = await asyncio.wait_for(
            anyio.to_thread.run_sync(lambda: probe_database(session_factory)),
            timeout=budget,
        )
    except (asyncio.TimeoutError, TimeoutError):
        # The worker thread is abandoned rather than cancelled — a thread
        # blocked in the driver cannot be interrupted. It exits on its own
        # socket timeout. What matters is that the *probe* answered on time.
        logger.warning("readiness: database probe timed out after %.1fs", budget)
        database = CheckResult(ok=False, detail=f"timed out after {budget:g}s")

    checks = {"database": database, "embedder": probe_embedder()}
    ready = database.ok
    metrics.READINESS_CHECKS.inc(result="ready" if ready else "not_ready")
    return Readiness(ready=ready, checks=checks)


@router.get("/readyz")
async def readyz() -> Response:
    """Dependency-aware readiness. 200 when servable, 503 when not."""
    result = await check_readiness()
    return JSONResponse(
        status_code=200 if result.ready else 503,
        content=result.as_dict(),
    )


# ── /metrics ──────────────────────────────────────────────────────────


def _configured_token() -> Optional[str]:
    """The configured ``/metrics`` bearer token, or ``None`` if there is none.

    An empty or whitespace-only value is treated as absent. An empty string is
    not a credential, and accepting one would make ``KLIO_METRICS_TOKEN=``
    silently equivalent to no gate at all.
    """
    raw = os.environ.get(METRICS_TOKEN_ENV)
    return raw.strip() or None if raw else None


def _token_accepted(header: Optional[str]) -> bool:
    """Whether the ``Authorization`` header carries the configured token.

    Compared with :func:`hmac.compare_digest`, so the response time does not
    reveal how many leading bytes of a guess were right.
    """
    expected = _configured_token()
    if expected is None:
        return False
    if not header or not header.lower().startswith("bearer "):
        return False
    presented = header[len("bearer ") :].strip()
    return hmac.compare_digest(presented, expected)


@router.get("/metrics")
def serve_metrics(request: Request) -> Response:
    """Prometheus text exposition, gated by a bearer token.

    **Not world-readable.** The exposition describes the service's operational
    shape — write and recall volume, error and quota rates, curator progress,
    pool pressure. That is reconnaissance for an attacker and commercial
    information for anyone else, so it is behind ``KLIO_METRICS_TOKEN``.

    **The unset case fails closed, deliberately.** With no token configured the
    endpoint answers 404 for everyone and logs a warning once per request-less
    startup path. The alternative — serving openly until someone remembers to
    set the variable — makes the safe state the one an operator has to opt into,
    and the default state a silent leak. Binding to localhost was rejected
    because the deployment target (Railway) exposes a single port and offers no
    second, private interface to bind to, so "bind-local" would be a comment
    rather than a control.

    **404, not 401.** A denied caller learns nothing, not even that the endpoint
    exists. There is no ``WWW-Authenticate`` challenge: this endpoint is for a
    scraper holding a pre-shared token, not for a browser to negotiate with.
    """
    if not _token_accepted(request.headers.get("authorization")):
        if _configured_token() is None:
            logger.warning(
                "%s is not set; /metrics is closed. Set it to enable scraping.",
                METRICS_TOKEN_ENV,
            )
        return PlainTextResponse("Not Found", status_code=404)
    return PlainTextResponse(
        metrics.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# ── remote_write to Grafana Cloud ─────────────────────────────────────


@dataclass(frozen=True)
class RemoteWriteConfig:
    """Grafana Cloud remote_write settings, read from the environment.

    The API key is deliberately excluded from ``repr``: a config object ends up
    in log lines and exception context, and a credential that can be printed
    eventually is printed.
    """

    url: str
    username: str
    api_key: str = field(repr=False)

    def __str__(self) -> str:
        """Never render the credential, whatever the caller does with us."""
        return f"RemoteWriteConfig(url={self.url!r}, username={self.username!r})"


def remote_write_config() -> Optional[RemoteWriteConfig]:
    """The remote_write configuration, or ``None`` when it is not configured.

    Inert unless all three of :data:`GRAFANA_ENV_VARS` are set. Partial
    configuration returns ``None`` rather than half-enabling a shipper: a URL
    with no credential would retry-loop against a 401 forever, and the
    resulting noise is worse than no metrics shipping.

    This returns configuration only. Nothing in this repository ships samples
    on its own — see ``docs/observability/grafana-cloud.md`` for the agent-side
    scrape config that consumes ``/metrics``.
    """
    url, username, api_key = (os.environ.get(v) for v in GRAFANA_ENV_VARS)
    if not (url and username and api_key):
        return None
    return RemoteWriteConfig(url=url, username=username, api_key=api_key)


def log_remote_write_state() -> None:
    """Log, once at startup, whether metrics shipping is configured.

    Only ever logs the URL and username. The API key is never logged, at any
    level, under any configuration.
    """
    config = remote_write_config()
    if config is None:
        logger.info(
            "Prometheus remote_write is not configured (%s unset); "
            "/metrics is still served for local scraping",
            ", ".join(GRAFANA_ENV_VARS),
        )
    else:
        logger.info("Prometheus remote_write configured for %s", config.url)
