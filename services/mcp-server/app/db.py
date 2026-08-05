"""Database engine and session factory for the MCP server.

Uses SQLAlchemy to manage PostgreSQL connections with a synchronous
engine, mirroring the storage worker. Exposes a ``get_db`` FastAPI
dependency that yields a request-scoped session and closes it after use.

Bounds
------
Every wait on the database is bounded, and each bound is configurable. The
pool was already *sized* (``pool_size=5, max_overflow=10``); what was missing
was any limit on how long something would wait:

* ``pool_timeout`` — how long a caller queues for a checkout when all 15 are
  busy. Left unset, SQLAlchemy's 30 s default meant a saturated pool held a
  request for half a minute before failing it.
* ``connect_timeout`` — how long opening a new connection waits. Left unset, an
  unreachable host is bounded only by the OS TCP timeout, which is minutes.
* ``statement_timeout`` — server-side, so one pathological query cannot hold
  its connection (and the worker thread running it) indefinitely. Applied as a
  ``SET`` on connect rather than as a libpq startup parameter: a PgBouncer
  transaction pooler (which is what Neon's pooled endpoint is) refuses any
  connection carrying an unrecognised startup parameter, so the startup-option
  form took production down while passing every test against a direct local
  Postgres.

``pool_pre_ping`` addresses none of these: it detects a connection that is
*dead*, not one that is *slow*, and slow is the failure mode that exhausts a
pool.

The knobs are parsed through :mod:`shared.env_config`, so a typo degrades to
the default with a WARNING rather than preventing import — a timeout must
never be the reason a service cannot boot.
"""

import logging
import os
import time

from shared import metrics
from shared.env_config import parse_float, parse_int
from sqlalchemy import create_engine, event
from sqlalchemy.exc import TimeoutError as SATimeoutError
from sqlalchemy.orm import sessionmaker

from app.config import DATABASE_URL

logger = logging.getLogger(__name__)

DEFAULT_POOL_TIMEOUT_S = 10.0
"""Seconds a caller may queue for a pooled connection.

Well below SQLAlchemy's 30 s default: a request that has already waited ten
seconds for a *connection* has no useful latency budget left, and failing fast
sheds load rather than accumulating it."""

DEFAULT_CONNECT_TIMEOUT_S = 10
"""Seconds to wait for a new TCP connection to Postgres."""

DEFAULT_STATEMENT_TIMEOUT_MS = 30_000
"""Server-side cap on a single statement, in milliseconds.

Generous for every query this service issues — the slowest is a pgvector
nearest-neighbour scan — while still bounding the pathological case."""


def pool_timeout_s() -> float:
    """Pool checkout timeout, from ``KLIO_DB_POOL_TIMEOUT_S``."""
    return parse_float(
        os.environ.get("KLIO_DB_POOL_TIMEOUT_S"),
        default=DEFAULT_POOL_TIMEOUT_S,
        name="KLIO_DB_POOL_TIMEOUT_S",
        minimum=0.1,
        maximum=60.0,
    )


def connect_timeout_s() -> int:
    """TCP connect timeout, from ``KLIO_DB_CONNECT_TIMEOUT_S``."""
    return parse_int(
        os.environ.get("KLIO_DB_CONNECT_TIMEOUT_S"),
        default=DEFAULT_CONNECT_TIMEOUT_S,
        name="KLIO_DB_CONNECT_TIMEOUT_S",
        minimum=1,
        maximum=60,
    )


def statement_timeout_ms() -> int:
    """Server-side statement timeout, from ``KLIO_DB_STATEMENT_TIMEOUT_MS``."""
    return parse_int(
        os.environ.get("KLIO_DB_STATEMENT_TIMEOUT_MS"),
        default=DEFAULT_STATEMENT_TIMEOUT_MS,
        name="KLIO_DB_STATEMENT_TIMEOUT_MS",
        minimum=100,
        maximum=300_000,
    )


#: Resolved once at import, because the engine is built once at import.
POOL_TIMEOUT_S = pool_timeout_s()
CONNECT_TIMEOUT_S = connect_timeout_s()
STATEMENT_TIMEOUT_MS = statement_timeout_ms()

CONNECT_ARGS = {
    "connect_timeout": CONNECT_TIMEOUT_S,
}
"""The libpq arguments handed to every new connection.

``statement_timeout`` is deliberately NOT here. libpq would forward it as a
startup parameter (``options: -c statement_timeout=...``), and a PgBouncer
transaction pooler rejects unknown startup parameters outright:

    ERROR: unsupported startup parameter in options: statement_timeout

We run against Neon's *pooled* endpoint, so setting it that way refused every
connection in production while passing every test against a direct local
Postgres. A bound that takes the database down is worse than no bound; it is
applied as an ordinary ``SET`` below instead, which the pooler accepts.

Named rather than inlined because SQLAlchemy does not expose an engine's
``connect_args`` for inspection, and these are exactly the settings that most
need to be verifiable."""

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_timeout=POOL_TIMEOUT_S,
    connect_args=CONNECT_ARGS,
)


@event.listens_for(engine, "connect")
def _apply_statement_timeout(dbapi_connection, _connection_record) -> None:
    """Bound each statement server-side, without a startup parameter.

    Issued as a plain ``SET`` on the fresh connection because that is what a
    transaction pooler tolerates. Two consequences worth stating:

    * On a direct connection this is session-wide and behaves exactly like the
      startup parameter it replaces.
    * Through a transaction pooler the setting lives on the *server* connection
      the pooler hands out, so it is best-effort rather than guaranteed for
      every client transaction. That is the strongest bound available on this
      topology, and it is strictly better than none.

    A failure here is logged and swallowed. This is a safety margin, and it
    must never become the reason a connection — and therefore a request — is
    refused. That is the exact failure this function exists to undo.
    """
    try:
        with dbapi_connection.cursor() as cursor:
            cursor.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
    except Exception:
        logger.warning(
            "could not apply statement_timeout=%s; continuing without it",
            STATEMENT_TIMEOUT_MS,
            exc_info=True,
        )


# ── pool instrumentation ──────────────────────────────────────────────
#
# Pool exhaustion is the failure that presents as "everything is slow": nothing
# errors, callers simply queue for a connection that never frees up. It is
# invisible in request-level metrics, and the bounds added above make it *fail*
# rather than hang without making it *visible*. These three listeners make it
# legible — depth, wait time, and give-ups.

#: Where a checkout's start time is stashed. On the connection record, so it
#: survives the trip through the pool and cannot collide across threads the way
#: a module-level dict would.
_CHECKOUT_STARTED_ATTR = "_klio_checkout_started"


def _on_checkout(_dbapi_connection, connection_record, _connection_proxy) -> None:
    """Raise the in-use gauge and record how long the checkout waited."""
    metrics.record(metrics.DB_POOL_IN_USE.inc)
    started = getattr(connection_record, _CHECKOUT_STARTED_ATTR, None)
    metrics.record(
        metrics.DB_POOL_WAIT_SECONDS.observe,
        0.0 if started is None else max(0.0, time.monotonic() - started),
    )
    setattr(connection_record, _CHECKOUT_STARTED_ATTR, time.monotonic())


def _on_checkin(_dbapi_connection, _connection_record) -> None:
    """Lower the in-use gauge when a connection returns to the pool."""
    metrics.record(metrics.DB_POOL_IN_USE.dec)


def instrument_pool(target_engine) -> None:
    """Attach the pool listeners to ``target_engine``.

    A function rather than decorators so a test can instrument a throwaway
    engine and assert on the arithmetic, instead of asserting against the real
    engine's live pool.
    """
    event.listen(target_engine, "checkout", _on_checkout)
    event.listen(target_engine, "checkin", _on_checkin)


def _raw_connect(target_engine):
    """The raw ``engine.connect()``, isolated so a test can make it fail.

    Distinguishing "the pool is exhausted" from "the database refused us" is
    the whole point of :func:`checkout_connection`, and the only way to exercise
    both branches is to control what connecting does.
    """
    return target_engine.connect()


def checkout_connection(target_engine=None):
    """Check out a connection, classifying a failure before re-raising it.

    Two failures look similar from a stack trace and mean opposite things:

    * ``TimeoutError`` — the pool is full. Every connection is in use and this
      caller queued past ``pool_timeout``. The fix is capacity or slower queries.
    * anything else (``OperationalError``, typically) — the database refused
      the connection. Nothing is exhausted; nothing can connect at all. This is
      the shape of the 2026-08-04 outage, where a transaction pooler rejected
      an unrecognised startup parameter on every single connection.

    Folding them into one counter would have shown "pool pressure" during an
    incident where the pool was entirely idle.
    """
    chosen = engine if target_engine is None else target_engine
    try:
        return _raw_connect(chosen)
    except SATimeoutError:
        metrics.record(metrics.DB_POOL_TIMEOUTS.inc)
        raise
    except Exception:
        metrics.record(metrics.DB_CONNECT_FAILURES.inc)
        raise


instrument_pool(engine)

SessionLocal = sessionmaker(bind=engine)


def get_db():
    """FastAPI dependency yielding a request-scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
