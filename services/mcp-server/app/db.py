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
  its connection (and the worker thread running it) indefinitely.

``pool_pre_ping`` addresses none of these: it detects a connection that is
*dead*, not one that is *slow*, and slow is the failure mode that exhausts a
pool.

The knobs are parsed through :mod:`shared.env_config`, so a typo degrades to
the default with a WARNING rather than preventing import — a timeout must
never be the reason a service cannot boot.
"""

import os

from shared.env_config import parse_float, parse_int
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import DATABASE_URL

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
    # libpq forwards ``options`` to the backend, so this is the *server*
    # enforcing the cap rather than the client hoping to.
    "options": f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
}
"""The libpq arguments handed to every new connection.

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
SessionLocal = sessionmaker(bind=engine)


def get_db():
    """FastAPI dependency yielding a request-scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
