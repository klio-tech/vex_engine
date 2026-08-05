"""Shared connection bounds for every long-lived SQLAlchemy engine.

Why this module exists
----------------------
``mcp-server/app/db.py`` bounds its waits (``pool_timeout``,
``connect_timeout``, ``pool_recycle``). Six other long-lived engines did not,
and three of them — the API-key resolver in :mod:`shared.auth`, the OAuth grant
resolver, and the directory lookup — sit on the request path. An unreachable
database on any of those was bounded only by the OS TCP timeout, which is
minutes, so a network partition presented as hung requests rather than as
errors.

``pool_pre_ping`` does not cover this. It detects a connection that is *dead*
before handing it out; it says nothing about how long *opening* one may take,
and nothing about a connection that is merely slow.

What is deliberately NOT here
-----------------------------
``statement_timeout``. libpq would carry it as a startup parameter, and Neon's
pooled endpoint is a PgBouncer transaction pooler that refuses any connection
carrying an unrecognised one — which took production down on 2026-08-04 while
every test passed against a direct local Postgres. Where a statement bound is
wanted it is applied as a plain ``SET`` on connect; see ``mcp-server/app/db.py``.
Keeping this module to settings that are safe on a *pooled* endpoint is the
point of centralising it.
"""

import os

from shared.env_config import parse_int

DEFAULT_CONNECT_TIMEOUT_S = 10
"""Seconds to wait for a new TCP connection before failing.

Ten seconds because a caller that has waited this long for a *connection* has
no useful latency budget left, and failing fast sheds load instead of
accumulating it."""

DEFAULT_POOL_RECYCLE_S = 3600
"""Seconds before a pooled connection is retired and reopened.

Matches the value ``mcp-server/app/db.py``, ``alert-service`` and
``storage-worker`` already use. Neon closes idle connections server-side, so
without this a long-idle pool fills with connections that are discovered dead
only when someone tries to use them."""


def connect_timeout_s() -> int:
    """TCP connect timeout, from ``KLIO_DB_CONNECT_TIMEOUT_S``.

    Guarded so a typo degrades to the default with a WARNING: a timeout must
    never be the reason a service cannot import.
    """
    return parse_int(
        os.environ.get("KLIO_DB_CONNECT_TIMEOUT_S"),
        default=DEFAULT_CONNECT_TIMEOUT_S,
        name="KLIO_DB_CONNECT_TIMEOUT_S",
        minimum=1,
        maximum=60,
    )


def pool_recycle_s() -> int:
    """Pooled-connection lifetime, from ``KLIO_DB_POOL_RECYCLE_S``."""
    return parse_int(
        os.environ.get("KLIO_DB_POOL_RECYCLE_S"),
        default=DEFAULT_POOL_RECYCLE_S,
        name="KLIO_DB_POOL_RECYCLE_S",
        minimum=60,
        maximum=86_400,
    )


def engine_bounds() -> dict:
    """Keyword arguments to splat into ``create_engine`` for a long-lived engine.

    Returns ``pool_recycle`` and ``connect_args``. Sizing (``pool_size``,
    ``max_overflow``) is deliberately left to each call site, because the right
    size depends on what the engine is for — a per-request auth lookup and a
    batch worker want different pools, and a shared default would be wrong for
    both.

    Read at call time rather than at import so an operator can change the
    bounds without a redeploy.
    """
    return {
        "pool_recycle": pool_recycle_s(),
        "connect_args": {"connect_timeout": connect_timeout_s()},
    }
