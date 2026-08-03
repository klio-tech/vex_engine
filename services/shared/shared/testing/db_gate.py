"""One place that decides whether a database-backed test runs, skips or fails.

Every tenancy-boundary suite in this repo needs a real Postgres — the boundary
IS the SQL, and a mocked session would pass against a query with no WHERE
clause at all. Each of those suites used to carry its own copy of::

    def _db_reachable() -> bool:
        try:
            ...
        except Exception:
            return False

    pytestmark = pytest.mark.skipif(not _db_reachable(), reason="...")

which is correct on a laptop and catastrophic in CI. A CI job with no Postgres
turned 112 adversarial tests — the entire graph, silo, compression-cache and
migration boundary suites — into skips and exited 0. The green check meant
"nothing ran", and nothing said so.

The fix is to make the *intent* explicit instead of inferring it from whether
a socket happened to connect:

* no ``KLIO_REQUIRE_DB`` (a developer's laptop) → unreachable Postgres skips,
  exactly as before;
* ``KLIO_REQUIRE_DB=1`` (CI sets this) → unreachable Postgres FAILS the test.

Usage — module-wide::

    from shared.testing import requires_db

    pytestmark = requires_db("the user-silo boundary tests")

or per test, for a module that also holds tests needing no database::

    needs_db = requires_db("the klio_retrieve boundary tests")

    @needs_db
    def test_cross_org_recall_returns_nothing(db): ...

This module is registered as a pytest plugin through vex-shared's ``pytest11``
entry point, so the marker is enforced anywhere vex-shared is installed —
including ``services/migrations``, which installs no package of its own.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional
from urllib.parse import urlsplit, urlunsplit
import os

import pytest
from sqlalchemy import create_engine, text

#: Matches docker-compose.yml. Only a fallback; CI and docker set DATABASE_URL.
DEFAULT_DATABASE_URL = (
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard"
)

#: Set this in CI. Its presence is the assertion "a database must be here".
REQUIRE_DB_ENV = "KLIO_REQUIRE_DB"

#: The pytest marker name registered by this plugin.
MARKER_NAME = "requires_db"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

RUN = "run"
SKIP = "skip"
FAIL = "fail"


def database_url() -> str:
    """The URL the database-backed suites connect to."""
    return os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL


def redacted_database_url(url: Optional[str] = None) -> str:
    """``url`` with any password removed, safe to print into a CI log."""
    parts = urlsplit(url if url is not None else database_url())
    if parts.password is None:
        return urlunsplit(parts)
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    netloc = f"{parts.username}:***@{host}" if parts.username else f"***@{host}"
    return urlunsplit(parts._replace(netloc=netloc))


def db_required() -> bool:
    """True when the environment asserts a database must be reachable."""
    return os.environ.get(REQUIRE_DB_ENV, "").strip().lower() in _TRUTHY


def gate_action(*, reachable: bool, required: bool) -> str:
    """The whole decision, as a pure function so it can be tested directly."""
    if reachable:
        return RUN
    return FAIL if required else SKIP


def probe(url: str) -> bool:
    """True if ``url`` accepts a connection and answers ``SELECT 1``."""
    try:
        engine = create_engine(url)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        finally:
            engine.dispose()
        return True
    except Exception:
        return False


@lru_cache(maxsize=None)
def _probe_cached(url: str) -> bool:
    return probe(url)


def db_reachable() -> bool:
    """True if the configured Postgres answers. Probed once per URL per run."""
    return _probe_cached(database_url())


def requires_db(reason: str):
    """Mark a test or module as needing the live Postgres.

    ``reason`` names the suite and is shown in both the skip and the failure,
    so a CI log says which boundary went untested rather than only that one
    did.
    """
    if not reason or not reason.strip():
        raise ValueError(
            "requires_db(reason) needs a reason naming the suite; it is what "
            "the CI log shows when the database is missing"
        )
    return getattr(pytest.mark, MARKER_NAME)(reason)


# ── pytest plugin hooks ───────────────────────────────────────────────


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        f"{MARKER_NAME}(reason): needs the live Postgres. Skips when it is "
        f"unreachable, unless {REQUIRE_DB_ENV} is set, in which case it fails.",
    )


def pytest_runtest_setup(item) -> None:
    marker = item.get_closest_marker(MARKER_NAME)
    if marker is None:
        return

    reason = marker.args[0] if marker.args else "this database-backed suite"
    action = gate_action(reachable=db_reachable(), required=db_required())
    if action == RUN:
        return

    where = redacted_database_url()
    if action == FAIL:
        pytest.fail(
            f"{REQUIRE_DB_ENV} is set, so {reason} must run — but Postgres is "
            f"unreachable at {where}. Skipping here would report a boundary as "
            f"tested when nothing tested it. Start the database, or unset "
            f"{REQUIRE_DB_ENV} to go back to skipping.",
            pytrace=False,
        )
    pytest.skip(
        f"Postgres unreachable at {where}; {reason} require it. "
        f"Set {REQUIRE_DB_ENV}=1 to make this a failure instead of a skip."
    )
