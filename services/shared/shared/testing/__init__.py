"""Test-support helpers shared across every service's suite.

Nothing in here is imported by production code. It lives inside the ``shared``
package so that the one copy is reachable from ``services/shared/tests``,
``services/mcp-server/tests`` and ``services/migrations/tests`` alike — the
alternative was the same helper pasted into eight files, which is how the
database gate drifted into silently skipping in the first place.
"""

from shared.testing.db_gate import (
    DEFAULT_DATABASE_URL,
    REQUIRE_DB_ENV,
    database_url,
    db_reachable,
    db_required,
    redacted_database_url,
    requires_db,
)

__all__ = [
    "DEFAULT_DATABASE_URL",
    "REQUIRE_DB_ENV",
    "database_url",
    "db_reachable",
    "db_required",
    "redacted_database_url",
    "requires_db",
]
