"""Brain session-id derivation for the MCP write tools.

The Klio brain stores every memory under a ``session_id``. Unlike the
verification path (where the session id is a real conversation id), brain
writes need a *stable* id that groups a client's writes so deduplication can
work across calls within the same connection.

The Streamable-HTTP transport assigns an ``mcp-session-id`` header per
connection, so we key on that when present. When it is absent (e.g. a client
that does not negotiate a session, or the in-memory transport), we fall back to
the principal ``org_id:agent_id`` so dedup stays stable per principal rather
than scattering across calls.

``session_id`` has no foreign key, so any string is acceptable — but it is
``VARCHAR(256)``, and the header is client input. Executed against Postgres 16::

    mcp-session-id header, 400 chars -> session_id len 411
    INSERT -> (psycopg2.errors.StringDataRightTruncation)
              value too long for type character varying(256)

The ``org_id:agent_id`` fallback was filed as overflowing the same column. It
does not: worst case is ``11 + 64 + 1 + 128 == 204``, executed and confirmed
inside 256. Only the header path is bounded here.
"""

import hashlib
from typing import Any

_BRAIN_SESSION_PREFIX = "klio-brain"

#: Matches ``session_memories.session_id`` (migration 014 widened it to 256).
MAX_BRAIN_SESSION_ID_LEN = 256

#: See ``shared.auth.normalize_agent_id`` for why a digest rather than a plain
#: truncation: two long headers truncating to the same prefix would silently
#: pool two connections into one dedup bucket.
_DIGEST_LEN = 12
_SEPARATOR = "~"


def _bounded_suffix(suffix: str) -> str:
    """Bound the session-id suffix so the full id fits the column."""
    budget = MAX_BRAIN_SESSION_ID_LEN - len(_BRAIN_SESSION_PREFIX) - 1
    if len(suffix) <= budget:
        return suffix
    digest = hashlib.sha256(suffix.encode("utf-8")).hexdigest()[:_DIGEST_LEN]
    keep = budget - len(_SEPARATOR) - _DIGEST_LEN
    return f"{suffix[:keep]}{_SEPARATOR}{digest}"


def brain_session_id(request: Any, org_id: str, agent_id: str) -> str:
    """Derive a stable brain ``session_id`` for the current connection.

    Args:
        request: The transport request; only ``request.headers.get(...)`` is
            read, so any dict-like-header carrier works.
        org_id: Resolved org principal (fallback key component).
        agent_id: Resolved agent principal (fallback key component).

    Returns:
        A ``"klio-brain:<suffix>"`` id of at most
        :data:`MAX_BRAIN_SESSION_ID_LEN` characters, where ``<suffix>`` is the
        ``mcp-session-id`` header when present, else ``"<org_id>:<agent_id>"``.
    """
    # The client-supplied ``mcp-session-id`` is trusted only as a dedup
    # *grouping* key, never for authorization: tenancy is enforced separately
    # by ``org_id`` (and ``agent_id``), which are filtered on every read/write.
    # A spoofed value can at most pool a client's own writes into a different
    # dedup bucket within its own org — it cannot cross the tenant boundary.
    # It is still client input reaching a bounded column, so it is bounded.
    mcp_sid = request.headers.get("mcp-session-id")
    suffix = mcp_sid if mcp_sid else f"{org_id}:{agent_id}"
    return f"{_BRAIN_SESSION_PREFIX}:{_bounded_suffix(suffix)}"
