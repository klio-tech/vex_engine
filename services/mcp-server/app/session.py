"""Brain session-id derivation for the MCP write tools.

The Klio brain stores every memory under a ``session_id``. Unlike the
verification path (where the session id is a real conversation id), brain
writes need a *stable* id that groups a client's writes so deduplication can
work across calls within the same connection.

The Streamable-HTTP transport assigns an ``mcp-session-id`` header per
connection, so we key on that when present. When it is absent (e.g. a client
that does not negotiate a session, or the in-memory transport), we fall back to
the principal ``org_id:agent_id`` so dedup stays stable per principal rather
than scattering across calls. ``session_id`` has no foreign key, so any string
is acceptable.
"""

from typing import Any

_BRAIN_SESSION_PREFIX = "klio-brain"


def brain_session_id(request: Any, org_id: str, agent_id: str) -> str:
    """Derive a stable brain ``session_id`` for the current connection.

    Args:
        request: The transport request; only ``request.headers.get(...)`` is
            read, so any dict-like-header carrier works.
        org_id: Resolved org principal (fallback key component).
        agent_id: Resolved agent principal (fallback key component).

    Returns:
        A ``"klio-brain:<suffix>"`` id, where ``<suffix>`` is the
        ``mcp-session-id`` header when present, else ``"<org_id>:<agent_id>"``.
    """
    mcp_sid = request.headers.get("mcp-session-id")
    suffix = mcp_sid if mcp_sid else f"{org_id}:{agent_id}"
    return f"{_BRAIN_SESSION_PREFIX}:{suffix}"
