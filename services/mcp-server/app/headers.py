"""Identity headers, with the Klio names first and the Vex names forever.

The product is Klio; the wire protocol said Vex. Every published connect
command was teaching a new user a legacy brand in their first minute, and the
brand mismatch read as "did I sign up for the right thing?" (2026-08-11).

This is an alias, not a rename. ``X-Vex-Key`` and friends live in configs we
do not control — MCP client settings, CI secrets, shell profiles — so the
legacy names can never stop working. ``X-AgentGuard-Key`` (older still) is
kept for the same reason. New documentation shows only ``X-Klio-*``.

Precedence: the Klio header wins when both are present. A caller sending both
is mid-migration, and the newer name is the one they mean.

Every read of these headers anywhere in the app goes through this module —
``test_klio_headers.py`` sweeps the source to keep it that way.
"""

from typing import Any, Optional

#: Ordered aliases per header: first match wins.
_KEY_HEADERS = ("X-Klio-Key", "X-Vex-Key", "X-AgentGuard-Key")
_AGENT_HEADERS = ("X-Klio-Agent", "X-Vex-Agent")
_PROJECT_HEADERS = ("X-Klio-Project", "X-Vex-Project")


def _first(request: Any, names: tuple[str, ...]) -> Optional[str]:
    for name in names:
        value = request.headers.get(name)
        if value:
            return value
    return None


def key_header(request: Any) -> Optional[str]:
    """The caller's API key, whichever name it arrived under."""
    return _first(request, _KEY_HEADERS)


def agent_header(request: Any) -> Optional[str]:
    """The caller's agent identity, or ``None`` for the shared default."""
    return _first(request, _AGENT_HEADERS)


def project_header(request: Any) -> Optional[str]:
    """The caller's project hint. Resolution of it still fails open."""
    return _first(request, _PROJECT_HEADERS)
