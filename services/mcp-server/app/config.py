"""Configuration for the Vex MCP server.

Configuration is driven by environment variables, falling back to local
development defaults that match the other Vex services.
"""

import os

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",
)


def _parse_bool(value: str | None, *, default: bool) -> bool:
    """Parse a truthy/falsy env string, falling back to ``default``."""
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _parse_csv(value: str | None) -> list[str]:
    """Parse a comma-separated env string into a list of trimmed tokens."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int(value: str | None, *, default: int, minimum: int) -> int:
    """Parse a non-negative int env string, clamped to ``minimum``.

    Falls back to ``default`` when unset or unparseable; never returns a value
    below ``minimum`` (a guard so a misconfigured tiny interval can't turn the
    background curator into a hot loop).
    """
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else minimum


# --- MCP transport security ------------------------------------------
#
# The MCP SDK ships DNS-rebinding protection (Host/Origin allow-listing)
# for its Streamable-HTTP transport.  It targets browser-driven attacks;
# this service is reached by agent backends and MCP clients (not
# browsers) and authenticates every operation via the ``X-Vex-Key``
# bridge, which is the real security boundary.  The deployment ``Host``
# also varies per environment (Docker service name, k8s ingress host),
# so a hardcoded localhost allow-list would reject legitimate traffic.
#
# We therefore default the SDK's protection to OFF and expose explicit
# env knobs so hardened deployments can turn it on with their own
# allow-lists.
MCP_DNS_REBINDING_PROTECTION = _parse_bool(
    os.environ.get("MCP_DNS_REBINDING_PROTECTION"),
    default=False,
)

# Comma-separated allow-lists, only consulted when protection is on.
# Supports the SDK's ``host:*`` / ``http://host:*`` wildcard-port syntax.
MCP_ALLOWED_HOSTS = _parse_csv(os.environ.get("MCP_ALLOWED_HOSTS"))
MCP_ALLOWED_ORIGINS = _parse_csv(os.environ.get("MCP_ALLOWED_ORIGINS"))


# --- Curator scheduler -----------------------------------------------
#
# The brain curator (``app.curator``) periodically synthesises each org's RAW
# memories into ``summary`` / ``fact`` rows.  It runs as an in-process
# background loop started from the app lifespan (``app.scheduler``).  This keeps
# the single-service Railway deployment self-contained (no external cron) and is
# safe for a single replica; multi-replica deployments should disable it here
# and drive ``run_curator_once`` from one external scheduler instead.
CURATOR_ENABLED = _parse_bool(os.environ.get("CURATOR_ENABLED"), default=True)

# Seconds between curator passes.  Floored at 60s so a misconfiguration can't
# spin the loop; defaults to 15 minutes.
CURATOR_INTERVAL_SECONDS = _parse_int(
    os.environ.get("CURATOR_INTERVAL_SECONDS"), default=900, minimum=60
)
