"""API key validation and principal resolution for the MCP server.

Bridges an incoming Vex API key to a tenant principal
``(org_id, agent_id)`` that the MCP tools use to scope memory
operations.  Authentication reuses the shared ``KeyValidator`` (the same
machinery the ingestion API uses) so the key contract stays identical
across services.

Scope: brain access is gated by the dedicated ``"memory"`` scope (NOT
``"ingest"``), so a key that can ingest cannot also drive the memory
tools unless it has been explicitly granted memory access.  Keys that
should reach the brain must carry ``"memory"`` in their ``scopes`` array.
That provisioning happens in the external dashboard (a separate repo —
no key-provisioning code lives here): it must grant the ``"memory"``
scope to any key the memory product is meant to authorize.

Project scoping: MCP clients have no current-working-directory concept,
so a workspace may declare its project via the optional ``X-Vex-Project``
header (a git remote URL or project slug).  Resolution is read-only
(``shared.brain.find_project_id``) and FAILS OPEN — an absent header,
unknown identifier, or lookup error all yield ``None`` (org-wide recall).
Project scoping is a relevance feature, not a security boundary
(``org_id`` enforces tenancy).
"""

import logging
import os
from typing import Any, Optional

from fastapi import Depends, HTTPException, Request
from shared import brain
from shared.auth import AuthError, KeyInfo, KeyValidator

from app.config import DATABASE_URL

logger = logging.getLogger("agentguard.mcp.auth")

SUPABASE_DATABASE_URL = os.environ.get("SUPABASE_DATABASE_URL")

# Agent principal used when a request does not carry an ``X-Vex-Agent``
# header.  MCP clients that never set an agent (e.g. a single-agent
# integration) all share this stable default so their memory stays
# scoped to one consistent agent_id rather than scattering across
# generated values.
DEFAULT_AGENT_ID = "mcp-default"

# Optional per-client project scoping. MCP clients have no cwd, so a
# workspace declares its project via this header (a git remote URL or
# project slug). Resolution is read-only and FAILS OPEN to org-wide:
# scoping is a relevance feature, not a security boundary (org_id is).
PROJECT_HEADER = "X-Vex-Project"


def resolve_project_from_request(
    request: Request, db: Any, *, org_id: str
) -> Optional[str]:
    """Resolve the X-Vex-Project header to a project id, or None.

    Read-only (never creates a project), tenant-scoped by org_id, and
    fails open: a missing header, unknown identifier, or lookup error all
    yield None (org-wide recall).
    """
    identifier = request.headers.get(PROJECT_HEADER)
    if not identifier:
        return None
    try:
        return brain.find_project_id(db, org_id=org_id, identifier=identifier)
    except Exception:
        logger.warning(
            "X-Vex-Project resolution failed for org %s; org-wide",
            org_id,
            exc_info=True,
        )
        return None


_validator: Optional[KeyValidator] = None


def get_validator() -> KeyValidator:
    """Return the module-level KeyValidator singleton (lazy init)."""
    global _validator
    if _validator is None:
        # Brain access is gated by the dedicated "memory" scope (see
        # module docstring); keys must be provisioned with it externally.
        _validator = KeyValidator(
            database_url=DATABASE_URL,
            required_scope="memory",
            supabase_database_url=SUPABASE_DATABASE_URL,
        )
    return _validator


def shutdown_validator() -> None:
    """Flush usage and close DB connections.  Called at app shutdown."""
    global _validator
    if _validator is not None:
        _validator.close()
        _validator = None


def verify_api_key(request: Request) -> KeyInfo:
    """FastAPI dependency that validates the API key and returns org info.

    Extracts the key from the ``X-Vex-Key`` header (falling back to the
    legacy ``X-AgentGuard-Key`` header), validates it through the
    ``KeyValidator``, and returns a ``KeyInfo`` containing the resolved
    ``org_id``, ``key_id``, and ``scopes``.

    Raises:
        HTTPException: 401 if the key is missing, invalid, revoked, or
            expired.  403 if the key lacks the required scope.  429 if
            the key's rate limit is exceeded (with a ``Retry-After``
            header).
    """
    api_key = request.headers.get("X-Vex-Key") or request.headers.get(
        "X-AgentGuard-Key"
    )
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-Vex-Key header")

    try:
        return get_validator().validate(api_key)
    except AuthError as e:
        headers = {}
        if e.retry_after_seconds is not None:
            headers["Retry-After"] = str(e.retry_after_seconds)
        raise HTTPException(
            status_code=e.status_code,
            detail=e.detail,
            headers=headers or None,
        ) from e


def resolve_principal(request: Request) -> tuple[str, str]:
    """Resolve the tenant principal ``(org_id, agent_id)`` for a request.

    Authenticates the request via :func:`verify_api_key` to derive the
    ``org_id`` from the validated key, then reads the ``agent_id`` from
    the ``X-Vex-Agent`` header.  When that header is absent the stable
    :data:`DEFAULT_AGENT_ID` is used so a client that never declares an
    agent still maps to one consistent principal.

    The MCP tools (D1+) call this to scope every memory operation to a
    single tenant.

    Returns:
        A ``(org_id, agent_id)`` tuple.

    Raises:
        HTTPException: Propagated from :func:`verify_api_key` on any
            authentication failure.
    """
    key_info = verify_api_key(request)
    agent_id = request.headers.get("X-Vex-Agent") or DEFAULT_AGENT_ID
    return key_info.org_id, agent_id


def resolve_principal_plan(request: Request) -> tuple[str, str, str, dict | None]:
    """Like :func:`resolve_principal` but also returns the resolved plan + overrides.

    Used by the memory-metering checks, which need the plan AND any per-account
    plan overrides to look up the effective per-plan memory quota. Returning the
    overrides alongside the plan keeps real-time enforcement consistent with the
    retention sweep, which also merges overrides via ``get_plan_config(plan,
    overrides)`` — without them a custom cap would be silently enforced at the
    base tier. Validates the key once (no extra round trip beyond what
    :func:`resolve_principal` already does).

    Returns:
        A ``(org_id, agent_id, plan, plan_overrides)`` tuple.
    """
    key_info = verify_api_key(request)
    agent_id = request.headers.get("X-Vex-Agent") or DEFAULT_AGENT_ID
    return key_info.org_id, agent_id, key_info.plan, key_info.plan_overrides


def resolve_principal_dep(
    request: Request,
    key_info: KeyInfo = Depends(verify_api_key),
) -> tuple[str, str]:
    """FastAPI-dependency form of :func:`resolve_principal` for REST routes.

    Unlike :func:`resolve_principal` (a plain function the MCP tool bodies call
    inline, because FastMCP tools are not FastAPI routes), this is wired through
    ``Depends(verify_api_key)``.  That has two concrete benefits on the REST
    capture path:

    * FastAPI runs the (blocking psycopg2) key validation in its threadpool,
      so the event loop is not blocked while the key is checked — the inline
      MCP pattern does block it.
    * Tests can override ``verify_api_key`` via ``app.dependency_overrides`` to
      bypass auth (see ``tests/conftest.py``), which a plain inline call could
      not intercept.

    The ``org_id`` comes from the validated key (never caller input); the
    ``agent_id`` is read from the ``X-Vex-Agent`` header, falling back to
    :data:`DEFAULT_AGENT_ID` so a client that never declares an agent still maps
    to one consistent principal.

    Returns:
        A ``(org_id, agent_id)`` tuple.

    Raises:
        HTTPException: Propagated from :func:`verify_api_key` on any
            authentication failure (401/403/429).
    """
    agent_id = request.headers.get("X-Vex-Agent") or DEFAULT_AGENT_ID
    return key_info.org_id, agent_id
