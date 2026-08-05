"""API key validation and principal resolution for the MCP server.

Bridges an incoming Vex API key (or an OAuth Bearer token resolved by
the middleware) to a tenant principal ``(org_id, agent_id)`` that the
MCP tools use to scope memory operations.

Authentication ladder
---------------------
``resolve_principal`` / ``resolve_principal_plan`` implement a two-rung
ladder:

1. **OAuth path** — when the ASGI middleware (``app.oauth_middleware``)
   has already validated a Bearer token and resolved it to an
   :class:`app.oauth.OAuthPrincipal`, the principal is found on
   ``request.state.oauth_principal`` and used directly (no key lookup).
   The OAuth ``agent_id`` is derived as
   ``oauth:<first8-of-user-uuid>/<slugified-client-name>`` where the
   client name comes from the MCP ``initialize`` handshake (set by
   ``_attach_client_name`` in ``mcp_app.py``), falling back to the OAuth
   ``client_id``.  ``X-Vex-Agent`` overrides this entirely.

2. **Key path** — when no OAuth principal is attached the request must
   carry ``X-Vex-Key`` / ``X-AgentGuard-Key``.  This is the original
   behavior, byte-for-byte unchanged.

``resolve_principal_dep`` (the REST capture dependency) is NOT changed:
capture stays key-only per the spec.

Key path scope: brain access is gated by the dedicated ``"memory"``
scope (NOT ``"ingest"``), so a key that can ingest cannot also drive the
memory tools unless it has been explicitly granted memory access.  Keys
that should reach the brain must carry ``"memory"`` in their ``scopes``
array.  That provisioning happens in the external dashboard (a separate
repo — no key-provisioning code lives here): it must grant the
``"memory"`` scope to any key the memory product is meant to authorize.

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
import re
from typing import Any, Optional

from fastapi import Depends, HTTPException, Request
from shared import brain
from shared.auth import AuthError, KeyInfo, KeyValidator, normalize_agent_id

from app.config import DATABASE_URL

logger = logging.getLogger("agentguard.mcp.auth")

SUPABASE_DATABASE_URL = os.environ.get("SUPABASE_DATABASE_URL")

# Every read of ``X-Vex-Agent`` in this module goes through
# ``shared.auth.normalize_agent_id``. The header (and, on the OAuth path,
# ``clientInfo.name``) is client input reaching ``session_memories.agent_id``,
# a ``String(128)``; unbounded, a 500-character header produced a 500-character
# id and Postgres answered "value too long for type character varying(128)",
# turning every write into a 500 rather than degrading.
#
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
        # This resolver runs on EVERY MCP request that carries the header, on
        # the session the tool body then does its real work with. Failing open
        # to org-wide is the documented degradation; leaving the transaction
        # aborted is not — it fails the tool for a reason that has nothing to
        # do with projects, on the widest path in the service.
        try:
            db.rollback()
        except Exception:
            logger.warning(
                "rollback after a failed X-Vex-Project resolution also failed "
                "for org %s",
                org_id,
                exc_info=True,
            )
        logger.warning(
            "X-Vex-Project resolution failed for org %s; org-wide",
            org_id,
            exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# OAuth agent-identity helpers
# ---------------------------------------------------------------------------


def _slugify_client_name(name: str) -> str:
    """Lowercase, spaces->dashes, strip anything outside ``[a-z0-9._-]``.

    Returns the slug, or the literal string ``"client"`` when the result
    would otherwise be empty (e.g. the input was all stripped characters).
    """
    slug = re.sub(r"\s+", "-", name.strip().lower())
    return re.sub(r"[^a-z0-9._-]", "", slug) or "client"


def _oauth_agent_id(request: Request, principal: Any) -> str:
    """Default agent identity for OAuth callers (spec section 4).

    ``oauth:<first 8 chars of user uuid>/<client name slug>`` where the
    client name comes from the MCP ``initialize`` handshake when the tool
    layer plumbed it onto ``request.state.mcp_client_name``; falls back
    to the OAuth ``client_id``.  The ``X-Vex-Agent`` header overrides
    entirely (checked by the callers before this is consulted).
    """
    client_name = getattr(request.state, "mcp_client_name", None)
    client_part = (
        _slugify_client_name(client_name) if client_name else principal.client_id
    )
    return f"oauth:{principal.user_id[:8]}/{client_part}"


def _oauth_principal_from(request: Request) -> Optional[Any]:
    """Return the middleware-attached OAuthPrincipal, or None.

    Tolerates requests whose ``.state`` object has no ``oauth_principal``
    attribute (e.g. pure key-path requests, test fakes without the attr).
    Only returns the principal when it is an actual :class:`~app.oauth.OAuthPrincipal`
    instance — this prevents MagicMock-patched requests in pre-existing tests
    (which return a truthy mock for any attribute access) from accidentally
    entering the OAuth branch.  Never raises.
    """
    candidate = getattr(getattr(request, "state", None), "oauth_principal", None)
    if candidate is None:
        return None
    # Lazy import to avoid making oauth.py a hard module-level dep of auth.py
    # (and to keep the import cycle-free — oauth.py never imports auth.py).
    from app.oauth import OAuthPrincipal  # noqa: PLC0415

    return candidate if isinstance(candidate, OAuthPrincipal) else None


# ---------------------------------------------------------------------------
# Key-validator singleton
# ---------------------------------------------------------------------------

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

    Implements the two-rung auth ladder (see module docstring):

    1. **OAuth path** — when ``request.state.oauth_principal`` is set by
       the middleware, the ``org_id`` is taken directly from the
       :class:`~app.oauth.OAuthPrincipal`.  The ``agent_id`` is the
       ``X-Vex-Agent`` header if provided, otherwise the derived OAuth
       agent id (``oauth:<user8>/<client-slug>`` via
       :func:`_oauth_agent_id`).

    2. **Key path** — when no OAuth principal is attached, authenticates
       via :func:`verify_api_key` to derive ``org_id`` from the validated
       key.  ``agent_id`` comes from ``X-Vex-Agent`` or falls back to
       :data:`DEFAULT_AGENT_ID`.  Byte-for-byte the original behavior.

    The MCP tools (D1+) call this to scope every memory operation to a
    single tenant.

    Returns:
        A ``(org_id, agent_id)`` tuple.

    Raises:
        HTTPException: Propagated from :func:`verify_api_key` on any
            authentication failure (key path only; the OAuth path never
            raises — the middleware already validated the token).
    """
    principal = _oauth_principal_from(request)
    if principal is not None:
        agent_id = normalize_agent_id(
            request.headers.get("X-Vex-Agent") or _oauth_agent_id(request, principal),
            default=DEFAULT_AGENT_ID,
        )
        return principal.org_id, agent_id
    key_info = verify_api_key(request)
    agent_id = normalize_agent_id(
        request.headers.get("X-Vex-Agent"), default=DEFAULT_AGENT_ID
    )
    return key_info.org_id, agent_id


def resolve_principal_plan(request: Request) -> tuple[str, str, str, dict | None]:
    """Like :func:`resolve_principal` but also returns the resolved plan + overrides.

    Implements the same two-rung auth ladder as :func:`resolve_principal`;
    additionally surfaces the plan and per-account overrides needed by the
    memory-metering checks.  Returning the overrides alongside the plan
    keeps real-time enforcement consistent with the retention sweep, which
    also merges overrides via ``get_plan_config(plan, overrides)`` —
    without them a custom cap would be silently enforced at the base tier.
    Validates the key (or reads the OAuth principal) once.

    Returns:
        A ``(org_id, agent_id, plan, plan_overrides)`` tuple.

    Raises:
        HTTPException: Propagated from :func:`verify_api_key` on any
            authentication failure (key path only).
    """
    principal = _oauth_principal_from(request)
    if principal is not None:
        agent_id = normalize_agent_id(
            request.headers.get("X-Vex-Agent") or _oauth_agent_id(request, principal),
            default=DEFAULT_AGENT_ID,
        )
        return principal.org_id, agent_id, principal.plan, principal.plan_overrides
    key_info = verify_api_key(request)
    agent_id = normalize_agent_id(
        request.headers.get("X-Vex-Agent"), default=DEFAULT_AGENT_ID
    )
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
    agent_id = normalize_agent_id(
        request.headers.get("X-Vex-Agent"), default=DEFAULT_AGENT_ID
    )
    return key_info.org_id, agent_id
