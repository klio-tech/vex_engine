"""ASGI auth ladder for the MCP endpoint + RFC 9728 metadata payload.

Why our own middleware instead of FastMCP's built-in auth: the SDK's
transport auth is all-or-nothing Bearer — it would 401 every legacy
``X-Vex-Key`` client. The spec requires a LADDER (Bearer first, key
fallback, challenge last), so enforcement lives here, scoped to the MCP
endpoint only (``/healthz``, ``/verify``, ``/capture/*`` keep their own
auth).

The middleware only ATTACHES the resolved principal to ``request.state``;
``app.auth.resolve_principal*`` consumes it. Tool-level behavior for key
clients is byte-for-byte unchanged.
"""

import logging
from typing import Any, Optional

import anyio
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from app import config
from app.oauth import (
    DEFAULT_OAUTH_RPM,
    GrantResolver,
    OAuthIdentity,
    OAuthPrincipal,
    OAuthRateLimiter,
    SupabaseTokenVerifier,
    VerifierUnavailable,
)

logger = logging.getLogger("agentguard.mcp.oauth")

METADATA_PATH = "/.well-known/oauth-protected-resource"

# The MCP transport endpoint path. The OAuth *resource identifier* is the full
# endpoint URL a client connects to (``<public-url>/mcp``), NOT the bare origin
# — Anthropic's MCP client validates that the metadata ``resource`` equals the
# server URL it dialed, so advertising the origin makes it reject discovery
# before any browser login. Per RFC 9728 the metadata for a resource that has a
# path is also published at the path-aware well-known location
# (``/.well-known/oauth-protected-resource/mcp``); we serve and advertise that
# as the canonical location while keeping the bare path for back-compat.
MCP_RESOURCE_PATH = "/mcp"
METADATA_PATH_FOR_RESOURCE = f"{METADATA_PATH}{MCP_RESOURCE_PATH}"


def mcp_resource_url() -> str:
    """Canonical OAuth resource identifier: the full MCP endpoint URL."""
    return f"{config.MCP_PUBLIC_URL}{MCP_RESOURCE_PATH}"


_verifier: Optional[SupabaseTokenVerifier] = None
_resolver: Optional[GrantResolver] = None
# Module-level singleton; in-memory, per-replica (matches the key-path limiter).
_rate_limiter = OAuthRateLimiter()


def _get_verifier() -> SupabaseTokenVerifier:
    global _verifier
    if _verifier is None:
        _verifier = SupabaseTokenVerifier(issuer=config.SUPABASE_AUTH_ISSUER)
    return _verifier


def _get_resolver() -> GrantResolver:
    global _resolver
    if _resolver is None:
        import os

        from sqlalchemy import create_engine

        _resolver = GrantResolver(
            supabase_engine=create_engine(
                os.environ["SUPABASE_DATABASE_URL"], pool_pre_ping=True
            ),
            tsdb_engine=create_engine(config.DATABASE_URL, pool_pre_ping=True),
        )
    return _resolver


# Seams for tests (monkeypatched in test_oauth_middleware.py).
#
# Both are SYNC functions so the middleware can offload them to a thread pool
# via ``anyio.to_thread.run_sync``, keeping blocking I/O (JWKS fetch,
# SQLAlchemy queries) off the asyncio event loop.
#
# ``_verify_token`` may raise ``VerifierUnavailable`` when the JWKS endpoint
# is unreachable; the middleware maps that to 503 + Retry-After.
def _verify_token(token: str) -> Optional[OAuthIdentity]:
    return _get_verifier()._verify_sync(token)


def _resolve_grant(identity: OAuthIdentity) -> Optional[OAuthPrincipal]:
    return _get_resolver().resolve(identity)


def protected_resource_metadata() -> dict[str, Any]:
    """RFC 9728 payload advertising Supabase as the authorization server.

    ``resource`` is the full MCP endpoint URL (``<public-url>/mcp``) so it
    matches the URL the client connected to (the value Anthropic's client
    validates against).
    """
    return {
        "resource": mcp_resource_url(),
        "authorization_servers": [config.SUPABASE_AUTH_ISSUER],
        "bearer_methods_supported": ["header"],
        "scopes_supported": ["memory"],
        "resource_documentation": "https://klio.tech/docs/mcp",
    }


def _challenge() -> Response:
    """401 with the WWW-Authenticate challenge that drives client discovery.

    Points at the path-aware metadata location (RFC 9728) for the ``/mcp``
    resource so a client that follows the hint loads metadata whose
    ``resource`` matches the URL it dialed.
    """
    metadata_url = f"{config.MCP_PUBLIC_URL}{METADATA_PATH_FOR_RESOURCE}"
    return Response(
        status_code=401,
        headers={"WWW-Authenticate": (f'Bearer resource_metadata="{metadata_url}"')},
    )


class OAuthLadderMiddleware:
    """Enforce the dual-mode auth ladder on the MCP endpoint only.

    Ladder (spec section 3):
      1. ``Authorization: Bearer`` -> verify + resolve grant; attach principal.
         Invalid token -> 401 challenge. Valid token, no grant -> 403
         ``not_connected``.
      2. ``X-Vex-Key`` / ``X-AgentGuard-Key`` present -> pass through
         (existing in-tool key validation, unchanged).
      3. Neither -> 401 challenge (drives OAuth discovery).
    """

    def __init__(self, app: ASGIApp, *, mcp_path: str) -> None:
        self._app = app
        self._mcp_path = mcp_path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] != self._mcp_path:
            await self._app(scope, receive, send)
            return

        request = Request(scope, receive)
        authorization = request.headers.get("Authorization", "")
        if authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
            try:
                identity = await anyio.to_thread.run_sync(_verify_token, token)
            except VerifierUnavailable:
                logger.warning("oauth: JWKS endpoint unreachable; returning 503")
                response = JSONResponse(
                    status_code=503,
                    content={"error": "verifier_unavailable"},
                    headers={"Retry-After": "5"},
                )
                await response(scope, receive, send)
                return
            if identity is None:
                await _challenge()(scope, receive, send)
                return
            principal = await anyio.to_thread.run_sync(_resolve_grant, identity)
            if principal is None:
                response = JSONResponse(
                    status_code=403,
                    content={
                        "error": "not_connected",
                        "detail": (
                            "No active Klio connection for this account. "
                            "Connect this workspace at "
                            "https://app.klio.tech/settings/connected-apps"
                        ),
                    },
                )
                await response(scope, receive, send)
                return
            # Rate-limit check: pure in-memory CPU work — called directly on
            # the event loop (no thread offload needed).  Placed after the
            # principal is confirmed valid and before it is attached so a
            # throttled request never reaches the tool layer.
            from shared.plan_limits import get_plan_config

            plan_config = get_plan_config(principal.plan, principal.plan_overrides)
            rpm = getattr(plan_config, "max_rpm", None) or DEFAULT_OAUTH_RPM
            retry_after = _rate_limiter.check(
                f"{principal.user_id}:{principal.client_id}", rpm=rpm
            )
            if retry_after is not None:
                response = JSONResponse(
                    status_code=429,
                    content={"error": "rate_limited"},
                    headers={"Retry-After": str(retry_after)},
                )
                await response(scope, receive, send)
                return
            scope.setdefault("state", {})["oauth_principal"] = principal
            await self._app(scope, receive, send)
            return

        if request.headers.get("X-Vex-Key") or request.headers.get("X-AgentGuard-Key"):
            await self._app(scope, receive, send)
            return

        await _challenge()(scope, receive, send)
