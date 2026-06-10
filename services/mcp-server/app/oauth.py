"""MCP OAuth resource-server pieces: token verification and grant resolution.

The engine is a pure OAuth RESOURCE server. Supabase Auth is the
authorization server: it issues and signs the tokens; we validate them
locally against Supabase's JWKS (no per-request Supabase round trip for the
signature check) and then resolve the (user, client) pair to a Klio org via
the ``oauth_grants`` table — see the spec at
docs/superpowers/specs/2026-06-10-mcp-oauth-design.md.

Fail-closed: any validation or lookup failure yields ``None`` / no
principal. Authentication is never fail-open (unlike the read-path
relevance lookups elsewhere in this service).
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import jwt as pyjwt

logger = logging.getLogger("agentguard.mcp.oauth")

_JWKS_TTL_S = 300.0
_JWKS_TIMEOUT_S = 5.0
# Allow small clock skew between Supabase and this host.
_LEEWAY_S = 30
# Minimum interval between FORCED JWKS refreshes (unknown-kid misses).
# Caps unauthenticated bad-kid floods at ~1 upstream fetch per window.
_JWKS_REFRESH_BACKOFF_S = 5.0


@dataclass(frozen=True)
class OAuthIdentity:
    """The validated (user, client) pair carried by a Supabase OAuth JWT."""

    user_id: str
    client_id: str


class SupabaseTokenVerifier:
    """Validate Supabase-issued OAuth JWTs against the project's JWKS.

    Mirrors the MCP SDK's ``TokenVerifier`` protocol shape
    (``async verify_token(token) -> identity | None``) but returns our own
    :class:`OAuthIdentity` since the dual-mode ladder is enforced by our
    middleware, not the SDK's transport auth.

    JWKS keys are cached for ``_JWKS_TTL_S`` with refresh-on-unknown-kid.
    Forced refreshes (triggered by an unknown kid) are additionally
    rate-limited to at most one upstream fetch per ``_JWKS_REFRESH_BACKOFF_S``
    window; subsequent unknown-kid misses within the window serve from the
    already-refreshed cache.  This caps the blast radius of unauthenticated
    bad-kid floods to ~1 Supabase round trip per window instead of one per
    request.  The TTL-based refresh path (stale or empty cache) is unchanged.
    All failures (bad signature, expiry, wrong issuer, missing client_id,
    JWKS unreachable) return ``None`` — fail closed.
    """

    def __init__(self, issuer: str) -> None:
        self._issuer = issuer.rstrip("/")
        self._jwks_url = f"{self._issuer}/.well-known/jwks.json"
        self._keys: dict[str, Any] = {}
        self._fetched_at = 0.0
        self._last_forced_refresh = 0.0
        self._lock = threading.Lock()

    # -- JWKS ---------------------------------------------------------------

    def _fetch_jwks(self) -> dict:
        """GET the JWKS document. Isolated for test monkeypatching."""
        response = httpx.get(self._jwks_url, timeout=_JWKS_TIMEOUT_S)
        response.raise_for_status()
        return response.json()

    def _key_for(self, kid: Optional[str], *, refresh: bool = False) -> Optional[Any]:
        """Return the signing key for ``kid``, refreshing the cache on miss.

        When ``refresh=True`` (forced refresh on unknown kid), the upstream
        fetch is skipped if one has already occurred within
        ``_JWKS_REFRESH_BACKOFF_S`` seconds; the method then serves whatever
        the cache currently holds for ``kid`` (likely ``None``).  This
        prevents a flood of tokens with random kids from causing one
        synchronous Supabase JWKS fetch per request.

        The TTL-based refresh path (``refresh=False`` but cache stale or
        empty) is not subject to the backoff — it proceeds unconditionally.
        """
        with self._lock:
            stale = (time.monotonic() - self._fetched_at) > _JWKS_TTL_S
            if refresh and not stale and self._keys:
                # Forced-refresh path: respect the backoff window.
                now = time.monotonic()
                if now - self._last_forced_refresh < _JWKS_REFRESH_BACKOFF_S:
                    # Already refreshed recently; serve from the current cache.
                    return self._keys.get(kid)
                jwks = self._fetch_jwks()
                self._keys = {
                    k.get("kid"): pyjwt.algorithms.RSAAlgorithm.from_jwk(k)
                    for k in jwks.get("keys", [])
                    if k.get("kty") == "RSA"
                }
                self._fetched_at = time.monotonic()
                self._last_forced_refresh = self._fetched_at
            elif stale or not self._keys:
                # TTL-based or cold-load refresh: no backoff applied.
                jwks = self._fetch_jwks()
                self._keys = {
                    k.get("kid"): pyjwt.algorithms.RSAAlgorithm.from_jwk(k)
                    for k in jwks.get("keys", [])
                    if k.get("kty") == "RSA"
                }
                self._fetched_at = time.monotonic()
            return self._keys.get(kid)

    # -- Verification ---------------------------------------------------------

    async def verify_token(self, token: str) -> Optional[OAuthIdentity]:
        """Validate ``token``; return the identity or ``None`` (fail closed)."""
        try:
            header = pyjwt.get_unverified_header(token)
            kid = header.get("kid")
            key = self._key_for(kid)
            if key is None:
                key = self._key_for(kid, refresh=True)
            if key is None:
                logger.info("oauth: unknown kid after JWKS refresh")
                return None
            claims = pyjwt.decode(
                token,
                key=key,
                algorithms=["RS256"],
                issuer=self._issuer,
                leeway=_LEEWAY_S,
                options={
                    "require": ["exp", "iat", "iss", "sub"],
                    # Supabase sets aud="authenticated" for user tokens; the
                    # OAuth-client check below (client_id) is the real gate
                    # for this resource, so we do not pin aud here.
                    "verify_aud": False,
                },
            )
        except Exception:
            logger.info("oauth: token validation failed", exc_info=True)
            return None

        # A dashboard session JWT has no OAuth client identity; only tokens
        # minted through the OAuth client flow carry client_id (or azp).
        client_id = claims.get("client_id") or claims.get("azp")
        sub = claims.get("sub")
        if not client_id or not sub:
            logger.info("oauth: token lacks client identity; rejected")
            return None
        return OAuthIdentity(user_id=str(sub), client_id=str(client_id))
