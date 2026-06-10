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

Availability sentinel
---------------------
``VerifierUnavailable`` is raised (never caught internally) when the JWKS
endpoint is unreachable due to a network or HTTP error.  It is distinct from a
token being invalid: invalid tokens yield ``None``; an unreachable JWKS endpoint
raises ``VerifierUnavailable`` so the caller can return 503 + Retry-After
instead of the 401 OAuth challenge (spec section 5).
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import jwt as pyjwt

logger = logging.getLogger("agentguard.mcp.oauth")


class VerifierUnavailable(Exception):
    """The JWKS endpoint is temporarily unreachable.

    Raised by :meth:`SupabaseTokenVerifier._verify_sync` (and therefore by
    :meth:`SupabaseTokenVerifier.verify_token`) when a network or HTTP error
    prevents key acquisition.  Distinct from an invalid token (which yields
    ``None``): callers should map this to 503 + Retry-After, not 401.
    """


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
        """GET the JWKS document. Isolated for test monkeypatching.

        Any exception raised here (network errors, HTTP errors, or those
        injected by test monkeypatches) is caught by ``_key_for`` and
        re-raised as :class:`VerifierUnavailable`.
        """
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

        Raises:
            VerifierUnavailable: when ``_fetch_jwks`` raises for any reason
                (network error, HTTP error, or test-injected exception).  An
                unknown kid after a *successful* refresh still returns ``None``
                (401 path), not this exception.
        """
        with self._lock:
            stale = (time.monotonic() - self._fetched_at) > _JWKS_TTL_S
            if refresh and not stale and self._keys:
                # Forced-refresh path: respect the backoff window.
                now = time.monotonic()
                if now - self._last_forced_refresh < _JWKS_REFRESH_BACKOFF_S:
                    # Already refreshed recently; serve from the current cache.
                    return self._keys.get(kid)
                try:
                    jwks = self._fetch_jwks()
                except Exception as exc:
                    raise VerifierUnavailable(
                        f"JWKS endpoint unreachable: {self._jwks_url}"
                    ) from exc
                self._keys = {
                    k.get("kid"): pyjwt.algorithms.RSAAlgorithm.from_jwk(k)
                    for k in jwks.get("keys", [])
                    if k.get("kty") == "RSA"
                }
                self._fetched_at = time.monotonic()
                self._last_forced_refresh = self._fetched_at
            elif stale or not self._keys:
                # TTL-based or cold-load refresh: no backoff applied.
                try:
                    jwks = self._fetch_jwks()
                except Exception as exc:
                    raise VerifierUnavailable(
                        f"JWKS endpoint unreachable: {self._jwks_url}"
                    ) from exc
                self._keys = {
                    k.get("kid"): pyjwt.algorithms.RSAAlgorithm.from_jwk(k)
                    for k in jwks.get("keys", [])
                    if k.get("kty") == "RSA"
                }
                self._fetched_at = time.monotonic()
            return self._keys.get(kid)

    # -- Verification ---------------------------------------------------------

    def _verify_sync(self, token: str) -> Optional[OAuthIdentity]:
        """Validate ``token`` synchronously; return the identity or ``None``.

        This is the real verification logic — all work is CPU-bound or makes
        a synchronous JWKS HTTP fetch.  Keeping it sync lets the middleware
        offload it to a thread pool via ``anyio.to_thread.run_sync`` without
        a nested-event-loop problem.

        Raises:
            VerifierUnavailable: when the JWKS endpoint cannot be reached due
                to a network or HTTP error (propagated from ``_key_for`` →
                ``_fetch_jwks``).  Token-structural failures (bad signature,
                expiry, wrong issuer, unknown kid after successful refresh) all
                return ``None`` — fail closed, not unavailable.
        """
        # VerifierUnavailable from _key_for/_fetch_jwks propagates to the
        # caller; all other exceptions are caught below (fail closed).
        try:
            header = pyjwt.get_unverified_header(token)
        except Exception:
            logger.info("oauth: token validation failed", exc_info=True)
            return None

        kid = header.get("kid")

        # _key_for may raise VerifierUnavailable on JWKS fetch failure;
        # we let that propagate intentionally.
        key = self._key_for(kid)
        if key is None:
            key = self._key_for(kid, refresh=True)
        if key is None:
            logger.info("oauth: unknown kid after JWKS refresh")
            return None

        try:
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

    async def verify_token(self, token: str) -> Optional[OAuthIdentity]:
        """Async wrapper around :meth:`_verify_sync`.

        Preserves the ``async def verify_token`` interface so external callers
        and tests that ``await verifier.verify_token(...)`` continue to work
        without modification.  The middleware uses ``_verify_sync`` directly
        (via ``anyio.to_thread.run_sync``) to avoid blocking the event loop.

        Raises:
            VerifierUnavailable: propagated from ``_verify_sync`` when the
                JWKS endpoint is temporarily unreachable.
        """
        return self._verify_sync(token)


# ---------------------------------------------------------------------------
# Grant resolution: OAuthIdentity -> OAuthPrincipal
# ---------------------------------------------------------------------------

_GRANT_CACHE_TTL_S = 60.0
# Negative results are cached briefly too, so a revoked client hammering the
# endpoint does not hammer Supabase; kept shorter than the positive TTL so a
# fresh consent becomes usable quickly.
_GRANT_NEGATIVE_TTL_S = 15.0


@dataclass(frozen=True)
class OAuthPrincipal:
    """A fully resolved OAuth caller, shaped to match the key path's data."""

    org_id: str
    account_slug: str
    user_id: str
    client_id: str
    plan: str
    plan_overrides: Optional[dict]


class GrantResolver:
    """Resolve a validated :class:`OAuthIdentity` to an :class:`OAuthPrincipal`.

    Chain (the same joins the key path already performs):
        oauth_grants(user_id, client_id, revoked_at IS NULL) -> account_slug
        organizations(account_slug)                          -> org_id
        accounts(slug)                                       -> plan, overrides

    Results are cached per (user_id, client_id) for ``_GRANT_CACHE_TTL_S``
    (60 s — the spec's revocation-lag bound, same TTL as the key cache).
    All lookup failures return ``None`` — authn fails closed.
    """

    def __init__(self, *, supabase_engine: Any, tsdb_engine: Any) -> None:
        self._supabase_engine = supabase_engine
        self._tsdb_engine = tsdb_engine
        self._cache: dict[
            tuple[str, str], tuple[Optional["OAuthPrincipal"], float]
        ] = {}
        self._lock = threading.Lock()

    def resolve(self, identity: OAuthIdentity) -> Optional["OAuthPrincipal"]:
        key = (identity.user_id, identity.client_id)
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                principal, cached_at = cached
                ttl = _GRANT_CACHE_TTL_S if principal else _GRANT_NEGATIVE_TTL_S
                if now - cached_at < ttl:
                    return principal
        try:
            principal = self._lookup(identity)
        except Exception:
            logger.warning("oauth: grant lookup failed (fail closed)", exc_info=True)
            return None
        with self._lock:
            self._cache[key] = (principal, now)
        return principal

    def _lookup(self, identity: OAuthIdentity) -> Optional["OAuthPrincipal"]:
        from sqlalchemy import text

        with self._supabase_engine.connect() as conn:
            grant = conn.execute(
                text(
                    """
                    SELECT account_slug FROM oauth_grants
                    WHERE user_id = :user_id
                      AND oauth_client_id = :client_id
                      AND revoked_at IS NULL
                    LIMIT 1
                    """
                ),
                {"user_id": identity.user_id, "client_id": identity.client_id},
            ).fetchone()
        if grant is None:
            return None
        account_slug = grant[0]

        with self._tsdb_engine.connect() as conn:
            org = conn.execute(
                text(
                    "SELECT org_id FROM organizations "
                    "WHERE account_slug = :slug LIMIT 1"
                ),
                {"slug": account_slug},
            ).fetchone()
        if org is None:
            # Spec: a grant whose account no longer resolves = revoked.
            logger.info("oauth: grant account %s has no org; rejected", account_slug)
            return None

        with self._supabase_engine.connect() as conn:
            plan_row = conn.execute(
                text(
                    "SELECT vex_plan, vex_plan_overrides FROM accounts "
                    "WHERE slug = :slug LIMIT 1"
                ),
                {"slug": account_slug},
            ).fetchone()
        plan = plan_row[0] if plan_row and plan_row[0] else "free"
        overrides = plan_row[1] if plan_row else None
        if isinstance(overrides, str):
            import json

            overrides = json.loads(overrides)

        return OAuthPrincipal(
            org_id=str(org[0]),
            account_slug=account_slug,
            user_id=identity.user_id,
            client_id=identity.client_id,
            plan=plan,
            plan_overrides=overrides,
        )


# ---------------------------------------------------------------------------
# Rate limiting for OAuth principals
# ---------------------------------------------------------------------------

_RATE_WINDOW_S = 60.0
# Default rpm for OAuth callers when the plan config exposes no explicit
# value; matches the key path's default per-key rpm.
DEFAULT_OAUTH_RPM = 1000


class OAuthRateLimiter:
    """Sliding-window request limiter keyed by ``user_id:client_id``.

    Same observable semantics as the API-key limiter (429 + Retry-After):
    at most ``rpm`` requests per trailing 60-second window per principal.
    In-memory and per-replica (matches the key limiter's deployment model).

    The ``check`` method is pure CPU/memory work — no I/O — so it is safe
    to call directly on the async event loop without thread offloading.
    """

    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(
        self, key: str, *, rpm: int, now: Optional[float] = None
    ) -> Optional[int]:
        """Record a hit; return ``None`` if allowed, else seconds to retry after.

        Args:
            key: Composite principal identifier, e.g. ``"<user_id>:<client_id>"``.
            rpm: Maximum requests allowed within the trailing ``_RATE_WINDOW_S``
                window for this key.
            now: Override the current timestamp (``time.monotonic()``) for
                deterministic testing.

        Returns:
            ``None`` when the request is within the rate limit.
            A positive integer (seconds) when the limit is exceeded; callers
            should surface this as the ``Retry-After`` response header value.
        """
        ts = time.monotonic() if now is None else now
        cutoff = ts - _RATE_WINDOW_S
        with self._lock:
            hits = [h for h in self._hits.get(key, []) if h > cutoff]
            if len(hits) >= rpm:
                # Oldest hit in the window determines when the window clears.
                self._hits[key] = hits
                return max(1, int(hits[0] + _RATE_WINDOW_S - ts) + 1)
            hits.append(ts)
            self._hits[key] = hits
            return None
