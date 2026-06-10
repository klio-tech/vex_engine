# MCP OAuth (Engine Side) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the engine half of `docs/superpowers/specs/2026-06-10-mcp-oauth-design.md`: validate Supabase-issued OAuth Bearer tokens at the MCP endpoint, resolve them to the same `(org_id, agent_id, plan, plan_overrides)` principal the key path produces, serve RFC 9728 protected-resource metadata, and 401-with-`WWW-Authenticate` unauthenticated requests — all behind `MCP_OAUTH_ENABLED` (default off), with `X-Vex-Key` auth untouched.

**Architecture:** Dual-mode auth ladder implemented as our own ASGI middleware on the FastAPI app (NOT FastMCP's built-in auth — that is all-or-nothing Bearer and would 401 every legacy key client). Bearer tokens are validated locally against Supabase's JWKS (PyJWT + cached keys), then resolved to an org via the `oauth_grants` table (Supabase) joined to `organizations.account_slug` (TimescaleDB) — the same join `KeyValidator` already uses. The OAuth principal rides on `request.state`; `resolve_principal*` in `app/auth.py` checks it before falling back to the key path.

**Tech Stack:** Python 3.11, FastAPI/Starlette middleware, PyJWT + cryptography (new deps), httpx (already present via tests; add as runtime dep for JWKS), SQLAlchemy raw SQL, pytest.

**Working directory:** `vex_private/vex_engine/`. Test commands: shared suite `cd services/shared && python3 -m pytest tests/ -q`; mcp-server suite from repo root `.venv/bin/python -m pytest services/mcp-server/tests -q`. Install new deps into the repo venv: `.venv/bin/python -m pip install "PyJWT[crypto]>=2.8" "httpx>=0.25"`.

**Out of scope (Dashboard repo, separate plan):** the `oauth_grants` Supabase migration, consent page, connected-apps settings, Supabase dashboard OAuth-server configuration. Engine tests fake the grants table; nothing here depends on the Dashboard shipping first because the flag stays off.

**Commit rule:** conventional commits, no co-author lines.

---

## File structure

| File | Responsibility |
|---|---|
| Modify `services/mcp-server/app/config.py` | `MCP_OAUTH_ENABLED`, `SUPABASE_AUTH_ISSUER`, `MCP_PUBLIC_URL` |
| Create `services/mcp-server/app/oauth.py` | JWKS cache + `SupabaseTokenVerifier` (JWT → `OAuthIdentity`), `GrantResolver` (identity → principal), in-memory caches |
| Create `services/mcp-server/app/oauth_middleware.py` | ASGI middleware (auth ladder at the transport) + RFC 9728 metadata payload builder |
| Modify `services/mcp-server/app/main.py` | Mount middleware + `/.well-known/oauth-protected-resource` route |
| Modify `services/mcp-server/app/auth.py` | `resolve_principal*` ladder: `request.state.oauth_principal` first, key path unchanged; OAuth agent-id derivation helper |
| Modify `services/mcp-server/app/mcp_app.py` | Pass MCP `clientInfo.name` into agent-id derivation for OAuth principals |
| Modify `services/mcp-server/pyproject.toml` | Add `PyJWT[crypto]`, `httpx` runtime deps |
| Create `services/mcp-server/tests/test_oauth_verifier.py` | Verifier unit tests (self-signed JWKS fixture) |
| Create `services/mcp-server/tests/test_oauth_grants.py` | GrantResolver unit tests (fake engines) |
| Create `services/mcp-server/tests/test_oauth_middleware.py` | Transport-level ladder + metadata tests (TestClient) |
| Modify `services/mcp-server/tests/test_auth.py` | Principal-ladder tests |

---

### Task 1: Config flags

**Files:**
- Modify: `services/mcp-server/app/config.py`
- Test: `services/mcp-server/tests/test_oauth_middleware.py` (created here with config tests; middleware tests appended in Task 4)

- [ ] **Step 1: Read `services/mcp-server/app/config.py`** to learn the existing env-parsing idiom (it already defines `MCP_DNS_REBINDING_PROTECTION`, `MCP_ALLOWED_HOSTS`, `MCP_ALLOWED_ORIGINS`, `DATABASE_URL`). Match it exactly.

- [ ] **Step 2: Write the failing test** — create `services/mcp-server/tests/test_oauth_config.py`:

```python
"""Config flags for MCP OAuth (engine side)."""

import importlib


def _reload_config(monkeypatch, **env):
    for key in ("MCP_OAUTH_ENABLED", "SUPABASE_AUTH_ISSUER", "MCP_PUBLIC_URL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import app.config as config

    importlib.reload(config)
    return config


def test_oauth_disabled_by_default(monkeypatch):
    config = _reload_config(monkeypatch)
    assert config.MCP_OAUTH_ENABLED is False


def test_oauth_enabled_flag(monkeypatch):
    config = _reload_config(monkeypatch, MCP_OAUTH_ENABLED="1")
    assert config.MCP_OAUTH_ENABLED is True


def test_issuer_and_public_url(monkeypatch):
    config = _reload_config(
        monkeypatch,
        SUPABASE_AUTH_ISSUER="https://proj.supabase.co/auth/v1",
        MCP_PUBLIC_URL="https://mcp.klio.tech",
    )
    assert config.SUPABASE_AUTH_ISSUER == "https://proj.supabase.co/auth/v1"
    assert config.MCP_PUBLIC_URL == "https://mcp.klio.tech"
    # Trailing slashes are normalized away so URL joins are predictable.
    config2 = _reload_config(
        monkeypatch,
        SUPABASE_AUTH_ISSUER="https://proj.supabase.co/auth/v1/",
        MCP_PUBLIC_URL="https://mcp.klio.tech/",
    )
    assert not config2.SUPABASE_AUTH_ISSUER.endswith("/")
    assert not config2.MCP_PUBLIC_URL.endswith("/")
```

Run: `.venv/bin/python -m pytest services/mcp-server/tests/test_oauth_config.py -v` → FAIL (`AttributeError: MCP_OAUTH_ENABLED`).

- [ ] **Step 3: Implement** — append to `services/mcp-server/app/config.py` (following its existing style):

```python
# --- MCP OAuth (engine = resource server; Supabase = authorization server) ---

# Master switch. Off (default) preserves today's behavior exactly: Bearer
# tokens are ignored and unauthenticated MCP requests reach the tools (which
# return their own unauthorized result dicts).
MCP_OAUTH_ENABLED: bool = os.environ.get("MCP_OAUTH_ENABLED", "0") == "1"

# The Supabase project's auth issuer, e.g. "https://<ref>.supabase.co/auth/v1".
# Used as the JWT `iss` check, the JWKS base (<issuer>/.well-known/jwks.json),
# and the advertised authorization server in RFC 9728 metadata.
SUPABASE_AUTH_ISSUER: str = os.environ.get("SUPABASE_AUTH_ISSUER", "").rstrip("/")

# This deployment's public base URL, e.g. "https://mcp.klio.tech". Used as the
# RFC 9728 `resource` identifier and in WWW-Authenticate challenges.
MCP_PUBLIC_URL: str = os.environ.get("MCP_PUBLIC_URL", "").rstrip("/")
```

- [ ] **Step 4: Run** the test → PASS. Then the full mcp-server suite → no regressions.

- [ ] **Step 5: Commit**

```bash
git add services/mcp-server/app/config.py services/mcp-server/tests/test_oauth_config.py
git commit -m "feat(config): MCP OAuth flags (disabled by default)"
```

---

### Task 2: `SupabaseTokenVerifier` (JWT → OAuthIdentity)

**Files:**
- Create: `services/mcp-server/app/oauth.py` (verifier half; GrantResolver added in Task 3)
- Create: `services/mcp-server/tests/test_oauth_verifier.py`
- Modify: `services/mcp-server/pyproject.toml` (deps)

- [ ] **Step 1: Add deps.** In `services/mcp-server/pyproject.toml` `[project] dependencies`, add `"PyJWT[crypto]>=2.8"` and `"httpx>=0.25"`. Install locally: `.venv/bin/python -m pip install "PyJWT[crypto]>=2.8" "httpx>=0.25"`.

- [ ] **Step 2: Write the failing tests** — `services/mcp-server/tests/test_oauth_verifier.py`:

```python
"""SupabaseTokenVerifier: local JWT validation against a JWKS fixture."""

import time
import uuid

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from app.oauth import OAuthIdentity, SupabaseTokenVerifier

ISSUER = "https://proj.supabase.co/auth/v1"
KID = "test-key-1"


@pytest.fixture(scope="module")
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def verifier(rsa_key, monkeypatch):
    v = SupabaseTokenVerifier(issuer=ISSUER)
    # Bypass the network: seed the JWKS cache from the fixture key.
    public_jwk = pyjwt.algorithms.RSAAlgorithm.to_jwk(
        rsa_key.public_key(), as_dict=True
    )
    public_jwk["kid"] = KID
    public_jwk["alg"] = "RS256"
    monkeypatch.setattr(v, "_fetch_jwks", lambda: {"keys": [public_jwk]})
    return v


def _mint(rsa_key, **overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "sub": str(uuid.uuid4()),
        "client_id": "client-abc",
        "aud": "authenticated",
        "iat": now,
        "exp": now + 3600,
    }
    claims.update(overrides)
    return pyjwt.encode(claims, rsa_key, algorithm="RS256", headers={"kid": KID})


async def test_valid_token_yields_identity(verifier, rsa_key):
    token = _mint(rsa_key, sub="11111111-2222-3333-4444-555555555555")
    identity = await verifier.verify_token(token)
    assert isinstance(identity, OAuthIdentity)
    assert identity.user_id == "11111111-2222-3333-4444-555555555555"
    assert identity.client_id == "client-abc"


async def test_expired_token_rejected(verifier, rsa_key):
    token = _mint(rsa_key, exp=int(time.time()) - 10)
    assert await verifier.verify_token(token) is None


async def test_wrong_issuer_rejected(verifier, rsa_key):
    token = _mint(rsa_key, iss="https://evil.example.com/auth/v1")
    assert await verifier.verify_token(token) is None


async def test_garbage_token_rejected(verifier):
    assert await verifier.verify_token("not-a-jwt") is None


async def test_token_without_client_id_rejected(verifier, rsa_key):
    """A plain dashboard session JWT (no OAuth client) must NOT authenticate
    the MCP endpoint — only tokens minted through the OAuth client flow do."""
    token = _mint(rsa_key)
    # Re-mint without client_id (and without azp).
    import time as _t

    now = int(_t.time())
    claims = {
        "iss": ISSUER,
        "sub": str(uuid.uuid4()),
        "aud": "authenticated",
        "iat": now,
        "exp": now + 3600,
    }
    token = pyjwt.encode(claims, rsa_key, algorithm="RS256", headers={"kid": KID})
    assert await verifier.verify_token(token) is None


async def test_unknown_kid_triggers_refetch_then_rejects(verifier, rsa_key, monkeypatch):
    calls = {"n": 0}
    real = verifier._fetch_jwks

    def counting():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(verifier, "_fetch_jwks", counting)
    token = pyjwt.encode(
        {"iss": ISSUER, "sub": "u", "client_id": "c",
         "iat": int(time.time()), "exp": int(time.time()) + 60},
        rsa_key, algorithm="RS256", headers={"kid": "unknown-kid"},
    )
    assert await verifier.verify_token(token) is None
    assert calls["n"] >= 1  # refreshed on miss


async def test_jwks_fetch_failure_fails_closed(rsa_key, monkeypatch):
    v = SupabaseTokenVerifier(issuer=ISSUER)

    def boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(v, "_fetch_jwks", boom)
    token = _mint(rsa_key)
    assert await v.verify_token(token) is None
```

Run: `.venv/bin/python -m pytest services/mcp-server/tests/test_oauth_verifier.py -v` → FAIL (`ModuleNotFoundError: app.oauth`).

- [ ] **Step 3: Implement the verifier half of `services/mcp-server/app/oauth.py`:**

```python
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
    All failures (bad signature, expiry, wrong issuer, missing client_id,
    JWKS unreachable) return ``None`` — fail closed.
    """

    def __init__(self, issuer: str) -> None:
        self._issuer = issuer.rstrip("/")
        self._jwks_url = f"{self._issuer}/.well-known/jwks.json"
        self._keys: dict[str, Any] = {}
        self._fetched_at = 0.0
        self._lock = threading.Lock()

    # -- JWKS ---------------------------------------------------------------

    def _fetch_jwks(self) -> dict:
        """GET the JWKS document. Isolated for test monkeypatching."""
        response = httpx.get(self._jwks_url, timeout=_JWKS_TIMEOUT_S)
        response.raise_for_status()
        return response.json()

    def _key_for(self, kid: Optional[str], *, refresh: bool = False) -> Optional[Any]:
        """Return the signing key for ``kid``, refreshing the cache on miss."""
        with self._lock:
            stale = (time.monotonic() - self._fetched_at) > _JWKS_TTL_S
            if refresh or stale or not self._keys:
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
```

- [ ] **Step 4: Run** the verifier tests → PASS. (If the suite's asyncio mode complains about the bare `async def` tests, they collect under `asyncio_mode = "auto"` — already this suite's config.)

- [ ] **Step 5: Commit**

```bash
git add services/mcp-server/app/oauth.py services/mcp-server/tests/test_oauth_verifier.py services/mcp-server/pyproject.toml
git commit -m "feat(oauth): Supabase JWT verifier with cached JWKS (fail closed)"
```

---

### Task 3: `GrantResolver` (identity → principal)

**Files:**
- Modify: `services/mcp-server/app/oauth.py` (append)
- Create: `services/mcp-server/tests/test_oauth_grants.py`

- [ ] **Step 1: Read** `services/shared/shared/auth.py` lines 250-380 — note how `KeyValidator` lazily creates `self._supabase_engine` from `supabase_database_url` and how `_query_plan_from_supabase(account_slug)` returns `(plan, plan_overrides)`. The resolver reuses the same two-engine pattern (Supabase engine for grants + plan; TimescaleDB engine for the `organizations.account_slug → org_id` join).

- [ ] **Step 2: Write the failing tests** — `services/mcp-server/tests/test_oauth_grants.py`:

```python
"""GrantResolver: (user, client) -> (org_id, account_slug, plan)."""

import time

from app.oauth import GrantResolver, OAuthIdentity, OAuthPrincipal


class FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):
        sql_s = str(sql)

        class R:
            def __init__(self, row):
                self._row = row

            def fetchone(self_inner):
                return self_inner._row

        for needle, row in self._rows:
            if needle in sql_s:
                return R(row)
        return R(None)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeEngine:
    def __init__(self, rows):
        self._rows = rows

    def connect(self):
        return FakeConn(self._rows)


def _resolver(grant_slug="acme", org_id="org-acme", plan=("pro", None)):
    grant_row = (grant_slug,) if grant_slug else None
    org_row = (org_id,) if org_id else None
    resolver = GrantResolver(
        supabase_engine=FakeEngine([("oauth_grants", grant_row),
                                    ("accounts", plan)]),
        tsdb_engine=FakeEngine([("organizations", org_row)]),
    )
    return resolver


def test_active_grant_resolves_principal():
    principal = _resolver().resolve(OAuthIdentity(user_id="u-1", client_id="c-1"))
    assert isinstance(principal, OAuthPrincipal)
    assert principal.org_id == "org-acme"
    assert principal.account_slug == "acme"
    assert principal.user_id == "u-1"
    assert principal.client_id == "c-1"
    assert principal.plan == "pro"


def test_no_grant_returns_none():
    assert _resolver(grant_slug=None).resolve(
        OAuthIdentity(user_id="u-1", client_id="c-1")
    ) is None


def test_grant_with_unknown_account_returns_none():
    """Spec: a grant whose account_slug no longer resolves = revoked."""
    assert _resolver(org_id=None).resolve(
        OAuthIdentity(user_id="u-1", client_id="c-1")
    ) is None


def test_lookup_error_fails_closed():
    class BoomEngine:
        def connect(self):
            raise RuntimeError("db down")

    resolver = GrantResolver(supabase_engine=BoomEngine(), tsdb_engine=BoomEngine())
    assert resolver.resolve(OAuthIdentity(user_id="u", client_id="c")) is None


def test_resolution_is_cached_for_ttl(monkeypatch):
    resolver = _resolver()
    calls = {"n": 0}
    real = resolver._lookup

    def counting(identity):
        calls["n"] += 1
        return real(identity)

    monkeypatch.setattr(resolver, "_lookup", counting)
    identity = OAuthIdentity(user_id="u-1", client_id="c-1")
    resolver.resolve(identity)
    resolver.resolve(identity)
    assert calls["n"] == 1  # second hit served from cache
```

Run: → FAIL (`ImportError: GrantResolver`).

- [ ] **Step 3: Implement** — append to `services/mcp-server/app/oauth.py`:

```python
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
        self._cache: dict[tuple[str, str], tuple[Optional["OAuthPrincipal"], float]] = {}
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
        plan = (plan_row[0] if plan_row and plan_row[0] else "free")
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
```

- [ ] **Step 4: Run** `test_oauth_grants.py` → PASS; full mcp-server suite → no regressions.

- [ ] **Step 5: Commit**

```bash
git add services/mcp-server/app/oauth.py services/mcp-server/tests/test_oauth_grants.py
git commit -m "feat(oauth): grant resolver maps (user, client) to org principal"
```

---

### Task 4: Auth-ladder middleware + RFC 9728 metadata

**Files:**
- Create: `services/mcp-server/app/oauth_middleware.py`
- Modify: `services/mcp-server/app/main.py`
- Create: `services/mcp-server/tests/test_oauth_middleware.py`

- [ ] **Step 1: Write the failing tests** — `services/mcp-server/tests/test_oauth_middleware.py` (reuse the suite's existing `TestClient` construction pattern from `tests/test_health.py` / `conftest.py`):

```python
"""Transport-level auth ladder + RFC 9728 protected-resource metadata."""

import pytest
from fastapi.testclient import TestClient

from app.oauth import OAuthIdentity, OAuthPrincipal


def _client(monkeypatch, *, enabled, verify=None, resolve=None):
    monkeypatch.setenv("MCP_OAUTH_ENABLED", "1" if enabled else "0")
    monkeypatch.setenv("SUPABASE_AUTH_ISSUER", "https://proj.supabase.co/auth/v1")
    monkeypatch.setenv("MCP_PUBLIC_URL", "https://mcp.klio.tech")
    import importlib

    import app.config as config
    import app.oauth_middleware as om
    import app.main as main

    importlib.reload(config)
    importlib.reload(om)
    importlib.reload(main)
    if verify is not None:
        monkeypatch.setattr(om, "_verify_token", verify)
    if resolve is not None:
        monkeypatch.setattr(om, "_resolve_grant", resolve)
    return TestClient(main.create_app())


def test_flag_off_preserves_today(monkeypatch):
    client = _client(monkeypatch, enabled=False)
    # /healthz works; no metadata route; no challenge on /mcp POSTs.
    assert client.get("/healthz").status_code == 200
    assert client.get("/.well-known/oauth-protected-resource").status_code == 404


def test_metadata_served_when_enabled(monkeypatch):
    client = _client(monkeypatch, enabled=True)
    resp = client.get("/.well-known/oauth-protected-resource")
    assert resp.status_code == 200
    body = resp.json()
    assert body["resource"] == "https://mcp.klio.tech"
    assert body["authorization_servers"] == ["https://proj.supabase.co/auth/v1"]
    assert "header" in body["bearer_methods_supported"]


def test_unauthenticated_mcp_gets_401_challenge(monkeypatch):
    client = _client(monkeypatch, enabled=True)
    resp = client.post("/mcp", json={})
    assert resp.status_code == 401
    challenge = resp.headers["WWW-Authenticate"]
    assert challenge.startswith("Bearer")
    assert "resource_metadata=" in challenge
    assert "/.well-known/oauth-protected-resource" in challenge


def test_key_header_passes_through(monkeypatch):
    """Legacy X-Vex-Key requests bypass Bearer enforcement entirely."""
    client = _client(monkeypatch, enabled=True)
    resp = client.post("/mcp", json={}, headers={"X-Vex-Key": "ag_test"})
    # Reaches the MCP transport (which rejects the malformed MCP body itself,
    # NOT our 401-with-challenge).
    assert resp.status_code != 401 or "resource_metadata" not in resp.headers.get(
        "WWW-Authenticate", ""
    )


def test_invalid_bearer_gets_401(monkeypatch):
    async def deny(token):
        return None

    client = _client(monkeypatch, enabled=True, verify=deny)
    resp = client.post(
        "/mcp", json={}, headers={"Authorization": "Bearer bad-token"}
    )
    assert resp.status_code == 401
    assert "resource_metadata=" in resp.headers["WWW-Authenticate"]


def test_valid_bearer_without_grant_gets_403(monkeypatch):
    async def allow(token):
        return OAuthIdentity(user_id="u-1", client_id="c-1")

    def no_grant(identity):
        return None

    client = _client(monkeypatch, enabled=True, verify=allow, resolve=no_grant)
    resp = client.post(
        "/mcp", json={}, headers={"Authorization": "Bearer good-token"}
    )
    assert resp.status_code == 403
    assert resp.json()["error"] == "not_connected"
    assert "app.klio.tech" in resp.json()["detail"]


def test_valid_bearer_with_grant_passes_through(monkeypatch):
    async def allow(token):
        return OAuthIdentity(user_id="u-1", client_id="c-1")

    def grant(identity):
        return OAuthPrincipal(
            org_id="org-1", account_slug="acme", user_id="u-1",
            client_id="c-1", plan="pro", plan_overrides=None,
        )

    client = _client(monkeypatch, enabled=True, verify=allow, resolve=grant)
    resp = client.post(
        "/mcp", json={}, headers={"Authorization": "Bearer good-token"}
    )
    # Passes our ladder; the MCP transport handles (and rejects) the
    # malformed MCP body itself — anything but our 401/403 challenge shapes.
    assert resp.status_code not in (401, 403)


def test_healthz_and_capture_unaffected(monkeypatch):
    client = _client(monkeypatch, enabled=True)
    assert client.get("/healthz").status_code == 200
    # capture keeps its own key auth (401 from ITS dependency, no challenge).
    resp = client.post("/capture/event", json={"content": "x"})
    assert resp.status_code == 401
    assert "resource_metadata" not in resp.headers.get("WWW-Authenticate", "")
```

Run: → FAIL (`ModuleNotFoundError: app.oauth_middleware`).

- [ ] **Step 2: Implement `services/mcp-server/app/oauth_middleware.py`:**

```python
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

import json
import logging
from typing import Any, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from app import config
from app.oauth import GrantResolver, OAuthIdentity, OAuthPrincipal, SupabaseTokenVerifier

logger = logging.getLogger("agentguard.mcp.oauth")

METADATA_PATH = "/.well-known/oauth-protected-resource"

_verifier: Optional[SupabaseTokenVerifier] = None
_resolver: Optional[GrantResolver] = None


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
async def _verify_token(token: str) -> Optional[OAuthIdentity]:
    return await _get_verifier().verify_token(token)


def _resolve_grant(identity: OAuthIdentity) -> Optional[OAuthPrincipal]:
    return _get_resolver().resolve(identity)


def protected_resource_metadata() -> dict[str, Any]:
    """RFC 9728 payload advertising Supabase as the authorization server."""
    return {
        "resource": config.MCP_PUBLIC_URL,
        "authorization_servers": [config.SUPABASE_AUTH_ISSUER],
        "bearer_methods_supported": ["header"],
        "scopes_supported": ["memory"],
        "resource_documentation": "https://klio.tech/docs/mcp",
    }


def _challenge() -> Response:
    """401 with the WWW-Authenticate challenge that drives client discovery."""
    metadata_url = f"{config.MCP_PUBLIC_URL}{METADATA_PATH}"
    return Response(
        status_code=401,
        headers={
            "WWW-Authenticate": (
                f'Bearer resource_metadata="{metadata_url}"'
            )
        },
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
            identity = await _verify_token(token)
            if identity is None:
                await _challenge()(scope, receive, send)
                return
            principal = _resolve_grant(identity)
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
            scope.setdefault("state", {})["oauth_principal"] = principal
            await self._app(scope, receive, send)
            return

        if request.headers.get("X-Vex-Key") or request.headers.get(
            "X-AgentGuard-Key"
        ):
            await self._app(scope, receive, send)
            return

        await _challenge()(scope, receive, send)
```

- [ ] **Step 3: Wire into `services/mcp-server/app/main.py`.** Inside `create_app()`, after the `/verify` route and BEFORE `app.include_router(capture_router)`, add:

```python
    # --- MCP OAuth (flag-gated; spec 2026-06-10-mcp-oauth-design.md) -------
    if config.MCP_OAUTH_ENABLED:

        @app.get(oauth_middleware.METADATA_PATH)
        def oauth_protected_resource():
            """RFC 9728 metadata: points MCP clients at the Supabase AS."""
            return oauth_middleware.protected_resource_metadata()
```

and change the mount line at the bottom to wrap the MCP app when the flag is on:

```python
    mcp_asgi = mcp_server.streamable_http_app()
    if config.MCP_OAUTH_ENABLED:
        mcp_asgi = oauth_middleware.OAuthLadderMiddleware(
            mcp_asgi, mcp_path=MCP_ENDPOINT_PATH
        )
    app.mount(MCP_MOUNT_PATH, mcp_asgi)
```

with imports added at the top of main.py:

```python
from app import config, oauth_middleware
```

(`app/config.py` is imported as a module here — check that nothing in main.py previously did `from app.config import ...`; if it did, keep both styles consistent with the file's existing imports.)

- [ ] **Step 4: Run** `test_oauth_middleware.py` → PASS, then the FULL mcp-server suite (flag defaults off in all pre-existing tests → zero behavior change) → all green.

- [ ] **Step 5: Commit**

```bash
git add services/mcp-server/app/oauth_middleware.py services/mcp-server/app/main.py services/mcp-server/tests/test_oauth_middleware.py
git commit -m "feat(oauth): dual-mode auth ladder middleware + RFC 9728 metadata (flag-gated)"
```

---

### Task 5: Principal ladder in `app/auth.py` + OAuth agent identity

**Files:**
- Modify: `services/mcp-server/app/auth.py`
- Modify: `services/mcp-server/app/mcp_app.py` (clientInfo plumb)
- Modify: `services/mcp-server/tests/test_auth.py` (append)

- [ ] **Step 1: Write the failing tests** — append to `services/mcp-server/tests/test_auth.py`:

```python
def _oauth_request(principal, headers=None):
    from app.oauth import OAuthPrincipal

    class _Req:
        def __init__(self):
            self.headers = headers or {}
            self.state = type("S", (), {})()
            self.state.oauth_principal = principal

    return _Req()


def _principal():
    from app.oauth import OAuthPrincipal

    return OAuthPrincipal(
        org_id="org-acme", account_slug="acme",
        user_id="11111111-2222-3333-4444-555555555555",
        client_id="client-abc", plan="pro", plan_overrides=None,
    )


def test_resolve_principal_prefers_oauth_state():
    from app.auth import resolve_principal

    org_id, agent_id = resolve_principal(_oauth_request(_principal()))
    assert org_id == "org-acme"
    assert agent_id == "oauth:11111111/client-abc"  # user_short/client_id fallback


def test_oauth_agent_uses_client_name_when_provided():
    from app.auth import resolve_principal

    request = _oauth_request(_principal())
    request.state.mcp_client_name = "Claude Code"
    org_id, agent_id = resolve_principal(request)
    assert agent_id == "oauth:11111111/claude-code"


def test_x_vex_agent_header_overrides_oauth_default():
    from app.auth import resolve_principal

    request = _oauth_request(_principal(), headers={"X-Vex-Agent": "my-bot"})
    org_id, agent_id = resolve_principal(request)
    assert agent_id == "my-bot"


def test_resolve_principal_plan_carries_oauth_plan():
    from app.auth import resolve_principal_plan

    org_id, agent_id, plan, overrides = resolve_principal_plan(
        _oauth_request(_principal())
    )
    assert (org_id, plan, overrides) == ("org-acme", "pro", None)


def test_no_oauth_state_falls_back_to_key_path(monkeypatch):
    """Requests without an OAuth principal use the existing key validation."""
    import app.auth as auth_mod

    class _Req:
        headers = {"X-Vex-Key": "ag_x", "X-Vex-Agent": "a1"}
        state = type("S", (), {})()  # no oauth_principal attribute

    class _KeyInfo:
        org_id = "org-key"
        plan = "free"
        plan_overrides = None

    monkeypatch.setattr(auth_mod, "verify_api_key", lambda request: _KeyInfo())
    assert auth_mod.resolve_principal(_Req()) == ("org-key", "a1")
```

Run: → FAIL.

- [ ] **Step 2: Implement in `services/mcp-server/app/auth.py`.** Add after `PROJECT_HEADER` / `resolve_project_from_request`:

```python
def _slugify_client_name(name: str) -> str:
    """Lowercase, spaces->dashes, strip anything outside [a-z0-9._-]."""
    import re

    slug = re.sub(r"\s+", "-", name.strip().lower())
    return re.sub(r"[^a-z0-9._-]", "", slug) or "client"


def _oauth_agent_id(request: Request, principal) -> str:
    """Default agent identity for OAuth callers (spec section 4).

    ``oauth:<first 8 chars of user uuid>/<client name slug>`` where the
    client name comes from the MCP ``initialize`` handshake when the tool
    layer plumbed it onto ``request.state.mcp_client_name``; falls back to
    the OAuth client id. The ``X-Vex-Agent`` header overrides entirely
    (checked by the callers before this is consulted).
    """
    client_name = getattr(request.state, "mcp_client_name", None)
    client_part = (
        _slugify_client_name(client_name) if client_name else principal.client_id
    )
    return f"oauth:{principal.user_id[:8]}/{client_part}"


def _oauth_principal_from(request: Request):
    """Return the middleware-attached OAuthPrincipal, or None."""
    return getattr(getattr(request, "state", None), "oauth_principal", None)
```

Then change `resolve_principal` and `resolve_principal_plan` to a ladder (key path untouched below the new branch):

```python
def resolve_principal(request: Request) -> tuple[str, str]:
    principal = _oauth_principal_from(request)
    if principal is not None:
        agent_id = request.headers.get("X-Vex-Agent") or _oauth_agent_id(
            request, principal
        )
        return principal.org_id, agent_id
    key_info = verify_api_key(request)
    agent_id = request.headers.get("X-Vex-Agent") or DEFAULT_AGENT_ID
    return key_info.org_id, agent_id


def resolve_principal_plan(request: Request) -> tuple[str, str, str, dict | None]:
    principal = _oauth_principal_from(request)
    if principal is not None:
        agent_id = request.headers.get("X-Vex-Agent") or _oauth_agent_id(
            request, principal
        )
        return principal.org_id, agent_id, principal.plan, principal.plan_overrides
    key_info = verify_api_key(request)
    agent_id = request.headers.get("X-Vex-Agent") or DEFAULT_AGENT_ID
    return key_info.org_id, agent_id, key_info.plan, key_info.plan_overrides
```

(Preserve both functions' existing docstrings, extended with the ladder description. `resolve_principal_dep` — the REST capture dependency — is deliberately NOT changed: capture stays key-only per the spec.)

- [ ] **Step 3: Plumb clientInfo in `services/mcp-server/app/mcp_app.py`.** In `_perform_write`, `_perform_recall`, `_perform_forget`, and `_perform_space`, right after `request = ctx.request_context.request` and its None-guard, add:

```python
        _attach_client_name(ctx, request)
```

and define once near the top of the module (after `_PROJECT_ANY`):

```python
def _attach_client_name(ctx: Context, request) -> None:
    """Expose the MCP client's self-reported name for OAuth agent identity.

    Best-effort: the Streamable-HTTP session carries the ``initialize``
    ``clientInfo``; absence (or any SDK shape change) leaves the attribute
    unset and the agent id falls back to the OAuth client id.
    """
    try:
        client_info = ctx.session.client_params.clientInfo
        if client_info and client_info.name:
            request.state.mcp_client_name = client_info.name
    except Exception:
        pass
```

(The implementer MUST verify the attribute path `ctx.session.client_params.clientInfo.name` against the installed SDK: `.venv/bin/python -c "import inspect, mcp.server.session as s; print(inspect.getsource(s.ServerSession.client_params.fget))"` — if the SDK exposes it differently, adapt the path inside `_attach_client_name` only; the try/except keeps any drift non-fatal.)

- [ ] **Step 4: Run** the new tests + FULL mcp-server suite → green. The key-path regression suite passing unchanged is the critical assertion of this task.

- [ ] **Step 5: Commit**

```bash
git add services/mcp-server/app/auth.py services/mcp-server/app/mcp_app.py services/mcp-server/tests/test_auth.py
git commit -m "feat(auth): OAuth principal ladder with per-user-per-client agent identity"
```

---

### Task 6: Rate limiting for OAuth principals

**Files:**
- Modify: `services/mcp-server/app/oauth.py` (append limiter)
- Modify: `services/mcp-server/app/oauth_middleware.py` (enforce after grant resolution)
- Create: `services/mcp-server/tests/test_oauth_rate_limit.py`

Spec §3: OAuth principals are rate-limited per `user_id:client_id` with the same sliding-window semantics as API keys (plan-capped rpm). The key path's limiter lives inside `KeyValidator.validate` and is not reusable without entangling the key cache, so this is a small standalone sliding window with the SAME observable behavior (429 + `Retry-After`).

- [ ] **Step 1: Read** `services/shared/shared/auth.py`'s sliding-window block (search for `rate` / `429` / `Retry-After` in `validate`) and `shared/plan_limits.py` for the per-plan rpm field name (used as the cap). Mirror the semantics; do not import the key limiter.

- [ ] **Step 2: Write the failing tests** — `services/mcp-server/tests/test_oauth_rate_limit.py`:

```python
"""Sliding-window rate limit per (user, client) for OAuth principals."""

from app.oauth import OAuthRateLimiter


def test_allows_under_limit():
    limiter = OAuthRateLimiter()
    for _ in range(5):
        retry_after = limiter.check("u-1:c-1", rpm=5, now=100.0)
        assert retry_after is None


def test_blocks_over_limit_with_retry_after():
    limiter = OAuthRateLimiter()
    for _ in range(5):
        assert limiter.check("u-1:c-1", rpm=5, now=100.0) is None
    retry_after = limiter.check("u-1:c-1", rpm=5, now=100.0)
    assert retry_after is not None and retry_after > 0


def test_window_slides():
    limiter = OAuthRateLimiter()
    for _ in range(5):
        assert limiter.check("u-1:c-1", rpm=5, now=100.0) is None
    assert limiter.check("u-1:c-1", rpm=5, now=100.0) is not None
    # 61 seconds later the window has slid past the burst.
    assert limiter.check("u-1:c-1", rpm=5, now=161.0) is None


def test_keys_are_isolated():
    limiter = OAuthRateLimiter()
    for _ in range(5):
        assert limiter.check("u-1:c-1", rpm=5, now=100.0) is None
    assert limiter.check("u-2:c-1", rpm=5, now=100.0) is None
```

Run: → FAIL (`ImportError: OAuthRateLimiter`).

- [ ] **Step 3: Implement** — append to `services/mcp-server/app/oauth.py`:

```python
_RATE_WINDOW_S = 60.0
# Default rpm for OAuth callers when the plan config exposes no explicit
# value; matches the key path's default per-key rpm.
DEFAULT_OAUTH_RPM = 1000


class OAuthRateLimiter:
    """Sliding-window request limiter keyed by ``user_id:client_id``.

    Same observable semantics as the API-key limiter (429 + Retry-After):
    at most ``rpm`` requests per trailing 60-second window per principal.
    In-memory and per-replica (matches the key limiter's deployment model).
    """

    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, *, rpm: int, now: Optional[float] = None) -> Optional[int]:
        """Record a hit; return None if allowed, else seconds to retry after."""
        ts = time.monotonic() if now is None else now
        cutoff = ts - _RATE_WINDOW_S
        with self._lock:
            hits = [h for h in self._hits.get(key, []) if h > cutoff]
            if len(hits) >= rpm:
                self._hits[key] = hits
                return max(1, int(hits[0] + _RATE_WINDOW_S - ts) + 1)
            hits.append(ts)
            self._hits[key] = hits
            return None
```

- [ ] **Step 4: Enforce in the middleware.** In `oauth_middleware.py`, add a module-level `_rate_limiter = OAuthRateLimiter()` (import it from `app.oauth`), and in `OAuthLadderMiddleware.__call__`, after the grant resolves successfully and before attaching the principal:

```python
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
```

(Verify the plan-config rpm attribute name against `shared/plan_limits.py` in Step 1 — if the field is named differently (e.g. `rate_limit_rpm`), use that name; if no rpm field exists, use `DEFAULT_OAUTH_RPM` unconditionally and note it in the commit body.)

Add a middleware-level test to `test_oauth_middleware.py`: monkeypatch `om._rate_limiter.check` to return `7` and assert a valid-Bearer-with-grant request yields 429 with `Retry-After: 7`.

- [ ] **Step 5: Run** the new tests + full mcp-server suite → green.

- [ ] **Step 6: Commit**

```bash
git add services/mcp-server/app/oauth.py services/mcp-server/app/oauth_middleware.py services/mcp-server/tests/test_oauth_rate_limit.py services/mcp-server/tests/test_oauth_middleware.py
git commit -m "feat(oauth): sliding-window rate limit per user:client principal"
```

---

### Task 7: Full verification

**Files:** none — verification only.

- [ ] **Step 1:** Both suites: `cd services/shared && python3 -m pytest tests/ -q` and `.venv/bin/python -m pytest services/mcp-server/tests -q` → ALL PASS.
- [ ] **Step 2:** Coverage on new modules: `.venv/bin/python -m pytest services/mcp-server/tests --cov=app.oauth --cov=app.oauth_middleware --cov-report=term-missing` → ≥80% each; add targeted tests for uncovered branches in NEW code.
- [ ] **Step 3:** `ruff check services/mcp-server/app/ && ruff format --check services/` → clean.
- [ ] **Step 4:** Flag-off regression proof: run the entire mcp-server suite with `MCP_OAUTH_ENABLED` unset (the default) — confirms zero behavior change for today's clients.
- [ ] **Step 5: Commit** any test additions, then hand off:

Rollout (spec section 8): deploy this dark (`MCP_OAUTH_ENABLED=0`); the Dashboard plan (separate repo) ships `oauth_grants` + consent page + Supabase OAuth-server config; THEN set `MCP_OAUTH_ENABLED=1`, `SUPABASE_AUTH_ISSUER`, `MCP_PUBLIC_URL` in production and dogfood from Claude Code + claude.ai.
