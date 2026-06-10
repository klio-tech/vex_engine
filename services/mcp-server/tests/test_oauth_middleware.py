"""Transport-level auth ladder + RFC 9728 protected-resource metadata."""

import importlib
import os

import pytest
from fastapi.testclient import TestClient

from app.oauth import OAuthIdentity, OAuthPrincipal, VerifierUnavailable


@pytest.fixture(autouse=True)
def _restore_modules():
    """Reload config / oauth_middleware / main from clean env after each test.

    The _client helper reloads these modules under patched env vars.
    Without this teardown fixture later test files (or later tests in this
    file) would observe a flag-on app because sys.modules retains the
    reloaded state even after monkeypatch restores env vars.
    """
    yield
    for key in (
        "MCP_OAUTH_ENABLED",
        "SUPABASE_AUTH_ISSUER",
        "MCP_PUBLIC_URL",
        "SUPABASE_DATABASE_URL",
    ):
        os.environ.pop(key, None)
    import app.config as config
    import app.oauth_middleware as om
    import app.main as main

    importlib.reload(config)
    importlib.reload(om)
    importlib.reload(main)


def _client(monkeypatch, *, enabled, verify=None, resolve=None):
    """Build a TestClient with the given OAuth flag state and optional seams."""
    monkeypatch.setenv("MCP_OAUTH_ENABLED", "1" if enabled else "0")
    monkeypatch.setenv("SUPABASE_AUTH_ISSUER", "https://proj.supabase.co/auth/v1")
    monkeypatch.setenv("MCP_PUBLIC_URL", "https://mcp.klio.tech")
    # Required by the startup config-validation check when OAuth is enabled.
    monkeypatch.setenv("SUPABASE_DATABASE_URL", "postgresql://test:test@localhost/test")

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
    return TestClient(main.create_app(), raise_server_exceptions=False)


def test_flag_off_preserves_today(monkeypatch):
    client = _client(monkeypatch, enabled=False)
    # /healthz works; no metadata route; no challenge on /mcp POSTs.
    assert client.get("/healthz").status_code == 200
    assert client.get("/.well-known/oauth-protected-resource").status_code == 404


def test_metadata_served_when_enabled(monkeypatch):
    client = _client(monkeypatch, enabled=True)
    # The `resource` identifier is the full MCP endpoint URL the client dials,
    # not the bare origin — served at both the bare and path-aware locations.
    for path in (
        "/.well-known/oauth-protected-resource",
        "/.well-known/oauth-protected-resource/mcp",
    ):
        resp = client.get(path)
        assert resp.status_code == 200, path
        body = resp.json()
        assert body["resource"] == "https://mcp.klio.tech/mcp"
        assert body["authorization_servers"] == ["https://proj.supabase.co/auth/v1"]
        assert "header" in body["bearer_methods_supported"]


def test_unauthenticated_mcp_gets_401_challenge(monkeypatch):
    client = _client(monkeypatch, enabled=True)
    resp = client.post("/mcp", json={})
    assert resp.status_code == 401
    challenge = resp.headers["WWW-Authenticate"]
    assert challenge.startswith("Bearer")
    assert "resource_metadata=" in challenge
    # Challenge points at the path-aware metadata location for the /mcp resource.
    assert "/.well-known/oauth-protected-resource/mcp" in challenge


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
    # _verify_token is now a sync seam; monkeypatch with a plain function.
    def deny(token):
        return None

    client = _client(monkeypatch, enabled=True, verify=deny)
    resp = client.post("/mcp", json={}, headers={"Authorization": "Bearer bad-token"})
    assert resp.status_code == 401
    assert "resource_metadata=" in resp.headers["WWW-Authenticate"]


def test_valid_bearer_without_grant_gets_403(monkeypatch):
    def allow(token):
        return OAuthIdentity(user_id="u-1", client_id="c-1")

    def no_grant(identity):
        return None

    client = _client(monkeypatch, enabled=True, verify=allow, resolve=no_grant)
    resp = client.post("/mcp", json={}, headers={"Authorization": "Bearer good-token"})
    assert resp.status_code == 403
    assert resp.json()["error"] == "not_connected"
    assert "app.klio.tech" in resp.json()["detail"]


def test_valid_bearer_with_grant_passes_through(monkeypatch):
    def allow(token):
        return OAuthIdentity(user_id="u-1", client_id="c-1")

    def grant(identity):
        return OAuthPrincipal(
            org_id="org-1",
            account_slug="acme",
            user_id="u-1",
            client_id="c-1",
            plan="pro",
            plan_overrides=None,
        )

    client = _client(monkeypatch, enabled=True, verify=allow, resolve=grant)
    resp = client.post("/mcp", json={}, headers={"Authorization": "Bearer good-token"})
    # Passes our ladder; the MCP transport handles (and rejects) the
    # malformed MCP body itself — anything but our 401/403 challenge shapes.
    assert resp.status_code not in (401, 403)


def test_jwks_unreachable_returns_503(monkeypatch):
    """JWKS endpoint down → 503 with Retry-After, never 401 (spec section 5)."""

    def unavailable(token):
        raise VerifierUnavailable("network down")

    client = _client(monkeypatch, enabled=True, verify=unavailable)
    resp = client.post("/mcp", json={}, headers={"Authorization": "Bearer any-token"})
    assert resp.status_code == 503
    assert resp.json()["error"] == "verifier_unavailable"
    assert resp.headers.get("Retry-After") == "5"


def test_flag_on_missing_config_fails_fast(monkeypatch):
    """MCP_OAUTH_ENABLED=1 with missing config must raise RuntimeError at startup.

    We reload config and oauth_middleware (to pick up the patched env vars) but
    call ``create_app()`` directly inside ``pytest.raises`` rather than via
    module reload, because ``main.py`` executes ``app = create_app()`` at module
    level — letting the reload raise outside the ``pytest.raises`` block would
    be a confusing test failure.
    """
    monkeypatch.setenv("MCP_OAUTH_ENABLED", "1")
    monkeypatch.setenv("MCP_PUBLIC_URL", "https://mcp.klio.tech")
    # Deliberately omit SUPABASE_AUTH_ISSUER and SUPABASE_DATABASE_URL.
    monkeypatch.delenv("SUPABASE_AUTH_ISSUER", raising=False)
    monkeypatch.delenv("SUPABASE_DATABASE_URL", raising=False)

    import app.config as config
    import app.oauth_middleware as om
    import app.main as main

    importlib.reload(config)
    importlib.reload(om)

    with pytest.raises(RuntimeError, match="SUPABASE_AUTH_ISSUER"):
        main.create_app()


def test_rate_limited_bearer_gets_429(monkeypatch):
    """When _rate_limiter.check returns a retry-after value, respond 429."""

    def allow(token):
        return OAuthIdentity(user_id="u-1", client_id="c-1")

    def grant(identity):
        return OAuthPrincipal(
            org_id="org-1",
            account_slug="acme",
            user_id="u-1",
            client_id="c-1",
            plan="pro",
            plan_overrides=None,
        )

    client = _client(monkeypatch, enabled=True, verify=allow, resolve=grant)

    import app.oauth_middleware as om

    monkeypatch.setattr(om._rate_limiter, "check", lambda key, **kw: 7)

    resp = client.post("/mcp", json={}, headers={"Authorization": "Bearer good-token"})
    assert resp.status_code == 429
    assert resp.json()["error"] == "rate_limited"
    assert resp.headers.get("Retry-After") == "7"


def test_healthz_and_capture_unaffected(monkeypatch):
    client = _client(monkeypatch, enabled=True)
    assert client.get("/healthz").status_code == 200
    # capture keeps its own key auth (401 from ITS dependency, no challenge).
    resp = client.post("/capture/event", json={"content": "x"})
    assert resp.status_code == 401
    assert "resource_metadata" not in resp.headers.get("WWW-Authenticate", "")


def test_agentguard_key_header_passes_through(monkeypatch):
    """Legacy X-AgentGuard-Key requests bypass Bearer enforcement entirely.

    This is the second key-header variant on the OR branch (line 180 of
    oauth_middleware.py).  The request must reach the MCP transport, not
    receive our 401 OAuth challenge.
    """
    client = _client(monkeypatch, enabled=True)
    resp = client.post("/mcp", json={}, headers={"X-AgentGuard-Key": "ag_legacy_key"})
    # Reaches the MCP transport (which may 400/422 the malformed JSON-RPC body,
    # but MUST NOT return our OAuth 401 challenge).
    assert resp.status_code != 401 or "resource_metadata" not in resp.headers.get(
        "WWW-Authenticate", ""
    )
