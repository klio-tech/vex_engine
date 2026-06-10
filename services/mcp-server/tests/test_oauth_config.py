"""Config flags for MCP OAuth (engine side)."""

import importlib
import os

import pytest


@pytest.fixture(autouse=True)
def _restore_config():
    """Reload app.config from a clean env after each test.

    The tests reload the module under patched env vars; monkeypatch restores
    the ENV at teardown but not the already-reloaded MODULE, so without this
    fixture later test files would observe stale flag values via
    sys.modules["app.config"].
    """
    yield
    import app.config as config

    for key in ("MCP_OAUTH_ENABLED", "SUPABASE_AUTH_ISSUER", "MCP_PUBLIC_URL"):
        os.environ.pop(key, None)
    importlib.reload(config)


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
