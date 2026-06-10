"""Unit tests for the MCP server's X-Vex-Key auth bridge.

These tests mock at the ``app.auth.get_validator`` boundary so that the
underlying ``KeyValidator`` (and its SQLAlchemy engine) is never
constructed — no live Postgres is required.  ``verify_api_key`` and
``resolve_principal`` are exercised directly with a lightweight fake
``Request`` carrying only the headers under test.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from shared.auth import AuthError, KeyInfo


def _request(headers: dict | None = None):
    """Build a minimal stand-in for a FastAPI ``Request``.

    ``verify_api_key`` only reads ``request.headers.get(...)``, so a
    ``SimpleNamespace`` wrapping a plain dict is sufficient and avoids
    constructing a real ASGI scope.
    """
    return SimpleNamespace(headers=headers or {})


def test_missing_key_raises_401():
    """No X-Vex-Key (and no fallback) -> HTTPException 401."""
    from app.auth import verify_api_key

    with pytest.raises(HTTPException) as exc_info:
        verify_api_key(_request({}))

    assert exc_info.value.status_code == 401
    assert "X-Vex-Key" in exc_info.value.detail


def test_invalid_key_raises_401():
    """Validator raising AuthError(401) -> HTTPException 401."""
    from app.auth import verify_api_key

    with patch("app.auth.get_validator") as mock_get_validator:
        mock_get_validator.return_value.validate.side_effect = AuthError(
            401, "Invalid API key"
        )
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key(_request({"X-Vex-Key": "bad-key"}))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid API key"


def test_valid_key_returns_org():
    """Validator returning KeyInfo -> verify_api_key returns it verbatim."""
    from app.auth import verify_api_key

    key_info = KeyInfo(org_id="org_x", key_id="key_1", scopes=["ingest"])
    with patch("app.auth.get_validator") as mock_get_validator:
        mock_get_validator.return_value.validate.return_value = key_info
        result = verify_api_key(_request({"X-Vex-Key": "good-key"}))

    assert result is key_info
    assert result.org_id == "org_x"
    mock_get_validator.return_value.validate.assert_called_once_with("good-key")


def test_resolve_principal_returns_org_and_agent():
    """resolve_principal returns (org_id, agent_id) from X-Vex-Agent;
    falls back to the default agent when the header is absent."""
    from app.auth import DEFAULT_AGENT_ID, resolve_principal

    key_info = KeyInfo(org_id="org_x", key_id="key_1", scopes=["ingest"])

    with patch("app.auth.get_validator") as mock_get_validator:
        mock_get_validator.return_value.validate.return_value = key_info

        # With explicit agent header
        org_id, agent_id = resolve_principal(
            _request({"X-Vex-Key": "good-key", "X-Vex-Agent": "agent-42"})
        )
        assert org_id == "org_x"
        assert agent_id == "agent-42"

        # Without agent header -> default
        org_id, agent_id = resolve_principal(_request({"X-Vex-Key": "good-key"}))
        assert org_id == "org_x"
        assert agent_id == DEFAULT_AGENT_ID


def test_resolve_principal_propagates_auth_error():
    """resolve_principal surfaces auth failures (no key) as HTTPException 401."""
    from app.auth import resolve_principal

    with pytest.raises(HTTPException) as exc_info:
        resolve_principal(_request({"X-Vex-Agent": "agent-42"}))

    assert exc_info.value.status_code == 401


def test_auth_error_retry_after_header():
    """AuthError carrying retry_after_seconds -> 429 with Retry-After header."""
    from app.auth import verify_api_key

    with patch("app.auth.get_validator") as mock_get_validator:
        mock_get_validator.return_value.validate.side_effect = AuthError(
            429, "Rate limit exceeded", retry_after_seconds=60
        )
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key(_request({"X-Vex-Key": "good-key"}))

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers == {"Retry-After": "60"}


def test_x_agentguard_key_fallback_header():
    """Legacy X-AgentGuard-Key header is accepted as a fallback."""
    from app.auth import verify_api_key

    key_info = KeyInfo(org_id="org_legacy", key_id="key_1", scopes=["ingest"])
    with patch("app.auth.get_validator") as mock_get_validator:
        mock_get_validator.return_value.validate.return_value = key_info
        result = verify_api_key(_request({"X-AgentGuard-Key": "ag_legacy_key"}))

    assert result.org_id == "org_legacy"
    mock_get_validator.return_value.validate.assert_called_once_with("ag_legacy_key")


def test_get_validator_uses_memory_scope():
    """Brain access is gated by the dedicated ``memory`` scope, not ``ingest``.

    ``get_validator`` must construct the shared ``KeyValidator`` with
    ``required_scope="memory"`` so a key needs the brain-specific scope to
    drive the MCP memory tools.
    """
    import app.auth as auth_module

    # The validator is a module-level singleton — reset around the test so
    # a previously-built instance (or one we build here) doesn't leak.
    auth_module._validator = None
    try:
        with patch("app.auth.KeyValidator") as mock_validator_cls:
            mock_validator_cls.return_value = MagicMock()
            auth_module.get_validator()

        mock_validator_cls.assert_called_once()
        _, kwargs = mock_validator_cls.call_args
        assert kwargs["required_scope"] == "memory"
    finally:
        auth_module._validator = None


# ---------------------------------------------------------------------------
# X-Vex-Project header resolution (Task 9)
# ---------------------------------------------------------------------------


def test_project_header_constant():
    from app.auth import PROJECT_HEADER

    assert PROJECT_HEADER == "X-Vex-Project"


def test_resolve_project_header_returns_none_without_header():
    from app.auth import resolve_project_from_request

    class _Req:
        headers = {}

    assert resolve_project_from_request(_Req(), object(), org_id="o") is None


def test_resolve_project_header_resolves_via_find(monkeypatch):
    import app.auth as auth_mod

    monkeypatch.setattr(
        auth_mod.brain, "find_project_id", lambda db, *, org_id, identifier: "proj-1"
    )

    class _Req:
        headers = {"X-Vex-Project": "git@github.com:acme/widgets.git"}

    assert (
        auth_mod.resolve_project_from_request(_Req(), object(), org_id="o")
        == "proj-1"
    )


def test_resolve_project_header_fails_open(monkeypatch):
    import app.auth as auth_mod

    def _boom(db, *, org_id, identifier):
        raise RuntimeError("db down")

    monkeypatch.setattr(auth_mod.brain, "find_project_id", _boom)

    class _Req:
        headers = {"X-Vex-Project": "anything"}

    assert auth_mod.resolve_project_from_request(_Req(), object(), org_id="o") is None
