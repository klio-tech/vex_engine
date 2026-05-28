"""Tests for the GET /verify endpoint.

``/verify`` is the lightweight check the ``klio init`` cloud flow calls to
confirm a pasted API key (valid + has the ``memory`` scope) before writing the
agent MCP config. It reuses the ``verify_api_key`` dependency, so a 200 means
the key authenticated AND carried the required scope; 401/403 otherwise.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient
from shared.auth import AuthError

from app.main import create_app


def test_verify_ok_with_valid_key(client):
    """A valid (auth-bypassed) key returns 200 with org + scopes."""
    resp = client.get("/verify", headers={"X-Vex-Key": "anything"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is True
    assert body["org_id"] == "test-org"
    assert "scopes" in body


def test_verify_missing_key_returns_401():
    """No X-Vex-Key header → 401 (dependency rejects before any validator call)."""
    app = create_app()
    with TestClient(app) as c:
        resp = c.get("/verify")
    assert resp.status_code == 401
    assert "X-Vex-Key" in resp.json()["detail"]


def test_verify_missing_memory_scope_returns_403():
    """A key the validator rejects for scope → 403 surfaced to the caller."""
    app = create_app()
    with patch("app.auth.get_validator") as mock_get_validator:
        mock_get_validator.return_value.validate.side_effect = AuthError(
            403, "API key missing required scope: memory"
        )
        with TestClient(app) as c:
            resp = c.get("/verify", headers={"X-Vex-Key": "ingest-only-key"})
    assert resp.status_code == 403
    assert "memory" in resp.json()["detail"]
