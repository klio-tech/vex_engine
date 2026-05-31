"""resolve_principal_plan returns (org_id, agent_id, plan)."""

from unittest.mock import MagicMock, patch

from shared.auth import KeyInfo

from app import auth


def _request(headers: dict):
    req = MagicMock()
    req.headers.get = lambda k, default=None: headers.get(k, default)
    return req


def test_resolve_principal_plan_returns_plan():
    info = KeyInfo(org_id="org-1", key_id="k1", scopes=["memory"], plan="pro")
    with patch.object(auth, "verify_api_key", return_value=info):
        org_id, agent_id, plan = auth.resolve_principal_plan(
            _request({"X-Vex-Agent": "claude-code"})
        )
    assert (org_id, agent_id, plan) == ("org-1", "claude-code", "pro")


def test_resolve_principal_plan_defaults_agent():
    info = KeyInfo(org_id="org-1", key_id="k1", scopes=["memory"], plan="free")
    with patch.object(auth, "verify_api_key", return_value=info):
        _, agent_id, _ = auth.resolve_principal_plan(_request({}))
    assert agent_id == auth.DEFAULT_AGENT_ID
