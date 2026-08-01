"""The single identity rule.

OAuth wins over the key because an OAuth principal is a verified human
session, while a key is a bearer token that a person created. Both are
trustworthy; neither is X-Vex-Agent.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.identity import resolve_user_id
from app.oauth import OAuthPrincipal
from shared.auth import KeyInfo

USER = "11111111-2222-3333-4444-555555555555"
OTHER = "99999999-8888-7777-6666-555555555555"


def _request(oauth_principal=None, headers=None):
    req = MagicMock()
    req.state = SimpleNamespace()
    if oauth_principal is not None:
        req.state.oauth_principal = oauth_principal
    req.headers = headers or {}
    return req


def _principal(user_id=USER):
    return OAuthPrincipal(
        org_id="acme",
        account_slug="acme",
        user_id=user_id,
        client_id="cli",
        plan="free",
        plan_overrides=None,
    )


def test_oauth_principal_supplies_the_full_uuid():
    """Not the 8-char prefix that agent_id embeds — the real value."""
    assert resolve_user_id(_request(oauth_principal=_principal())) == USER


def test_api_key_created_by_is_used_when_there_is_no_oauth():
    info = KeyInfo(org_id="acme", key_id="k1", scopes=[], user_id=USER)
    assert resolve_user_id(_request(), key_info=info) == USER


def test_oauth_wins_when_both_are_present():
    info = KeyInfo(org_id="acme", key_id="k1", scopes=[], user_id=OTHER)
    got = resolve_user_id(_request(oauth_principal=_principal(USER)), key_info=info)
    assert got == USER


def test_returns_none_when_neither_credential_identifies_a_person():
    """Legal state. The caller writes org-scoped and logs; it does not fail."""
    assert resolve_user_id(_request()) is None
    assert resolve_user_id(_request(), key_info=KeyInfo("acme", "k1", [])) is None


def test_x_vex_agent_is_never_an_identity_source():
    """The security property of this module.

    X-Vex-Agent is declared by the client. If it could set identity, any
    caller could read another person's private memories by setting a header.
    """
    req = _request(headers={"X-Vex-Agent": OTHER, "x-vex-agent": OTHER})
    assert resolve_user_id(req) is None


def test_x_vex_agent_cannot_override_a_real_credential():
    req = _request(oauth_principal=_principal(USER), headers={"X-Vex-Agent": OTHER})
    assert resolve_user_id(req) == USER


def test_missing_state_attribute_is_tolerated():
    """Key-path requests and test fakes have no oauth_principal attribute."""
    req = MagicMock()
    req.state = SimpleNamespace()
    req.headers = {}
    assert resolve_user_id(req) is None


def test_a_non_principal_object_on_state_is_ignored():
    """MagicMock-patched requests in pre-existing tests return a truthy mock
    for any attribute access; that must not be read as an identity."""
    req = _request(oauth_principal=object())
    assert resolve_user_id(req) is None


# ---------------------------------------------------------------------------
# Phase 2 turns the routing on.
#
# The two tests below replace phase 1's no-op guards, which asserted the exact
# opposite: that NOTHING called resolve_user_id and that capture.py contained
# no private scope. Those guards existed to prove phase 1 was schema-only and
# therefore a safe place to stop. Phase 2 is the phase they were waiting for,
# so they are inverted rather than deleted — the file still mechanically
# asserts where identity is and is not wired, it now asserts the other state.
# ---------------------------------------------------------------------------


def test_identity_is_wired_into_the_write_and_read_paths():
    """The inverse of phase 1's guard: routing now depends on identity.

    Named explicitly rather than asserted as "non-empty" so that a module
    silently LOSING its identity call is a failure. capture.py routing to
    private without consulting resolve_user_id would be an owner-less private
    row; curator.py without it would read across users.
    """
    import pathlib

    app_dir = pathlib.Path(__file__).resolve().parents[1] / "app"
    callers = {
        p.name
        for p in app_dir.glob("*.py")
        if p.name != "identity.py"
        and (
            "resolve_user_id" in p.read_text()
            or "resolve_request_user_id" in p.read_text()
        )
    }
    assert {"capture.py", "mcp_app.py"} <= callers, (
        f"identity must be wired into capture and the MCP tools; found: {callers}"
    )


def test_capture_routes_hook_writes_to_private():
    """The 33,037 hook rows a month now land in their author's private scope."""
    import pathlib

    capture = (
        pathlib.Path(__file__).resolve().parents[1] / "app" / "capture.py"
    ).read_text()
    assert '"private", user_id' in capture
    assert "resolve_user_id" in capture
