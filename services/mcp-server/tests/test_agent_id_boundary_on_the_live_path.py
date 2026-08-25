"""The ``agent_id`` bound has to hold on the module that serves requests.

``test_agent_id_boundary_applied.py`` pins the four readers in ``app.auth`` and
passes. It has never covered production: ``mcp_app.py`` imports
``resolve_principal`` / ``resolve_principal_plan`` from ``app.identity``, and
``app.auth.resolve_principal`` has no production caller at all. So the suite
asserted a boundary that the live path did not apply.

Executed pre-fix, against ``app.identity``::

    X-Vex-Agent header, 500 chars   -> agent_id len 500   (column is VARCHAR(128) PK)
    X-Vex-Agent header, "   "       -> agent_id '   '     (expected the default)

The first overflows ``agents.agent_id``; the second mints a junk principal that
dedup and agent-scoped recall then key on.
"""

import pytest
from shared.auth import MAX_AGENT_ID_LEN

from app import identity as identity_module
from app.auth import DEFAULT_AGENT_ID


class _State:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Req:
    def __init__(self, headers=None, state=None):
        self.headers = headers or {}
        self.state = state or _State()


class _KeyInfo:
    org_id = "org-1"
    plan = "pro"
    plan_overrides = None
    user_id = "user-1"


class _OAuthPrincipal:
    org_id = "org-oauth"
    user_id = "11111111-2222-3333-4444-555555555555"
    client_id = "client-1"
    plan = "pro"
    plan_overrides = None


@pytest.fixture
def key_path(monkeypatch):
    monkeypatch.setattr(identity_module, "request_key_info", lambda request: _KeyInfo())


LONG = "a" * 500


def test_resolve_principal_bounds_the_header(key_path):
    _, agent_id = identity_module.resolve_principal(_Req({"X-Vex-Agent": LONG}))
    assert len(agent_id) <= MAX_AGENT_ID_LEN


def test_resolve_principal_plan_bounds_the_header(key_path):
    _, agent_id, _, _ = identity_module.resolve_principal_plan(
        _Req({"X-Vex-Agent": LONG})
    )
    assert len(agent_id) <= MAX_AGENT_ID_LEN


def test_the_klio_header_is_bounded_too(key_path):
    # headers.py accepts X-Klio-Agent as well; the bound cannot depend on which
    # spelling the client used.
    _, agent_id = identity_module.resolve_principal(_Req({"X-Klio-Agent": LONG}))
    assert len(agent_id) <= MAX_AGENT_ID_LEN


def test_the_oauth_slug_path_is_bounded_too(monkeypatch):
    # The derived oauth:<user8>/<slug> id is built from the client-declared
    # clientInfo.name, so it is caller-influenced and needs the same bound.
    #
    # Patch app.auth, not this module: resolve_principal_plan imports
    # _oauth_principal_from from app.auth inside the function body, so a
    # monkeypatch on app.identity is never consulted.
    from app import auth as auth_module

    principal = _OAuthPrincipal()
    monkeypatch.setattr(auth_module, "_oauth_principal_from", lambda request: principal)
    request = _Req({}, state=_State(mcp_client_name=LONG))
    _, agent_id = identity_module.resolve_principal(request)
    assert len(agent_id) <= MAX_AGENT_ID_LEN


def test_a_whitespace_header_yields_the_default(key_path):
    # "   " is truthy, so an `or DEFAULT` guard lets it through as a principal.
    _, agent_id = identity_module.resolve_principal(_Req({"X-Vex-Agent": "   "}))
    assert agent_id == DEFAULT_AGENT_ID


def test_a_normal_header_is_still_passed_through(key_path):
    _, agent_id = identity_module.resolve_principal(
        _Req({"X-Vex-Agent": "claude-code"})
    )
    assert agent_id == "claude-code"


def test_a_missing_header_still_yields_the_default(key_path):
    _, agent_id = identity_module.resolve_principal(_Req({}))
    assert agent_id == DEFAULT_AGENT_ID


def test_the_bound_is_deterministic(key_path):
    # dedup and agent-scoped recall key on this string, so the same header must
    # map to the same principal on every request and across restarts.
    first = identity_module.resolve_principal(_Req({"X-Vex-Agent": LONG}))[1]
    second = identity_module.resolve_principal(_Req({"X-Vex-Agent": LONG}))[1]
    assert first == second
