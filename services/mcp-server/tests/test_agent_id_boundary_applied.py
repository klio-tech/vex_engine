"""The ``agent_id`` bound has to be applied where the header is read.

``shared.auth.normalize_agent_id`` bounding the value is necessary and not
sufficient — the finding is only closed if every place that reads
``X-Vex-Agent`` (or derives an id from the OAuth ``clientInfo.name``) runs it
through. There are four such places in ``app.auth``; this pins all of them.

Executed pre-fix, both paths overflowed ``VARCHAR(128)``::

    X-Vex-Agent header, 500 chars           -> agent_id len 500
    OAuth slug path, clientInfo.name 500 ch -> agent_id len 515
"""

import pytest
from shared.auth import MAX_AGENT_ID_LEN

from app import auth as auth_module
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


@pytest.fixture
def key_path(monkeypatch):
    monkeypatch.setattr(auth_module, "verify_api_key", lambda request: _KeyInfo())


LONG = "a" * 500


def test_resolve_principal_bounds_the_header(key_path):
    _, agent_id = auth_module.resolve_principal(_Req({"X-Vex-Agent": LONG}))
    assert len(agent_id) <= MAX_AGENT_ID_LEN


def test_resolve_principal_plan_bounds_the_header(key_path):
    _, agent_id, _, _ = auth_module.resolve_principal_plan(_Req({"X-Vex-Agent": LONG}))
    assert len(agent_id) <= MAX_AGENT_ID_LEN


def test_resolve_principal_dep_bounds_the_header():
    _, agent_id = auth_module.resolve_principal_dep(
        _Req({"X-Vex-Agent": LONG}), key_info=_KeyInfo()
    )
    assert len(agent_id) <= MAX_AGENT_ID_LEN


def test_the_oauth_slug_path_is_bounded_too(monkeypatch):
    """``clientInfo.name`` is client input as much as the header is.

    ``oauth:<user8>/<slug>`` added 15 characters on top of it, which is why a
    500-char name produced a 515-char id.
    """
    from app.oauth import OAuthPrincipal

    principal = OAuthPrincipal(
        org_id="org-1",
        account_slug="acct-1",
        user_id="uuuuuuuu-1111-2222-3333-444444444444",
        client_id="client-1",
        plan="pro",
        plan_overrides=None,
    )
    request = _Req(state=_State(oauth_principal=principal, mcp_client_name="n" * 500))
    _, agent_id = auth_module.resolve_principal(request)
    assert len(agent_id) <= MAX_AGENT_ID_LEN


def test_a_normal_header_is_still_passed_through(key_path):
    """Positive control: bounding must not rewrite real agent ids."""
    _, agent_id = auth_module.resolve_principal(_Req({"X-Vex-Agent": "claude-code"}))
    assert agent_id == "claude-code"


def test_a_missing_header_still_yields_the_default(key_path):
    """Positive control on the other branch."""
    _, agent_id = auth_module.resolve_principal(_Req({}))
    assert agent_id == DEFAULT_AGENT_ID
