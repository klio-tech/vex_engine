"""An explicit `scope="agent"` declaration must survive to the row.

`agent_only` recall filters `scope = 'agent' AND agent_id = :agent_id`. If no
row is ever written with `scope='agent'`, that predicate matches nothing --
forever -- and the isolation it promises is silently inert rather than
enforced. A consumer switching to it sees memory stop working, with no error.

That is what was happening. `_hook_write_scope` discards the declared scope
for any caller it can attribute to a person, and an engine key IS attributable:
`shared.auth` sets `KeyInfo.user_id` from the key's `created_by`. So every
capture from an integration key -- `scope="agent"` included -- was stored as
`private`, and the API reported the downgrade in its response while the client
carried on believing it had partitioned its users.

Why honouring 'agent' does not re-open the widening the discard rule prevents:
a default read's ladder is (org | private-mine | project-mine) with NO agent
arm, so an agent row never reaches the team brain. It is not a publication,
which is why -- unlike 'org' -- it needs no service token. Hooks are untouched
either way: they never declare a scope.
"""

from app.capture import _hook_write_scope


class _ServiceKey:
    org_id = "org-1"
    key_id = "key-1"
    user_id = "11111111-2222-3333-4444-555555555555"
    is_service = True


class _IntegrationKey(_ServiceKey):
    """An `ag_live_` engine key: attributable (user_id = created_by) but NOT
    a workspace service token."""

    is_service = False


class _Req:
    headers: dict = {}
    state = type("S", (), {})()


def test_an_explicit_agent_declaration_is_honoured():
    """The blocker: this returned ('private', ...) and made agent_only inert."""
    scope, user = _hook_write_scope(_Req(), _IntegrationKey(), "agent")

    assert scope == "agent"
    assert user == _IntegrationKey.user_id


def test_agent_does_not_require_a_service_token():
    """Unlike 'org', which publishes to the team brain, an agent row reaches
    no default read — so gating it on is_service would only break the
    integration keys that need it, for no privacy gain."""
    service, _ = _hook_write_scope(_Req(), _ServiceKey(), "agent")
    integration, _ = _hook_write_scope(_Req(), _IntegrationKey(), "agent")

    assert service == integration == "agent"


def test_attribution_is_kept_on_an_agent_row():
    """Provenance survives. The agent arm carries no user predicate, so this
    does not change who can read the row."""
    _scope, user = _hook_write_scope(_Req(), _IntegrationKey(), "agent")

    assert user == _IntegrationKey.user_id


def test_an_undeclared_capture_is_still_private():
    """The rule this exception sits inside is unchanged: nobody chose to
    publish when a Stop hook fired, and hooks never declare a scope."""
    scope, user = _hook_write_scope(_Req(), _IntegrationKey(), "private")

    assert scope == "private"
    assert user == _IntegrationKey.user_id


def test_org_still_needs_a_service_token():
    """Regression pin on the pre-existing exception."""
    assert _hook_write_scope(_Req(), _ServiceKey(), "org")[0] == "org"
    assert _hook_write_scope(_Req(), _IntegrationKey(), "org")[0] == "private"


def test_an_unattributable_caller_is_unchanged():
    """The other branch keeps honouring the declared scope."""

    class _Anon(_IntegrationKey):
        user_id = None

    scope, user = _hook_write_scope(_Req(), _Anon(), "agent")

    assert scope == "agent"
    assert user is None
