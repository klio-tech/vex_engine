"""An attributable caller's EXPLICIT org declaration is publication.

The discard rule protected against silent widening: hooks never chose to
publish, so an attributable capture is private no matter what the client
wrote in the payload. Chat broke the assumption behind it: a plot message
was said in the open room, the bridge declares scope='org' deliberately,
and forcing it private made every chat capture visible to nobody but the
workspace bot -- the team's own conversation, dark to the team.

The rule now: an attributable caller declaring EXACTLY 'org' gets org (an
explicit publication); anything else declared -- or nothing -- keeps the
private default. Hooks never declare, so their privacy is untouched.
"""

from app.capture import _hook_write_scope


class _Key:
    org_id = "org-1"
    key_id = "key-1"
    user_id = "11111111-2222-3333-4444-555555555555"
    is_service = True


class _HumanKey(_Key):
    is_service = False


class _Req:
    headers: dict = {}
    state = type("S", (), {})()


def test_service_token_declaring_org_gets_org():
    scope, user = _hook_write_scope(_Req(), _Key(), "org")
    assert scope == "org"
    assert user == _Key.user_id  # attribution is kept even when publishing


def test_a_human_key_declaring_org_stays_private():
    # Nobody chose to publish when a Stop hook fired: a human credential
    # declaring org is a client bug or an attack, never a publication.
    scope, user = _hook_write_scope(_Req(), _HumanKey(), "org")
    assert scope == "private" and user == _Key.user_id


def test_attributable_declaring_private_stays_private():
    scope, user = _hook_write_scope(_Req(), _Key(), "private")
    assert scope == "private" and user == _Key.user_id


def test_attributable_declaring_agent_is_honoured():
    # This asserted `private` while 'agent' was a deprecated read-only idea,
    # so collapsing it lost nothing. `agent_only` recall changed the premise:
    # it filters `scope = 'agent' AND agent_id = :agent_id`, so collapsing the
    # write makes that predicate match nothing forever and the isolation it
    # promises silently inert. 'agent' is now an explicit destination, not an
    # odd scope. See test_capture_agent_scope.py for the full rule and for why
    # it is not a widening (a default read's ladder has no agent arm).
    scope, user = _hook_write_scope(_Req(), _Key(), "agent")
    assert scope == "agent" and user == _Key.user_id


def test_an_unrecognised_scope_still_falls_back_to_private():
    # The discard rule itself is unchanged: only the two named exceptions
    # ('org' for a service token, 'agent') survive attribution.
    scope, user = _hook_write_scope(_Req(), _Key(), "project")
    assert scope == "private" and user == _Key.user_id


def test_project_resolution_never_renarrows_an_explicit_org_publication():
    """The end-to-end shape of the live bug: the gate said org, then the
    project-default resolution re-resolved the attributable capture and
    silently returned it to private. Found in production with a spy on
    _hook_write_scope: gate -> ('org', bot), response -> scope private.
    The resolution may only run on the private default."""
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "app" / "capture.py"
    text = src.read_text()
    bare = len(re.findall(r"if user_id:\n\s+scope = await anyio", text))
    guarded = len(re.findall(r'if user_id and scope == "private":', text))
    assert bare == 0, "an unguarded resolve_capture_scope override survives"
    assert guarded >= 2, "both capture routes must carry the guard"
