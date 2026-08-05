"""``MCP_OAUTH_ENABLED`` was the one bare ``== "1"`` with a security surface.

Executed against the pre-fix ``os.environ.get("MCP_OAUTH_ENABLED", "0") == "1"``::

    'true'  bare==False  _parse_bool=True
    'TRUE'  bare==False  _parse_bool=True
    'yes'   bare==False  _parse_bool=True
    'on'    bare==False  _parse_bool=True
    '1'     bare==True   _parse_bool=True

Two things went wrong at once. The operator asked for OAuth and did not get it;
and ``main.py:115`` gates its "required OAuth config is missing" startup check
on the *same* flag, so a deployment that set ``MCP_OAUTH_ENABLED=true`` without
``SUPABASE_AUTH_ISSUER`` started clean and served unauthenticated MCP requests.
There was no signal anywhere.
"""

import importlib

import pytest

from app import config as config_module


def _reload(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("MCP_OAUTH_ENABLED", raising=False)
    else:
        monkeypatch.setenv("MCP_OAUTH_ENABLED", value)
    return importlib.reload(config_module)


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    yield
    monkeypatch.delenv("MCP_OAUTH_ENABLED", raising=False)
    importlib.reload(config_module)


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "True", "yes", "on", " true "])
def test_every_reasonable_way_to_say_yes_enables_oauth(monkeypatch, raw):
    assert _reload(monkeypatch, raw).MCP_OAUTH_ENABLED is True


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off"])
def test_every_reasonable_way_to_say_no_disables_oauth(monkeypatch, raw):
    assert _reload(monkeypatch, raw).MCP_OAUTH_ENABLED is False


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_absent_or_blank_keeps_oauth_off(monkeypatch, raw):
    """Positive control: the off-by-default contract is unchanged.

    ``MCP_OAUTH_ENABLED`` defaulting to off is what preserves today's
    behaviour for every deployment that has not opted in; widening the truthy
    set must not disturb it.
    """
    assert _reload(monkeypatch, raw).MCP_OAUTH_ENABLED is False


@pytest.mark.parametrize("raw", ["treu", "enable", "2"])
def test_a_typo_keeps_the_default_and_does_not_enable_oauth(monkeypatch, raw):
    assert _reload(monkeypatch, raw).MCP_OAUTH_ENABLED is False
