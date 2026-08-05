"""``MEMORY_ENABLED`` is the on/off switch for the entire memory product.

Executed against the pre-fix ``os.environ.get("MEMORY_ENABLED", "false").lower()
== "true"``::

    MEMORY_ENABLED='true'   -> True
    MEMORY_ENABLED='true '  -> False     <- trailing space
    MEMORY_ENABLED='TRUE'   -> True
    MEMORY_ENABLED='1'      -> False
    MEMORY_ENABLED='yes'    -> False
    MEMORY_ENABLED='on'     -> False

Production has ``'true'`` exactly, so production is safe today. The blast
radius of any of the other four is the whole memory layer, silently, with
nothing logged — ``sync-gateway`` and ``async-worker`` both gate on this and
simply skip the write.
"""

import logging

import pytest

from shared.memory_config import is_memory_enabled


@pytest.mark.parametrize(
    "raw", ["true", "TRUE", "True", "true ", " true", "1", "yes", "on"]
)
def test_every_reasonable_way_to_say_yes_enables_memory(monkeypatch, raw):
    monkeypatch.setenv("MEMORY_ENABLED", raw)
    assert is_memory_enabled() is True


@pytest.mark.parametrize("raw", ["false", "FALSE", "0", "no", "off", " off "])
def test_every_reasonable_way_to_say_no_disables_memory(monkeypatch, raw):
    """Positive control: the fix did not simply make the flag always true."""
    monkeypatch.setenv("MEMORY_ENABLED", raw)
    assert is_memory_enabled() is False


@pytest.mark.parametrize("raw", ["", "   "])
def test_a_blank_value_keeps_the_off_default(monkeypatch, raw):
    monkeypatch.setenv("MEMORY_ENABLED", raw)
    assert is_memory_enabled() is False


def test_unset_keeps_the_off_default(monkeypatch):
    monkeypatch.delenv("MEMORY_ENABLED", raising=False)
    assert is_memory_enabled() is False


def test_a_typo_is_logged_rather_than_silently_disabling_the_product(
    monkeypatch, caplog
):
    """The finding is the silence, not the value."""
    monkeypatch.setenv("MEMORY_ENABLED", "treu")
    with caplog.at_level(logging.WARNING, logger="agentguard.shared.env_config"):
        assert is_memory_enabled() is False
    assert "MEMORY_ENABLED" in caplog.text
