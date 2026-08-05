"""The guarded parsers, and the exact inputs that defeated the unguarded ones.

Every case marked "executed" below was reproduced against the pre-fix code
before this file was written.
"""

import logging

import pytest

from shared.env_config import parse_bool, parse_float, parse_int


# ---------------------------------------------------------------------------
# parse_bool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "True", "yes", "on", " true "])
def test_truthy_tokens(raw):
    """Executed: every one of these gave ``False`` under a bare ``== "1"``.

    ``MCP_OAUTH_ENABLED=true`` is the live instance — the operator turned OAuth
    on and the service left it off, with no startup complaint because the same
    flag gated the required-config check.
    """
    assert parse_bool(raw, default=False, name="X") is True


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off", " off "])
def test_falsy_tokens(raw):
    assert parse_bool(raw, default=True, name="X") is False


@pytest.mark.parametrize("raw", [None, "", "   ", "\t\n"])
def test_absent_or_blank_takes_the_default(raw):
    """A declared-but-unset variable in Railway/compose arrives as ``""``.

    That is an absence, not an instruction to disable.
    """
    assert parse_bool(raw, default=True, name="X") is True
    assert parse_bool(raw, default=False, name="X") is False


@pytest.mark.parametrize("raw", ["treu", "maybe", "enabled", "2", "-1"])
def test_unrecognised_tokens_take_the_default_not_false(raw):
    """Executed: the old ``_parse_bool`` returned ``False`` for all of these.

    The docstring said "falling back to ``default``" while the code fell back
    to ``False``, so a typo silently disabled every feature whose default was
    on.
    """
    assert parse_bool(raw, default=True, name="X") is True
    assert parse_bool(raw, default=False, name="X") is False


def test_an_unrecognised_token_is_logged(caplog):
    """Silence is the actual defect: a wrong flag must leave a trace."""
    with caplog.at_level(logging.WARNING, logger="agentguard.shared.env_config"):
        parse_bool("treu", default=True, name="KLIO_SOMETHING")
    assert "KLIO_SOMETHING" in caplog.text
    assert "treu" in caplog.text


def test_a_recognised_token_is_not_logged(caplog):
    """Positive control: the warning fires on typos, not on every read."""
    with caplog.at_level(logging.WARNING, logger="agentguard.shared.env_config"):
        parse_bool("true", default=False, name="KLIO_SOMETHING")
    assert caplog.text == ""


# ---------------------------------------------------------------------------
# parse_float / parse_int
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["10s", "five", "", "  ", "1,5"])
def test_a_malformed_float_falls_back_instead_of_raising(raw):
    """Executed: ``VERIFICATION_TIMEOUT_S='10s'`` raised
    ``ValueError: could not convert string to float: '10s'`` out of every
    ``call_llm`` at both sites, because the bare ``float()`` sat above the
    ``try`` whose whole job is to make LLM failure degrade gracefully.
    """
    assert parse_float(raw, default=5.0, name="X") == 5.0


def test_a_valid_float_parses():
    """Positive control."""
    assert parse_float("2.5", default=5.0, name="X") == 2.5


@pytest.mark.parametrize("raw", ["0", "-1", "-0.5"])
def test_a_non_positive_timeout_falls_back(raw):
    """A zero or negative timeout is not a shorter timeout, it is a broken one."""
    assert parse_float(raw, default=5.0, name="X", minimum=0.1) == 5.0


def test_an_out_of_range_float_falls_back():
    assert parse_float("9999", default=5.0, name="X", maximum=600) == 5.0


@pytest.mark.parametrize("raw", ["twelve", "12.5", "", None])
def test_a_malformed_int_falls_back(raw):
    assert parse_int(raw, default=12, name="X") == 12


def test_a_negative_int_falls_back_when_bounded():
    """Executed shape: a negative entity cap silently drops every entity."""
    assert parse_int("-5", default=12, name="X", minimum=1) == 12


def test_a_valid_int_parses():
    """Positive control."""
    assert parse_int("25", default=12, name="X", minimum=1) == 25
