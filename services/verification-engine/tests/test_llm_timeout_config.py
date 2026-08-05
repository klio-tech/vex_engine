"""The second ``VERIFICATION_TIMEOUT_S`` site. The reviewers found one of two.

Executed against the pre-fix code with ``VERIFICATION_TIMEOUT_S='10s'``::

    verification-engine/engine/llm_client.py:66 -> ValueError: could not convert string to float: '10s'

The bare ``float()`` sits above the ``try`` that exists to make LLM failure
return ``None``, so the typo escaped every ``call_llm`` caller as a hard error
instead of degrading.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from engine import llm_client


def _call():
    return asyncio.run(llm_client.call_llm("hi"))


@pytest.mark.parametrize("raw", ["10s", "", "   ", "five", "0", "-1"])
def test_a_malformed_timeout_degrades_instead_of_raising(monkeypatch, raw):
    monkeypatch.setenv("VERIFICATION_TIMEOUT_S", raw)
    captured = {}

    async def _fake(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("no network in tests")

    with patch.object(llm_client, "acompletion", AsyncMock(side_effect=_fake)):
        assert _call() is None

    assert captured["timeout"] == llm_client.DEFAULT_TIMEOUT_S, (
        f"VERIFICATION_TIMEOUT_S={raw!r} must fall back to the default timeout"
    )


def test_a_valid_timeout_is_honoured(monkeypatch):
    """Positive control."""
    monkeypatch.setenv("VERIFICATION_TIMEOUT_S", "12.5")
    captured = {}

    async def _fake(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("no network in tests")

    with patch.object(llm_client, "acompletion", AsyncMock(side_effect=_fake)):
        assert _call() is None

    assert captured["timeout"] == 12.5
