"""``VERIFICATION_TIMEOUT_S`` must not be able to break every LLM call.

Executed against the pre-fix code, with ``VERIFICATION_TIMEOUT_S='10s'``::

    shared/memory.py:324                        -> ValueError: could not convert string to float: '10s'
    verification-engine/engine/llm_client.py:66 -> ValueError: could not convert string to float: '10s'

Two sites, not one. Both put the bare ``float()`` *above* the ``try`` whose
entire purpose is to make LLM failure degrade to ``None`` — so a one-character
typo in an env var converted graceful degradation into an exception escaping
every caller.

This file covers the ``shared/memory.py`` site. The verification-engine site is
covered by ``services/verification-engine/tests/test_llm_timeout_config.py``,
which is where that module's suite lives.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from shared import memory


def _call(prompt="hi"):
    return asyncio.run(memory.call_llm(prompt))


@pytest.mark.parametrize("raw", ["10s", "", "   ", "five", "0", "-1"])
def test_a_malformed_timeout_degrades_instead_of_raising(monkeypatch, raw):
    monkeypatch.setenv("VERIFICATION_TIMEOUT_S", raw)
    captured = {}

    async def _fake(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("no network in tests")

    with patch.object(memory, "acompletion", AsyncMock(side_effect=_fake)):
        assert _call() is None

    assert captured["timeout"] == memory.DEFAULT_LLM_TIMEOUT_S, (
        f"VERIFICATION_TIMEOUT_S={raw!r} must fall back to the default timeout"
    )


def test_a_valid_timeout_is_honoured(monkeypatch):
    """Positive control: the guard did not pin the timeout to the default."""
    monkeypatch.setenv("VERIFICATION_TIMEOUT_S", "12.5")
    captured = {}

    async def _fake(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("no network in tests")

    with patch.object(memory, "acompletion", AsyncMock(side_effect=_fake)):
        assert _call() is None

    assert captured["timeout"] == 12.5
