"""One key validation per request, and a 429 that stays a 429.

A key-path MCP tool body resolves the caller twice: once for the tenant
principal (now ``resolve_active_principal``) and once for the human
(``resolve_request_user_id``). Both used to reach ``verify_api_key``
independently, and ``KeyValidator.validate`` appends to the sliding-window
rate counter on EVERY call — the entry cache short-circuits the database
lookup, not the limiter. So each tool call spent two tokens and the effective
rate limit was half the configured number.

That is a billing-shaped bug with a visibility-shaped consequence. The second
validation is the one that trips, and its ``429`` was swallowed into
``user_id = None``. ``None`` is a legal identity — it means "cannot be
attributed" — so the write proceeded, and a memory the caller asked to file
into their project landed at ``org`` scope instead: readable by the whole
tenant, with nothing in the response saying the caller was rate limited or
that the scope had been widened.

Both halves are tested here because fixing either alone is wrong. Re-raising
the 429 without removing the second call makes rate limiting twice as strict
as configured; removing the double call without re-raising leaves the
swallowing in place for every other path that resolves an identity.

The sequence under test is spelled as ``mcp_app``'s own bound names, so it
follows the tool bodies wherever those names are imported from.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from shared.auth import AuthError, KeyInfo

from app import mcp_app

USER = "11111111-2222-3333-4444-555555555555"


def _request(key: str = "vex_live_abc", agent: str = "agent-1"):
    """A key-path request. No OAuth principal, so the key ladder is used."""
    headers = {"X-Vex-Key": key, "X-Vex-Agent": agent}
    return SimpleNamespace(
        headers=SimpleNamespace(get=lambda k, default=None: headers.get(k, default)),
        state=SimpleNamespace(),
    )


class _CountingValidator:
    """Mimics ``KeyValidator``: every ``validate`` spends a rate-limit token."""

    def __init__(self, rate_limit_rpm: int = 60, user_id: str | None = USER):
        self.calls = 0
        self.rate_limit_rpm = rate_limit_rpm
        self._user_id = user_id

    def validate(self, api_key, agent_id=None):
        self.calls += 1
        if self.calls > self.rate_limit_rpm:
            raise AuthError(429, "Rate limit exceeded", retry_after_seconds=30)
        return KeyInfo(
            org_id="acme",
            key_id="k1",
            scopes=["memory"],
            plan="free",
            user_id=self._user_id,
        )


def _tool_call(request):
    """Exactly what a key-path MCP tool body does to resolve its caller.

    The bound name is ``resolve_active_principal`` since the node wiring: the
    wrapper consults the principal twice internally (once directly, once
    inside ``resolve_active_org``), which is exactly why this test matters --
    the per-request memo must keep that at ONE key validation. The session is
    a stand-in: without a node header it is never touched.
    """
    principal = mcp_app.resolve_active_principal(request, MagicMock())
    user_id = mcp_app.resolve_request_user_id(request)
    return principal, user_id


def test_one_tool_call_validates_the_key_once():
    validator = _CountingValidator()
    request = _request()

    with patch("app.auth.get_validator", return_value=validator):
        (org_id, agent_id, plan, _overrides), user_id = _tool_call(request)

    assert validator.calls == 1, (
        f"the key was validated {validator.calls} times for one tool call; "
        "every validation spends a rate-limit token, so N validations make the "
        "effective limit rpm/N"
    )
    # Positive control: one validation still answers both questions correctly.
    assert (org_id, agent_id, plan) == ("acme", "agent-1", "free")
    assert user_id == USER


def test_the_configured_rate_limit_is_the_effective_rate_limit():
    """Two tool calls must fit under a limit of two. They used to spend four."""
    validator = _CountingValidator(rate_limit_rpm=2)

    with patch("app.auth.get_validator", return_value=validator):
        for i in range(2):
            _, user_id = _tool_call(_request())
            assert user_id == USER, f"call {i + 1} lost the caller's identity"

    assert validator.calls == 2, (
        f"two tool calls spent {validator.calls} of the 2 rpm allowance"
    )


def test_a_rate_limited_request_raises_rather_than_becoming_anonymous():
    """A 429 must surface as a 429, never as a silent widening to org scope."""
    validator = _CountingValidator(rate_limit_rpm=0)
    request = _request()

    with patch("app.auth.get_validator", return_value=validator):
        with pytest.raises(HTTPException) as excinfo:
            mcp_app.resolve_request_user_id(request)

    assert excinfo.value.status_code == 429, (
        "the rate-limit refusal was converted into an unattributed caller, "
        "which writes org-scoped instead of failing"
    )


def test_a_second_resolution_reuses_the_first_answer():
    """Identity is resolved once per request even when asked for repeatedly."""
    validator = _CountingValidator()
    request = _request()

    with patch("app.auth.get_validator", return_value=validator):
        first = mcp_app.resolve_request_user_id(request)
        second = mcp_app.resolve_request_user_id(request)
        third = mcp_app.resolve_active_principal(request, MagicMock())

    assert first == second == USER
    assert third[0] == "acme"
    assert validator.calls == 1, f"{validator.calls} validations for one request"


def test_a_key_with_no_owner_still_resolves_to_no_user():
    """The legal ``None``: a legacy key with no ``created_by``.

    This is the case the swallowing was originally justified by, and it still
    behaves the same way — an unattributable caller writes org-scoped. What
    changed is that a rate-limited caller is no longer disguised as one.
    """
    validator = _CountingValidator(user_id=None)
    request = _request()

    with patch("app.auth.get_validator", return_value=validator):
        _, user_id = _tool_call(request)

    assert user_id is None
    assert validator.calls == 1


def test_each_request_is_validated_on_its_own():
    """The cache is per request, not a process-wide one that outlives it."""
    validator = _CountingValidator()

    with patch("app.auth.get_validator", return_value=validator):
        mcp_app.resolve_request_user_id(_request())
        mcp_app.resolve_request_user_id(_request())

    assert validator.calls == 2, (
        "two separate requests must each be validated; a shared cache would "
        "let a revoked key ride on another request's result"
    )


# ---------------------------------------------------------------------------
# The same contract, through ``onboarding``'s own bound names
# ---------------------------------------------------------------------------
#
# ``mcp_app`` is not the only module with tool bodies. ``app.onboarding``
# resolves the caller with exactly the same two calls, and it kept importing
# ``resolve_principal_plan`` from ``app.auth`` — the un-memoized copy — while
# taking ``resolve_request_user_id`` from ``app.identity``. That pairing is
# precisely the shape the memoization exists to remove: the auth-module
# principal never writes the memo, so the identity call finds nothing cached
# and validates a second time. ``init`` therefore still spent two tokens per
# request, and on the token that trips it is the SECOND call that raises —
# the one that decides whether the caller is a person or nobody.
#
# Spelled as ``onboarding``'s bound names for the same reason the block above
# is spelled as ``mcp_app``'s: the test follows the import, so re-pointing it
# at the un-memoized module fails here rather than in production.


def _init_tool_call(request):
    """Exactly what ``perform_init`` does to resolve its caller."""
    from app import onboarding

    principal = onboarding.resolve_principal_plan(request)
    user_id = onboarding.resolve_request_user_id(request)
    return principal, user_id


def test_init_validates_the_key_once():
    validator = _CountingValidator()
    request = _request()

    with patch("app.auth.get_validator", return_value=validator):
        (org_id, agent_id, plan, _overrides), user_id = _init_tool_call(request)

    assert validator.calls == 1, (
        f"init validated the key {validator.calls} times for one request; "
        "app.onboarding must import resolve_principal_plan from app.identity "
        "so both questions share one validation"
    )
    # Positive control: one validation still answers both questions correctly.
    assert (org_id, agent_id, plan) == ("acme", "agent-1", "free")
    assert user_id == USER


def test_init_is_not_rate_limited_at_half_the_configured_rpm():
    """Two init calls must fit under a limit of two. They used to spend four."""
    validator = _CountingValidator(rate_limit_rpm=2)

    with patch("app.auth.get_validator", return_value=validator):
        for i in range(2):
            _, user_id = _init_tool_call(_request())
            assert user_id == USER, f"init call {i + 1} lost its identity"

    assert validator.calls == 2
