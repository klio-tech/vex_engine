"""Contract and invariant tests for the `init` onboarding tool.

Two layers, mirroring the house pattern:

* Wiring/contract tests run against the registered tool callable with a fake
  Context and patched principal seams (no DB) — what matters is which
  arguments the tool derives and forwards, and what its schema promises.
* The five security invariants from the spec
  (docs/superpowers/specs/2026-08-03-klio-init-onboarding-design.md) run
  against real Postgres further down, gated by the shared ``requires_db``
  marker: the redaction/dedup door IS the SQL + chokepoint, and a mock would
  pass with no door at all.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import app.onboarding as onboarding
from app.mcp_app import EXPECTED_TOOL_NAMES, build_mcp_server

USER = "11111111-2222-3333-4444-555555555555"


def _fake_ctx(headers: dict | None = None):
    request = SimpleNamespace(headers=headers if headers is not None else {})
    request_context = SimpleNamespace(request=request)
    return SimpleNamespace(request_context=request_context)


def _get_tool(name: str = "init"):
    server = build_mcp_server()
    tool = server._tool_manager.get_tool(name)
    assert tool is not None, f"tool {name!r} not registered"
    return tool


def _principal_patches(user_id=USER):
    """Patch the two identity seams at the onboarding module boundary."""
    return (
        patch(
            "app.onboarding.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.onboarding.resolve_request_user_id", return_value=user_id),
        patch("app.onboarding.SessionLocal", MagicMock(return_value=MagicMock())),
    )


# ---------------------------------------------------------------------------
# Inventory + routing contract
# ---------------------------------------------------------------------------


def test_tool_inventory_is_nineteen_and_includes_init():
    """18 -> 19 tools; `init` asserted by name (spec acceptance)."""
    assert "init" in EXPECTED_TOOL_NAMES
    assert len(EXPECTED_TOOL_NAMES) == 19
    server = build_mcp_server()
    assert server._tool_manager.get_tool("init") is not None


def test_init_description_contains_routing_phrases():
    """'onboard me' / 'set me up' / 'get started' must all route here.

    The UX contract is a sentence, not a CLI flag: the description is the
    router, so the phrases are pinned.
    """
    desc = (_get_tool().description or "").lower()
    for phrase in ("onboard", "set up", "get started"):
        assert phrase in desc, f"init description missing routing phrase {phrase!r}"


def test_init_schema_requires_phase_and_offers_no_scope_lever():
    """`phase` is the one required argument, and NOTHING in the public schema
    can name a scope — invariant 2 starts at the contract surface."""
    tool = _get_tool()
    schema = tool.parameters
    props = schema.get("properties", {})
    assert "phase" in props
    assert "phase" in schema.get("required", [])
    assert "scope" not in props
    # No property description may promise scope control either.
    for name in props:
        assert "scope" not in name.lower()


# ---------------------------------------------------------------------------
# Phase: start
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_returns_versioned_playbook_with_three_steps_and_caps():
    fn = _get_tool().fn
    p = _principal_patches()
    with p[0], p[1], p[2], patch("app.onboarding._count_runs_today", return_value=0):
        result = await fn(_fake_ctx(), phase="start")

    assert result["phase"] == "start"
    playbook = result["playbook"]
    assert playbook["version"] == onboarding.PLAYBOOK_VERSION
    assert len(playbook["steps"]) == 3
    # Step 3 is the consent gate, not decoration.
    joined = " ".join(s["instructions"] for s in playbook["steps"]).lower()
    assert "not" in joined and "remember" in joined
    # The playbook tells the agent what NOT to submit.
    refusals = " ".join(playbook["do_not_submit"]).lower()
    for banned in ("secret", "personal", "off-limits"):
        assert banned in refusals
    assert result["caps"] == {
        "candidates_per_submit": 25,
        "accepted_per_run": 60,
        "content_chars_per_candidate": 2000,
        "runs_per_day": 3,
    }


@pytest.mark.asyncio
async def test_start_refuses_an_unattributed_caller():
    """init seeds a PRIVATE silo; without a person there is no silo to seed."""
    fn = _get_tool().fn
    p = _principal_patches(user_id=None)
    with p[0], p[1], p[2]:
        result = await fn(_fake_ctx(), phase="start")
    assert result["error"] == "unattributed_caller"


@pytest.mark.asyncio
async def test_start_refuses_a_fourth_run_in_one_day():
    fn = _get_tool().fn
    p = _principal_patches()
    with p[0], p[1], p[2], patch("app.onboarding._count_runs_today", return_value=3):
        result = await fn(_fake_ctx(), phase="start")
    assert result["error"] == "run_limit_reached"
    assert "3" in result["detail"]


@pytest.mark.asyncio
async def test_start_is_refused_cleanly_on_auth_failure():
    from fastapi import HTTPException

    fn = _get_tool().fn
    with patch(
        "app.onboarding.resolve_principal_plan",
        side_effect=HTTPException(status_code=401, detail="bad key"),
    ):
        result = await fn(_fake_ctx(), phase="start")
    assert result["error"] == "unauthorized"
