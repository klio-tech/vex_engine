"""is_tool_call_noise: matches tool-call/build-artifact shapes, NEVER durable facts."""

import pytest

from shared.fact_noise import is_tool_call_noise

# Real noise observed in the production brain — must be retracted.
NOISE = [
    "Used tool Bash\ninput: {\"command\":\"git show 96519d9\"}",
    "Tool Bash was used with command: grep -n memory file.json",
    "Tool Read was used on file /Users/x/y/env.py",
    "Agent executed Bash command to find JSON files in the sdk directory.",
    'The agent used the Bash tool with command "npx oxlint packages/...".',
    "Used tool mcp__plugin_stripe_stripe__get_stripe_account_info\ninput: {}",
    "response: {\"stdout\":\"\",\"stderr\":\"\",\"interrupted\":false}",
]

# Durable, useful facts — must SURVIVE (false-positive guards).
LEGIT = [
    "The team decided to use Stripe for billing.",
    "The user prefers production-ready changes only, no quick fixes.",
    "Bash scripts for deployment live in services/scripts.",
    "AgentGuard was renamed to Klio.tech in June 2026.",
    "The user's name is Abhishek and they work on Klio.",
    "The engine validates Supabase tokens via JWKS.",
    "MoonForge uses a Stripe metered billing meter named moonforge_events.",
]


@pytest.mark.parametrize("content", NOISE)
def test_matches_tool_call_noise(content):
    assert is_tool_call_noise(content) is True


@pytest.mark.parametrize("content", LEGIT)
def test_spares_durable_facts(content):
    assert is_tool_call_noise(content) is False


def test_empty_and_none_safe():
    assert is_tool_call_noise("") is False
    assert is_tool_call_noise(None) is False
