"""is_tool_call_noise: matches tool-call/build-artifact shapes, NEVER durable facts."""

import pytest

from shared.fact_noise import is_tool_call_noise

# Real noise observed in the production brain — must be retracted.
NOISE = [
    'Used tool Bash\ninput: {"command":"git show 96519d9"}',
    "Tool Bash was used with command: grep -n memory file.json",
    "Tool Read was used on file /Users/x/y/env.py",
    "Agent executed Bash command to find JSON files in the sdk directory.",
    'The agent used the Bash tool with command "npx oxlint packages/...".',
    "Used tool mcp__plugin_stripe_stripe__get_stripe_account_info\ninput: {}",
    'response: {"stdout":"","stderr":"","interrupted":false}',
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
    # Adversarial: durable facts that name a tool with the common noun "tool"
    # mid-sentence followed by "was used". These describe WHAT the team used,
    # not a captured tool invocation, and must survive.
    "The migration tool db-migrate was used to roll out the schema change.",
    "The team's preferred deploy tool flyctl was used for all production releases.",
    "Our internal tool deploybot was used to ship the release.",
    "The tool terraform was used to provision the staging cluster.",
    "The reporting tool metabase was used to build the Q3 dashboard.",
    "The team's tool eslint was used to lint the frontend.",
    "The data pipeline tool airflow-cli was used to orchestrate ETL jobs.",
    "We documented that the tool kubectl was used to debug the outage.",
    # Adversarial: "was used with command-..." continuations that are not a
    # raw "command:" tool dump.
    "A custom tool was used with command-line flags for the import.",
    "The user noted the tool was used with command palette shortcuts in VS Code.",
    "The keyboard tool was used with command-key bindings on macOS.",
    "The new tool was used with command-line arguments only, no GUI.",
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
