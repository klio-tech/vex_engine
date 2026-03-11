"""Tests for the verification pipeline."""

from unittest.mock import AsyncMock, patch

import pytest
from engine.models import CheckResult, ConversationTurn, GuardrailRule, VerificationConfig
from engine.pipeline import _rebalance_weights, route_action, verify
from shared.models import StepRecord


def test_route_action_pass():
    assert route_action(0.9, pass_threshold=0.8, flag_threshold=0.5) == "pass"


def test_route_action_flag():
    assert route_action(0.6, pass_threshold=0.8, flag_threshold=0.5) == "flag"


def test_route_action_block():
    assert route_action(0.3, pass_threshold=0.8, flag_threshold=0.5) == "block"


def test_route_action_none_returns_flag():
    assert route_action(None, pass_threshold=0.8, flag_threshold=0.5) == "flag"


def test_route_action_at_threshold_boundary():
    assert route_action(0.8, pass_threshold=0.8, flag_threshold=0.5) == "pass"
    assert route_action(0.5, pass_threshold=0.8, flag_threshold=0.5) == "flag"
    assert route_action(0.4999, pass_threshold=0.8, flag_threshold=0.5) == "block"


@pytest.mark.asyncio
async def test_full_pipeline_all_pass():
    hallucination_response = {
        "claims": ["The revenue is $1M"],
        "grounded": ["The revenue is $1M"],
        "ungrounded": [],
        "score": 1.0,
    }
    drift_response = {"score": 0.95, "explanation": "Relevant output"}

    with (
        patch("engine.hallucination.call_llm", new_callable=AsyncMock, return_value=hallucination_response),
        patch("engine.drift.call_llm", new_callable=AsyncMock, return_value=drift_response),
    ):
        result = await verify(
            output={"revenue": 1000000},
            task="Generate financial report",
            schema={"type": "object", "required": ["revenue"]},
            ground_truth={"revenue": 1000000},
        )

    assert result.action == "pass"
    assert result.confidence is not None
    assert result.confidence >= 0.8
    assert "schema" in result.checks
    assert "hallucination" in result.checks
    assert "drift" in result.checks
    assert result.checks["schema"].passed is True


@pytest.mark.asyncio
async def test_full_pipeline_schema_failure_causes_block():
    hallucination_response = {
        "claims": [],
        "grounded": [],
        "ungrounded": [],
        "score": 1.0,
    }
    drift_response = {"score": 0.9, "explanation": "Relevant"}

    with (
        patch("engine.hallucination.call_llm", new_callable=AsyncMock, return_value=hallucination_response),
        patch("engine.drift.call_llm", new_callable=AsyncMock, return_value=drift_response),
    ):
        result = await verify(
            output={"wrong_field": "data"},
            task="Generate report",
            schema={"type": "object", "required": ["revenue"]},
            ground_truth={"revenue": 1000000},
        )

    assert result.checks["schema"].passed is False
    assert result.checks["schema"].score == 0.0
    # confidence: (0.3*0.0 + 0.4*1.0 + 0.3*0.9) = 0.67 → flag
    assert result.action == "flag"


@pytest.mark.asyncio
async def test_pipeline_with_no_optional_inputs():
    """No schema, ground_truth, or task — all checks skip, action=pass."""
    result = await verify(output="hello world")
    assert result.action == "pass"
    assert result.confidence is not None
    assert result.confidence == 1.0
    assert result.checks["schema"].details.get("skipped") is True
    assert result.checks["hallucination"].details.get("skipped") is True
    assert result.checks["drift"].details.get("skipped") is True


@pytest.mark.asyncio
async def test_pipeline_llm_timeout_graceful():
    """LLM timeout returns None scores — pipeline still succeeds."""
    with (
        patch("engine.hallucination.call_llm", new_callable=AsyncMock, return_value=None),
        patch("engine.drift.call_llm", new_callable=AsyncMock, return_value=None),
    ):
        result = await verify(
            output="hello",
            task="greet user",
            ground_truth={"greeting": "hello"},
        )

    assert result.checks["hallucination"].score is None
    assert result.checks["drift"].score is None
    # Only schema has a score (1.0, no schema defined → skip → score 1.0)
    assert result.confidence is not None
    assert result.action == "pass"


@pytest.mark.asyncio
async def test_pipeline_custom_config():
    config = VerificationConfig(
        weights={"schema": 1.0, "hallucination": 0.0, "drift": 0.0},
        pass_threshold=0.9,
        flag_threshold=0.5,
    )
    result = await verify(
        output={"name": "Alice"},
        schema={"type": "object", "required": ["name"]},
        config=config,
    )
    assert result.confidence is not None
    assert result.confidence == 1.0
    assert result.action == "pass"


# --- Conversation-aware pipeline tests ---


@pytest.mark.asyncio
async def test_pipeline_with_history_runs_four_checks():
    """When conversation_history is provided, coherence check is included."""
    hallucination_response = {
        "claims": ["Revenue is $5.2B"],
        "grounded": ["Revenue is $5.2B"],
        "ungrounded": [],
        "score": 1.0,
    }
    drift_response = {
        "immediate_relevance": 0.95,
        "trajectory_drift": 0.9,
        "explanation": "On topic.",
    }
    coherence_response = {
        "contradictions": [],
        "score": 1.0,
    }

    history = [
        ConversationTurn(sequence_number=0, input="Revenue?", output="$5.2B"),
    ]

    with (
        patch("engine.hallucination.call_llm", new_callable=AsyncMock, return_value=hallucination_response),
        patch("engine.drift.call_llm", new_callable=AsyncMock, return_value=drift_response),
        patch("engine.coherence.call_llm", new_callable=AsyncMock, return_value=coherence_response),
    ):
        result = await verify(
            output="Revenue is still $5.2B.",
            task="Financial summary",
            ground_truth={"revenue": "$5.2B"},
            conversation_history=history,
        )

    assert "schema" in result.checks
    assert "hallucination" in result.checks
    assert "drift" in result.checks
    assert "coherence" in result.checks
    assert result.action == "pass"
    assert result.confidence is not None
    assert result.confidence >= 0.8


@pytest.mark.asyncio
async def test_pipeline_without_history_runs_three_checks():
    """Without history, coherence is NOT included — same as Phase 2 behavior."""
    hallucination_response = {
        "claims": ["Revenue is $5.2B"],
        "grounded": ["Revenue is $5.2B"],
        "ungrounded": [],
        "score": 1.0,
    }
    drift_response = {"score": 0.95, "explanation": "On topic."}

    with (
        patch("engine.hallucination.call_llm", new_callable=AsyncMock, return_value=hallucination_response),
        patch("engine.drift.call_llm", new_callable=AsyncMock, return_value=drift_response),
    ):
        result = await verify(
            output="Revenue is $5.2B.",
            task="Financial summary",
            ground_truth={"revenue": "$5.2B"},
            conversation_history=None,
        )

    assert "schema" in result.checks
    assert "hallucination" in result.checks
    assert "drift" in result.checks
    assert "coherence" not in result.checks
    assert result.action == "pass"


@pytest.mark.asyncio
async def test_pipeline_custom_weights_with_coherence():
    """Custom weights should be rebalanced when coherence is active."""
    hallucination_response = {
        "claims": [],
        "grounded": [],
        "ungrounded": [],
        "score": 1.0,
    }
    drift_response = {
        "immediate_relevance": 1.0,
        "trajectory_drift": 1.0,
        "explanation": "Perfect.",
    }
    coherence_response = {
        "contradictions": [],
        "score": 0.3,
    }

    history = [
        ConversationTurn(sequence_number=0, input="q", output="a"),
    ]

    config = VerificationConfig(
        weights={"schema": 0.5, "hallucination": 0.25, "drift": 0.25},
    )

    with (
        patch("engine.hallucination.call_llm", new_callable=AsyncMock, return_value=hallucination_response),
        patch("engine.drift.call_llm", new_callable=AsyncMock, return_value=drift_response),
        patch("engine.coherence.call_llm", new_callable=AsyncMock, return_value=coherence_response),
    ):
        result = await verify(
            output="output",
            task="task",
            conversation_history=history,
            config=config,
        )

    assert "coherence" in result.checks
    # Coherence score is 0.3 which should pull confidence down
    assert result.confidence is not None
    assert result.confidence < 1.0


@pytest.mark.asyncio
async def test_pipeline_runs_tool_loop_check_when_steps_provided():
    """Pipeline includes tool_loop check when steps are passed."""
    steps = [
        StepRecord(step_type="tool_call", name="search", input="q", output="r"),
        StepRecord(step_type="tool_call", name="read", input="f", output="d"),
    ]

    with (
        patch("engine.hallucination.check", new_callable=AsyncMock) as mock_hal,
        patch("engine.drift.check", new_callable=AsyncMock) as mock_drift,
    ):
        mock_hal.return_value = CheckResult(check_type="hallucination", score=1.0, passed=True, details={})
        mock_drift.return_value = CheckResult(check_type="drift", score=1.0, passed=True, details={})

        result = await verify(output="test output", steps=steps)

    assert "tool_loop" in result.checks
    assert result.checks["tool_loop"].passed is True


@pytest.mark.asyncio
async def test_pipeline_skips_tool_loop_when_no_steps():
    """Pipeline does not include tool_loop when no steps provided."""
    with (
        patch("engine.hallucination.check", new_callable=AsyncMock) as mock_hal,
        patch("engine.drift.check", new_callable=AsyncMock) as mock_drift,
    ):
        mock_hal.return_value = CheckResult(check_type="hallucination", score=1.0, passed=True, details={})
        mock_drift.return_value = CheckResult(check_type="drift", score=1.0, passed=True, details={})

        result = await verify(output="test output")

    assert "tool_loop" not in result.checks


@pytest.mark.asyncio
async def test_pipeline_tool_loop_failure_affects_confidence():
    """A tool loop detection failure lowers composite confidence."""
    steps = [StepRecord(step_type="tool_call", name="search", input="q", output="r")] * 30

    with (
        patch("engine.hallucination.check", new_callable=AsyncMock) as mock_hal,
        patch("engine.drift.check", new_callable=AsyncMock) as mock_drift,
    ):
        mock_hal.return_value = CheckResult(check_type="hallucination", score=1.0, passed=True, details={})
        mock_drift.return_value = CheckResult(check_type="drift", score=1.0, passed=True, details={})

        result = await verify(output="test output", steps=steps)

    assert result.checks["tool_loop"].passed is False
    assert result.confidence is not None
    assert result.confidence < 1.0


def test_rebalance_weights():
    """_rebalance_weights preserves proportions and adds the named check."""
    original = {"schema": 0.3, "hallucination": 0.4, "drift": 0.3}
    # Default check_name="coherence" for backward compatibility
    rebalanced = _rebalance_weights(original, 0.20)

    # coherence should be present
    assert rebalanced["coherence"] == 0.20
    # total should still be 1.0
    assert abs(sum(rebalanced.values()) - 1.0) < 1e-9
    # proportions of original keys should be preserved
    assert abs(rebalanced["schema"] / rebalanced["drift"] - 1.0) < 1e-9
    assert rebalanced["hallucination"] > rebalanced["schema"]


def test_rebalance_weights_custom_check_name():
    """_rebalance_weights works with custom check names."""
    original = {"schema": 0.3, "hallucination": 0.4, "drift": 0.3}
    rebalanced = _rebalance_weights(original, 0.20, check_name="memory_consistency")

    assert rebalanced["memory_consistency"] == 0.20
    assert "coherence" not in rebalanced
    assert abs(sum(rebalanced.values()) - 1.0) < 1e-9


# --- Guardrails pipeline tests ---


@pytest.mark.asyncio
async def test_pipeline_with_guardrails_keyword_block():
    """A keyword block guardrail should force action to block."""
    config = VerificationConfig(
        guardrails=[
            GuardrailRule(
                name="no-competitor", rule_type="keyword", condition={"keywords": ["RivalCo"]}, action="block"
            ),
        ],
    )

    with (
        patch("engine.hallucination.check", new_callable=AsyncMock) as mock_hal,
        patch("engine.drift.check", new_callable=AsyncMock) as mock_drift,
    ):
        mock_hal.return_value = CheckResult(check_type="hallucination", score=1.0, passed=True, details={})
        mock_drift.return_value = CheckResult(check_type="drift", score=1.0, passed=True, details={})

        result = await verify(output="Try RivalCo for better results", config=config)

    assert "guardrails" in result.checks
    assert result.checks["guardrails"].passed is False
    assert result.action == "block"


@pytest.mark.asyncio
async def test_pipeline_with_guardrails_flag_only():
    """A flag guardrail violation should escalate pass to flag."""
    config = VerificationConfig(
        guardrails=[
            GuardrailRule(name="no-emails", rule_type="regex", condition={"pattern": r"\S+@\S+"}, action="flag"),
        ],
    )

    with (
        patch("engine.hallucination.check", new_callable=AsyncMock) as mock_hal,
        patch("engine.drift.check", new_callable=AsyncMock) as mock_drift,
    ):
        mock_hal.return_value = CheckResult(check_type="hallucination", score=1.0, passed=True, details={})
        mock_drift.return_value = CheckResult(check_type="drift", score=1.0, passed=True, details={})

        result = await verify(output="Email me at test@example.com", config=config)

    assert "guardrails" in result.checks
    assert result.checks["guardrails"].passed is False
    assert result.action in ("flag", "block")


@pytest.mark.asyncio
async def test_pipeline_without_guardrails_no_check():
    """No guardrails configured — guardrails check should not be in results."""
    with (
        patch("engine.hallucination.check", new_callable=AsyncMock) as mock_hal,
        patch("engine.drift.check", new_callable=AsyncMock) as mock_drift,
    ):
        mock_hal.return_value = CheckResult(check_type="hallucination", score=1.0, passed=True, details={})
        mock_drift.return_value = CheckResult(check_type="drift", score=1.0, passed=True, details={})

        result = await verify(output="clean output")

    assert "guardrails" not in result.checks


@pytest.mark.asyncio
async def test_pipeline_guardrails_passing_does_not_affect_action():
    """Guardrails that pass should not change the action."""
    config = VerificationConfig(
        guardrails=[
            GuardrailRule(name="no-bad", rule_type="keyword", condition={"keywords": ["bad"]}, action="block"),
        ],
    )

    with (
        patch("engine.hallucination.check", new_callable=AsyncMock) as mock_hal,
        patch("engine.drift.check", new_callable=AsyncMock) as mock_drift,
    ):
        mock_hal.return_value = CheckResult(check_type="hallucination", score=1.0, passed=True, details={})
        mock_drift.return_value = CheckResult(check_type="drift", score=1.0, passed=True, details={})

        result = await verify(output="all good here", config=config)

    assert "guardrails" in result.checks
    assert result.checks["guardrails"].passed is True
    assert result.action == "pass"


# --- Memory consistency pipeline tests ---


@pytest.mark.asyncio
async def test_pipeline_with_memories_runs_memory_consistency():
    """When session_memories is provided, memory_consistency check is included."""
    import uuid

    from shared.memory import MemoryRecord

    memories = [
        MemoryRecord(
            id=uuid.uuid4(),
            session_id="s1",
            org_id="o1",
            agent_id="a1",
            memory_type="fact",
            content="Refund takes 3-5 days",
            sequence_number=1,
        ),
    ]
    memory_response = {"contradictions": [], "forgotten_context": [], "score": 1.0}

    with (
        patch("engine.hallucination.check", new_callable=AsyncMock) as mock_hal,
        patch("engine.drift.check", new_callable=AsyncMock) as mock_drift,
        patch("engine.memory_consistency.call_llm", new_callable=AsyncMock, return_value=memory_response),
    ):
        mock_hal.return_value = CheckResult(check_type="hallucination", score=1.0, passed=True, details={})
        mock_drift.return_value = CheckResult(check_type="drift", score=1.0, passed=True, details={})

        result = await verify(
            output="Your refund takes 3-5 days.",
            task="Handle refund",
            session_memories=memories,
        )

    assert "memory_consistency" in result.checks
    assert result.checks["memory_consistency"].score == 1.0
    assert result.action == "pass"


@pytest.mark.asyncio
async def test_pipeline_without_memories_skips_memory_consistency():
    """Without memories, memory_consistency is NOT included."""
    with (
        patch("engine.hallucination.check", new_callable=AsyncMock) as mock_hal,
        patch("engine.drift.check", new_callable=AsyncMock) as mock_drift,
    ):
        mock_hal.return_value = CheckResult(check_type="hallucination", score=1.0, passed=True, details={})
        mock_drift.return_value = CheckResult(check_type="drift", score=1.0, passed=True, details={})

        result = await verify(output="test output")

    assert "memory_consistency" not in result.checks


@pytest.mark.asyncio
async def test_pipeline_memory_contradiction_lowers_confidence():
    """A memory contradiction should lower composite confidence."""
    import uuid

    from shared.memory import MemoryRecord

    memories = [
        MemoryRecord(
            id=uuid.uuid4(),
            session_id="s1",
            org_id="o1",
            agent_id="a1",
            memory_type="fact",
            content="Refund takes 3-5 days",
            sequence_number=1,
        ),
    ]
    memory_response = {
        "contradictions": [
            {"memory": "Refund takes 3-5 days", "current_statement": "tomorrow", "explanation": "conflict"}
        ],
        "forgotten_context": [],
        "score": 0.2,
    }

    with (
        patch("engine.hallucination.check", new_callable=AsyncMock) as mock_hal,
        patch("engine.drift.check", new_callable=AsyncMock) as mock_drift,
        patch("engine.memory_consistency.call_llm", new_callable=AsyncMock, return_value=memory_response),
    ):
        mock_hal.return_value = CheckResult(check_type="hallucination", score=1.0, passed=True, details={})
        mock_drift.return_value = CheckResult(check_type="drift", score=1.0, passed=True, details={})

        result = await verify(
            output="Your refund arrives tomorrow.",
            task="Handle refund",
            session_memories=memories,
        )

    assert "memory_consistency" in result.checks
    assert result.checks["memory_consistency"].score == 0.2
    assert result.confidence is not None
    assert result.confidence < 1.0


@pytest.mark.asyncio
async def test_pipeline_with_memories_and_history_runs_both():
    """When both conversation_history and session_memories are provided, both checks run."""
    import uuid

    from shared.memory import MemoryRecord

    memories = [
        MemoryRecord(
            id=uuid.uuid4(),
            session_id="s1",
            org_id="o1",
            agent_id="a1",
            memory_type="fact",
            content="User is in NYC",
            sequence_number=1,
        ),
    ]
    history = [
        ConversationTurn(sequence_number=0, input="Where are you?", output="NYC"),
    ]

    memory_response = {"contradictions": [], "forgotten_context": [], "score": 1.0}
    coherence_response = {"contradictions": [], "score": 1.0}

    with (
        patch("engine.hallucination.check", new_callable=AsyncMock) as mock_hal,
        patch("engine.drift.check", new_callable=AsyncMock) as mock_drift,
        patch("engine.coherence.call_llm", new_callable=AsyncMock, return_value=coherence_response),
        patch("engine.memory_consistency.call_llm", new_callable=AsyncMock, return_value=memory_response),
    ):
        mock_hal.return_value = CheckResult(check_type="hallucination", score=1.0, passed=True, details={})
        mock_drift.return_value = CheckResult(check_type="drift", score=1.0, passed=True, details={})

        result = await verify(
            output="I'm in NYC.",
            task="Location check",
            conversation_history=history,
            session_memories=memories,
        )

    assert "coherence" in result.checks
    assert "memory_consistency" in result.checks
    assert result.action == "pass"
