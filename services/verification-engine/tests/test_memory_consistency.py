"""Tests for the memory consistency check."""

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from engine.memory_consistency import check
from shared.memory import MemoryRecord


def _make_memory(content: str, memory_type: str = "fact", seq: int = 1) -> MemoryRecord:
    return MemoryRecord(
        id=uuid.uuid4(),
        session_id="s1",
        org_id="o1",
        agent_id="a1",
        memory_type=memory_type,
        content=content,
        sequence_number=seq,
    )


@pytest.mark.asyncio
async def test_no_memories_skips():
    result = await check(output="any output", session_memories=None)
    assert result.check_type == "memory_consistency"
    assert result.score == 1.0
    assert result.passed is True
    assert result.details["skipped"] is True


@pytest.mark.asyncio
async def test_empty_memories_skips():
    result = await check(output="any output", session_memories=[])
    assert result.check_type == "memory_consistency"
    assert result.score == 1.0
    assert result.passed is True
    assert result.details["skipped"] is True


@pytest.mark.asyncio
async def test_contradiction_detected():
    mock_response = {
        "contradictions": [
            {
                "memory": "User's name is Alice",
                "current_statement": "Hello Bob",
                "explanation": "Agent addressed user as Bob but memory says Alice",
            }
        ],
        "forgotten_context": [],
        "score": 0.2,
    }
    memories = [_make_memory("User's name is Alice")]
    with patch(
        "engine.memory_consistency.call_llm",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await check(output="Hello Bob, how can I help?", session_memories=memories)

    assert result.check_type == "memory_consistency"
    assert result.score == 0.2
    assert result.passed is False
    assert len(result.details["contradictions"]) == 1
    assert result.details["memories_evaluated"] == 1


@pytest.mark.asyncio
async def test_consistent_output_passes():
    mock_response = {
        "contradictions": [],
        "forgotten_context": [],
        "score": 1.0,
    }
    memories = [_make_memory("User's name is Alice")]
    with patch(
        "engine.memory_consistency.call_llm",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await check(output="Hello Alice!", session_memories=memories)

    assert result.score == 1.0
    assert result.passed is True
    assert len(result.details["contradictions"]) == 0
    assert len(result.details["forgotten_context"]) == 0


@pytest.mark.asyncio
async def test_timeout_returns_none_score():
    memories = [_make_memory("some fact")]
    with patch(
        "engine.memory_consistency.call_llm",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = await check(output="some output", session_memories=memories)

    assert result.check_type == "memory_consistency"
    assert result.score is None
    assert result.passed is True
    assert result.details["error"] == "timeout"


@pytest.mark.asyncio
async def test_score_clamped_to_valid_range():
    mock_response = {
        "contradictions": [],
        "forgotten_context": [],
        "score": 1.5,
    }
    memories = [_make_memory("some fact")]
    with patch(
        "engine.memory_consistency.call_llm",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await check(output="output", session_memories=memories)

    assert result.score == 1.0


@pytest.mark.asyncio
async def test_forgotten_context_lowers_score():
    mock_response = {
        "contradictions": [],
        "forgotten_context": [
            {
                "memory": "User asked to always respond in French",
                "explanation": "Agent responded in English ignoring language preference",
            }
        ],
        "score": 0.6,
    }
    memories = [
        _make_memory("User asked to always respond in French", seq=1),
        _make_memory("User is located in Paris", seq=2),
    ]
    with patch(
        "engine.memory_consistency.call_llm",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await check(output="Here is the answer in English.", session_memories=memories)

    assert result.score == 0.6
    assert result.passed is True  # >= 0.5
    assert len(result.details["forgotten_context"]) == 1
    assert result.details["memories_evaluated"] == 2
