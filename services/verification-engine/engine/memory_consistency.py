"""LLM-based memory consistency check.

Evaluates whether the latest agent output contradicts established session
memories.  Returns a consistency score where 1.0 means fully consistent
and 0.0 means severe contradictions with stored memories.

Explicit corrections (e.g. "I was wrong earlier...") are NOT penalized.
"""

import json
import logging
from typing import Any, Optional

from shared.memory import MemoryRecord

from engine.llm_client import call_llm
from engine.models import CheckResult

logger = logging.getLogger("agentguard.verification-engine.memory_consistency")

SYSTEM_PROMPT = (
    "You are a memory consistency evaluator.  Given a set of session "
    "memories and the agent's latest output, determine whether the output "
    "contradicts any established facts or ignores important context from "
    "prior turns.\n\n"
    "Evaluate for:\n"
    "1. Contradictions with established facts in memory\n"
    "2. Forgotten context from earlier turns that should have been applied\n"
    "3. Fabricated information that conflicts with session history\n\n"
    "IMPORTANT: Do NOT penalize explicit corrections where the agent "
    "acknowledges and corrects a prior mistake (e.g. 'I was wrong earlier', "
    "'Let me correct that', 'Actually...').\n\n"
    "Respond in JSON with this schema:\n"
    '{"contradictions": [\n'
    '  {"memory": "...", "current_statement": "...", "explanation": "..."}\n'
    '], "forgotten_context": [\n'
    '  {"memory": "...", "explanation": "..."}\n'
    '], "score": 0.8}\n\n'
    "The score should be between 0 (severe contradictions / forgotten "
    "context) and 1 (fully consistent with all memories).  If there are "
    "no issues, return score 1.0 with empty contradictions and "
    "forgotten_context lists."
)


def _format_memories(memories: list[MemoryRecord]) -> str:
    """Format memory records into a human-readable string for the LLM prompt.

    Each memory is rendered as ``[memory_type] (turn N): content`` on its
    own line.
    """
    lines: list[str] = []
    for mem in memories:
        turn = mem.sequence_number if mem.sequence_number is not None else "?"
        lines.append(f"[{mem.memory_type}] (turn {turn}): {mem.content}")
    return "\n".join(lines)


async def check(
    output: Any,
    session_memories: Optional[list[MemoryRecord]] = None,
    task: Optional[str] = None,
    input_data: Optional[Any] = None,
) -> CheckResult:
    """Check agent output for contradictions against session memories.

    Args:
        output: The agent's latest output to check.
        session_memories: Stored session memories.  If None or empty,
            the check is skipped.
        task: Optional task description for additional context.
        input_data: Optional input data for additional context.

    Returns:
        CheckResult with a memory consistency score.
        If no memories, returns a passed result with skipped=True.
        On LLM timeout, returns score=None with error details.
    """
    if not session_memories:
        return CheckResult(
            check_type="memory_consistency",
            score=1.0,
            passed=True,
            details={"skipped": True},
        )

    output_str = output if isinstance(output, str) else json.dumps(output)
    memories_str = _format_memories(session_memories)

    prompt_parts = [
        f"Session memories:\n{memories_str}\n",
        f"Latest agent output:\n{output_str}\n",
    ]
    if task:
        prompt_parts.append(f"Task context:\n{task}\n")
    if input_data:
        input_str = input_data if isinstance(input_data, str) else json.dumps(input_data)
        prompt_parts.append(f"User input:\n{input_str}\n")

    prompt_parts.append(
        "Identify any contradictions with stored memories or important forgotten context.  Return JSON as specified."
    )

    prompt = "\n".join(prompt_parts)

    result = await call_llm(prompt, system=SYSTEM_PROMPT)

    if result is None:
        return CheckResult(
            check_type="memory_consistency",
            score=None,
            passed=True,
            details={"error": "timeout"},
        )

    score = result.get("score")
    if score is None:
        score = 1.0

    # Clamp score to [0, 1]
    score = max(0.0, min(1.0, float(score)))

    contradictions = result.get("contradictions", [])
    forgotten_context = result.get("forgotten_context", [])

    return CheckResult(
        check_type="memory_consistency",
        score=score,
        passed=score >= 0.5,
        details={
            "contradictions": contradictions,
            "forgotten_context": forgotten_context,
            "memories_evaluated": len(session_memories),
        },
    )
