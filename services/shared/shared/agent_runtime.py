"""One agent turn: gather context, think, act, reply.

Structured so the parts that need judgement are testable without a model,
a queue, or a network. :func:`run_turn` is the loop; everything it needs
from the outside arrives through :class:`TurnContext` and the two callables
on it, so a test supplies a scripted model and records what was posted.

Two bounds are not optional. ``max_steps`` and a wall-clock deadline both
terminate the loop, because a runaway agent in a team channel is not a
degraded experience — it is an incident someone has to explain.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("klio.agent_runtime")

#: How many model calls one turn may make. Eight is enough for recall,
#: two or three tool calls, and an answer; far short of a loop.
DEFAULT_MAX_STEPS = 8

#: Wall clock for the whole turn. A turn nobody is watching still costs
#: money and holds a worker slot.
DEFAULT_DEADLINE_SECONDS = 90.0


@dataclass
class TurnResult:
    """What happened, for the caller to log and for tests to assert on."""

    answered: bool
    steps: int
    stopped_early: bool = False
    reason: str = ""
    text: str = ""
    tool_calls: list[str] = field(default_factory=list)


@dataclass
class TurnContext:
    """Everything a turn needs, injected so the loop stays pure-ish.

    Attributes:
        system_prompt: The agent's instructions.
        user_message: What was said to it.
        history: Prior thread messages, oldest first, as ``(who, text)``.
        call_model: ``(messages, tools) -> dict`` — the model call. Returns
            either ``{"tool": name, "arguments": {...}}`` or
            ``{"text": "..."}``. Keeping the shape this narrow is what lets
            a test script a turn in three lines.
        run_tool: ``(name, arguments) -> str`` — executes a tool and returns
            a result string. Raising is allowed; the loop feeds the error
            back to the model rather than dying, so an agent can explain or
            recover instead of going silent.
        tools: Tool schemas handed to the model.
        on_progress: Called with a human-readable line each step, so the
            thread can show work happening. Failures here are swallowed:
            progress is a courtesy and must never end a turn.
    """

    system_prompt: str
    user_message: str
    call_model: Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]
    run_tool: Callable[[str, dict[str, Any]], str]
    tools: list[dict[str, Any]] = field(default_factory=list)
    history: list[tuple[str, str]] = field(default_factory=list)
    on_progress: Optional[Callable[[str], None]] = None
    max_steps: int = DEFAULT_MAX_STEPS
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS


def _report(context: TurnContext, line: str) -> None:
    if context.on_progress is None:
        return
    try:
        context.on_progress(line)
    except Exception:
        logger.warning("agent progress report failed", exc_info=True)


def run_turn(
    context: TurnContext, *, now: Callable[[], float] = time.monotonic
) -> TurnResult:
    """Run one turn to an answer, a bound, or a failure.

    Args:
        context: See :class:`TurnContext`.
        now: Clock, injectable so a deadline can be tested without waiting.

    Returns:
        A :class:`TurnResult`. ``answered`` is False when the turn hit a
        bound or the model never produced text — the caller still posts
        something, because an agent that goes silent is indistinguishable
        from one still thinking, and that ambiguity is what makes people
        stop trusting it.
    """
    started = now()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": context.system_prompt},
    ]
    for who, text in context.history:
        messages.append({"role": "user", "content": f"{who}: {text}"})
    messages.append({"role": "user", "content": context.user_message})

    tool_calls: list[str] = []

    for step in range(1, context.max_steps + 1):
        if now() - started > context.deadline_seconds:
            return TurnResult(
                answered=False,
                steps=step - 1,
                stopped_early=True,
                reason="deadline",
                tool_calls=tool_calls,
            )

        try:
            reply = context.call_model(messages, context.tools)
        except Exception as exc:
            logger.warning("agent model call failed: %s", exc, exc_info=True)
            return TurnResult(
                answered=False,
                steps=step,
                stopped_early=True,
                reason="model_failed",
                tool_calls=tool_calls,
            )

        text = (reply or {}).get("text")
        tool = (reply or {}).get("tool")

        if tool:
            arguments = (reply or {}).get("arguments") or {}
            tool_calls.append(tool)
            _report(context, f"using {tool}")
            try:
                result = context.run_tool(tool, arguments)
            except Exception as exc:
                # Fed back rather than fatal: the model can often explain
                # or try another way, and an error it can see beats a turn
                # that vanishes.
                result = f"error: {exc}"
                logger.info("agent tool %s failed: %s", tool, exc)
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps({"tool": tool, "arguments": arguments}),
                }
            )
            messages.append({"role": "user", "content": f"result of {tool}: {result}"})
            continue

        if text:
            return TurnResult(
                answered=True, steps=step, text=text, tool_calls=tool_calls
            )

        # Neither a tool nor text: nothing to act on and nothing to say.
        return TurnResult(
            answered=False,
            steps=step,
            stopped_early=True,
            reason="empty_reply",
            tool_calls=tool_calls,
        )

    return TurnResult(
        answered=False,
        steps=context.max_steps,
        stopped_early=True,
        reason="max_steps",
        tool_calls=tool_calls,
    )
