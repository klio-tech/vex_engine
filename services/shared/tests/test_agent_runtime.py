"""The turn loop, driven by a scripted model.

No network, no queue, no LLM. Every case here is one someone would notice
in a team channel: an agent that loops forever, one that dies on a tool
error, one that goes silent, one that answers.
"""

import pytest

from shared.agent_runtime import DEFAULT_MAX_STEPS, TurnContext, run_turn


def context(script, **kw):
    """A turn whose model returns `script` entries in order."""
    calls = iter(script)
    tool_log = []

    def call_model(_messages, _tools):
        return next(calls)

    def run_tool(name, arguments):
        tool_log.append((name, arguments))
        return kw.pop("tool_result", "ok")

    base = dict(
        system_prompt="be helpful",
        user_message="what did we decide?",
        call_model=call_model,
        run_tool=run_tool,
    )
    base.update(kw)
    ctx = TurnContext(**base)
    ctx._tool_log = tool_log  # type: ignore[attr-defined]
    return ctx


class TestAnswering:
    def test_a_direct_answer_ends_the_turn(self):
        result = run_turn(context([{"text": "we chose retries"}]))
        assert result.answered is True
        assert result.text == "we chose retries"
        assert result.steps == 1

    def test_a_tool_call_then_an_answer(self):
        ctx = context([{"tool": "recall", "arguments": {"query": "retries"}}, {"text": "found it"}])
        result = run_turn(ctx)
        assert result.answered is True
        assert result.tool_calls == ["recall"]
        assert ctx._tool_log == [("recall", {"query": "retries"})]

    def test_the_tool_result_is_fed_back(self):
        seen = {}

        def call_model(messages, _tools):
            seen["last"] = messages[-1]["content"]
            return {"text": "done"} if len(messages) > 2 else {"tool": "recall", "arguments": {}}

        ctx = context([], call_model=call_model)
        run_turn(ctx)
        assert "result of recall" in seen["last"]


class TestBounds:
    def test_it_stops_at_max_steps(self):
        # A model that only ever asks for tools must not loop forever.
        endless = [{"tool": "recall", "arguments": {}}] * (DEFAULT_MAX_STEPS + 5)
        result = run_turn(context(endless))
        assert result.answered is False
        assert result.stopped_early is True
        assert result.reason == "max_steps"
        assert result.steps == DEFAULT_MAX_STEPS

    def test_it_stops_at_the_deadline(self):
        clock = iter([0.0, 0.0, 1000.0, 1000.0])
        endless = [{"tool": "recall", "arguments": {}}] * 5
        result = run_turn(context(endless), now=lambda: next(clock))
        assert result.stopped_early is True
        assert result.reason == "deadline"

    def test_a_custom_step_bound_is_honoured(self):
        result = run_turn(context([{"tool": "x", "arguments": {}}] * 4, max_steps=2))
        assert result.steps == 2


class TestFailures:
    def test_a_tool_error_does_not_end_the_turn(self):
        # The model should get a chance to explain or try something else.
        def run_tool(_name, _arguments):
            raise RuntimeError("work item not found")

        ctx = context(
            [{"tool": "work_items", "arguments": {}}, {"text": "that ticket does not exist"}],
            run_tool=run_tool,
        )
        result = run_turn(ctx)
        assert result.answered is True
        assert result.text == "that ticket does not exist"

    def test_the_tool_error_reaches_the_model(self):
        seen = {}

        def run_tool(_name, _arguments):
            raise RuntimeError("boom")

        def call_model(messages, _tools):
            if len(messages) > 2:
                seen["last"] = messages[-1]["content"]
                return {"text": "ok"}
            return {"tool": "work_items", "arguments": {}}

        run_turn(context([], call_model=call_model, run_tool=run_tool))
        assert "error: boom" in seen["last"]

    def test_a_model_failure_ends_the_turn_cleanly(self):
        def call_model(_messages, _tools):
            raise RuntimeError("gateway down")

        result = run_turn(context([], call_model=call_model))
        assert result.answered is False
        assert result.reason == "model_failed"

    def test_an_empty_reply_is_not_an_answer(self):
        result = run_turn(context([{}]))
        assert result.answered is False
        assert result.reason == "empty_reply"


class TestProgress:
    def test_each_tool_call_reports_progress(self):
        lines = []
        ctx = context(
            [{"tool": "recall", "arguments": {}}, {"text": "done"}],
            on_progress=lines.append,
        )
        run_turn(ctx)
        assert lines == ["using recall"]

    def test_a_failing_progress_report_never_ends_the_turn(self):
        # Progress is a courtesy. Losing it must not lose the answer.
        def on_progress(_line):
            raise RuntimeError("thread deleted")

        ctx = context(
            [{"tool": "recall", "arguments": {}}, {"text": "done"}],
            on_progress=on_progress,
        )
        result = run_turn(ctx)
        assert result.answered is True
