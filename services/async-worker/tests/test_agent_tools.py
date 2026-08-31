"""Tests for the agent tool surface (`app.agent_tools.build_tools`).

Each tool returns a plain string to the model, so these assert on that
string rather than on internal call shapes wherever the string alone
proves the behavior. Where the boundary itself is the point — remember's
scope, list_work_items's truncation signal — the mock call arguments are
asserted directly.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.agent_tools import MAX_WORK_ITEMS, build_tools
from shared.audience_scope import AudienceScope

ORG_ID = "org-123"
AGENT_ID = "22222222-2222-2222-2222-222222222222"
TOKEN = "plane_api_test"
SLUG = "acme"


def _scope(**overrides):
    defaults = {"project_id": None, "work_item_id": None, "user_id": None}
    defaults.update(overrides)
    return AudienceScope(**defaults)


def _record(content):
    return SimpleNamespace(content=content)


# --- recall -----------------------------------------------------------


def test_recall_needs_a_query():
    _, run_tool = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
    assert run_tool("recall", {"query": "  "}) == "recall needs a query"


def test_recall_with_no_org_says_so_rather_than_guessing():
    _, run_tool = build_tools(_scope(), None, TOKEN, SLUG, AGENT_ID)
    assert run_tool("recall", {"query": "anything"}) == (
        "no memory is connected to this conversation"
    )


def test_recall_reads_at_org_scope_matching_the_audience():
    scope = _scope(project_id="proj-1", work_item_id="wi-1", user_id="user-1")
    with (
        patch("app.db.SessionLocal", return_value=MagicMock()),
        patch(
            "shared.memory.retrieve_memories_scoped",
            new_callable=AsyncMock,
            return_value=[_record("the deploy moved to Railway")],
        ) as mock_retrieve,
    ):
        _, run_tool = build_tools(scope, ORG_ID, TOKEN, SLUG, AGENT_ID)
        result = run_tool("recall", {"query": "deploy"})

    assert "the deploy moved to Railway" in result
    mock_retrieve.assert_awaited_once()
    _, kwargs = mock_retrieve.call_args
    assert kwargs["scope"] == "org"
    assert kwargs["user_id"] == "user-1"
    assert kwargs["project_id"] == "proj-1"
    assert kwargs["work_item_id"] == "wi-1"


def test_recall_records_a_count_when_it_finds_something():
    """`run_tool.recalled_memory_counts` is what `agent_turn.execute_turn`
    reads to build the provenance line -- a hit must be logged."""
    with (
        patch("app.db.SessionLocal", return_value=MagicMock()),
        patch(
            "shared.memory.retrieve_memories_scoped",
            new_callable=AsyncMock,
            return_value=[_record("a"), _record("b"), _record("c")],
        ),
    ):
        _, run_tool = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
        run_tool("recall", {"query": "anything"})

    assert run_tool.recalled_memory_counts == [3]


def test_recall_records_nothing_when_it_finds_nothing():
    """Zero memories is not a citation -- see `_extract_agent_provenance`
    on the workspace app side. Nothing should even be logged for it."""
    with (
        patch("app.db.SessionLocal", return_value=MagicMock()),
        patch(
            "shared.memory.retrieve_memories_scoped",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        _, run_tool = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
        run_tool("recall", {"query": "anything"})

    assert run_tool.recalled_memory_counts == []


def test_recall_counts_accumulate_across_calls_in_one_turn():
    with (
        patch("app.db.SessionLocal", return_value=MagicMock()),
        patch(
            "shared.memory.retrieve_memories_scoped",
            new_callable=AsyncMock,
            side_effect=[[_record("a")], [_record("b"), _record("c")]],
        ),
    ):
        _, run_tool = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
        run_tool("recall", {"query": "first"})
        run_tool("recall", {"query": "second"})

    assert run_tool.recalled_memory_counts == [1, 2]


def test_recall_failure_is_returned_not_raised():
    with (
        patch("app.db.SessionLocal", return_value=MagicMock()),
        patch(
            "shared.memory.retrieve_memories_scoped",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db down"),
        ),
    ):
        _, run_tool = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
        result = run_tool("recall", {"query": "anything"})

    assert "memory search failed" in result


# --- remember -----------------------------------------------------------


def test_remember_needs_content():
    _, run_tool = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
    assert run_tool("remember", {"content": "   "}) == "remember needs content"


def test_remember_with_no_org_or_agent_declines_rather_than_writing_unscoped():
    _, run_tool = build_tools(_scope(), None, TOKEN, SLUG, None)
    assert run_tool("remember", {"content": "the sky is blue"}) == (
        "no memory is connected to this conversation"
    )


def test_remember_writes_at_org_scope_matching_recall():
    """The security-critical assertion: a remember write must land at
    exactly the scope `_recall` reads — never private, never another org."""
    scope = _scope(project_id="proj-1", work_item_id="wi-1", user_id="user-1")
    with (
        patch("app.db.SessionLocal", return_value=MagicMock()),
        patch(
            "shared.brain.write_scoped_memory",
            new_callable=AsyncMock,
            return_value={"stored": True, "id": "mem-1"},
        ) as mock_write,
    ):
        _, run_tool = build_tools(scope, ORG_ID, TOKEN, SLUG, AGENT_ID)
        result = run_tool("remember", {"content": "the team ships on Fridays"})

    assert result == "stored"
    mock_write.assert_awaited_once()
    _, kwargs = mock_write.call_args
    assert kwargs["org_id"] == ORG_ID
    assert kwargs["agent_id"] == AGENT_ID
    assert kwargs["scope"] == "org"
    assert kwargs["project_id"] == "proj-1"
    assert kwargs["work_item_id"] == "wi-1"
    # Authorship, never a visibility predicate on a shared row — see
    # write_scoped_memory's docstring.
    assert kwargs["user_id"] == "user-1"
    assert kwargs["content"] == "the team ships on Fridays"
    assert kwargs["metadata"]["source"] == "agent_chat"


def test_remember_never_files_a_work_item_without_a_project():
    """AudienceScope currently never produces this combination, but the
    tool must not forward it blindly: write_scoped_memory raises on a
    work_item_id with no project_id."""
    scope = _scope(project_id=None, work_item_id="wi-orphan")
    with (
        patch("app.db.SessionLocal", return_value=MagicMock()),
        patch(
            "shared.brain.write_scoped_memory",
            new_callable=AsyncMock,
            return_value={"stored": True, "id": "mem-1"},
        ) as mock_write,
    ):
        _, run_tool = build_tools(scope, ORG_ID, TOKEN, SLUG, AGENT_ID)
        run_tool("remember", {"content": "x"})

    assert mock_write.call_args.kwargs["work_item_id"] is None


def test_remember_deduplicated_is_reported_as_already_known():
    with (
        patch("app.db.SessionLocal", return_value=MagicMock()),
        patch(
            "shared.brain.write_scoped_memory",
            new_callable=AsyncMock,
            return_value={"stored": False, "deduplicated": True, "id": "mem-1"},
        ),
    ):
        _, run_tool = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
        result = run_tool("remember", {"content": "the team ships on Fridays"})

    assert "already knows" in result


def test_remember_store_failure_is_returned_not_raised():
    with (
        patch("app.db.SessionLocal", return_value=MagicMock()),
        patch(
            "shared.brain.write_scoped_memory",
            new_callable=AsyncMock,
            return_value={"stored": False, "error": "embedding_failed"},
        ),
    ):
        _, run_tool = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
        result = run_tool("remember", {"content": "x"})

    assert "could not store" in result
    assert "embedding_failed" in result


def test_remember_exception_is_returned_not_raised():
    with (
        patch("app.db.SessionLocal", return_value=MagicMock()),
        patch(
            "shared.brain.write_scoped_memory",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db down"),
        ),
    ):
        _, run_tool = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
        result = run_tool("remember", {"content": "x"})

    assert "could not store that" in result


# --- list_projects -----------------------------------------------------


def test_list_projects_reports_non_200():
    response = MagicMock(status_code=500)
    with patch("app.agent_tools.requests.get", return_value=response):
        _, run_tool = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
        result = run_tool("list_projects", {})

    assert "could not list projects" in result
    assert "500" in result


def test_list_projects_returns_id_and_name():
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "results": [{"id": "p1", "name": "Launch", "extra": "dropped"}]
    }
    with patch("app.agent_tools.requests.get", return_value=response):
        _, run_tool = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
        result = run_tool("list_projects", {})

    assert json.loads(result) == [{"id": "p1", "name": "Launch"}]


# --- list_work_items -----------------------------------------------------


def test_list_work_items_needs_a_project_id():
    _, run_tool = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
    assert run_tool("list_work_items", {}) == "list_work_items needs a project_id"


def test_list_work_items_reports_non_200():
    response = MagicMock(status_code=404)
    with patch("app.agent_tools.requests.get", return_value=response):
        _, run_tool = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
        result = run_tool("list_work_items", {"project_id": "proj-1"})

    assert "could not list work items" in result
    assert "404" in result


def _issues_response(rows, total_count=None):
    payload = {"results": rows}
    if total_count is not None:
        payload["total_count"] = total_count
    response = MagicMock(status_code=200)
    response.json.return_value = payload
    return response


def _states_response(rows):
    response = MagicMock(status_code=200)
    response.json.return_value = {"results": rows}
    return response


def test_list_work_items_resolves_state_id_to_a_label():
    issues = _issues_response(
        [{"id": "wi-1", "name": "Ship it", "state": "state-done", "priority": "high"}]
    )
    states = _states_response([{"id": "state-done", "name": "Done"}])
    with patch("app.agent_tools.requests.get", side_effect=[issues, states]):
        _, run_tool = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
        result = run_tool("list_work_items", {"project_id": "proj-1"})

    items = json.loads(result.split("\n…")[0])
    assert items == [
        {"id": "wi-1", "name": "Ship it", "state": "Done", "priority": "high"}
    ]


def test_list_work_items_falls_back_to_raw_state_id_when_states_call_fails():
    issues = _issues_response(
        [{"id": "wi-1", "name": "Ship it", "state": "state-done"}]
    )
    states = MagicMock(status_code=500)
    with patch("app.agent_tools.requests.get", side_effect=[issues, states]):
        _, run_tool = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
        result = run_tool("list_work_items", {"project_id": "proj-1"})

    items = json.loads(result.split("\n…")[0])
    assert items[0]["state"] == "state-done"


def test_list_work_items_truncation_is_visible_when_total_count_exceeds_the_page():
    rows = [{"id": f"wi-{i}", "name": f"Item {i}"} for i in range(MAX_WORK_ITEMS)]
    issues = _issues_response(rows, total_count=MAX_WORK_ITEMS + 37)
    states = _states_response([])
    with patch("app.agent_tools.requests.get", side_effect=[issues, states]):
        _, run_tool = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
        result = run_tool("list_work_items", {"project_id": "proj-1"})

    assert f"showing {MAX_WORK_ITEMS} of {MAX_WORK_ITEMS + 37}" in result


def test_list_work_items_says_nothing_extra_when_nothing_was_cut():
    issues = _issues_response([{"id": "wi-1", "name": "Only item"}], total_count=1)
    states = _states_response([])
    with patch("app.agent_tools.requests.get", side_effect=[issues, states]):
        _, run_tool = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
        result = run_tool("list_work_items", {"project_id": "proj-1"})

    assert "showing" not in result


def test_list_work_items_requests_a_full_page():
    issues = _issues_response([])
    states = _states_response([])
    with patch(
        "app.agent_tools.requests.get", side_effect=[issues, states]
    ) as mock_get:
        _, run_tool = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
        run_tool("list_work_items", {"project_id": "proj-1"})

    first_call = mock_get.call_args_list[0]
    assert first_call.kwargs["params"] == {"per_page": MAX_WORK_ITEMS}


# --- unknown tool -----------------------------------------------------


def test_unknown_tool_is_named_not_raised():
    _, run_tool = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
    assert run_tool("delete_everything", {}) == "no such tool: delete_everything"


def test_schemas_include_all_four_tools():
    schemas, _ = build_tools(_scope(), ORG_ID, TOKEN, SLUG, AGENT_ID)
    names = {s["function"]["name"] for s in schemas}
    assert names == {"recall", "list_projects", "list_work_items", "remember"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
