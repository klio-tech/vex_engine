"""Tests for `app.agent_turn` — loading a turn's context and reporting
what it actually recalled.

Two things are pinned here:

- `load_turn` resolves a `scope_label`: the real, human-readable scope a
  turn's recall is narrowed to — the linked project's `display_name` when
  there is one, else the workspace's own `name`. Never a guess, never a
  constant.
- `execute_turn` turns what `recall` actually returned during the turn
  into a `provenance` object on the FINAL PATCH only — never on a
  progress edit, never fabricated when recall found nothing or when no
  real scope label was available, and correctly summed across multiple
  `recall` calls in the same turn (the number is a total memory count,
  because `scope_label` — and therefore `source_name` — is the turn's
  audience scope and does not vary call to call).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from app.agent_turn import execute_turn, load_turn
from cryptography.fernet import Fernet
from shared.agent_runtime import TurnResult
from shared.token_crypto import derive_key


# --- load_turn: scope_label --------------------------------------------


#: The workspace stores a bot's token Fernet-encrypted; the row a turn loads
#: carries that ciphertext, and load_turn decrypts it with the shared secret.
#: Real crypto here on purpose -- a stub would hide a KDF drift between the
#: two repos, which is the exact failure this contract exists to catch.
_TEST_SECRET = "klio-test-secret"
_RAW_BOT_TOKEN = "klio_bot_test"
_ENCRYPTED_BOT_TOKEN = (
    Fernet(derive_key(_TEST_SECRET)).encrypt(_RAW_BOT_TOKEN.encode()).decode()
)


@pytest.fixture(autouse=True)
def _shared_secret(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN_CIPHER_KEY", _TEST_SECRET)


def _agent_row(**overrides):
    defaults = dict(
        id="agent-1",
        name="Researcher",
        instructions="be helpful",
        model="",
        is_enabled=True,
        bot_user_id="bot-user-1",
        agent_token_encrypted=_ENCRYPTED_BOT_TOKEN,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _message_row(**overrides):
    defaults = dict(
        id="msg-1",
        content="what's the ship date",
        actor_id="human-1",
        parent_id=None,
        channel_id="chan-1",
        kind="plot",
        is_private=False,
        project_id=None,
        workspace_id="ws-1",
        workspace_slug="acme",
        workspace_name="Acme Co",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _conn(fetchone_results=None, fetchall_results=None):
    """A fake sqlalchemy Connection: `.execute(...)` returns an object
    whose `.fetchone()`/`.fetchall()` come off two queues, consumed in
    call order — matching how `load_turn` issues its queries."""
    fetchone_results = list(fetchone_results or [])
    fetchall_results = list(fetchall_results or [])
    conn = MagicMock()

    def _execute(*_args, **_kwargs):
        result = MagicMock()
        result.fetchone.side_effect = lambda: fetchone_results.pop(0)
        result.fetchall.side_effect = lambda: fetchall_results.pop(0)
        return result

    conn.execute.side_effect = _execute
    return conn


@pytest.mark.parametrize("channel_kind", ["plot"])
def test_scope_label_is_the_linked_project_display_name(channel_kind):
    message = _message_row(project_id="plane-proj-1")
    workspace_conn = _conn(
        fetchone_results=[_agent_row(), message],
        fetchall_results=[[], []],  # participants, history
    )
    linked = SimpleNamespace(
        id="proj-1", org_id="org-1", display_name="Launch Readiness"
    )
    engine_conn = _conn(
        fetchone_results=[SimpleNamespace(org_id="org-1"), linked],
    )

    loaded = load_turn(
        workspace_conn, engine_conn, agent_id="agent-1", message_id="msg-1"
    )

    assert loaded["scope_label"] == "Launch Readiness"


def test_scope_label_falls_back_to_the_workspace_name_with_no_linked_project():
    message = _message_row(project_id=None)
    workspace_conn = _conn(
        fetchone_results=[_agent_row(), message],
        fetchall_results=[[], []],
    )
    engine_conn = _conn(fetchone_results=[SimpleNamespace(org_id="org-1")])

    loaded = load_turn(
        workspace_conn, engine_conn, agent_id="agent-1", message_id="msg-1"
    )

    assert loaded["scope_label"] == "Acme Co"


def test_scope_label_falls_back_to_workspace_name_when_project_id_does_not_resolve():
    """A `project_id` on the message that this org has never linked --
    `_LINKED_PROJECT_SQL` returns nothing, so the label falls back to the
    workspace rather than staying stuck on an unresolved project."""
    message = _message_row(project_id="plane-proj-orphan")
    workspace_conn = _conn(
        fetchone_results=[_agent_row(), message],
        fetchall_results=[[], []],
    )
    engine_conn = _conn(fetchone_results=[SimpleNamespace(org_id="org-1"), None])

    loaded = load_turn(
        workspace_conn, engine_conn, agent_id="agent-1", message_id="msg-1"
    )

    assert loaded["scope_label"] == "Acme Co"


# --- execute_turn: provenance -------------------------------------------


def _loaded(**overrides):
    defaults = dict(
        agent=_agent_row(),
        # What load_turn produces from the row's ciphertext: the raw
        # credential execute_turn presents as X-Api-Key.
        agent_token=_RAW_BOT_TOKEN,
        message=_message_row(),
        scope=SimpleNamespace(project_id=None, work_item_id=None, user_id=None),
        org_id="org-1",
        history=[],
        thread_root="msg-1",
        scope_label="Acme Co",
    )
    defaults.update(overrides)
    return defaults


def _build_tools_returning(run_tool):
    def build_tools(_scope, _org_id, _token, _slug, _agent_id):
        return [], run_tool

    return build_tools


def _run_tool_with_counts(counts):
    """A `run_tool` stand-in carrying the `recalled_memory_counts`
    attribute the real `agent_tools.build_tools` exposes."""
    fn = MagicMock(return_value="ignored")
    fn.recalled_memory_counts = list(counts)
    return fn


@pytest.fixture(autouse=True)
def _no_real_http():
    with patch("app.agent_turn.requests") as mock_requests:
        post_response = MagicMock(status_code=201)
        post_response.json.return_value = {"id": "placeholder-msg"}
        mock_requests.post.return_value = post_response
        yield mock_requests


def test_a_recalled_turn_reports_memory_count_and_the_real_scope_label(_no_real_http):
    run_tool = _run_tool_with_counts([3])
    with patch(
        "app.agent_turn.run_turn",
        return_value=TurnResult(answered=True, steps=2, text="the 12th"),
    ):
        execute_turn(_loaded(), build_tools=_build_tools_returning(run_tool))

    final_patch = _no_real_http.patch.call_args_list[-1]
    assert final_patch.kwargs["json"]["provenance"] == {
        "memory_count": 3,
        "source_name": "Acme Co",
    }


def test_multiple_recalls_in_one_turn_sum_to_a_total_count(_no_real_http):
    """Two `recall` calls in the same turn -- the number reported is the
    TOTAL memories recalled, because `source_name` (the audience scope)
    is identical across every call in one turn by construction."""
    run_tool = _run_tool_with_counts([3, 2])
    with patch(
        "app.agent_turn.run_turn",
        return_value=TurnResult(answered=True, steps=3, text="the 12th"),
    ):
        execute_turn(_loaded(), build_tools=_build_tools_returning(run_tool))

    final_patch = _no_real_http.patch.call_args_list[-1]
    assert final_patch.kwargs["json"]["provenance"]["memory_count"] == 5


def test_a_turn_that_never_recalled_anything_sends_no_provenance(_no_real_http):
    run_tool = _run_tool_with_counts([])
    with patch(
        "app.agent_turn.run_turn",
        return_value=TurnResult(answered=True, steps=1, text="I don't know"),
    ):
        execute_turn(_loaded(), build_tools=_build_tools_returning(run_tool))

    final_patch = _no_real_http.patch.call_args_list[-1]
    assert "provenance" not in final_patch.kwargs["json"]


def test_a_recall_that_matched_nothing_sends_no_provenance(_no_real_http):
    """`recall` only records a count when it actually found something --
    see `agent_tools._recall`. Zero memories is not a citation."""
    run_tool = _run_tool_with_counts([])  # recall ran, found nothing, logged nothing
    run_tool.side_effect = None
    with patch(
        "app.agent_turn.run_turn",
        return_value=TurnResult(answered=True, steps=1, text="I don't have that"),
    ):
        execute_turn(_loaded(), build_tools=_build_tools_returning(run_tool))

    final_patch = _no_real_http.patch.call_args_list[-1]
    assert "provenance" not in final_patch.kwargs["json"]


def test_no_real_scope_label_means_no_provenance_even_with_a_real_recall(_no_real_http):
    """Never a fabricated or constant `source_name` -- if `load_turn`
    could not resolve a real one, the citation is dropped entirely rather
    than guessed."""
    run_tool = _run_tool_with_counts([4])
    with patch(
        "app.agent_turn.run_turn",
        return_value=TurnResult(answered=True, steps=2, text="the 12th"),
    ):
        execute_turn(
            _loaded(scope_label=None), build_tools=_build_tools_returning(run_tool)
        )

    final_patch = _no_real_http.patch.call_args_list[-1]
    assert "provenance" not in final_patch.kwargs["json"]


def test_progress_edits_never_carry_provenance(_no_real_http):
    """Provenance belongs on the final answer only -- a `…using recall`
    progress edit must never claim a citation before the turn is done."""
    run_tool = _run_tool_with_counts([3])

    def _fake_run_turn(context, **_kwargs):
        context.on_progress("using recall")
        return TurnResult(answered=True, steps=2, text="the 12th")

    with patch("app.agent_turn.run_turn", side_effect=_fake_run_turn):
        execute_turn(_loaded(), build_tools=_build_tools_returning(run_tool))

    progress_patch = _no_real_http.patch.call_args_list[0]
    assert "provenance" not in progress_patch.kwargs["json"]
    final_patch = _no_real_http.patch.call_args_list[-1]
    assert final_patch.kwargs["json"]["provenance"]["memory_count"] == 3


def test_an_unanswered_turn_still_reports_a_real_recall(_no_real_http):
    """`max_steps`/`deadline` still get the honest fallback text, and if
    the turn genuinely recalled something before it ran out, that
    recall really happened -- the citation is still true."""
    run_tool = _run_tool_with_counts([2])
    with patch(
        "app.agent_turn.run_turn",
        return_value=TurnResult(
            answered=False, steps=8, reason="max_steps", stopped_early=True
        ),
    ):
        execute_turn(_loaded(), build_tools=_build_tools_returning(run_tool))

    final_patch = _no_real_http.patch.call_args_list[-1]
    assert final_patch.kwargs["json"]["provenance"] == {
        "memory_count": 2,
        "source_name": "Acme Co",
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
