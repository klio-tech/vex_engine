"""Tests for the evidence-loop publish path added to ``/capture/transcript``.

Mock-based, mirroring ``tests/test_capture.py``: no live Postgres, no real
LLM calls. ``app.capture.SessionLocal`` is patched to a fake session;
``generate_summary`` / ``extract_typed_memories`` / ``ensure_project`` are patched so
the distillation path is deterministic; ``app.capture.enqueue_trace_outbox``
is the mockable seam for the evidence publish (patched directly rather than
reaching into the Redis client, per the module's docstring).

Auth is bypassed by the shared ``client`` fixture (``conftest.py``), which
overrides ``verify_api_key`` — the fixture the brief calls ``app_with_auth``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import app.capture as capture


def _patch_db():
    """Patch SessionLocal with a fake session; return (patcher, session)."""
    session = MagicMock()
    return patch("app.capture.SessionLocal", MagicMock(return_value=session)), session


def _body():
    return {
        "session_id": "s1",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "done"},
        ],
        "tool_calls": [{"tool": "Edit", "args": {"file_path": "a.py"}}],
    }


def _distillation_mocks():
    """Patch the distillation seams so no real DB/LLM call happens."""
    return (
        patch.object(capture.brain, "ensure_project", MagicMock(return_value=None)),
        patch("app.capture.generate_summary", AsyncMock(return_value=None)),
        patch("app.capture.extract_typed_memories", AsyncMock(return_value=[])),
    )


def test_transcript_capture_enqueues_outbox(client):
    """A successful transcript capture also publishes to the evidence outbox."""
    db_patch, _ = _patch_db()
    ensure_project, gen_summary, ext_facts = _distillation_mocks()
    with (
        db_patch,
        ensure_project,
        gen_summary,
        ext_facts,
        patch("app.capture.enqueue_trace_outbox", AsyncMock()) as enq,
    ):
        r = client.post("/capture/transcript", json=_body())

    assert r.status_code == 200
    assert enq.called
    payload = enq.call_args.kwargs["payload"]
    # The published session_id is the DERIVED (principal-scoped) identity,
    # not the raw client-supplied "s1" — see _capture_session_id. Using the
    # raw value would collide across principals that happen to send the same
    # client string, or the same "no session_id" default.
    expected_session_id = capture._capture_session_id(
        "test-org", "mcp-default", "s1", user_id=None
    )
    assert payload["session_id"] == expected_session_id
    assert payload["session_id"] != "s1"
    assert len(payload["tool_calls"]) == 1


def test_transcript_capture_sessionless_uses_derived_identity_not_unknown(client):
    """A request with no session_id gets the derived per-principal identity,
    not a shared literal "unknown" that every sessionless capture in the org
    would otherwise collapse onto (silently overwriting each other's
    session_traces row on the richer-wins upsert)."""
    db_patch, _ = _patch_db()
    ensure_project, gen_summary, ext_facts = _distillation_mocks()
    body = _body()
    del body["session_id"]
    with (
        db_patch,
        ensure_project,
        gen_summary,
        ext_facts,
        patch("app.capture.enqueue_trace_outbox", AsyncMock()) as enq,
    ):
        r = client.post("/capture/transcript", json=body)

    assert r.status_code == 200
    payload = enq.call_args.kwargs["payload"]
    expected_session_id = capture._capture_session_id(
        "test-org", "mcp-default", None, user_id=None
    )
    assert payload["session_id"] == expected_session_id
    assert payload["session_id"] != "unknown"


def test_transcript_capture_publishes_redacted_content(client):
    """The published payload is redacted, not the raw request body."""
    db_patch, _ = _patch_db()
    ensure_project, gen_summary, ext_facts = _distillation_mocks()
    body = _body()
    body["messages"][0]["content"] = "here is my key sk-ant-abcdefghijklmnopqrstuvwx"
    with (
        db_patch,
        ensure_project,
        gen_summary,
        ext_facts,
        patch("app.capture.enqueue_trace_outbox", AsyncMock()) as enq,
    ):
        r = client.post("/capture/transcript", json=body)

    assert r.status_code == 200
    payload = enq.call_args.kwargs["payload"]
    assert "sk-ant-abcdefghijklmnopqrstuvwx" not in str(payload)


def test_transcript_capture_omitted_tool_calls_defaults_empty(client):
    """tool_calls is optional; omitting it is backward compatible."""
    db_patch, _ = _patch_db()
    ensure_project, gen_summary, ext_facts = _distillation_mocks()
    body = _body()
    del body["tool_calls"]
    with (
        db_patch,
        ensure_project,
        gen_summary,
        ext_facts,
        patch("app.capture.enqueue_trace_outbox", AsyncMock()) as enq,
    ):
        r = client.post("/capture/transcript", json=body)

    assert r.status_code == 200
    payload = enq.call_args.kwargs["payload"]
    assert payload["tool_calls"] == []


def test_outbox_failure_does_not_break_capture(client):
    """A publish failure never fails the capture request itself."""
    db_patch, _ = _patch_db()
    ensure_project, gen_summary, ext_facts = _distillation_mocks()
    with (
        db_patch,
        ensure_project,
        gen_summary,
        ext_facts,
        patch(
            "app.capture.enqueue_trace_outbox",
            AsyncMock(side_effect=RuntimeError("redis down")),
        ),
    ):
        r = client.post("/capture/transcript", json=_body())

    assert r.status_code == 200


def test_empty_transcript_does_not_enqueue_outbox(client):
    """The empty-transcript short-circuit returns before evidence publish."""
    with patch("app.capture.enqueue_trace_outbox", AsyncMock()) as enq:
        r = client.post("/capture/transcript", json={"messages": [{"content": "  "}]})

    assert r.status_code == 200
    assert r.json()["reason"] == "empty_transcript"
    enq.assert_not_called()
