"""Tests for the REST capture layer (``app.capture``).

Mock-based, mirroring ``tests/test_write_tools.py`` and ``tests/test_curator.py``:
no live Postgres, no real embeddings/LLM.  The brain/memory seams are patched at
the ``app.capture`` module boundary:

* ``app.capture.SessionLocal`` → a ``MagicMock`` so ``SessionLocal()`` yields a
  fake session that is ``.close()``-d in the ``finally``.
* ``app.capture.brain.write_scoped_memory`` → an ``AsyncMock`` capturing kwargs.
* ``app.capture.extract_typed_memories`` / ``generate_summary`` /
  ``retrieve_memories_scoped`` → ``AsyncMock``s with canned returns.
* ``app.capture.brain.ensure_project`` / ``find_project_id`` / ``find_space_id``
  → ``MagicMock``s.

Auth is bypassed by the shared ``client`` fixture (``conftest.py``), which
overrides ``verify_api_key`` to a fixed ``org_id='test-org'`` — the override
flows through ``resolve_principal_dep`` because that dependency is wired via
``Depends(verify_api_key)``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from datetime import datetime, timezone

import app.capture as capture
from app.main import create_app
from fastapi.testclient import TestClient


def _memrow(id="m1", memory_type="observation", content="hi", created_at=None):
    """A memory row/record stand-in exposing the attrs the serializer reads."""
    return SimpleNamespace(
        id=id,
        memory_type=memory_type,
        content=content,
        created_at=created_at or datetime(2026, 5, 1, tzinfo=timezone.utc),
    )


def _patch_db():
    """Patch SessionLocal with a fake session; return (patcher, session)."""
    session = MagicMock()
    return patch("app.capture.SessionLocal", MagicMock(return_value=session)), session


# ---------------------------------------------------------------------------
# /capture/event
# ---------------------------------------------------------------------------


def test_event_stores_observation_with_source_tag(client):
    """A bare event writes an 'observation' tagged source=hook, scope org."""
    write = AsyncMock(return_value={"stored": True, "id": "x", "scope": "org"})
    db_patch, session = _patch_db()
    with (
        db_patch,
        patch.object(capture.brain, "write_scoped_memory", write),
        patch.object(capture.brain, "ensure_project", MagicMock(return_value=None)),
    ):
        resp = client.post("/capture/event", json={"content": "do the thing"})

    assert resp.status_code == 200
    assert resp.json()["stored"] is True
    write.assert_awaited_once()
    kw = write.await_args.kwargs
    assert kw["org_id"] == "test-org"
    assert kw["memory_type"] == "observation"
    assert kw["content"] == "do the thing"
    assert kw["metadata"]["source"] == "hook"
    assert kw["project_id"] is None
    session.close.assert_called_once()


def test_event_coerces_unknown_memory_type_to_observation(client):
    """A memory_type outside the RAW set is coerced to 'observation'."""
    write = AsyncMock(return_value={"stored": True})
    db_patch, _ = _patch_db()
    with (
        db_patch,
        patch.object(capture.brain, "write_scoped_memory", write),
        patch.object(capture.brain, "ensure_project", MagicMock(return_value=None)),
    ):
        client.post(
            "/capture/event",
            json={"content": "x", "memory_type": "summary"},
        )
    assert write.await_args.kwargs["memory_type"] == "observation"


def test_event_allows_raw_memory_type(client):
    """A valid RAW type (e.g. 'decision') is passed through unchanged."""
    write = AsyncMock(return_value={"stored": True})
    db_patch, _ = _patch_db()
    with (
        db_patch,
        patch.object(capture.brain, "write_scoped_memory", write),
        patch.object(capture.brain, "ensure_project", MagicMock(return_value=None)),
    ):
        client.post(
            "/capture/event", json={"content": "ship", "memory_type": "decision"}
        )
    assert write.await_args.kwargs["memory_type"] == "decision"


def test_event_tags_project_when_git_remote_present(client):
    """git_remote → ensure_project resolves a project_id that tags the write."""
    write = AsyncMock(return_value={"stored": True})
    ensure = MagicMock(return_value="proj-123")
    db_patch, _ = _patch_db()
    with (
        db_patch,
        patch.object(capture.brain, "write_scoped_memory", write),
        patch.object(capture.brain, "ensure_project", ensure),
    ):
        client.post(
            "/capture/event",
            json={"content": "x", "git_remote": "git@github.com:o/repo.git"},
        )
    assert ensure.call_args.kwargs["display_name"] == "repo"
    assert write.await_args.kwargs["project_id"] == "proj-123"


def test_event_empty_content_returns_422(client):
    """Empty content violates the min_length schema → 422 before any write."""
    db_patch, _ = _patch_db()
    with db_patch:
        resp = client.post("/capture/event", json={"content": ""})
    assert resp.status_code == 422


def test_event_missing_key_returns_401():
    """No X-Vex-Key (no auth override) → 401 from the principal dependency."""
    app = create_app()
    c = TestClient(app)
    resp = c.post("/capture/event", json={"content": "x"})
    assert resp.status_code == 401


def test_event_write_exception_returns_clean_error(client):
    """An unexpected write failure surfaces a clean error, not a 500 traceback."""
    write = AsyncMock(side_effect=RuntimeError("boom"))
    db_patch, session = _patch_db()
    with (
        db_patch,
        patch.object(capture.brain, "write_scoped_memory", write),
        patch.object(capture.brain, "ensure_project", MagicMock(return_value=None)),
    ):
        resp = client.post("/capture/event", json={"content": "x"})
    assert resp.status_code == 200
    assert resp.json() == {"stored": False, "error": "capture_failed"}
    session.close.assert_called_once()


# ---------------------------------------------------------------------------
# /capture/transcript
# ---------------------------------------------------------------------------


JOB_ID = "77777777-7777-7777-7777-777777777777"


def test_transcript_queues_a_job_and_calls_no_model_inline(client):
    """The request path resolves and enqueues; it does not distil.

    Distillation was two LLM calls inside this handler, with the Stop hook
    waiting on them. It now happens in ``async-worker``, so the endpoint must
    write NOTHING to memory itself -- a write here would mean the model work
    is still on the caller's clock.
    """
    write = AsyncMock(return_value={"stored": True})
    create = MagicMock(return_value=JOB_ID)
    publish = AsyncMock()
    db_patch, session = _patch_db()
    with (
        db_patch,
        patch.object(capture.brain, "write_scoped_memory", write),
        patch.object(capture.brain, "ensure_project", MagicMock(return_value=None)),
        patch("app.capture.create_job", create),
        patch("app.capture.publish_job", publish),
    ):
        resp = client.post(
            "/capture/transcript",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert resp.json() == {
        "stored": 0,
        "facts": 0,
        "by_kind": {},
        "summary": False,
        "queued": True,
        "job_id": JOB_ID,
    }
    write.assert_not_awaited()
    create.assert_called_once()
    publish.assert_awaited_once_with(JOB_ID)
    assert create.call_args.kwargs["transcript"] == "user: hello"
    assert create.call_args.kwargs["org_id"] == "test-org"
    session.close.assert_called_once()


def test_transcript_survives_a_publish_failure_because_the_row_is_durable(client):
    """A dead Redis delays the job to the next sweep; it must not fail the
    capture, and the response must say so rather than claiming success."""
    create = MagicMock(return_value=JOB_ID)
    db_patch, _ = _patch_db()
    with (
        db_patch,
        patch.object(capture.brain, "write_scoped_memory", AsyncMock()),
        patch.object(capture.brain, "ensure_project", MagicMock(return_value=None)),
        patch("app.capture.create_job", create),
        patch(
            "app.capture.publish_job", AsyncMock(side_effect=RuntimeError("no redis"))
        ),
    ):
        resp = client.post(
            "/capture/transcript", json={"messages": [{"content": "hi"}]}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["queued"] is False
    assert body["job_id"] == JOB_ID
    create.assert_called_once()


def test_transcript_empty_messages_short_circuits(client):
    """No usable message text → nothing stored, no LLM call, no session work.

    Since the node wiring the session is OPENED before the empty check — the
    active-node resolution needs it, and ``SessionLocal`` connects lazily so
    an unselected request still touches no database — but it must be closed
    on this path and no query, write, or LLM call may happen.
    """
    write = AsyncMock()
    sl = MagicMock()
    with (
        patch("app.capture.SessionLocal", sl),
        patch.object(capture.brain, "write_scoped_memory", write),
        patch(
            "app.capture.create_job",
            MagicMock(return_value="77777777-7777-7777-7777-777777777777"),
        ) as gs,
        patch("app.capture.publish_job", AsyncMock()) as ef,
    ):
        resp = client.post(
            "/capture/transcript", json={"messages": [{"content": "  "}]}
        )

    assert resp.json()["reason"] == "empty_transcript"
    write.assert_not_called()
    gs.assert_not_called()
    ef.assert_not_called()
    sl.return_value.execute.assert_not_called()
    sl.return_value.close.assert_called_once()


# ---------------------------------------------------------------------------
# /capture/recall
# ---------------------------------------------------------------------------


def test_recall_with_query_uses_vector_search(client):
    """A non-empty query routes to retrieve_memories_scoped and serializes rows."""
    retr = AsyncMock(return_value=[_memrow(id="a"), _memrow(id="b")])
    db_patch, _ = _patch_db()
    with (
        db_patch,
        patch("app.capture.retrieve_memories_scoped", retr),
        patch.object(capture.brain, "find_project_id", MagicMock(return_value=None)),
    ):
        resp = client.post("/capture/recall", json={"query": "auth flow", "limit": 5})

    body = resp.json()
    assert body["count"] == 2
    assert body["memories"][0]["id"] == "a"
    assert retr.await_args.kwargs["limit"] == 5


def test_recall_empty_query_falls_back_to_recency(client):
    """An empty query reads most-recent rows directly (no vector search)."""
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = [
        _memrow(id="r1"),
        _memrow(id="r2"),
    ]
    retr = AsyncMock()
    with (
        patch("app.capture.SessionLocal", MagicMock(return_value=session)),
        patch("app.capture.retrieve_memories_scoped", retr),
        patch.object(capture.brain, "find_project_id", MagicMock(return_value=None)),
    ):
        resp = client.post("/capture/recall", json={"query": "   "})

    body = resp.json()
    assert body["count"] == 2
    assert [m["id"] for m in body["memories"]] == ["r1", "r2"]
    retr.assert_not_called()  # recency path, not vector search


def test_recall_resolves_project_filter_from_git_remote(client):
    """git_remote that resolves to a project scopes the recall by project_id."""
    retr = AsyncMock(return_value=[])
    find = MagicMock(return_value="proj-9")
    db_patch, _ = _patch_db()
    with (
        db_patch,
        patch("app.capture.retrieve_memories_scoped", retr),
        patch.object(capture.brain, "find_project_id", find),
    ):
        client.post(
            "/capture/recall",
            json={"query": "x", "git_remote": "git@github.com:o/repo.git"},
        )
    assert find.call_args.kwargs["identifier"] == "git@github.com:o/repo.git"
    assert retr.await_args.kwargs["project_id"] == "proj-9"


def test_recall_caps_limit(client):
    """A wildly large limit is clamped to the hard cap before hitting the DB."""
    retr = AsyncMock(return_value=[])
    db_patch, _ = _patch_db()
    with (
        db_patch,
        patch("app.capture.retrieve_memories_scoped", retr),
        patch.object(capture.brain, "find_project_id", MagicMock(return_value=None)),
    ):
        client.post("/capture/recall", json={"query": "x", "limit": 9999})
    assert retr.await_args.kwargs["limit"] == capture._MAX_RECALL_LIMIT


def test_recall_logs_event(client):
    """The vector-search path logs a recall event tagged source=hook with the
    returned result_count, query_text, resolved project_id, and the agent_id
    resolved from the X-Vex-Agent header."""
    retr = AsyncMock(return_value=[_memrow(id="a"), _memrow(id="b")])
    log = MagicMock()
    db_patch, session = _patch_db()
    with (
        db_patch,
        patch("app.capture.retrieve_memories_scoped", retr),
        patch("app.capture.log_recall_event", log),
        patch.object(capture.brain, "find_project_id", MagicMock(return_value=None)),
    ):
        resp = client.post(
            "/capture/recall",
            json={"query": "auth flow", "limit": 5},
            headers={"X-Vex-Agent": "klio-test-agent"},
        )

    assert resp.json()["count"] == 2
    log.assert_called_once()
    kw = log.call_args.kwargs
    assert kw["source"] == "hook"
    assert kw["result_count"] == 2
    assert kw["org_id"] == "test-org"
    assert kw["agent_id"] == "klio-test-agent"
    assert kw["query_text"] == "auth flow"
    assert kw["project_id"] is None
    session.close.assert_called_once()


def test_recall_logs_event_on_recency_branch(client):
    """The empty-query recency path ALSO logs a recall event (source=hook) with
    the recency result_count, a null query_text, and the agent_id resolved from
    the X-Vex-Agent header."""
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = [
        _memrow(id="r1"),
        _memrow(id="r2"),
    ]
    log = MagicMock()
    retr = AsyncMock()
    with (
        patch("app.capture.SessionLocal", MagicMock(return_value=session)),
        patch("app.capture.retrieve_memories_scoped", retr),
        patch("app.capture.log_recall_event", log),
        patch.object(capture.brain, "find_project_id", MagicMock(return_value=None)),
    ):
        resp = client.post(
            "/capture/recall",
            json={"query": "   "},
            headers={"X-Vex-Agent": "klio-test-agent"},
        )

    assert resp.json()["count"] == 2
    retr.assert_not_called()  # recency path, not vector search
    log.assert_called_once()
    kw = log.call_args.kwargs
    assert kw["source"] == "hook"
    assert kw["result_count"] == 2
    assert kw["agent_id"] == "klio-test-agent"
    assert kw["query_text"] is None


# ---------------------------------------------------------------------------
# /capture/event scope forwarding (agent isolation)
# ---------------------------------------------------------------------------


def test_event_scope_agent_is_forwarded(client):
    """scope='agent' in the request flows to write_scoped_memory."""
    write = AsyncMock(return_value={"stored": True, "scope": "agent"})
    db_patch, _ = _patch_db()
    with (
        db_patch,
        patch.object(capture.brain, "write_scoped_memory", write),
        patch.object(capture.brain, "ensure_project", MagicMock(return_value=None)),
    ):
        resp = client.post("/capture/event", json={"content": "x", "scope": "agent"})
    assert resp.status_code == 200
    assert write.await_args.kwargs["scope"] == "agent"


def test_event_scope_defaults_to_org(client):
    """Omitting scope writes scope='org' (backward compatible)."""
    write = AsyncMock(return_value={"stored": True})
    db_patch, _ = _patch_db()
    with (
        db_patch,
        patch.object(capture.brain, "write_scoped_memory", write),
        patch.object(capture.brain, "ensure_project", MagicMock(return_value=None)),
    ):
        client.post("/capture/event", json={"content": "x"})
    assert write.await_args.kwargs["scope"] == "org"


def test_event_rejects_unknown_scope(client):
    """A scope outside org|agent is a 422 (Literal validation)."""
    db_patch, _ = _patch_db()
    with db_patch:
        resp = client.post("/capture/event", json={"content": "x", "scope": "session"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Pre-LLM redaction (Task 3)
# ---------------------------------------------------------------------------

SECRET_T3 = (
    "sk_test_51TcfsTP4cXbUl5jnQJG2Ou9BUcBChSVqEDNj7d65uQjZXtAVzI1XR2"
    "eewh5KDq8GmCfrr1a1xRNcSdB1c7TtWrU00GsK8y8Je"
)


def test_combine_messages_redacts_before_llm():
    """_combine_messages must redact secrets before the text is sent to the LLM.

    The raw Stripe test key must be replaced by a [REDACTED:stripe_secret_key:…]
    placeholder; non-secret content is preserved unchanged.
    """
    from app.capture import TranscriptMessage, _combine_messages

    combined = _combine_messages(
        [
            TranscriptMessage(role="user", content=f"here is my key {SECRET_T3}"),
            TranscriptMessage(role="assistant", content="thanks, configuring now"),
        ]
    )
    assert SECRET_T3 not in combined
    assert "[REDACTED:stripe_secret_key:" in combined
    assert "configuring now" in combined


# ---------------------------------------------------------------------------
# recall_hidden threading (Task 7)
# ---------------------------------------------------------------------------


def test_capture_event_observation_is_recall_hidden(client, monkeypatch):
    """Passive hook observations are written recall_hidden=True.

    Observations are raw curator fuel and must not surface in agent recall,
    so capture_event must pass recall_hidden=True for memory_type='observation'.
    """
    import shared.brain as brain_mod

    seen = {}

    async def _fake_write(db, **kwargs):
        seen.update(kwargs)
        return {"stored": True, "deduplicated": False, "id": "x", "scope": "org"}

    monkeypatch.setattr(brain_mod, "write_scoped_memory", _fake_write)
    db_patch, _ = _patch_db()
    with db_patch:
        resp = client.post(
            "/capture/event",
            json={"content": "Used tool Bash", "memory_type": "observation"},
        )
    assert resp.status_code == 200
    assert seen["recall_hidden"] is True


def test_capture_event_note_is_visible(client, monkeypatch):
    """Deliberate writes (note/plan/decision/memory) are NOT recall-hidden.

    Only raw observations feed the curator as hidden rows; note/plan/decision/
    memory types must remain visible (recall_hidden=False) so agents can
    recall them directly.
    """
    import shared.brain as brain_mod

    seen = {}

    async def _fake_write(db, **kwargs):
        seen.update(kwargs)
        return {"stored": True, "deduplicated": False, "id": "x", "scope": "org"}

    monkeypatch.setattr(brain_mod, "write_scoped_memory", _fake_write)
    db_patch, _ = _patch_db()
    with db_patch:
        resp = client.post(
            "/capture/event", json={"content": "decision made", "memory_type": "note"}
        )
    assert resp.status_code == 200
    assert seen["recall_hidden"] is False


def test_recent_memories_sql_excludes_hidden(client):
    """_recent_memories must include recall_hidden = FALSE in its WHERE clause.

    The recency fallback (empty-query path) must not surface hidden curator-fuel
    rows to agents at session start.
    """
    captured_sql = {}
    session = MagicMock()

    def _fake_execute(sql, params=None):
        captured_sql["sql"] = str(sql)
        result = MagicMock()
        result.fetchall.return_value = []
        return result

    session.execute.side_effect = _fake_execute

    from app.capture import _recent_memories

    _recent_memories(session, org_id="org-1", project_id=None, limit=5)
    assert "recall_hidden = FALSE" in captured_sql["sql"]


# ---------------------------------------------------------------------------
# /capture/forget
# ---------------------------------------------------------------------------


def test_capture_forget_returns_404_for_unknown_id(client):
    """forget returns 404 when retract_memory reports not_found."""
    import uuid

    db_patch, _ = _patch_db()
    with (
        db_patch,
        patch(
            "app.capture.retract_memory",
            return_value={"forgotten": False, "error": "not_found"},
        ),
    ):
        resp = client.post("/capture/forget", json={"memory_id": str(uuid.uuid4())})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "not_found"


def test_capture_forget_returns_200_on_success(client):
    """forget returns 200 with forgotten=True on a valid retraction."""
    import uuid

    mem_id = str(uuid.uuid4())
    db_patch, _ = _patch_db()
    with (
        db_patch,
        patch(
            "app.capture.retract_memory", return_value={"forgotten": True, "id": mem_id}
        ),
    ):
        resp = client.post("/capture/forget", json={"memory_id": mem_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["forgotten"] is True
    assert body["id"] == mem_id
