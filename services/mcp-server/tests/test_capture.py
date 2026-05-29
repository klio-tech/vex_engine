"""Tests for the REST capture layer (``app.capture``).

Mock-based, mirroring ``tests/test_write_tools.py`` and ``tests/test_curator.py``:
no live Postgres, no real embeddings/LLM.  The brain/memory seams are patched at
the ``app.capture`` module boundary:

* ``app.capture.SessionLocal`` → a ``MagicMock`` so ``SessionLocal()`` yields a
  fake session that is ``.close()``-d in the ``finally``.
* ``app.capture.brain.write_scoped_memory`` → an ``AsyncMock`` capturing kwargs.
* ``app.capture.extract_facts`` / ``generate_summary`` /
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


def test_transcript_extracts_facts_and_summary(client):
    """Transcript distils to one summary + N facts, each stored; counts returned."""
    write = AsyncMock(return_value={"stored": True})
    db_patch, session = _patch_db()
    with (
        db_patch,
        patch.object(capture.brain, "write_scoped_memory", write),
        patch.object(capture.brain, "ensure_project", MagicMock(return_value=None)),
        patch("app.capture.generate_summary", AsyncMock(return_value="a summary")),
        patch("app.capture.extract_facts", AsyncMock(return_value=["f1", "f2"])),
    ):
        resp = client.post(
            "/capture/transcript",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    body = resp.json()
    assert body == {"stored": 3, "facts": 2, "summary": True}
    types = [c.kwargs["memory_type"] for c in write.await_args_list]
    assert types == ["summary", "fact", "fact"]
    session.close.assert_called_once()


def test_transcript_empty_messages_short_circuits(client):
    """No usable message text → nothing stored, no LLM call, no DB session."""
    write = AsyncMock()
    sl = MagicMock()
    with (
        patch("app.capture.SessionLocal", sl),
        patch.object(capture.brain, "write_scoped_memory", write),
        patch("app.capture.generate_summary", AsyncMock()) as gs,
        patch("app.capture.extract_facts", AsyncMock()) as ef,
    ):
        resp = client.post(
            "/capture/transcript", json={"messages": [{"content": "  "}]}
        )

    assert resp.json()["reason"] == "empty_transcript"
    write.assert_not_called()
    gs.assert_not_called()
    ef.assert_not_called()
    sl.assert_not_called()


def test_transcript_summary_failure_still_stores_facts(client):
    """A summary-extraction failure does not block fact storage (independent)."""
    write = AsyncMock(return_value={"stored": True})
    db_patch, _ = _patch_db()
    with (
        db_patch,
        patch.object(capture.brain, "write_scoped_memory", write),
        patch.object(capture.brain, "ensure_project", MagicMock(return_value=None)),
        patch("app.capture.generate_summary", AsyncMock(side_effect=RuntimeError())),
        patch("app.capture.extract_facts", AsyncMock(return_value=["only-fact"])),
    ):
        resp = client.post(
            "/capture/transcript",
            json={"messages": [{"content": "hi"}]},
        )

    body = resp.json()
    assert body == {"stored": 1, "facts": 1, "summary": False}
    assert write.await_args.kwargs["memory_type"] == "fact"


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
