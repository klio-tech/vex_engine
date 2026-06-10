"""Unit tests for :func:`shared.brain.write_scoped_memory`.

These tests follow the mock pattern in ``tests/test_memory.py``: the
embedding/DB-backed primitives (``write_memory`` / ``find_similar_memory``)
are patched at the ``shared.brain`` module boundary so no real embeddings or
Postgres are touched. They assert the scope-promotion behaviour
(``write_memory`` hardcodes ``scope='session'``; ``write_scoped_memory`` then
issues an UPDATE that promotes the row to the brain ``scope='org'`` and binds
``space_id`` / ``project_id``).
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared.brain import write_scoped_memory
from shared.memory import MemoryRecord


def _make_record(
    *, memory_id=None, memory_type="memory", content="a fact"
) -> MemoryRecord:
    """Build a minimal MemoryRecord as write_memory/find_similar_memory return."""
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=memory_id or uuid.uuid4(),
        session_id="klio-brain:org-1:agent-1",
        org_id="org-1",
        agent_id="agent-1",
        memory_type=memory_type,
        content=content,
        execution_id=None,
        scope="session",
        status="active",
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# Dedup hit: an existing similar memory suppresses the write entirely.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_scoped_memory_dedup_hit_suppresses_write():
    """When find_similar_memory returns a record, write_memory is NOT called
    and the result reports deduplication with the existing id."""
    existing = _make_record(memory_type="note", content="already stored")
    db_session = MagicMock()

    with (
        patch(
            "shared.brain.find_similar_memory",
            new_callable=AsyncMock,
            return_value=existing,
        ) as mock_find,
        patch("shared.brain.write_memory", new_callable=AsyncMock) as mock_write,
    ):
        result = await write_scoped_memory(
            db_session,
            org_id="org-1",
            agent_id="agent-1",
            content="already stored (rephrased)",
            memory_type="note",
            session_id="klio-brain:org-1:agent-1",
        )

    mock_find.assert_awaited_once()
    mock_write.assert_not_called()
    db_session.execute.assert_not_called()
    assert result == {
        "stored": False,
        "deduplicated": True,
        "id": str(existing.id),
        "memory_type": "note",
        "scope": "org",
        "redacted": False,
        "redaction_kinds": [],
    }


@pytest.mark.asyncio
async def test_write_scoped_memory_dedup_disabled_skips_lookup():
    """With dedup=False, find_similar_memory is never called."""
    record = _make_record()
    db_session = MagicMock()

    with (
        patch("shared.brain.find_similar_memory", new_callable=AsyncMock) as mock_find,
        patch(
            "shared.brain.write_memory",
            new_callable=AsyncMock,
            return_value=record,
        ),
    ):
        result = await write_scoped_memory(
            db_session,
            org_id="org-1",
            agent_id="agent-1",
            content="a fact",
            memory_type="memory",
            session_id="sess-1",
            dedup=False,
        )

    mock_find.assert_not_called()
    assert result["stored"] is True


# ---------------------------------------------------------------------------
# Happy path: write then promote to org scope.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_scoped_memory_happy_path_promotes_to_org():
    """On a fresh write the row is created via write_memory(execution_id=None)
    then an UPDATE promotes scope='org' and binds space_id/project_id/id."""
    record = _make_record(memory_type="decision", content="ship on friday")
    db_session = MagicMock()

    with (
        patch(
            "shared.brain.find_similar_memory",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "shared.brain.write_memory",
            new_callable=AsyncMock,
            return_value=record,
        ) as mock_write,
    ):
        result = await write_scoped_memory(
            db_session,
            org_id="org-1",
            agent_id="agent-1",
            content="ship on friday",
            memory_type="decision",
            session_id="klio-brain:org-1:agent-1",
            space_id="space-9",
            project_id="proj-7",
            metadata={"rationale": "team aligned"},
        )

    # (a) write_memory called with positional principal args + execution_id=None.
    mock_write.assert_awaited_once()
    call = mock_write.await_args
    assert call.args[0] is db_session
    assert call.args[1] == "klio-brain:org-1:agent-1"  # session_id
    assert call.args[2] == "org-1"  # org_id
    assert call.args[3] == "agent-1"  # agent_id
    assert call.args[4] is None  # execution_id
    assert call.args[5] == "decision"  # memory_type
    assert call.args[6] == "ship on friday"  # content
    assert call.kwargs.get("metadata") == {"rationale": "team aligned"}

    # (b) An UPDATE was issued that promotes scope='org' and binds the scope keys.
    assert db_session.execute.call_count == 1
    update_call = db_session.execute.call_args_list[0]
    sql = str(update_call.args[0])
    assert "UPDATE session_memories" in sql
    assert "scope" in sql
    params = update_call.args[1]
    assert params["space_id"] == "space-9"
    assert params["project_id"] == "proj-7"
    assert params["id"] == str(record.id)
    db_session.commit.assert_called_once()

    # (c) Result reports the promoted row.
    assert result == {
        "stored": True,
        "deduplicated": False,
        "id": str(record.id),
        "memory_type": "decision",
        "scope": "org",
        "space_id": "space-9",
        "project_id": "proj-7",
        "redacted": False,
        "redaction_kinds": [],
    }


@pytest.mark.asyncio
async def test_write_scoped_memory_promotes_with_null_scope_keys():
    """When space_id/project_id are omitted the UPDATE binds them as None."""
    record = _make_record()
    db_session = MagicMock()

    with (
        patch(
            "shared.brain.find_similar_memory",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "shared.brain.write_memory",
            new_callable=AsyncMock,
            return_value=record,
        ),
    ):
        result = await write_scoped_memory(
            db_session,
            org_id="org-1",
            agent_id="agent-1",
            content="a fact",
            memory_type="memory",
            session_id="sess-1",
        )

    params = db_session.execute.call_args_list[0].args[1]
    assert params["space_id"] is None
    assert params["project_id"] is None
    assert result["space_id"] is None
    assert result["project_id"] is None
    assert result["scope"] == "org"


# ---------------------------------------------------------------------------
# Embedding failure: write_memory returns None -> no UPDATE, clean error.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_scoped_memory_embedding_failure():
    """When write_memory returns None (embedding failed), report the error
    and issue NO UPDATE."""
    db_session = MagicMock()

    with (
        patch(
            "shared.brain.find_similar_memory",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "shared.brain.write_memory",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        result = await write_scoped_memory(
            db_session,
            org_id="org-1",
            agent_id="agent-1",
            content="a fact",
            memory_type="memory",
            session_id="sess-1",
        )

    assert result == {"stored": False, "error": "embedding_failed"}
    db_session.execute.assert_not_called()
    db_session.commit.assert_not_called()


# ---------------------------------------------------------------------------
# Agent scope: write_scoped_memory(scope='agent') keeps the row private.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_scoped_memory_agent_scope_promotes_to_agent():
    """scope='agent' promotes the row to scope='agent' (not 'org'), dedups
    against the same agent's private rows, and echoes scope='agent'."""
    record = _make_record(memory_type="observation", content="userA D7 retention")
    db_session = MagicMock()

    with (
        patch(
            "shared.brain.find_similar_memory",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_find,
        patch(
            "shared.brain.write_memory",
            new_callable=AsyncMock,
            return_value=record,
        ),
    ):
        result = await write_scoped_memory(
            db_session,
            org_id="org-1",
            agent_id="mf-userA-game1",
            content="userA D7 retention",
            memory_type="observation",
            session_id="klio-hook:s1",
            scope="agent",
        )

    # Dedup window scoped to THIS agent's private rows.
    assert mock_find.await_args.kwargs["scope"] == "agent"
    assert mock_find.await_args.kwargs["agent_id"] == "mf-userA-game1"

    # The promote UPDATE binds scope='agent'.
    update_params = db_session.execute.call_args_list[0].args[1]
    assert update_params["scope"] == "agent"
    assert update_params["id"] == str(record.id)
    db_session.commit.assert_called_once()

    assert result["stored"] is True
    assert result["scope"] == "agent"


@pytest.mark.asyncio
async def test_write_scoped_memory_default_dedup_is_session_scoped():
    """Default (org) write keeps the historical session-scoped dedup: it does
    NOT pass scope/agent_id to find_similar_memory (regression guard)."""
    record = _make_record()
    db_session = MagicMock()

    with (
        patch(
            "shared.brain.find_similar_memory",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_find,
        patch(
            "shared.brain.write_memory",
            new_callable=AsyncMock,
            return_value=record,
        ),
    ):
        result = await write_scoped_memory(
            db_session,
            org_id="org-1",
            agent_id="agent-1",
            content="a fact",
            memory_type="memory",
            session_id="sess-1",
        )

    # Org write → dedup not scoped to an agent.
    assert mock_find.await_args.kwargs.get("scope") is None
    assert mock_find.await_args.kwargs.get("agent_id") is None
    # Promote still lands at org (backward compatible).
    assert db_session.execute.call_args_list[0].args[1]["scope"] == "org"
    assert result["scope"] == "org"


@pytest.mark.asyncio
async def test_write_scoped_memory_agent_dedup_hit_returns_agent_scope():
    """An agent-scoped dedup hit reports scope='agent' (not 'org')."""
    existing = _make_record(memory_type="observation", content="already private")
    db_session = MagicMock()

    with (
        patch(
            "shared.brain.find_similar_memory",
            new_callable=AsyncMock,
            return_value=existing,
        ) as mock_find,
        patch("shared.brain.write_memory", new_callable=AsyncMock) as mock_write,
    ):
        result = await write_scoped_memory(
            db_session,
            org_id="org-1",
            agent_id="mf-userA-game1",
            content="already private (rephrased)",
            memory_type="observation",
            session_id="klio-hook:s1",
            scope="agent",
        )

    assert mock_find.await_args.kwargs["scope"] == "agent"
    mock_write.assert_not_called()
    assert result == {
        "stored": False,
        "deduplicated": True,
        "id": str(existing.id),
        "memory_type": "observation",
        "scope": "agent",
        "redacted": False,
        "redaction_kinds": [],
    }


@pytest.mark.asyncio
async def test_write_scoped_memory_rejects_unknown_scope():
    """A scope other than 'org'/'agent' fails fast before any DB work."""
    db_session = MagicMock()
    with pytest.raises(ValueError, match="unsupported scope"):
        await write_scoped_memory(
            db_session,
            org_id="org-1",
            agent_id="agent-1",
            content="x",
            memory_type="memory",
            session_id="sess-1",
            scope="session",
        )
    db_session.execute.assert_not_called()
