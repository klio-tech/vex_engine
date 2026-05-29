"""Tests for the shared session memory module."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared.memory import (
    DEDUP_THRESHOLD,
    DEFAULT_EMBEDDING_MODEL,
    MemoryRecord,
    _row_to_record,
    call_llm,
    embed_text,
    extract_facts,
    find_similar_memory,
    generate_summary,
    retrieve_memories,
    supersede_memory,
    write_memory,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_EMBEDDING = [0.1, 0.2, 0.3, 0.4, 0.5]


def _make_db_row(
    *,
    memory_id=None,
    session_id="sess-1",
    org_id="org-1",
    agent_id="agent-1",
    execution_id="exec-1",
    memory_type="summary",
    content="test content",
    embedding=None,
    confidence=0.95,
    sequence_number=1,
    scope="session",
    status="active",
    superseded_by=None,
    metadata=None,
    created_at=None,
    updated_at=None,
):
    """Build a tuple matching the expected DB row order."""
    return (
        memory_id or uuid.uuid4(),
        session_id,
        org_id,
        agent_id,
        execution_id,
        memory_type,
        content,
        embedding,
        confidence,
        sequence_number,
        scope,
        status,
        superseded_by,
        metadata or {},
        created_at or datetime.now(timezone.utc),
        updated_at or datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_text_calls_litellm():
    """embed_text should call litellm.aembedding and return the embedding list."""
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=SAMPLE_EMBEDDING)]

    with patch(
        "shared.memory.aembedding", new_callable=AsyncMock, return_value=mock_response
    ):
        result = await embed_text("hello world")

    assert result == SAMPLE_EMBEDDING


@pytest.mark.asyncio
async def test_embed_text_returns_none_on_error():
    """embed_text should return None when litellm raises an exception."""
    with patch(
        "shared.memory.aembedding",
        new_callable=AsyncMock,
        side_effect=Exception("API down"),
    ):
        result = await embed_text("hello world")

    assert result is None


# ---------------------------------------------------------------------------
# call_llm
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_llm_returns_parsed_json():
    """call_llm should parse JSON from the LLM response."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"summary": "test"}'))
    ]

    with patch(
        "shared.memory.acompletion", new_callable=AsyncMock, return_value=mock_response
    ):
        result = await call_llm("summarize this")

    assert result == {"summary": "test"}


@pytest.mark.asyncio
async def test_call_llm_returns_none_on_error():
    """call_llm should return None when the LLM call fails."""
    with patch(
        "shared.memory.acompletion",
        new_callable=AsyncMock,
        side_effect=Exception("timeout"),
    ):
        result = await call_llm("summarize this")

    assert result is None


# ---------------------------------------------------------------------------
# generate_summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_summary_returns_string():
    """generate_summary should return the summary string from call_llm."""
    with patch(
        "shared.memory.call_llm",
        new_callable=AsyncMock,
        return_value={"summary": "Agent searched for X and found Y."},
    ):
        result = await generate_summary(output="Found Y", task="search for X")

    assert result == "Agent searched for X and found Y."


@pytest.mark.asyncio
async def test_generate_summary_returns_none_on_timeout():
    """generate_summary should return None when call_llm returns None."""
    with patch("shared.memory.call_llm", new_callable=AsyncMock, return_value=None):
        result = await generate_summary(output="something")

    assert result is None


# ---------------------------------------------------------------------------
# extract_facts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_facts_returns_list():
    """extract_facts should return a list of fact strings."""
    with patch(
        "shared.memory.call_llm",
        new_callable=AsyncMock,
        return_value={"facts": ["User prefers dark mode", "Account is premium"]},
    ):
        result = await extract_facts(
            output="User said they like dark mode and have premium"
        )

    assert result == ["User prefers dark mode", "Account is premium"]


@pytest.mark.asyncio
async def test_extract_facts_returns_empty_on_timeout():
    """extract_facts should return empty list when call_llm returns None."""
    with patch("shared.memory.call_llm", new_callable=AsyncMock, return_value=None):
        result = await extract_facts(output="something")

    assert result == []


# ---------------------------------------------------------------------------
# MemoryRecord
# ---------------------------------------------------------------------------


def test_memory_record_model():
    """MemoryRecord should have correct defaults."""
    record = MemoryRecord(
        id=uuid.uuid4(),
        session_id="sess-1",
        org_id="org-1",
        agent_id="agent-1",
        memory_type="fact",
        content="test",
    )
    assert record.scope == "session"
    assert record.status == "active"
    assert record.metadata == {}
    assert record.embedding is None
    assert record.confidence is None
    assert record.sequence_number is None
    assert record.superseded_by is None
    assert record.execution_id is None
    assert record.created_at is None
    assert record.updated_at is None


# ---------------------------------------------------------------------------
# _row_to_record
# ---------------------------------------------------------------------------


def test_row_to_record():
    """_row_to_record should map tuple fields to MemoryRecord."""
    mid = uuid.uuid4()
    now = datetime.now(timezone.utc)
    row = _make_db_row(memory_id=mid, created_at=now, updated_at=now)
    record = _row_to_record(row)

    assert record.id == mid
    assert record.session_id == "sess-1"
    assert record.org_id == "org-1"
    assert record.agent_id == "agent-1"
    assert record.memory_type == "summary"
    assert record.content == "test content"
    assert record.created_at == now


# ---------------------------------------------------------------------------
# write_memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_memory_inserts_row():
    """write_memory should embed, insert, and commit."""
    db_session = MagicMock()
    db_session.execute = MagicMock()
    db_session.commit = MagicMock()

    with patch(
        "shared.memory.embed_text",
        new_callable=AsyncMock,
        return_value=SAMPLE_EMBEDDING,
    ):
        result = await write_memory(
            db_session=db_session,
            session_id="sess-1",
            org_id="org-1",
            agent_id="agent-1",
            execution_id="exec-1",
            memory_type="summary",
            content="Agent completed task",
            confidence=0.9,
            sequence_number=1,
        )

    assert result is not None
    assert isinstance(result, MemoryRecord)
    assert result.content == "Agent completed task"
    assert result.embedding == SAMPLE_EMBEDDING
    db_session.execute.assert_called_once()
    db_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_write_memory_returns_none_when_embedding_fails():
    """write_memory should return None when embed_text returns None."""
    db_session = MagicMock()

    with patch("shared.memory.embed_text", new_callable=AsyncMock, return_value=None):
        result = await write_memory(
            db_session=db_session,
            session_id="sess-1",
            org_id="org-1",
            agent_id="agent-1",
            execution_id="exec-1",
            memory_type="fact",
            content="some fact",
        )

    assert result is None
    db_session.execute.assert_not_called()


# ---------------------------------------------------------------------------
# retrieve_memories
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_memories_semantic_search():
    """retrieve_memories should return combined semantic + summary results."""
    row1 = _make_db_row(
        memory_id=uuid.uuid4(), content="semantic match", memory_type="fact"
    )
    row2 = _make_db_row(
        memory_id=uuid.uuid4(), content="recent summary", memory_type="summary"
    )

    db_session = MagicMock()
    # First execute returns semantic results, second returns summaries
    db_session.execute = MagicMock(
        side_effect=[
            MagicMock(fetchall=MagicMock(return_value=[row1])),
            MagicMock(fetchall=MagicMock(return_value=[row2])),
        ]
    )

    with patch(
        "shared.memory.embed_text",
        new_callable=AsyncMock,
        return_value=SAMPLE_EMBEDDING,
    ):
        results = await retrieve_memories(
            db_session=db_session,
            session_id="sess-1",
            org_id="org-1",
            query_text="what happened?",
            limit=10,
        )

    assert len(results) == 2
    contents = {r.content for r in results}
    assert "semantic match" in contents
    assert "recent summary" in contents


@pytest.mark.asyncio
async def test_retrieve_memories_returns_empty_when_embedding_fails():
    """retrieve_memories should return empty list when embedding fails."""
    db_session = MagicMock()

    with patch("shared.memory.embed_text", new_callable=AsyncMock, return_value=None):
        results = await retrieve_memories(
            db_session=db_session,
            session_id="sess-1",
            org_id="org-1",
            query_text="what happened?",
        )

    assert results == []


# ---------------------------------------------------------------------------
# find_similar_memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_similar_memory_above_threshold():
    """find_similar_memory should return record when similarity >= threshold."""
    row = _make_db_row(content="existing fact")
    db_session = MagicMock()
    # fetchone returns (row..., similarity_score)
    mock_result = MagicMock()
    mock_result.fetchone = MagicMock(return_value=(*row, 0.95))
    db_session.execute = MagicMock(return_value=mock_result)

    with patch(
        "shared.memory.embed_text",
        new_callable=AsyncMock,
        return_value=SAMPLE_EMBEDDING,
    ):
        result = await find_similar_memory(
            db_session=db_session,
            session_id="sess-1",
            org_id="org-1",
            content="existing fact rephrased",
        )

    assert result is not None
    assert result.content == "existing fact"


@pytest.mark.asyncio
async def test_find_similar_memory_below_threshold():
    """find_similar_memory should return None when similarity < threshold."""
    row = _make_db_row(content="unrelated")
    db_session = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchone = MagicMock(return_value=(*row, 0.5))
    db_session.execute = MagicMock(return_value=mock_result)

    with patch(
        "shared.memory.embed_text",
        new_callable=AsyncMock,
        return_value=SAMPLE_EMBEDDING,
    ):
        result = await find_similar_memory(
            db_session=db_session,
            session_id="sess-1",
            org_id="org-1",
            content="something different",
        )

    assert result is None


# ---------------------------------------------------------------------------
# supersede_memory
# ---------------------------------------------------------------------------


def test_supersede_memory():
    """supersede_memory should update the old memory's status."""
    db_session = MagicMock()
    old_id = uuid.uuid4()
    new_id = uuid.uuid4()

    supersede_memory(db_session, old_id, new_id)

    db_session.execute.assert_called_once()
    db_session.commit.assert_called_once()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_constants():
    """Verify module constants are set correctly."""
    assert DEDUP_THRESHOLD == 0.92
    assert DEFAULT_EMBEDDING_MODEL == "intfloat/multilingual-e5-large-instruct"


# ---------------------------------------------------------------------------
# find_similar_memory: dedup window (default session-scoped vs agent-scoped)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_similar_memory_default_is_session_scoped():
    """Default dedup query filters by session_id (historical behavior)."""
    db_session = MagicMock()
    db_session.execute.return_value.fetchone.return_value = None

    with patch(
        "shared.memory.embed_text",
        new_callable=AsyncMock,
        return_value=SAMPLE_EMBEDDING,
    ):
        await find_similar_memory(db_session, "sess-1", "org-1", "some content")

    sql = str(db_session.execute.call_args.args[0])
    params = db_session.execute.call_args.args[1]
    assert "session_id = :session_id" in sql
    assert "scope = 'agent'" not in sql
    assert params["session_id"] == "sess-1"
    assert "agent_id" not in params


@pytest.mark.asyncio
async def test_find_similar_memory_agent_scope_filters_agent():
    """scope='agent' dedup query filters scope='agent' AND agent_id, and drops
    the session_id filter (dedups against the same agent's private rows)."""
    db_session = MagicMock()
    db_session.execute.return_value.fetchone.return_value = None

    with patch(
        "shared.memory.embed_text",
        new_callable=AsyncMock,
        return_value=SAMPLE_EMBEDDING,
    ):
        await find_similar_memory(
            db_session,
            "sess-1",
            "org-1",
            "some content",
            scope="agent",
            agent_id="mf-userA-game1",
        )

    sql = str(db_session.execute.call_args.args[0])
    params = db_session.execute.call_args.args[1]
    assert "scope = 'agent'" in sql
    assert "agent_id = :agent_id" in sql
    assert "session_id = :session_id" not in sql
    assert params["agent_id"] == "mf-userA-game1"
    assert "session_id" not in params
