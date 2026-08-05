"""Tests for the shared session memory module."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
import shared.memory as memory_module
from shared.memory import (
    DEDUP_THRESHOLD,
    DEFAULT_EMBEDDING_MODEL,
    EMBED_MIN_CHARS,
    MAX_EMBED_CHARS,
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
    """embed_text should return None when the backend raises a non-over-length
    exception (e.g. the service is down): such errors are not fixed by shrinking,
    so it returns None immediately without retrying.
    """
    with patch(
        "shared.memory.aembedding",
        new_callable=AsyncMock,
        side_effect=Exception("API down"),
    ):
        result = await embed_text("hello world")

    assert result is None


def _overlength_error() -> litellm.ContextWindowExceededError:
    """Build the litellm over-length error the embedding backend raises when the
    input exceeds the model's token cap.

    This is the ONLY error shape ``embed_text`` should shrink-and-retry on;
    every other exception must return ``None`` on the first attempt.
    """
    return litellm.ContextWindowExceededError(
        message="input exceeds model token limit",
        model="test-embed-model",
        llm_provider="openai",
    )


def _embedding_backend(success_max: int, call_sizes: list[int]):
    """Build an aembedding side_effect that records input sizes and raises an
    over-length error for any input longer than ``success_max`` (simulating the
    512-token cap that embed_text is expected to shrink under)."""

    def _side_effect(*args, **kwargs):
        text_in = kwargs["input"][0]
        size = len(text_in) if text_in else 0
        call_sizes.append(size)
        if size > success_max:
            raise _overlength_error()
        resp = MagicMock()
        resp.data = [MagicMock(embedding=SAMPLE_EMBEDDING)]
        return resp

    return _side_effect


@pytest.mark.asyncio
async def test_embed_text_progressively_truncates_long_input():
    """A very long/dense input still embeds: embed_text shrinks and retries
    until the backend accepts the smaller input (the write is never lost)."""
    success_max = 400
    call_sizes: list[int] = []

    with patch(
        "shared.memory.aembedding",
        new_callable=AsyncMock,
        side_effect=_embedding_backend(success_max, call_sizes),
    ):
        result = await embed_text("x" * 10_000)

    assert result == SAMPLE_EMBEDDING
    # More than one attempt: the initial head-truncation still exceeded the cap.
    assert len(call_sizes) > 1
    # First attempt is the MAX_EMBED_CHARS head-truncation.
    assert call_sizes[0] == MAX_EMBED_CHARS
    # Sizes shrink monotonically until one fits under the cap.
    assert call_sizes[-1] <= success_max
    assert all(later <= earlier for earlier, later in zip(call_sizes, call_sizes[1:]))


@pytest.mark.asyncio
async def test_embed_text_short_input_single_call():
    """A normal short input embeds in a single backend call — no needless
    shrinking when the input already fits."""
    call_sizes: list[int] = []

    with patch(
        "shared.memory.aembedding",
        new_callable=AsyncMock,
        side_effect=_embedding_backend(success_max=10_000, call_sizes=call_sizes),
    ):
        result = await embed_text("a short note")

    assert result == SAMPLE_EMBEDDING
    assert len(call_sizes) == 1


@pytest.mark.asyncio
async def test_embed_text_returns_none_when_overlength_at_every_size():
    """embed_text returns None when the input is rejected as over-length even at
    the floor, and it stops shrinking at EMBED_MIN_CHARS (floor logic)."""
    call_sizes: list[int] = []

    def _always_overlength(*args, **kwargs):
        text_in = kwargs["input"][0]
        call_sizes.append(len(text_in) if text_in else 0)
        raise _overlength_error()

    with patch(
        "shared.memory.aembedding",
        new_callable=AsyncMock,
        side_effect=_always_overlength,
    ):
        result = await embed_text("y" * 10_000)

    assert result is None
    # Retried down to the floor and no further.
    assert call_sizes[-1] == EMBED_MIN_CHARS
    assert all(size >= EMBED_MIN_CHARS for size in call_sizes)


@pytest.mark.asyncio
async def test_embed_text_non_overlength_error_returns_none_single_call():
    """A non-over-length backend error (service down, timeout, transient gateway
    error) is NOT retried with truncation: embed_text returns None after exactly
    ONE call, so it never fans out against a failing backend nor silently embeds
    truncated content that a plain retry would have accepted at full size."""
    call_sizes: list[int] = []

    def _transient_failure(*args, **kwargs):
        text_in = kwargs["input"][0]
        call_sizes.append(len(text_in) if text_in else 0)
        raise Exception("embedding service unavailable")

    with patch(
        "shared.memory.aembedding",
        new_callable=AsyncMock,
        side_effect=_transient_failure,
    ):
        # A long input would be shrunk IF the error were over-length shaped; a
        # generic error must still short-circuit after a single attempt.
        result = await embed_text("y" * 10_000)

    assert result is None
    assert len(call_sizes) == 1
    # No shrinking happened: the single attempt was the head-truncated size.
    assert call_sizes == [MAX_EMBED_CHARS]


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
    """write_memory should release, embed, insert, and commit.

    Two commits, not one. The first is :func:`shared.memory.release_connection`
    returning the caller's pooled connection before the embed is awaited — the
    caller reaches here with a transaction already open from its quota check,
    and holding it across a network call to the embedding backend is what
    exhausted the pool at 16 concurrent writes. The second is the insert's own.
    """
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
    assert db_session.commit.call_count == 2


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

    supersede_memory(db_session, old_id, new_id, org_id="org-1")

    db_session.execute.assert_called_once()
    # The tenancy predicate and its bound value, together: a WHERE naming
    # :org_id with nothing bound to it would silently match every tenant.
    sql, params = db_session.execute.call_args.args
    assert "org_id = :org_id" in str(sql)
    assert params["org_id"] == "org-1"
    db_session.commit.assert_called_once()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_constants():
    """Verify module constants are set correctly."""
    assert DEDUP_THRESHOLD == 0.95
    assert DEFAULT_EMBEDDING_MODEL == "intfloat/multilingual-e5-large-instruct"


def test_dedup_threshold_env_override(monkeypatch):
    """KLIO_DEDUP_THRESHOLD overrides the default when the module is reloaded.

    The threshold is read from the environment at import time, so an operator
    override only takes effect on a fresh import/reload — which is exactly how
    it is picked up at process start in production.
    """
    import importlib

    import shared.memory as memory_mod

    monkeypatch.setenv("KLIO_DEDUP_THRESHOLD", "0.97")
    try:
        reloaded = importlib.reload(memory_mod)
        assert reloaded.DEDUP_THRESHOLD == 0.97
        # find_similar_memory's default binding tracks the (reloaded) constant.
        import inspect

        sig = inspect.signature(reloaded.find_similar_memory)
        assert sig.parameters["threshold"].default == 0.97
    finally:
        # Restore module to its default state so later tests see 0.95.
        monkeypatch.delenv("KLIO_DEDUP_THRESHOLD", raising=False)
        importlib.reload(memory_mod)


def test_dedup_threshold_malformed_falls_back_to_default(monkeypatch):
    """A non-numeric KLIO_DEDUP_THRESHOLD must NOT crash module import: it falls
    back to the 0.95 default (with a logged warning)."""
    import importlib

    import shared.memory as memory_mod

    monkeypatch.setenv("KLIO_DEDUP_THRESHOLD", "not-a-number")
    try:
        reloaded = importlib.reload(memory_mod)
        assert reloaded.DEDUP_THRESHOLD == 0.95
    finally:
        monkeypatch.delenv("KLIO_DEDUP_THRESHOLD", raising=False)
        importlib.reload(memory_mod)


@pytest.mark.parametrize("bad_value", ["1.5", "-0.1", "2", "-3.0"])
def test_dedup_threshold_out_of_range_falls_back_to_default(monkeypatch, bad_value):
    """A KLIO_DEDUP_THRESHOLD outside [0.0, 1.0] is clamped back to the 0.95
    default (a similarity threshold outside that range is meaningless)."""
    import importlib

    import shared.memory as memory_mod

    monkeypatch.setenv("KLIO_DEDUP_THRESHOLD", bad_value)
    try:
        reloaded = importlib.reload(memory_mod)
        assert reloaded.DEDUP_THRESHOLD == 0.95
    finally:
        monkeypatch.delenv("KLIO_DEDUP_THRESHOLD", raising=False)
        importlib.reload(memory_mod)


def test_dedup_threshold_boundary_values_accepted(monkeypatch):
    """The inclusive boundaries 0.0 and 1.0 are valid and preserved (not clamped
    to the default), so the [0,1] guard does not reject legitimate extremes."""
    import importlib

    import shared.memory as memory_mod

    for boundary in ("0.0", "1.0"):
        monkeypatch.setenv("KLIO_DEDUP_THRESHOLD", boundary)
        try:
            reloaded = importlib.reload(memory_mod)
            assert reloaded.DEDUP_THRESHOLD == float(boundary)
        finally:
            monkeypatch.delenv("KLIO_DEDUP_THRESHOLD", raising=False)
            importlib.reload(memory_mod)


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


# ---------------------------------------------------------------------------
# Prompt hardening: credential reproduction guard
# ---------------------------------------------------------------------------


async def test_extract_facts_prompt_forbids_credentials(monkeypatch):
    """extract_facts prompt must instruct the LLM never to reproduce credentials."""
    import shared.memory as memory_mod

    seen: dict = {}

    async def _fake_call_llm(prompt, system=None):
        seen["prompt"] = prompt
        return {"facts": []}

    monkeypatch.setattr(memory_mod, "call_llm", _fake_call_llm)
    await memory_mod.extract_facts("some output")
    assert "Never reproduce credentials" in seen["prompt"]


async def test_generate_summary_prompt_forbids_credentials(monkeypatch):
    """generate_summary prompt must instruct the LLM never to reproduce credentials."""
    import shared.memory as memory_mod

    seen: dict = {}

    async def _fake_call_llm(prompt, system=None):
        seen["prompt"] = prompt
        return {"summary": "ok"}

    monkeypatch.setattr(memory_mod, "call_llm", _fake_call_llm)
    await memory_mod.generate_summary("some output")
    assert "Never reproduce credentials" in seen["prompt"]


# ---------------------------------------------------------------------------
# Prompt input bounds
# ---------------------------------------------------------------------------
#
# ``generate_summary`` and ``extract_facts`` interpolate caller-supplied
# ``output`` / ``task`` / ``input_data`` straight into a prompt. WP-B bounded
# them at the two call sites it owned (sync-gateway, async-worker); every other
# caller — the capture transcript path, and anything added later — is still
# unbounded. An agent turn that returns a 40MB blob otherwise becomes a 40MB
# request to the vendor: a cost and latency fault, and a timeout that looks
# like an LLM outage. The bound belongs in the function so it holds for all
# callers, not only the careful ones.


@pytest.mark.asyncio
async def test_generate_summary_bounds_every_interpolated_field():
    from shared.memory import MAX_PROMPT_INPUT_CHARS

    captured = {}

    async def _fake_llm(prompt, system=None):
        captured["prompt"] = prompt
        return {"summary": "ok"}

    huge = "x" * (MAX_PROMPT_INPUT_CHARS * 3)
    with patch("shared.memory.call_llm", new=_fake_llm):
        await memory_module.generate_summary(huge, task=huge, input_data=huge)

    # Three bounded fields plus a fixed instruction preamble.
    assert len(captured["prompt"]) < 3 * MAX_PROMPT_INPUT_CHARS + 2_000, (
        f"prompt was {len(captured['prompt'])} chars"
    )
    assert "[truncated]" in captured["prompt"]


@pytest.mark.asyncio
async def test_extract_facts_bounds_every_interpolated_field():
    from shared.memory import MAX_PROMPT_INPUT_CHARS

    captured = {}

    async def _fake_llm(prompt, system=None):
        captured["prompt"] = prompt
        return {"facts": []}

    huge = "y" * (MAX_PROMPT_INPUT_CHARS * 3)
    with patch("shared.memory.call_llm", new=_fake_llm):
        await memory_module.extract_facts(huge, task=huge, input_data=huge)

    assert len(captured["prompt"]) < 3 * MAX_PROMPT_INPUT_CHARS + 2_000, (
        f"prompt was {len(captured['prompt'])} chars"
    )
    assert "[truncated]" in captured["prompt"]


@pytest.mark.asyncio
async def test_ordinary_input_is_passed_through_untouched():
    """Positive control: the bound must not edit normal turns."""
    captured = {}

    async def _fake_llm(prompt, system=None):
        captured["prompt"] = prompt
        return {"summary": "ok"}

    with patch("shared.memory.call_llm", new=_fake_llm):
        await memory_module.generate_summary(
            "wrote the migration", task="ship phase 2", input_data={"n": 1}
        )

    assert "wrote the migration" in captured["prompt"]
    assert "ship phase 2" in captured["prompt"]
    assert "[truncated]" not in captured["prompt"]
