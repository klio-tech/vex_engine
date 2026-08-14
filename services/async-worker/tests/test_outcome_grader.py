from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared.memory import MemoryRecord

from app import outcome_grader as og
from engine.memory_consistency import _format_memories


def _mem_row(mem_id, content, memory_type="fact", sequence_number=1):
    """A row tuple matching shared.memory._row_to_record's expected column
    order: id, session_id, org_id, agent_id, execution_id, memory_type,
    content, embedding, confidence, sequence_number, scope, status,
    superseded_by, metadata, created_at, updated_at.
    """
    return (
        mem_id, "s", "o", "a", None, memory_type, content, None, None,
        sequence_number, "session", "active", None, {}, None, None,
    )


def _db(recalls, memory_rows, session_outcome_returns=None):
    """``session_outcome_returns`` lets a test control what the
    session_outcomes INSERT ... RETURNING id call yields, in order, across
    repeated calls to grade_stored_session against the same mock — used to
    simulate a second (duplicate) grade hitting the UNIQUE constraint and
    getting nothing back from ON CONFLICT DO NOTHING.
    """
    db = MagicMock()
    returns_iter = iter(session_outcome_returns or [])

    def execute(stmt, params=None):
        m = MagicMock()
        s = str(stmt)
        if "FROM brain_recall_events" in s:
            m.mappings.return_value.all.return_value = recalls
        elif "FROM session_memories" in s:
            m.fetchall.return_value = memory_rows
        elif "INSERT INTO session_outcomes" in s and session_outcome_returns is not None:
            returned_id = next(returns_iter, None)
            m.first.return_value = (returned_id,) if returned_id else None
        return m
    db.execute.side_effect = execute
    return db


TRACE = {"messages": [{"role": "user", "content": "q"},
                      {"role": "assistant", "content": "final answer about postgres"}],
         "tool_calls": [], "row": {}}


@pytest.mark.anyio
async def test_one_session_verdict_and_per_memory_attribution():
    recalls = [{"id": "re1", "memory_ids": ["m1", "m2"], "session_id": "s"}]
    memory_rows = [_mem_row("m1", "postgres decision"),
                   _mem_row("m2", "unrelated fact")]
    verdict = MagicMock(confidence=0.9, action="pass")
    checks = {"memory_consistency": MagicMock(score=0.8)}
    verdict.checks = checks
    with patch.object(og, "fetch_trace", return_value=TRACE), \
         patch.object(og, "verify", new=AsyncMock(return_value=verdict)) as v, \
         patch.object(og, "embed_text", new=AsyncMock(side_effect=[
             [1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])):  # output, m1, m2
        result = await og.grade_stored_session(
            _db(recalls, memory_rows), MagicMock(), org_id="o", session_id="s")
    # verify called ONCE, with the served memories
    assert v.await_count == 1
    assert v.await_args.kwargs["session_memories"] is not None
    assert result["attributions"] == 2


@pytest.mark.anyio
async def test_session_verdict_written_once_not_per_recall():
    recalls = [{"id": f"re{i}", "memory_ids": ["m1"], "session_id": "s"}
               for i in range(8)]
    memory_rows = [_mem_row("m1", "x")]
    verdict = MagicMock(confidence=0.5, action="flag", checks={})
    db = _db(recalls, memory_rows)
    with patch.object(og, "fetch_trace", return_value=TRACE), \
         patch.object(og, "verify", new=AsyncMock(return_value=verdict)), \
         patch.object(og, "embed_text", new=AsyncMock(return_value=[1.0, 0.0])):
        await og.grade_stored_session(db, MagicMock(), org_id="o", session_id="s")
    session_inserts = [c for c in db.execute.call_args_list
                       if "INSERT INTO session_outcomes" in str(c.args[0])]
    assert len(session_inserts) == 1  # eight recalls, ONE session verdict


@pytest.mark.anyio
async def test_missing_trace_writes_unknown_session_outcome():
    recalls = [{"id": "re1", "memory_ids": ["m1"], "session_id": "s"}]
    db = _db(recalls, [_mem_row("m1", "x")])
    with patch.object(og, "fetch_trace", return_value=None):
        result = await og.grade_stored_session(db, MagicMock(), org_id="o",
                                               session_id="s")
    inserts = [c for c in db.execute.call_args_list
               if "INSERT INTO session_outcomes" in str(c.args[0])]
    assert inserts and inserts[0].args[1]["verdict"] == "unknown"


@pytest.mark.anyio
async def test_verify_receives_memory_record_instances():
    """CRITICAL regression guard: verify() must receive real MemoryRecord
    objects (attribute access: .content, .memory_type, .sequence_number),
    not bare {id, content} dicts — engine.memory_consistency._format_memories
    does attribute access and raises AttributeError on a dict, which the
    surrounding except Exception silently swallows, degrading every grade
    to verdict='unknown'.
    """
    recalls = [{"id": "re1", "memory_ids": ["m1"], "session_id": "s"}]
    memory_rows = [_mem_row("m1", "postgres decision", memory_type="decision",
                             sequence_number=3)]
    verdict = MagicMock(confidence=0.9, action="pass", checks={})
    with patch.object(og, "fetch_trace", return_value=TRACE), \
         patch.object(og, "verify", new=AsyncMock(return_value=verdict)) as v, \
         patch.object(og, "embed_text", new=AsyncMock(return_value=[1.0, 0.0])):
        await og.grade_stored_session(
            _db(recalls, memory_rows), MagicMock(), org_id="o", session_id="s")
    served = v.await_args.kwargs["session_memories"]
    assert len(served) == 1
    assert all(isinstance(m, MemoryRecord) for m in served)
    assert served[0].content == "postgres decision"
    assert served[0].memory_type == "decision"
    assert served[0].sequence_number == 3


def test_format_memories_accepts_real_records_no_attribute_error():
    """Calls the REAL engine.memory_consistency._format_memories (not
    mocked) with MemoryRecord objects built the same way
    grade_stored_session builds them, proving the shape is actually
    compatible rather than merely type-correct in isolation.
    """
    records = [
        og._row_to_record(_mem_row("m1", "postgres decision", "decision", 3)),
        og._row_to_record(_mem_row("m2", "unrelated fact", "fact", None)),
    ]
    formatted = _format_memories(records)
    assert "[decision] (turn 3): postgres decision" in formatted
    assert "[fact] (turn ?): unrelated fact" in formatted


@pytest.mark.anyio
async def test_regrade_skips_attribution_and_does_not_raise():
    """Two sequential grade calls for the same (org, session, grader_version)
    simulate a duplicate sessions.stored delivery (storage-worker emits one
    even when its own trace upsert lost the richer-wins race). The second
    call must find zero rows returned by ON CONFLICT DO NOTHING, skip ALL
    recall_outcomes inserts, and return cleanly rather than raising a
    ForeignKeyViolation against an id that was never persisted.
    """
    recalls = [{"id": "re1", "memory_ids": ["m1"], "session_id": "s"}]
    memory_rows = [_mem_row("m1", "x")]
    verdict = MagicMock(confidence=0.5, action="flag", checks={})

    # First call: RETURNING id yields the minted id (insert won).
    # Second call: RETURNING id yields nothing (ON CONFLICT DO NOTHING hit).
    db = _db(recalls, memory_rows, session_outcome_returns=["some-id", None])
    with patch.object(og, "fetch_trace", return_value=TRACE), \
         patch.object(og, "verify", new=AsyncMock(return_value=verdict)), \
         patch.object(og, "embed_text", new=AsyncMock(return_value=[1.0, 0.0])):
        first = await og.grade_stored_session(db, MagicMock(), org_id="o", session_id="s")
        db.execute.call_args_list.clear()
        second = await og.grade_stored_session(db, MagicMock(), org_id="o", session_id="s")

    assert first["attributions"] == 1
    assert second == {"session_outcome_id": None, "attributions": 0}

    recall_inserts = [c for c in db.execute.call_args_list
                       if "INSERT INTO recall_outcomes" in str(c.args[0])]
    assert recall_inserts == []
