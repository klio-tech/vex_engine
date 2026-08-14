from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import outcome_grader as og


def _db(recalls, memories):
    db = MagicMock()
    def execute(stmt, params=None):
        m = MagicMock()
        s = str(stmt)
        if "FROM brain_recall_events" in s:
            m.mappings.return_value.all.return_value = recalls
        elif "FROM session_memories" in s:
            m.mappings.return_value.all.return_value = memories
        return m
    db.execute.side_effect = execute
    return db


TRACE = {"messages": [{"role": "user", "content": "q"},
                      {"role": "assistant", "content": "final answer about postgres"}],
         "tool_calls": [], "row": {}}


@pytest.mark.anyio
async def test_one_session_verdict_and_per_memory_attribution():
    recalls = [{"id": "re1", "memory_ids": ["m1", "m2"], "session_id": "s"}]
    memories = [{"id": "m1", "content": "postgres decision"},
                {"id": "m2", "content": "unrelated fact"}]
    verdict = MagicMock(confidence=0.9, action="pass")
    checks = {"memory_consistency": MagicMock(score=0.8)}
    verdict.checks = checks
    with patch.object(og, "fetch_trace", return_value=TRACE), \
         patch.object(og, "verify", new=AsyncMock(return_value=verdict)) as v, \
         patch.object(og, "embed_text", new=AsyncMock(side_effect=[
             [1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])):  # output, m1, m2
        result = await og.grade_stored_session(
            _db(recalls, memories), MagicMock(), org_id="o", session_id="s")
    # verify called ONCE, with the served memories
    assert v.await_count == 1
    assert v.await_args.kwargs["session_memories"] is not None
    assert result["attributions"] == 2


@pytest.mark.anyio
async def test_session_verdict_written_once_not_per_recall():
    recalls = [{"id": f"re{i}", "memory_ids": ["m1"], "session_id": "s"}
               for i in range(8)]
    memories = [{"id": "m1", "content": "x"}]
    verdict = MagicMock(confidence=0.5, action="flag", checks={})
    db = _db(recalls, memories)
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
    db = _db(recalls, [{"id": "m1", "content": "x"}])
    with patch.object(og, "fetch_trace", return_value=None):
        result = await og.grade_stored_session(db, MagicMock(), org_id="o",
                                               session_id="s")
    inserts = [c for c in db.execute.call_args_list
               if "INSERT INTO session_outcomes" in str(c.args[0])]
    assert inserts and inserts[0].args[1]["verdict"] == "unknown"
