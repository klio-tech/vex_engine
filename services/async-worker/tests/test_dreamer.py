import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import dreamer


def _sessions_with_tool_failures(n):
    return [{"session_id": f"s{i}",
             "tool_calls": [{"tool": "Bash", "status": "error",
                             "args": "npm test"}]} for i in range(n)]


@pytest.mark.anyio
async def test_repeated_tool_failure_becomes_proposal_with_prevalence():
    db = MagicMock()
    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": _sessions_with_tool_failures(3),
            "bad_memories": [], "stale_clusters": []}), \
         patch.object(dreamer, "_draft_diff", new=AsyncMock(return_value="Add a note: npm test fails without X")):
        written = await dreamer.dream_for_org(db, MagicMock(), org_id="o")
    assert written == 1
    ins = [c for c in db.execute.call_args_list
           if "INSERT INTO memory_proposals" in str(c.args[0])][0]
    evidence = json.loads(ins.args[1]["evidence"])
    assert evidence["prevalence"] == 3
    assert ins.args[1]["status"] == "open"


@pytest.mark.anyio
async def test_two_failures_is_below_threshold_no_proposal():
    db = MagicMock()
    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": _sessions_with_tool_failures(2),
            "bad_memories": [], "stale_clusters": []}):
        written = await dreamer.dream_for_org(db, MagicMock(), org_id="o")
    assert written == 0


@pytest.mark.anyio
async def test_llm_failure_still_files_proposal_with_raw_evidence():
    db = MagicMock()
    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": _sessions_with_tool_failures(4),
            "bad_memories": [], "stale_clusters": []}), \
         patch.object(dreamer, "_draft_diff", new=AsyncMock(side_effect=RuntimeError)):
        written = await dreamer.dream_for_org(db, MagicMock(), org_id="o")
    assert written == 1  # detection is arithmetic; drafting is optional prose


@pytest.mark.anyio
async def test_llm_failure_diff_falls_back_to_deterministic_summary():
    db = MagicMock()
    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": _sessions_with_tool_failures(4),
            "bad_memories": [], "stale_clusters": []}), \
         patch.object(dreamer, "_draft_diff", new=AsyncMock(side_effect=RuntimeError)):
        await dreamer.dream_for_org(db, MagicMock(), org_id="o")
    ins = [c for c in db.execute.call_args_list
           if "INSERT INTO memory_proposals" in str(c.args[0])][0]
    assert "Bash" in ins.args[1]["diff"]
    assert "4" in ins.args[1]["diff"]


@pytest.mark.anyio
async def test_bad_outcome_memory_detector_emits_revise_proposal():
    db = MagicMock()
    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": [],
            "bad_memories": [{"memory_id": "m1", "attributions": 6,
                               "used_rate": 0.5, "pass_rate": 0.2}],
            "stale_clusters": []}), \
         patch.object(dreamer, "_draft_diff", new=AsyncMock(return_value="revise m1")):
        written = await dreamer.dream_for_org(db, MagicMock(), org_id="o")
    assert written == 1
    ins = [c for c in db.execute.call_args_list
           if "INSERT INTO memory_proposals" in str(c.args[0])][0]
    assert ins.args[1]["kind"] == "revise"
    assert ins.args[1]["target_memory_id"] == "m1"


@pytest.mark.anyio
async def test_bad_outcome_memory_below_thresholds_no_proposal():
    db = MagicMock()
    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": [],
            # attributions high enough, but used_rate too low
            "bad_memories": [{"memory_id": "m1", "attributions": 6,
                               "used_rate": 0.1, "pass_rate": 0.2}],
            "stale_clusters": []}):
        written = await dreamer.dream_for_org(db, MagicMock(), org_id="o")
    assert written == 0


@pytest.mark.anyio
async def test_stale_serve_cluster_detector_emits_retire_proposal():
    db = MagicMock()
    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": [], "bad_memories": [],
            "stale_clusters": [{"memory_id": "m2", "stale_count": 3}]}), \
         patch.object(dreamer, "_draft_diff", new=AsyncMock(return_value="retire m2")):
        written = await dreamer.dream_for_org(db, MagicMock(), org_id="o")
    assert written == 1
    ins = [c for c in db.execute.call_args_list
           if "INSERT INTO memory_proposals" in str(c.args[0])][0]
    assert ins.args[1]["kind"] == "retire"
    assert ins.args[1]["target_memory_id"] == "m2"


@pytest.mark.anyio
async def test_agent_scope_revise_auto_applies_via_cas():
    db = MagicMock()

    def execute(stmt, params=None):
        m = MagicMock()
        s = str(stmt)
        if "FROM session_memories" in s:
            m.mappings.return_value.first.return_value = {
                "scope": "agent", "agent_id": "agent-1", "revision": "rev-1",
            }
        return m
    db.execute.side_effect = execute

    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": [],
            "bad_memories": [{"memory_id": "m1", "attributions": 6,
                               "used_rate": 0.5, "pass_rate": 0.2}],
            "stale_clusters": []}), \
         patch.object(dreamer, "_draft_diff", new=AsyncMock(return_value="revise m1")), \
         patch.object(dreamer, "supersede_with_revision",
                      return_value={"conflict": False, "stored": True,
                                    "new_id": "m1-new", "new_revision": "rev-2"}) as sup:
        written = await dreamer.dream_for_org(db, MagicMock(), org_id="o")

    assert written == 1
    sup.assert_called_once()
    kwargs = sup.call_args.kwargs
    assert kwargs["org_id"] == "o"
    assert kwargs["memory_id"] == "m1"
    assert kwargs["base_revision"] == "rev-1"
    assert kwargs["agent_id"] == "agent-1"

    updates = [c for c in db.execute.call_args_list
               if "UPDATE memory_proposals" in str(c.args[0])]
    assert len(updates) == 1
    assert updates[0].args[1]["status"] == "auto"


@pytest.mark.anyio
async def test_org_scope_revise_never_auto_applies():
    db = MagicMock()

    def execute(stmt, params=None):
        m = MagicMock()
        s = str(stmt)
        if "FROM session_memories" in s:
            m.mappings.return_value.first.return_value = {
                "scope": "org", "agent_id": None, "revision": "rev-1",
            }
        return m
    db.execute.side_effect = execute

    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": [],
            "bad_memories": [{"memory_id": "m1", "attributions": 6,
                               "used_rate": 0.5, "pass_rate": 0.2}],
            "stale_clusters": []}), \
         patch.object(dreamer, "_draft_diff", new=AsyncMock(return_value="revise m1")), \
         patch.object(dreamer, "supersede_with_revision") as sup:
        written = await dreamer.dream_for_org(db, MagicMock(), org_id="o")

    assert written == 1
    sup.assert_not_called()
    updates = [c for c in db.execute.call_args_list
               if "UPDATE memory_proposals" in str(c.args[0])]
    assert len(updates) == 0
