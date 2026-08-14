import json
from unittest.mock import MagicMock, patch

from shared import memory as memory_mod

from app import dreamer

_MEMORY_ID = "11111111-1111-1111-1111-111111111111"
_BASE_REVISION = "22222222-2222-2222-2222-222222222222"


def _sessions_with_tool_failures(n):
    return [{"session_id": f"s{i}",
             "tool_calls": [{"tool": "Bash", "status": "error",
                             "args": "npm test"}]} for i in range(n)]


def _insert_first_result(proposal_id):
    """A ``db.execute(...)`` result mock for the proposal INSERT ...
    RETURNING id — mimics ``ON CONFLICT ... DO NOTHING`` succeeding."""
    m = MagicMock()
    m.first.return_value = (proposal_id,)
    return m


def test_repeated_tool_failure_becomes_proposal_with_prevalence():
    db = MagicMock()
    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": _sessions_with_tool_failures(3),
            "bad_memories": [], "stale_clusters": []}), \
         patch.object(dreamer, "_draft_diff", return_value="Add a note: npm test fails without X"):
        written = dreamer.dream_for_org(db, MagicMock(), org_id="o")
    assert written == 1
    ins = [c for c in db.execute.call_args_list
           if "INSERT INTO memory_proposals" in str(c.args[0])][0]
    evidence = json.loads(ins.args[1]["evidence"])
    assert evidence["prevalence"] == 3
    assert ins.args[1]["status"] == "open"
    # (a) tool-failure sets proposed_content deterministically — no LLM
    # needed for this value at all.
    assert ins.args[1]["proposed_content"] is not None
    assert ins.args[1]["detector"] == "tool_failure"


def test_two_failures_is_below_threshold_no_proposal():
    db = MagicMock()
    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": _sessions_with_tool_failures(2),
            "bad_memories": [], "stale_clusters": []}):
        written = dreamer.dream_for_org(db, MagicMock(), org_id="o")
    assert written == 0


def test_llm_failure_still_files_proposal_with_raw_evidence():
    db = MagicMock()
    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": _sessions_with_tool_failures(4),
            "bad_memories": [], "stale_clusters": []}), \
         patch.object(dreamer, "_draft_diff", side_effect=RuntimeError):
        written = dreamer.dream_for_org(db, MagicMock(), org_id="o")
    assert written == 1  # detection is arithmetic; drafting is optional prose


def test_llm_failure_diff_falls_back_to_deterministic_summary():
    db = MagicMock()
    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": _sessions_with_tool_failures(4),
            "bad_memories": [], "stale_clusters": []}), \
         patch.object(dreamer, "_draft_diff", side_effect=RuntimeError):
        dreamer.dream_for_org(db, MagicMock(), org_id="o")
    ins = [c for c in db.execute.call_args_list
           if "INSERT INTO memory_proposals" in str(c.args[0])][0]
    assert "Bash" in ins.args[1]["diff"]
    assert "4" in ins.args[1]["diff"]
    # LLM failure never touches proposed_content — it was never the LLM's.
    assert ins.args[1]["proposed_content"] is not None


def test_sample_args_are_truncated_before_reaching_the_prompt_or_row():
    db = MagicMock()
    huge_args = "x" * 5000
    sessions = [{"session_id": f"s{i}",
                 "tool_calls": [{"tool": "Bash", "status": "error", "args": huge_args}]}
                for i in range(3)]
    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": sessions, "bad_memories": [], "stale_clusters": []}), \
         patch.object(dreamer, "_draft_diff", side_effect=RuntimeError):
        dreamer.dream_for_org(db, MagicMock(), org_id="o")
    ins = [c for c in db.execute.call_args_list
           if "INSERT INTO memory_proposals" in str(c.args[0])][0]
    evidence = json.loads(ins.args[1]["evidence"])
    assert len(evidence["sample_args"]) <= dreamer.MAX_SAMPLE_ARGS_CHARS + len("…[truncated]")


def test_bad_outcome_memory_detector_emits_revise_proposal_with_null_content():
    db = MagicMock()
    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": [],
            "bad_memories": [{"memory_id": "m1", "attributions": 6,
                               "used_rate": 0.5, "decided_count": 5,
                               "pass_rate": 0.2}],
            "stale_clusters": []}), \
         patch.object(dreamer, "_draft_diff", return_value="revise m1"):
        written = dreamer.dream_for_org(db, MagicMock(), org_id="o")
    assert written == 1
    ins = [c for c in db.execute.call_args_list
           if "INSERT INTO memory_proposals" in str(c.args[0])][0]
    assert ins.args[1]["kind"] == "revise"
    assert ins.args[1]["target_memory_id"] == "m1"
    # (b) bad-outcome-memory never gets an LLM-drafted (or any) replacement
    # belief — a human must supply one.
    assert ins.args[1]["proposed_content"] is None


def test_bad_outcome_memory_below_thresholds_no_proposal():
    db = MagicMock()
    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": [],
            # attributions high enough, but used_rate too low
            "bad_memories": [{"memory_id": "m1", "attributions": 6,
                               "used_rate": 0.1, "decided_count": 5,
                               "pass_rate": 0.2}],
            "stale_clusters": []}):
        written = dreamer.dream_for_org(db, MagicMock(), org_id="o")
    assert written == 0


def test_bad_outcome_memory_with_too_few_decided_verdicts_is_skipped():
    """A grader outage (mostly verdict='unknown') must not manufacture a
    proposal — decided_count below MIN_DECIDED_VERDICTS skips detection
    entirely, even though attributions/used_rate/pass_rate all clear."""
    db = MagicMock()
    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": [],
            "bad_memories": [{"memory_id": "m1", "attributions": 6,
                               "used_rate": 0.9, "decided_count": 1,
                               "pass_rate": 0.0}],
            "stale_clusters": []}):
        written = dreamer.dream_for_org(db, MagicMock(), org_id="o")
    assert written == 0


def test_stale_serve_cluster_detector_emits_retire_proposal():
    db = MagicMock()
    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": [], "bad_memories": [],
            "stale_clusters": [{"memory_id": "m2", "stale_count": 3}]}), \
         patch.object(dreamer, "_draft_diff", return_value="retire m2"):
        written = dreamer.dream_for_org(db, MagicMock(), org_id="o")
    assert written == 1
    ins = [c for c in db.execute.call_args_list
           if "INSERT INTO memory_proposals" in str(c.args[0])][0]
    assert ins.args[1]["kind"] == "retire"
    assert ins.args[1]["target_memory_id"] == "m2"
    assert ins.args[1]["proposed_content"] is None


def test_retire_never_auto_applies_even_at_agent_scope():
    """HIGH fix: retire never auto-applies, regardless of scope."""
    db = MagicMock()

    def execute(stmt, params=None):
        m = MagicMock()
        s = str(stmt)
        if "FROM session_memories" in s:
            m.mappings.return_value.first.return_value = {
                "scope": "agent", "agent_id": "agent-1", "revision": _BASE_REVISION,
                "memory_type": "fact",
            }
        elif "INSERT INTO memory_proposals" in s:
            return _insert_first_result("p1")
        return m
    db.execute.side_effect = execute

    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": [], "bad_memories": [],
            "stale_clusters": [{"memory_id": "m2", "stale_count": 3}]}), \
         patch.object(dreamer, "_draft_diff", return_value="retire m2"), \
         patch.object(dreamer, "retract_memory") as rm, \
         patch.object(dreamer, "supersede_with_revision") as sup:
        written = dreamer.dream_for_org(db, MagicMock(), org_id="o")

    assert written == 1
    rm.assert_not_called()
    sup.assert_not_called()
    updates = [c for c in db.execute.call_args_list
               if "UPDATE memory_proposals" in str(c.args[0])]
    assert len(updates) == 0


def test_org_scope_revise_never_auto_applies():
    db = MagicMock()

    def execute(stmt, params=None):
        m = MagicMock()
        s = str(stmt)
        if "FROM session_memories" in s:
            m.mappings.return_value.first.return_value = {
                "scope": "org", "agent_id": None, "revision": _BASE_REVISION,
                "memory_type": "fact",
            }
        elif "INSERT INTO memory_proposals" in s:
            return _insert_first_result("p1")
        return m
    db.execute.side_effect = execute

    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": [],
            "bad_memories": [{"memory_id": "m1", "attributions": 6,
                               "used_rate": 0.5, "decided_count": 5,
                               "pass_rate": 0.2}],
            "stale_clusters": []}), \
         patch.object(dreamer, "_draft_diff", return_value="revise m1"), \
         patch.object(dreamer, "supersede_with_revision") as sup:
        written = dreamer.dream_for_org(db, MagicMock(), org_id="o")

    assert written == 1
    sup.assert_not_called()
    updates = [c for c in db.execute.call_args_list
               if "UPDATE memory_proposals" in str(c.args[0])]
    assert len(updates) == 0


def test_agent_scope_revise_with_no_proposed_content_never_auto_applies():
    """bad_outcome_memory detections have proposed_content=None — even at
    scope='agent' they must never auto-apply (nothing to write)."""
    db = MagicMock()

    def execute(stmt, params=None):
        m = MagicMock()
        s = str(stmt)
        if "FROM session_memories" in s:
            m.mappings.return_value.first.return_value = {
                "scope": "agent", "agent_id": "agent-1", "revision": _BASE_REVISION,
                "memory_type": "fact",
            }
        elif "INSERT INTO memory_proposals" in s:
            return _insert_first_result("p1")
        return m
    db.execute.side_effect = execute

    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": [],
            "bad_memories": [{"memory_id": "m1", "attributions": 6,
                               "used_rate": 0.5, "decided_count": 5,
                               "pass_rate": 0.2}],
            "stale_clusters": []}), \
         patch.object(dreamer, "_draft_diff", return_value="revise m1"), \
         patch.object(dreamer, "supersede_with_revision") as sup:
        written = dreamer.dream_for_org(db, MagicMock(), org_id="o")

    assert written == 1
    sup.assert_not_called()


def test_duplicate_open_detection_is_deduplicated_by_conflict():
    """ON CONFLICT DO NOTHING against the partial unique index: a detection
    matching an already-open proposal for the same org/detector/target
    inserts nothing and does not count toward `written`."""
    db = MagicMock()

    def execute(stmt, params=None):
        m = MagicMock()
        s = str(stmt)
        if "INSERT INTO memory_proposals" in s:
            m.first.return_value = None  # ON CONFLICT DO NOTHING: no row
        return m
    db.execute.side_effect = execute

    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": _sessions_with_tool_failures(3),
            "bad_memories": [], "stale_clusters": []}), \
         patch.object(dreamer, "_draft_diff", return_value="note"):
        written = dreamer.dream_for_org(db, MagicMock(), org_id="o")
    assert written == 0


def test_agent_scope_revise_auto_applies_via_real_supersede_with_required_kwargs():
    """CRITICAL regression test: exercises the REAL
    shared.memory.supersede_with_revision (never patched out itself) so a
    call site missing a required kwarg (memory_type/session_id) raises,
    same as it would in production. Only the impure internals
    (_prepare_cas_write, _apply_supersession_for_cas) are patched, mirroring
    services/shared/tests/test_cas_write.py's own pattern.
    """
    db = MagicMock()

    def execute(stmt, params=None):
        m = MagicMock()
        s = str(stmt)
        if "FROM session_memories" in s and "revision, memory_type" in s.replace(
                "\n", " "):
            m.mappings.return_value.first.return_value = {
                "scope": "agent", "agent_id": "agent-1", "revision": _BASE_REVISION,
                "memory_type": "fact",
            }
        elif "SELECT scope, agent_id, user_id, project_id" in s:
            # supersede_with_revision's own phase-2 authorization SELECT —
            # attribute access, per shared.memory._authorize_scoped_action.
            row = MagicMock()
            row.scope = "agent"
            row.agent_id = "agent-1"
            row.user_id = None
            row.project_id = None
            m.fetchone.return_value = row
        elif "SET revision = gen_random_uuid()" in s:
            m.rowcount = 1  # CAS guard hit
        elif "INSERT INTO memory_proposals" in s:
            return _insert_first_result("proposal-1")
        return m
    db.execute.side_effect = execute

    with (
        patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": [],
            "bad_memories": [{"memory_id": _MEMORY_ID, "attributions": 6,
                               "used_rate": 0.5, "decided_count": 5,
                               "pass_rate": 0.2}],
            "stale_clusters": []}),
        patch.object(dreamer, "_draft_diff", return_value="revise"),
        patch.object(memory_mod, "_prepare_cas_write",
                     return_value=("mine", {}, [0.1] * 4, False, [])),
        patch.object(memory_mod, "_apply_supersession_for_cas",
                     return_value={"new_id": "m1-new", "new_revision": "rev-2"}) as ap,
    ):
        # bad_outcome_memory files with proposed_content=None so it never
        # auto-applies on its own; force a proposed_content to exercise the
        # auto-apply branch directly instead of relaying on the detector.
        with patch.object(dreamer, "_detect_bad_memories", return_value=[{
                "detector": "bad_outcome_memory", "kind": "revise",
                "target_memory_id": _MEMORY_ID, "prevalence": 6,
                "evidence": {"memory_id": _MEMORY_ID}, "summary": "s",
                "proposed_content": "the corrected belief"}]):
            written = dreamer.dream_for_org(db, MagicMock(), org_id="o")

    assert written == 1
    ap.assert_called_once()
    kwargs = ap.call_args.kwargs
    assert kwargs["memory_type"] == "fact"
    assert kwargs["session_id"].startswith("proposal-")
    assert kwargs["content"] == "mine"

    updates = [c for c in db.execute.call_args_list
               if "UPDATE memory_proposals" in str(c.args[0])]
    assert len(updates) == 1
    assert updates[0].args[1]["status"] == "auto"
