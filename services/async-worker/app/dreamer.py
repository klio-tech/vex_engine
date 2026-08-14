"""The dreamer: deterministic detection over evidence, LLM prose only (Build 5).

Detection is ARITHMETIC. Three detectors run over already-graded evidence
(``session_traces``' tool_calls, ``session_outcomes``/``recall_outcomes``)
and each one either counts past a fixed threshold or it does not — there is
no LLM in that path. Only AFTER a detector has already decided a pattern is
real does an LLM get a turn, and even then its only job is prose: turning
the counted evidence into a human-readable ``diff`` sentence
(:func:`_draft_diff`). If the LLM call fails or returns nothing useful, the
proposal is filed anyway with the detector's own deterministic summary
sentence as the diff. Dreaming cannot hallucinate a pattern, because the
pattern was never the LLM's to invent.

Every detector reads only org-predicated evidence (see
:func:`_load_recent_evidence`) and every ``INSERT``/``UPDATE`` below carries
``org_id`` — dreaming for one org can never see or touch another org's
memories, sessions, or proposals.

Proposal application mirrors every other writer in this schema: it goes
through :func:`shared.memory.supersede_with_revision`, obeying the same CAS
guard (stale ``base_revision`` → refused, not overwritten) and the same
scope-authorization ladder (:func:`shared.memory._authorize_scoped_action`)
as a human-driven revise. The dreamer holds no user principal, so that
ladder only ever lets it through for ``scope='agent'`` targets (an
``agent``-scoped row's authorization check is "does the caller's
``agent_id`` match the row's" — trivially true here because the dreamer
supplies the row's OWN ``agent_id`` back to it, i.e. this is the agent
correcting its own belief, not a third party). ``private`` and ``project``
rows require a ``user_id`` the dreamer never has, and fall through to a
refusal — which is exactly why ``status='auto'`` is restricted to
``scope='agent'`` by construction here, not just by convention: every other
scope's CAS call would be refused anyway. ``org``-scoped proposals (and
``add`` proposals, which have no target to CAS against) are always filed
``status='open'`` for a human to approve via the REST endpoints in
``app.mcp_app`` (mcp-server), which resolve a real ``user_id`` and record it
as ``decided_by``.

No network call ever sits inside an open transaction on ``db`` (the Task
8/9 lesson, binding here too): each proposal's target-row lookup is
committed before :func:`_draft_diff` (an LLM call) runs, the proposal INSERT
is committed before :func:`shared.memory.supersede_with_revision` runs (that
function does its own internal embed call), and the final status UPDATE is
its own transaction.
"""

import json
import logging
import uuid
from typing import Any, Optional

from sqlalchemy import text

from shared.memory import call_llm, supersede_with_revision
from shared.traces import fetch_trace

logger = logging.getLogger("agentguard.dreamer")

#: (a) repeated tool failure — same tool erroring in this many DISTINCT
#: sessions' tool_calls.
TOOL_FAILURE_MIN_SESSIONS = 3

#: (b) bad-outcome memory — a memory attributed at least this many times,
#: used at least this often, in sessions that mostly did NOT pass.
BAD_MEMORY_MIN_ATTRIBUTIONS = 5
BAD_MEMORY_MIN_USED_RATE = 0.4
BAD_MEMORY_MAX_PASS_RATE = 0.5

#: (c) stale-serve cluster — a memory served stale at least this many times.
STALE_SERVE_MIN_ROWS = 3

#: How many recent sessions' traces a single dream pass inspects for the
#: tool-failure detector. Not the same knob as ``batch_limit`` (that one caps
#: proposals written); this caps evidence gathered.
DEFAULT_SESSION_LOOKBACK = 200

DEFAULT_BATCH_LIMIT = 50

_APPLICABLE_KINDS = ("revise", "retire")


def _load_recent_evidence(
    db: Any, s3: Any, *, org_id: str, session_lookback: int = DEFAULT_SESSION_LOOKBACK
) -> dict[str, Any]:
    """Gather this org's evidence for all three detectors, org-predicated.

    Returns ``{"sessions": [...], "bad_memories": [...], "stale_clusters": [...]}``:

    * ``sessions`` — recent ``session_traces`` rows for this org, each with
      its S3-hydrated ``tool_calls`` list (via :func:`shared.traces.fetch_trace`),
      feeding the repeated-tool-failure detector.
    * ``bad_memories`` — one row per memory with >= ``BAD_MEMORY_MIN_ATTRIBUTIONS``
      ``recall_outcomes`` attributions in this org, joined to
      ``session_outcomes`` for the session pass-rate.
    * ``stale_clusters`` — one row per memory with >= ``STALE_SERVE_MIN_ROWS``
      ``served_stale`` rows in this org.
    """
    session_rows = (
        db.execute(
            text(
                """
                SELECT session_id FROM session_traces
                WHERE org_id = :org_id
                ORDER BY created_at DESC
                LIMIT :limit
                """
            ),
            {"org_id": org_id, "limit": session_lookback},
        )
        .mappings()
        .all()
    )
    sessions: list[dict[str, Any]] = []
    for row in session_rows:
        session_id = row["session_id"]
        trace = fetch_trace(db, s3, org_id=org_id, session_id=session_id)
        if trace is None:
            continue
        sessions.append(
            {"session_id": session_id, "tool_calls": trace.get("tool_calls", [])}
        )

    bad_memory_rows = (
        db.execute(
            text(
                """
                SELECT ro.memory_id AS memory_id,
                       COUNT(*) AS attributions,
                       AVG(CASE WHEN ro.used THEN 1.0 ELSE 0.0 END) AS used_rate,
                       AVG(CASE WHEN so.verdict = 'pass' THEN 1.0 ELSE 0.0 END)
                           AS pass_rate
                FROM recall_outcomes ro
                JOIN session_outcomes so
                  ON so.id = ro.session_outcome_id AND so.org_id = ro.org_id
                WHERE ro.org_id = :org_id
                GROUP BY ro.memory_id
                HAVING COUNT(*) >= :min_attrib
                """
            ),
            {"org_id": org_id, "min_attrib": BAD_MEMORY_MIN_ATTRIBUTIONS},
        )
        .mappings()
        .all()
    )

    stale_rows = (
        db.execute(
            text(
                """
                SELECT memory_id, COUNT(*) AS stale_count
                FROM recall_outcomes
                WHERE org_id = :org_id AND served_stale = true
                GROUP BY memory_id
                HAVING COUNT(*) >= :min_stale
                """
            ),
            {"org_id": org_id, "min_stale": STALE_SERVE_MIN_ROWS},
        )
        .mappings()
        .all()
    )

    return {
        "sessions": sessions,
        "bad_memories": [dict(r) for r in bad_memory_rows],
        "stale_clusters": [dict(r) for r in stale_rows],
    }


def _detect_tool_failures(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """(a) Same ``tool`` erroring in >= :data:`TOOL_FAILURE_MIN_SESSIONS`
    DISTINCT sessions' ``tool_calls``. A tool erroring 5 times in ONE session
    counts once toward prevalence — prevalence is sessions affected, not raw
    error count, so a single flaky session cannot itself trip the detector.
    """
    by_tool: dict[str, dict[str, Any]] = {}
    for session in sessions:
        session_id = session.get("session_id")
        seen_this_session: set[str] = set()
        for call in session.get("tool_calls") or []:
            if call.get("status") != "error":
                continue
            tool = call.get("tool")
            if not tool or tool in seen_this_session:
                continue
            seen_this_session.add(tool)
            entry = by_tool.setdefault(tool, {"sessions": [], "sample_args": call.get("args")})
            entry["sessions"].append(session_id)

    detections = []
    for tool, entry in by_tool.items():
        prevalence = len(entry["sessions"])
        if prevalence < TOOL_FAILURE_MIN_SESSIONS:
            continue
        detections.append(
            {
                "detector": "tool_failure",
                "kind": "add",
                "target_memory_id": None,
                "prevalence": prevalence,
                "evidence": {"sessions": entry["sessions"], "tool": tool,
                             "sample_args": entry["sample_args"]},
                "summary": (
                    f"[auto] {tool} failed identically in {prevalence} sessions: "
                    f"{entry['sample_args']}"
                ),
            }
        )
    return detections


def _detect_bad_memories(bad_memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """(b) A memory attributed >= :data:`BAD_MEMORY_MIN_ATTRIBUTIONS` times,
    used at a rate >= :data:`BAD_MEMORY_MIN_USED_RATE`, whose sessions pass at
    a rate < :data:`BAD_MEMORY_MAX_PASS_RATE`. The memory is being SERVED and
    USED, but the sessions it's used in are mostly failing — evidence it is
    actively steering agents wrong, not merely irrelevant/ignored.
    """
    detections = []
    for row in bad_memories:
        attributions = int(row.get("attributions") or 0)
        used_rate = float(row.get("used_rate") or 0.0)
        pass_rate = float(row.get("pass_rate") or 0.0)
        if (
            attributions >= BAD_MEMORY_MIN_ATTRIBUTIONS
            and used_rate >= BAD_MEMORY_MIN_USED_RATE
            and pass_rate < BAD_MEMORY_MAX_PASS_RATE
        ):
            memory_id = str(row["memory_id"])
            detections.append(
                {
                    "detector": "bad_outcome_memory",
                    "kind": "revise",
                    "target_memory_id": memory_id,
                    "prevalence": attributions,
                    "evidence": {
                        "memory_id": memory_id,
                        "attributions": attributions,
                        "used_rate": used_rate,
                        "pass_rate": pass_rate,
                    },
                    "summary": (
                        f"[auto] memory {memory_id} used in {attributions} "
                        f"attributions (used_rate={used_rate:.2f}) but session "
                        f"pass-rate is {pass_rate:.2f}"
                    ),
                }
            )
    return detections


def _detect_stale_clusters(stale_clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """(c) A memory served stale (curator-lag window) >= :data:`STALE_SERVE_MIN_ROWS`
    times — evidence the belief itself has drifted and should be retired
    rather than kept serving a known-stale answer.
    """
    detections = []
    for row in stale_clusters:
        stale_count = int(row.get("stale_count") or 0)
        if stale_count < STALE_SERVE_MIN_ROWS:
            continue
        memory_id = str(row["memory_id"])
        detections.append(
            {
                "detector": "stale_serve_cluster",
                "kind": "retire",
                "target_memory_id": memory_id,
                "prevalence": stale_count,
                "evidence": {"memory_id": memory_id, "stale_count": stale_count},
                "summary": (
                    f"[auto] memory {memory_id} served stale in {stale_count} recalls"
                ),
            }
        )
    return detections


async def _draft_diff(detection: dict[str, Any]) -> str:
    """LLM prose pass over ALREADY-computed detector evidence.

    Uses the same shared ``call_llm`` (``litellm.acompletion`` config) the
    curator's own synthesis passes use — see ``app.curator`` for the
    established config (``VERIFICATION_MODEL`` / ``LITELLM_API_URL`` /
    ``LITELLM_API_KEY`` env vars). The prompt supplies ONLY the detector's
    own evidence and forbids inventing facts; the LLM's role is to phrase a
    diff sentence, not to decide whether a pattern exists — that already
    happened, arithmetically, before this function is ever called.

    Raises on any failure (no result, malformed JSON, missing ``diff``
    field) so the caller's ``except Exception`` can fall back to the
    detector's own deterministic ``summary`` — never returns an empty or
    placeholder diff itself.
    """
    prompt = (
        "You are drafting a proposed memory-store change for a human "
        "reviewer to approve or reject. Write 1-3 concise sentences "
        "describing the proposed change, based ONLY on the evidence below. "
        "Do not invent facts, numbers, or session ids not present here.\n\n"
        f"Detector: {detection['detector']}\n"
        f"Proposal kind: {detection['kind']}\n"
        f"Evidence: {json.dumps(detection['evidence'])}\n"
        f"Deterministic summary: {detection['summary']}\n\n"
        'Respond with JSON: {"diff": "<your proposed diff text>"}'
    )
    result = await call_llm(prompt)
    if not result:
        raise RuntimeError("dreamer: LLM drafting returned no result")
    diff = result.get("diff")
    if not diff or not isinstance(diff, str):
        raise RuntimeError("dreamer: LLM drafting returned no diff field")
    return diff


def _target_row(db: Any, *, org_id: str, memory_id: str) -> Optional[dict[str, Any]]:
    """Org-predicated lookup of a proposal target's current scope/agent_id/revision.

    Committed immediately after reading (no lingering open transaction) —
    the caller may go on to make a network call (drafting, then possibly
    :func:`shared.memory.supersede_with_revision`'s internal embed), and
    nothing here should still be holding this connection's transaction open
    when that happens.
    """
    row = (
        db.execute(
            text(
                """
                SELECT scope, agent_id, revision FROM session_memories
                WHERE id = :memory_id AND org_id = :org_id
                """
            ),
            {"memory_id": memory_id, "org_id": org_id},
        )
        .mappings()
        .first()
    )
    db.commit()
    return dict(row) if row is not None else None


async def dream_for_org(
    db: Any, s3: Any, *, org_id: str, batch_limit: int = DEFAULT_BATCH_LIMIT
) -> int:
    """Run all three deterministic detectors for ``org_id`` and file one
    ``memory_proposals`` row per detection.

    Returns the number of proposals written. Detection is arithmetic and
    always runs to completion; an LLM drafting failure degrades only that
    one proposal's ``diff`` text to the detector's deterministic summary
    sentence — it never suppresses the proposal.

    ``batch_limit`` caps proposals written per call (a defensive ceiling,
    matching the curator's own per-tick reconciliation budget pattern —
    see ``app.curator._reconcile_budget``); detection itself is not capped,
    only how many of its findings get filed in one pass.
    """
    evidence = _load_recent_evidence(db, s3, org_id=org_id)

    detections = (
        _detect_tool_failures(evidence.get("sessions", []))
        + _detect_bad_memories(evidence.get("bad_memories", []))
        + _detect_stale_clusters(evidence.get("stale_clusters", []))
    )

    written = 0
    for detection in detections[:batch_limit]:
        target_memory_id = detection.get("target_memory_id")
        target = (
            _target_row(db, org_id=org_id, memory_id=target_memory_id)
            if target_memory_id
            else None
        )
        scope = target["scope"] if target else "org"

        try:
            diff = await _draft_diff(detection)
        except Exception:
            logger.warning(
                "dreamer: drafting failed org=%s detector=%s; using deterministic "
                "summary",
                org_id, detection["detector"], exc_info=True,
            )
            diff = detection["summary"]

        proposal_id = str(uuid.uuid4())
        db.execute(
            text(
                """
                INSERT INTO memory_proposals
                  (id, org_id, scope, kind, target_memory_id, diff, evidence,
                   confidence, status)
                VALUES (:id, :org_id, :scope, :kind, :target_memory_id, :diff,
                        :evidence, :confidence, :status)
                """
            ),
            {
                "id": proposal_id,
                "org_id": org_id,
                "scope": scope,
                "kind": detection["kind"],
                "target_memory_id": target_memory_id,
                "diff": diff,
                "evidence": json.dumps(
                    {"prevalence": detection["prevalence"],
                     "detector": detection["detector"],
                     **detection["evidence"]}
                ),
                "confidence": None,
                "status": "open",
            },
        )
        db.commit()
        written += 1

        can_auto = (
            detection["kind"] in _APPLICABLE_KINDS
            and target is not None
            and scope == "agent"
        )
        if can_auto:
            _try_auto_apply(
                db,
                org_id=org_id,
                proposal_id=proposal_id,
                memory_id=target_memory_id,
                base_revision=str(target["revision"]),
                agent_id=target["agent_id"],
                diff=diff,
            )

    return written


def _try_auto_apply(
    db: Any,
    *,
    org_id: str,
    proposal_id: str,
    memory_id: str,
    base_revision: str,
    agent_id: Optional[str],
    diff: str,
) -> None:
    """Attempt to apply an ``agent``-scope proposal via the CAS path.

    Called only when the target's scope is ``agent`` (see
    :func:`dream_for_org`) — every other scope falls through
    ``shared.memory._authorize_scoped_action`` because the dreamer holds no
    ``user_id``, so this function does not need to re-check scope itself.
    The proposal's own INSERT has already committed by the time this runs,
    so a refused/conflicted apply just leaves the proposal ``status='open'``
    for a human to review through the normal REST path — never a lost write.
    """
    try:
        result = supersede_with_revision(
            db,
            org_id=org_id,
            memory_id=memory_id,
            base_revision=base_revision,
            new_content=diff,
            agent_id=agent_id,
            scope="agent",
        )
    except Exception:
        logger.warning(
            "dreamer: auto-apply raised org=%s proposal=%s memory=%s",
            org_id, proposal_id, memory_id, exc_info=True,
        )
        return

    if not result.get("stored"):
        logger.info(
            "dreamer: auto-apply not stored org=%s proposal=%s memory=%s error=%s",
            org_id, proposal_id, memory_id, result.get("error"),
        )
        return

    db.execute(
        text(
            """
            UPDATE memory_proposals
            SET status = :status, decided_at = now()
            WHERE id = :id AND org_id = :org_id
            """
        ),
        {"status": "auto", "id": proposal_id, "org_id": org_id},
    )
    db.commit()
