"""The dreamer: deterministic detection over evidence, LLM prose only (Build 5).

Detection is ARITHMETIC. Three detectors run over already-graded evidence
(``session_traces``' tool_calls, ``session_outcomes``/``recall_outcomes``)
and each one either counts past a fixed threshold or it does not — there is
no LLM in that path. Only AFTER a detector has already decided a pattern is
real does an LLM get a turn, and even then its only job is prose
(:func:`_draft_diff`) written to ``diff`` — a REVIEWER-FACING RATIONALE
column, never memory content. What actually gets written to
``session_memories`` on approval is ``proposed_content``: for the
tool-failure detector that is a deterministic note the detector itself
composed (no LLM in that value at all); for the bad-outcome-memory detector
it is deliberately left ``NULL`` — that detector only has evidence that an
existing belief is hurting outcomes, not a proposed replacement belief, so a
human must supply one before it can ever be applied (``needs_content``, see
``app.proposals``). If the LLM drafting call fails or returns nothing
useful, ``diff`` falls back to the detector's own deterministic summary
sentence; ``proposed_content`` is never touched by that fallback, because
it was never the LLM's to begin with. Dreaming cannot hallucinate a
pattern, and it cannot hallucinate a belief either.

Every detector reads only org-predicated, time-bounded evidence (see
:func:`_load_recent_evidence` — a rolling :data:`EVIDENCE_LOOKBACK_DAYS`
window) and every ``INSERT``/``UPDATE`` below carries ``org_id`` — dreaming
for one org can never see or touch another org's memories, sessions, or
proposals. The bad-outcome-memory detector additionally requires at least
:data:`MIN_DECIDED_VERDICTS` non-``unknown`` session verdicts before it
will fire at all: a grading outage (every session comes back
``verdict='unknown'``) must never manufacture a "sessions are failing"
proposal out of missing data.

Proposal application:

* ``kind='add'`` and ``kind='revise'`` both require ``proposed_content``.
* ``kind='revise'``/``kind='retire'`` apply against ``target_memory_id``,
  but through TWO different primitives — this is deliberate, not an
  oversight: ``revise`` goes through the CAS path
  (:func:`shared.memory.supersede_with_revision`), obeying the same
  scope-authorization ladder and optimistic-concurrency guard as any other
  writer; ``retire`` goes through :func:`shared.memory.retract_memory`,
  because superseding an active belief with prose ("this memory looks
  wrong") would leave that meta-commentary AS the belief's new content —
  exactly the failure mode retiring exists to prevent. A retired memory
  disappears from recall; it does not get replaced with a note about
  itself.
* Auto-apply (``status='auto'``, no human in the loop) is restricted
  FURTHER than "scope='agent'" alone: only ``kind='revise'`` ever
  auto-applies. ``retire`` NEVER auto-applies, at any scope — the dreamer
  can propose that a belief is stale, but taking it out of recall
  unattended is a human decision even for an agent's own memory, so every
  ``retire`` proposal is filed ``status='open'`` and reviewed through
  ``app.proposals`` regardless of scope. This mirrors why the dreamer holds
  no user principal in the first place: it has no accountable identity to
  put in ``decided_by`` for a destructive action, only for a corrective
  one it can immediately show its own working for (the new content IS the
  evidence).

No network call ever sits inside an open transaction on ``db`` (the Task
8/9 lesson, binding here too): the target-row lookup commits before the LLM
drafting call runs; the proposal INSERT commits before
:func:`shared.memory.supersede_with_revision` runs (that function does its
own internal embed call); the atomic status CLAIM (see
:func:`_try_auto_apply`) commits before the CAS apply call, and reverts
with its own commit if that apply fails. ``_load_recent_evidence`` commits
immediately after its session-id lookup and rolls back after every single
``fetch_trace`` call in its loop, so no transaction spans more than one S3
GET at a time — never all of them.

Entirely synchronous (the one network call, LLM drafting, uses
``litellm.completion`` — the SAME sync-completion pattern
``shared.memory._embed_text_sync`` already established for CAS writes, for
the identical reason: this function is designed to run inside a worker
thread via ``anyio.to_thread.run_sync``, where there is no event loop to
``await`` against). ``app.main`` calls ``dream_for_org`` that way rather
than ``await``-ing it inline, so a slow dream pass (up to
:data:`DEFAULT_SESSION_LOOKBACK` S3 GETs plus one LLM call per detection)
blocks only its own worker thread — never the event loop the grader and
verification consumers share.
"""

import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import litellm
from sqlalchemy import text

from shared.env_config import parse_float
from shared.memory import supersede_with_revision
from shared.traces import fetch_trace

logger = logging.getLogger("agentguard.dreamer")

#: (a) repeated tool failure — same tool erroring in this many DISTINCT
#: sessions' tool_calls.
TOOL_FAILURE_MIN_SESSIONS = 3

#: (b) bad-outcome memory — a memory attributed at least this many times,
#: used at least this often, in DECIDED sessions that mostly did NOT pass.
BAD_MEMORY_MIN_ATTRIBUTIONS = 5
BAD_MEMORY_MIN_USED_RATE = 0.4
BAD_MEMORY_MAX_PASS_RATE = 0.5
#: A grader outage (everything ``verdict='unknown'``) must not manufacture
#: a proposal — at least this many non-``unknown`` verdicts are required
#: before the pass-rate check even runs.
MIN_DECIDED_VERDICTS = 3

#: (c) stale-serve cluster — a memory served stale at least this many times.
STALE_SERVE_MIN_ROWS = 3

#: Evidence older than this never feeds a detector — a memory that was bad
#: two years ago and has not been attributed since is not "currently bad".
EVIDENCE_LOOKBACK_DAYS = 30

#: How many recent sessions' traces a single dream pass inspects for the
#: tool-failure detector. Not the same knob as ``batch_limit`` (that one caps
#: proposals written); this caps evidence gathered.
DEFAULT_SESSION_LOOKBACK = 200

DEFAULT_BATCH_LIMIT = 50

_DEFAULT_LLM_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_LLM_TIMEOUT_S = 10.0

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _since(days: int = EVIDENCE_LOOKBACK_DAYS) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


def _load_recent_evidence(
    db: Any, s3: Any, *, org_id: str, session_lookback: int = DEFAULT_SESSION_LOOKBACK
) -> dict[str, Any]:
    """Gather this org's evidence for all three detectors, org-predicated
    and time-bounded to the last :data:`EVIDENCE_LOOKBACK_DAYS` days
    (attribution/staleness queries only — the session-id lookup below is
    already bounded by recency + ``session_lookback``, so it is not
    re-bounded by calendar date on top of that).

    Returns ``{"sessions": [...], "bad_memories": [...], "stale_clusters": [...]}``.

    No transaction spans more than one S3 GET: the session-id SELECT
    commits immediately (closing its own transaction before the loop's
    first network call), and each ``fetch_trace`` call — which does its own
    SELECT immediately before its S3 GET — is followed by a ``rollback()``
    that closes THAT call's transaction before the next iteration's GET.
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
    db.commit()

    sessions: list[dict[str, Any]] = []
    for row in session_rows:
        session_id = row["session_id"]
        trace = fetch_trace(db, s3, org_id=org_id, session_id=session_id)
        db.rollback()
        if trace is None:
            continue
        sessions.append(
            {"session_id": session_id, "tool_calls": trace.get("tool_calls", [])}
        )

    since = _since()

    bad_memory_rows = (
        db.execute(
            text(
                """
                SELECT ro.memory_id AS memory_id,
                       COUNT(*) AS attributions,
                       AVG(CASE WHEN ro.used THEN 1.0 ELSE 0.0 END) AS used_rate,
                       COUNT(*) FILTER (WHERE so.verdict != 'unknown')
                           AS decided_count,
                       AVG(CASE WHEN so.verdict = 'pass' THEN 1.0 ELSE 0.0 END)
                           FILTER (WHERE so.verdict != 'unknown') AS pass_rate
                FROM recall_outcomes ro
                JOIN session_outcomes so
                  ON so.id = ro.session_outcome_id AND so.org_id = ro.org_id
                WHERE ro.org_id = :org_id AND ro.created_at >= :since
                GROUP BY ro.memory_id
                HAVING COUNT(*) >= :min_attrib
                """
            ),
            {
                "org_id": org_id,
                "min_attrib": BAD_MEMORY_MIN_ATTRIBUTIONS,
                "since": since,
            },
        )
        .mappings()
        .all()
    )
    db.commit()

    stale_rows = (
        db.execute(
            text(
                """
                SELECT memory_id, COUNT(*) AS stale_count
                FROM recall_outcomes
                WHERE org_id = :org_id AND served_stale = true
                  AND created_at >= :since
                GROUP BY memory_id
                HAVING COUNT(*) >= :min_stale
                """
            ),
            {"org_id": org_id, "min_stale": STALE_SERVE_MIN_ROWS, "since": since},
        )
        .mappings()
        .all()
    )
    db.commit()

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

    ``proposed_content`` is set here, deterministically — no LLM involved in
    this value at all — so a ``kind='add'`` proposal from this detector can
    be approved even if LLM drafting fails entirely.

    Evidence and content NEVER carry raw tool-call ``args``: those come
    from individual (possibly private-scope) captured sessions, but every
    proposal this detector files is org-scoped (``scope='org'`` — see
    :func:`_insert_proposal`) and readable by any key in the org via
    ``GET /proposals``. Only the tool name, how many distinct sessions it
    failed in, and the session ids themselves (references, not content)
    are ever recorded here.
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
            entry = by_tool.setdefault(tool, {"sessions": []})
            entry["sessions"].append(session_id)

    detections = []
    for tool, entry in by_tool.items():
        prevalence = len(entry["sessions"])
        if prevalence < TOOL_FAILURE_MIN_SESSIONS:
            continue
        summary = f"[auto] {tool} failed in {prevalence} sessions"
        detections.append(
            {
                "detector": "tool_failure",
                "kind": "add",
                "target_memory_id": None,
                "prevalence": prevalence,
                "evidence": {"sessions": entry["sessions"], "tool": tool},
                "summary": summary,
                "proposed_content": summary,
            }
        )
    return detections


def _detect_bad_memories(bad_memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """(b) A memory attributed >= :data:`BAD_MEMORY_MIN_ATTRIBUTIONS` times,
    used at a rate >= :data:`BAD_MEMORY_MIN_USED_RATE`, whose DECIDED
    sessions pass at a rate < :data:`BAD_MEMORY_MAX_PASS_RATE`. The memory
    is being SERVED and USED, but the sessions it's used in are mostly
    failing — evidence it is actively steering agents wrong, not merely
    irrelevant/ignored.

    Requires >= :data:`MIN_DECIDED_VERDICTS` non-``unknown`` verdicts among
    its attributed sessions (``decided_count``, computed in
    ``_load_recent_evidence``'s SQL) — a grading outage must not manufacture
    a proposal out of missing data.

    ``proposed_content`` is deliberately left ``None``: this detector has
    evidence an existing belief is hurting outcomes, not a drafted
    replacement belief. A human supplies the replacement text before this
    kind of proposal can be approved (see ``app.proposals``'s
    ``needs_content`` refusal).
    """
    detections = []
    for row in bad_memories:
        attributions = int(row.get("attributions") or 0)
        used_rate = float(row.get("used_rate") or 0.0)
        decided_count = int(row.get("decided_count") or 0)
        pass_rate = row.get("pass_rate")
        if decided_count < MIN_DECIDED_VERDICTS or pass_rate is None:
            continue
        pass_rate = float(pass_rate)
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
                        "decided_count": decided_count,
                        "pass_rate": pass_rate,
                    },
                    "summary": (
                        f"[auto] memory {memory_id} used in {attributions} "
                        f"attributions (used_rate={used_rate:.2f}) but "
                        f"{decided_count} decided sessions pass at "
                        f"{pass_rate:.2f}"
                    ),
                    "proposed_content": None,
                }
            )
    return detections


def _detect_stale_clusters(
    stale_clusters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """(c) A memory served stale (curator-lag window) >= :data:`STALE_SERVE_MIN_ROWS`
    times — evidence the belief itself has drifted and should be retired
    rather than kept serving a known-stale answer.

    ``kind='retire'``: retiring is a retraction (:func:`shared.memory.retract_memory`),
    never a supersede — there is no ``proposed_content`` for a retraction,
    and this kind never auto-applies regardless of scope (see module
    docstring).
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
                "proposed_content": None,
            }
        )
    return detections


def _extract_json_object(content: Optional[str]) -> Optional[dict[str, Any]]:
    """Best-effort JSON object extraction from an LLM's raw text response.

    Local copy of ``shared.memory``'s own ``_extract_json`` shape (try a
    straight parse, fall back to the first ``{...}`` span) — duplicated
    rather than imported because it is a private helper of another module;
    see the module docstring's "copy the pattern" note re: the curator's
    LLM-call config, same rationale here.
    """
    if not content:
        return None
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        pass
    match = _JSON_OBJECT_RE.search(content)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _draft_diff(detection: dict[str, Any]) -> str:
    """Synchronous LLM prose pass over ALREADY-computed detector evidence.

    ``diff`` is reviewer-facing rationale ONLY — never written to
    ``session_memories`` (see module docstring). Uses ``litellm.completion``
    (sync), the same sync-call pattern ``shared.memory._embed_text_sync``
    established for the CAS write path, and the SAME env-driven config the
    curator's own LLM calls use (``VERIFICATION_MODEL`` /
    ``VERIFICATION_TIMEOUT_S`` / ``LITELLM_API_URL`` / ``LITELLM_API_KEY``) —
    copied here rather than imported, since ``app.curator`` lives in
    mcp-server and this service does not import across service boundaries.

    Raises on any failure (network error, no result, missing ``diff`` field)
    so the caller's ``except Exception`` can fall back to the detector's own
    deterministic ``summary`` — never returns an empty or placeholder diff.
    """
    model = os.environ.get("VERIFICATION_MODEL", _DEFAULT_LLM_MODEL)
    timeout_s = parse_float(
        os.environ.get("VERIFICATION_TIMEOUT_S"),
        default=_DEFAULT_LLM_TIMEOUT_S,
        name="VERIFICATION_TIMEOUT_S",
        minimum=0.1,
        maximum=600.0,
    )
    api_base = os.environ.get("LITELLM_API_URL") or os.environ.get("OPENAI_API_BASE")
    api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")

    prompt = (
        "You are drafting a REVIEW NOTE for a human deciding whether to "
        "approve a proposed memory-store change. Write 1-3 concise "
        "sentences explaining the proposed change, based ONLY on the "
        "evidence below. Do not invent facts, numbers, or session ids not "
        "present here. This note is shown to the reviewer only — it is "
        "never itself stored as memory content.\n\n"
        f"Detector: {detection['detector']}\n"
        f"Proposal kind: {detection['kind']}\n"
        f"Evidence: {json.dumps(detection['evidence'])}\n"
        f"Deterministic summary: {detection['summary']}\n\n"
        'Respond with JSON: {"diff": "<your review note>"}'
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "timeout": timeout_s,
        "temperature": 0.0,
    }
    if api_base:
        kwargs["custom_llm_provider"] = "openai"
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key

    response = litellm.completion(**kwargs)
    content = response.choices[0].message.content
    result = _extract_json_object(content)
    if not result:
        raise RuntimeError("dreamer: LLM drafting returned no parseable result")
    diff = result.get("diff")
    if not diff or not isinstance(diff, str):
        raise RuntimeError("dreamer: LLM drafting returned no diff field")
    return diff


def _target_row(db: Any, *, org_id: str, memory_id: str) -> Optional[dict[str, Any]]:
    """Org-predicated lookup of a proposal target's current scope/agent_id/
    revision/memory_type. Committed immediately after reading (no
    lingering open transaction) — the caller may go on to make a network
    call (drafting, then possibly a CAS apply), and nothing here should
    still be holding this connection's transaction open when that happens.
    """
    row = (
        db.execute(
            text(
                """
                SELECT scope, agent_id, revision, memory_type
                FROM session_memories
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


def _dedup_key(detection: dict[str, Any]) -> str:
    """A synthetic, always-non-NULL dedup key for the partial unique index
    (migration 045, ``ux_memory_proposals_org_detector_dedup_open``).

    ``target_memory_id`` cannot serve this role on its own: Postgres treats
    every ``NULL`` as DISTINCT from every other ``NULL`` under uniqueness,
    and ``tool_failure`` detections always have ``target_memory_id = None``
    (``kind='add'`` has no target) — so an index keyed on that column alone
    never actually deduplicated ``tool_failure`` proposals; every dream pass
    filed a fresh one for the same still-failing tool.

    ``tool_failure``: ``'tool:' + sha256(tool)[:16]`` — the tool name IS
    "the same ongoing pattern" for this detector; hashed (not embedded
    verbatim) to keep the key a short, bounded, opaque token rather than an
    unbounded tool-name string leaking into an index/ON CONFLICT clause.

    Every other detector: ``target_memory_id`` as text — already a correct,
    always-non-NULL dedup key on its own (each finding names exactly one
    memory row).
    """
    if detection["detector"] == "tool_failure":
        tool = detection["evidence"]["tool"]
        digest = hashlib.sha256(tool.encode("utf-8")).hexdigest()[:16]
        return f"tool:{digest}"
    return str(detection["target_memory_id"])


def _insert_proposal(
    db: Any, *, org_id: str, detection: dict[str, Any], diff: str
) -> Optional[str]:
    """Insert one proposal, deduplicated against any already-OPEN proposal
    for the same ``(org_id, detector, dedup_key)`` via the partial unique
    index (migration 045). Returns the new proposal's id, or ``None`` if a
    conflicting open proposal already existed (no insert).
    """
    target_memory_id = detection.get("target_memory_id")
    scope = "org"
    proposal_id = str(uuid.uuid4())
    dedup_key = _dedup_key(detection)
    inserted = db.execute(
        text(
            """
            INSERT INTO memory_proposals
              (id, org_id, scope, kind, detector, target_memory_id, dedup_key,
               diff, proposed_content, evidence, confidence, status)
            VALUES (:id, :org_id, :scope, :kind, :detector, :target_memory_id,
                    :dedup_key, :diff, :proposed_content, :evidence,
                    :confidence, :status)
            ON CONFLICT (org_id, detector, dedup_key)
            WHERE status = 'open'
            DO NOTHING
            RETURNING id
            """
        ),
        {
            "id": proposal_id,
            "org_id": org_id,
            "scope": scope,
            "kind": detection["kind"],
            "detector": detection["detector"],
            "target_memory_id": target_memory_id,
            "dedup_key": dedup_key,
            "diff": diff,
            "proposed_content": detection.get("proposed_content"),
            "evidence": json.dumps(
                {
                    "prevalence": detection["prevalence"],
                    "detector": detection["detector"],
                    **detection["evidence"],
                }
            ),
            "confidence": None,
            "status": "open",
        },
    ).first()
    db.commit()
    return str(inserted[0]) if inserted is not None else None


def dream_for_org(
    db: Any, s3: Any, *, org_id: str, batch_limit: int = DEFAULT_BATCH_LIMIT
) -> int:
    """Run all three deterministic detectors for ``org_id`` and file one
    ``memory_proposals`` row per NEW detection (deduplicated against any
    already-open proposal for the same org/detector/target).

    Returns the number of proposals actually inserted. Detection is
    arithmetic and always runs to completion; an LLM drafting failure
    degrades only that one proposal's ``diff`` rationale to the detector's
    deterministic summary sentence — it never suppresses the proposal.

    Synchronous by design — see the module docstring for why (this must run
    inside ``anyio.to_thread.run_sync``, not on the event loop).

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

        try:
            diff = _draft_diff(detection)
        except Exception:
            logger.warning(
                "dreamer: drafting failed org=%s detector=%s; using deterministic "
                "summary",
                org_id,
                detection["detector"],
                exc_info=True,
            )
            diff = detection["summary"]

        proposal_id = _insert_proposal(
            db, org_id=org_id, detection=detection, diff=diff
        )
        if proposal_id is None:
            # An open proposal for this org/detector/target already exists
            # (partial unique index) — not a new finding, nothing to count.
            continue
        written += 1

        can_auto = (
            detection["kind"] == "revise"
            and detection.get("proposed_content") is not None
            and target is not None
            and target.get("scope") == "agent"
        )
        if can_auto:
            _try_auto_apply(
                db,
                org_id=org_id,
                proposal_id=proposal_id,
                memory_id=target_memory_id,
                base_revision=str(target["revision"]),
                agent_id=target.get("agent_id"),
                memory_type=target["memory_type"],
                proposed_content=detection["proposed_content"],
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
    memory_type: str,
    proposed_content: str,
) -> None:
    """Attempt to apply an ``agent``-scope ``revise`` proposal via the CAS
    path — the ONLY kind/scope combination the dreamer ever auto-applies
    (see module docstring).

    Claims the proposal FIRST via an atomic ``UPDATE ... WHERE status =
    'open'`` (the same claim-before-apply discipline ``app.proposals`` uses
    for human decisions) — closing the double-decide race between this
    auto-apply and a human approving/rejecting the same proposal through
    the REST endpoints in the gap between insert and this call. A lost
    claim (rowcount 0) means a human already decided it; this function
    backs off silently. A won claim that then fails to apply (CAS conflict,
    an exception) is reverted back to ``status='open'`` so the proposal
    stays reviewable rather than stuck ``auto`` with nothing actually
    written.
    """
    claimed = db.execute(
        text(
            """
            UPDATE memory_proposals
            SET status = :status, decided_at = now()
            WHERE id = :id AND org_id = :org_id AND status = 'open'
            RETURNING id
            """
        ),
        {"status": "auto", "id": proposal_id, "org_id": org_id},
    ).first()
    db.commit()
    if claimed is None:
        logger.info(
            "dreamer: auto-apply lost the claim org=%s proposal=%s (already "
            "decided elsewhere)",
            org_id,
            proposal_id,
        )
        return

    try:
        result = supersede_with_revision(
            db,
            org_id=org_id,
            memory_id=memory_id,
            base_revision=base_revision,
            new_content=proposed_content,
            agent_id=agent_id,
            memory_type=memory_type,
            session_id=f"proposal-{proposal_id}",
            scope="agent",
        )
    except Exception:
        logger.warning(
            "dreamer: auto-apply raised org=%s proposal=%s memory=%s",
            org_id,
            proposal_id,
            memory_id,
            exc_info=True,
        )
        result = {"stored": False}

    if result.get("stored"):
        return

    logger.info(
        "dreamer: auto-apply not stored org=%s proposal=%s memory=%s error=%s; "
        "reverting claim to open",
        org_id,
        proposal_id,
        memory_id,
        result.get("error"),
    )
    # Roll back FIRST: we reach here either after supersede_with_revision
    # raised (this session's transaction may hold a half-applied write or
    # sit in a failed/aborted state) or after it returned stored=False
    # (no exception, but rolling back first is still correct — cheap and
    # never wrong). The revert UPDATE must start from a clean transaction.
    db.rollback()
    result_row = db.execute(
        text(
            """
            UPDATE memory_proposals
            SET status = :status, decided_at = NULL
            WHERE id = :id AND org_id = :org_id AND status = 'auto'
            """
        ),
        {"status": "open", "id": proposal_id, "org_id": org_id},
    )
    db.commit()
    if not result_row.rowcount:
        logger.warning(
            "dreamer: auto-apply revert matched no row (already decided "
            "elsewhere) org=%s proposal=%s",
            org_id,
            proposal_id,
        )
