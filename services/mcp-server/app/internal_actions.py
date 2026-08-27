"""Internal branch-merge and proposal-review actions for the workspace app.

The read-only status endpoints live in :mod:`app.internal_api`; this module
is the WRITE half of the same surface — merge a context branch, and decide
the merge's review proposals. Same guard (``X-Klio-Internal``), same 404
refusal shape, same per-request secret read.

Merge semantics ("the GitHub model", 2026-08 design call):

* similarity ≥ :data:`shared.memory.DEDUP_THRESHOLD` — auto-resolved:
  :func:`shared.branches.merge_branch` supersedes the main row (the branch
  version is the newer belief).
* :data:`PROPOSAL_BAND_LOW` ≤ similarity < ``DEDUP_THRESHOLD`` — a human
  decision, filed BEFORE the merge as a ``memory_proposals`` row with
  ``kind='retire'`` targeting the OLD MAIN row. Both rows live on main
  after the merge; approving retires the old belief (take the branch's),
  rejecting keeps both. Retire composes with existing primitives and can
  never duplicate content, which is why the proposal is not a ``revise``.
* below the band — unrelated; both rows simply coexist on main.

``decided_by`` / ``merged_by`` carry the workspace user id the Django
proxy authenticated; the workspace's member gate is the authorization,
exactly as the capture bridge trusts the workspace on attribution.
"""

import json
import logging
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends
from sqlalchemy import text

from app.db import get_db
from app.internal_api import require_internal_secret
from app.proposals import (
    _claim_proposal,
    _list_proposals,
    _load_proposal,
    _revert_claim,
)

logger = logging.getLogger("agentguard.mcp.internal_actions")

#: The review band's floor; its ceiling is the dedup threshold.
PROPOSAL_BAND_LOW = 0.80

#: Band pairs: branch rows whose nearest main row is similar enough to be
#: suspicious but not similar enough for the automatic supersede.
_BAND_PAIRS_SQL = text(
    """
    SELECT bm.id AS branch_memory_id,
           bm.content AS branch_content,
           nearest.id AS main_memory_id,
           nearest.content AS main_content,
           nearest.similarity
    FROM session_memories bm
    JOIN LATERAL (
        SELECT mm.id, mm.content, 1 - (mm.embedding <=> bm.embedding) AS similarity
        FROM session_memories mm
        WHERE mm.org_id = :org_id
          AND mm.project_id = CAST(:project_id AS uuid)
          AND mm.branch_id IS NULL
          AND mm.status = 'active'
          AND mm.embedding IS NOT NULL
          AND 1 - (mm.embedding <=> bm.embedding) >= :band_low
          AND 1 - (mm.embedding <=> bm.embedding) < :threshold
        ORDER BY mm.embedding <=> bm.embedding
        LIMIT 1
    ) nearest ON TRUE
    WHERE bm.org_id = :org_id
      AND bm.branch_id = CAST(:branch_id AS uuid)
      AND bm.status = 'active'
      AND bm.embedding IS NOT NULL
    ORDER BY bm.created_at, bm.id
    """
)

_INSERT_PROPOSAL_SQL = text(
    """
    INSERT INTO memory_proposals
      (id, org_id, scope, kind, detector, target_memory_id, dedup_key,
       diff, proposed_content, evidence, confidence, status)
    VALUES (:id, :org_id, 'org', 'retire', 'branch_merge', :target_memory_id,
            :dedup_key, :diff, NULL, :evidence, :confidence, 'open')
    ON CONFLICT (org_id, detector, dedup_key)
    WHERE status = 'open'
    DO NOTHING
    RETURNING id
    """
)

#: Mirrors shared.memory.retract_memory's metadata discipline, applied on
#: the internal trusted path (see proposal_decide for why the ladder is
#: not used here).
_RETRACT_TARGET_SQL = text(
    """
    UPDATE session_memories
    SET status = 'retracted',
        metadata = COALESCE(metadata, '{}'::jsonb)
                   || jsonb_build_object(
                       'retracted_by', CAST(:decided_by AS text),
                       'retracted_at', NOW()::text,
                       'retracted_via', 'branch_merge_review'
                   ),
        updated_at = NOW()
    WHERE org_id = :org_id
      AND id = CAST(:memory_id AS uuid)
      AND status = 'active'
    """
)

_LINKED_PROJECT_SQL = text(
    """
    SELECT id, org_id
    FROM projects
    WHERE plane_project_id = CAST(:plane_project_id AS uuid)
    LIMIT 1
    """
)


router = APIRouter(
    prefix="/internal",
    include_in_schema=False,
    dependencies=[Depends(require_internal_secret)],
)


def _linked_project(db: Any, plane_project_id: uuid.UUID) -> Optional[Any]:
    return db.execute(
        _LINKED_PROJECT_SQL, {"plane_project_id": str(plane_project_id)}
    ).fetchone()


def file_band_proposals(
    db: Any, *, org_id: str, project_id: str, branch_id: str
) -> int:
    """File one retire-proposal per review-band pair; returns how many."""
    from shared.memory import DEDUP_THRESHOLD  # noqa: PLC0415 -- embedding stack

    pairs = db.execute(
        _BAND_PAIRS_SQL,
        {
            "org_id": org_id,
            "project_id": project_id,
            "branch_id": branch_id,
            "band_low": PROPOSAL_BAND_LOW,
            "threshold": DEDUP_THRESHOLD,
        },
    ).fetchall()

    filed = 0
    for pair in pairs:
        similarity = float(pair.similarity)
        diff = (
            "Branch merge found a likely conflict "
            f"(similarity {similarity:.2f}).\n\n"
            f"KEEP (approve retires this): {pair.main_content}\n\n"
            f"NEW (already merged to main): {pair.branch_content}"
        )
        inserted = db.execute(
            _INSERT_PROPOSAL_SQL,
            {
                "id": str(uuid.uuid4()),
                "org_id": org_id,
                "target_memory_id": str(pair.main_memory_id),
                "dedup_key": f"{pair.branch_memory_id}:{pair.main_memory_id}",
                "diff": diff,
                "evidence": json.dumps(
                    {
                        "branch_memory_id": str(pair.branch_memory_id),
                        "branch_id": branch_id,
                        "similarity": similarity,
                    }
                ),
                "confidence": similarity,
            },
        ).fetchone()
        if inserted is not None:
            filed += 1
    db.commit()
    return filed


@router.post("/branch-merge")
def branch_merge(
    workspace_project: uuid.UUID = Body(...),
    branch_id: uuid.UUID = Body(...),
    merged_by: Optional[str] = Body(default=None),
    db: Any = Depends(get_db),
) -> dict:
    """Merge one context branch, filing review proposals for the band first."""
    from shared.branches import merge_branch  # noqa: PLC0415

    project = _linked_project(db, workspace_project)
    if project is None:
        return {"merged": False, "reason": "not_linked"}

    proposals = file_band_proposals(
        db,
        org_id=str(project.org_id),
        project_id=str(project.id),
        branch_id=str(branch_id),
    )
    result = merge_branch(
        db,
        org_id=str(project.org_id),
        project_id=str(project.id),
        branch_id=str(branch_id),
        merged_by=merged_by,
    )
    result["proposals_filed"] = proposals
    return result


@router.get("/merge-proposals")
def merge_proposals(workspace_project: uuid.UUID, db: Any = Depends(get_db)) -> dict:
    """Open branch-merge proposals for the org of this linked project."""
    project = _linked_project(db, workspace_project)
    if project is None:
        return {"linked": False, "proposals": []}
    rows = _list_proposals(db, org_id=str(project.org_id), status="open", limit=100)
    proposals = [
        {
            "id": str(row["id"]),
            "target_memory_id": str(row["target_memory_id"])
            if row["target_memory_id"]
            else None,
            "diff": row["diff"],
            "confidence": float(row["confidence"])
            if row["confidence"] is not None
            else None,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }
        for row in rows
        if row["detector"] == "branch_merge"
    ]
    return {"linked": True, "proposals": proposals}


@router.post("/proposal-decide")
def proposal_decide(
    workspace_project: uuid.UUID = Body(...),
    proposal_id: uuid.UUID = Body(...),
    action: str = Body(...),
    decided_by: Optional[str] = Body(default=None),
    db: Any = Depends(get_db),
) -> dict:
    """Approve (retire the old belief) or reject (keep both) one proposal."""
    if action not in {"approve", "reject"}:
        return {"decided": False, "reason": "unknown_action"}

    project = _linked_project(db, workspace_project)
    if project is None:
        return {"decided": False, "reason": "not_linked"}
    org_id = str(project.org_id)

    proposal = _load_proposal(db, org_id=org_id, proposal_id=str(proposal_id))
    if proposal is None or proposal["status"] != "open":
        return {"decided": False, "reason": "not_open"}
    if proposal["kind"] != "retire" or not proposal["target_memory_id"]:
        return {"decided": False, "reason": "unsupported_kind"}

    status = "approved" if action == "approve" else "rejected"
    claimed = _claim_proposal(
        db,
        org_id=org_id,
        proposal_id=str(proposal_id),
        status=status,
        decided_by=decided_by or "workspace",
    )
    if not claimed:
        return {"decided": False, "reason": "already_decided"}

    if action == "approve":
        # Authorization happened in the workspace (the Django proxy member-
        # gates by plane project role); the engine-side retract ladder keys
        # off ENGINE identities and would refuse a workspace uuid. This is
        # the same trust the capture bridge extends on attribution, so the
        # retraction is applied directly, with the same audit metadata
        # discipline as every other retraction path.
        try:
            result = db.execute(
                _RETRACT_TARGET_SQL,
                {
                    "org_id": org_id,
                    "memory_id": str(proposal["target_memory_id"]),
                    "decided_by": decided_by or "workspace",
                },
            )
            db.commit()
            if not result.rowcount:
                _revert_claim(db, org_id=org_id, proposal_id=str(proposal_id))
                return {"decided": False, "reason": "target_gone"}
        except Exception:
            try:
                db.rollback()
            except Exception:
                logger.debug("proposal-decide: rollback also failed")
            _revert_claim(db, org_id=org_id, proposal_id=str(proposal_id))
            logger.warning(
                "proposal-decide: retract failed for %s; claim reverted",
                proposal_id,
                exc_info=True,
            )
            return {"decided": False, "reason": "apply_failed"}

    return {"decided": True, "status": status}


# ---------------------------------------------------------------------------
# Agent turns
# ---------------------------------------------------------------------------


@router.post("/agent-turn")
async def agent_turn(
    payload: dict[str, Any] = Body(...),
    _: None = Depends(require_internal_secret),
) -> dict:
    """Queue one agent turn.

    The workspace decides WHETHER an agent was summoned — it owns mentions
    and knows which member is an agent. This endpoint decides nothing; it
    validates the shape and puts the turn on the stream the worker reads.
    Keeping the judgement on one side and the execution on the other is
    what stops the two services growing a second, disagreeing copy of the
    rule about who gets woken.

    Answers 200 with ``queued: false`` rather than an error when the
    payload is unusable: the caller is a fire-and-forget task that must not
    retry, and a 4xx would only produce noise in its logs.
    """
    from shared.agent_turns import build_turn, publish_turn  # noqa: PLC0415

    required = ("workspace_slug", "channel_id", "message_id", "agent_id")
    missing = [key for key in required if not str(payload.get(key) or "").strip()]
    if missing:
        logger.warning("agent turn rejected; missing %s", ", ".join(missing))
        return {"queued": False, "reason": "incomplete"}

    turn = build_turn(
        workspace_slug=str(payload["workspace_slug"]).strip(),
        channel_id=str(payload["channel_id"]).strip(),
        message_id=str(payload["message_id"]).strip(),
        agent_id=str(payload["agent_id"]).strip(),
    )
    try:
        await publish_turn(turn)
    except Exception:
        # Loud in the log, quiet to the caller: the message it came from is
        # already sent and must not be affected by our queue being down.
        logger.exception("could not queue agent turn for %s", turn["agent_id"])
        return {"queued": False, "reason": "queue_unavailable"}

    logger.info(
        "queued agent turn: agent=%s channel=%s", turn["agent_id"], turn["channel_id"]
    )
    return {"queued": True}
