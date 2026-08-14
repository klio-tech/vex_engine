"""REST review surface for the dreamer's proposals (Build 5).

``app.dreamer`` (async-worker) files ``memory_proposals`` rows; this module
is where a human reviews them. Three endpoints, mounted under ``/proposals``:

* ``GET /proposals?status=open&limit=50`` — list this org's proposals,
  filtered by status (default ``open``), capped at ``limit`` (default 50).
* ``POST /proposals/{id}/approve`` — apply the proposal, through the SAME
  primitive every other writer in this service uses for that ``kind``:

  * ``kind='add'`` / ``kind='revise'`` — require ``proposed_content``
    (``diff`` is reviewer-facing rationale ONLY, never memory content; see
    ``app.dreamer``'s module docstring). A ``NULL`` ``proposed_content``
    refuses with 422 ``needs_content`` — nothing to write yet.
  * ``kind='add'`` writes a brand-new memory
    (:func:`shared.brain.write_scoped_memory`).
  * ``kind='revise'`` applies through the CAS path
    (:func:`shared.memory.supersede_with_revision`) against
    ``target_memory_id`` — subject to the identical authorization ladder
    and optimistic-concurrency guard as any other write; a target that
    moved since the dreamer detected it fails with a conflict rather than
    silently overwriting.
  * ``kind='retire'`` retracts (:func:`shared.memory.retract_memory`) —
    NOT a supersede. Superseding an active belief with prose ("this memory
    looks wrong") would leave that meta-commentary AS the belief's new
    content; retiring removes it from recall instead. No content needed or
    accepted.
* ``POST /proposals/{id}/reject`` — marks the proposal decided, no write.

Both decision endpoints require an attributable caller
(:func:`app.identity.resolve_user_id`) and record it in ``decided_by`` — a
decision with no accountable human defeats the point of a review queue.

Every decision CLAIMS the proposal first, via an atomic ``UPDATE ... WHERE
status = 'open'`` (see :func:`_claim_proposal`), before attempting the
underlying write. A lost claim (someone else already decided it, or the
same request retried after a partial failure) is a clean 409 — there is no
window where two concurrent approvals, or an approve racing a reject, can
both believe they own the decision. If the underlying apply then fails
(CAS conflict, retraction refused), the claim is REVERTED back to
``status='open'`` so the proposal stays reviewable rather than stuck
half-decided with nothing actually written.

Auth mirrors ``app.capture``: the same ``X-Vex-Key`` (+ optional
``X-Vex-Agent``) bridge via :func:`app.auth.resolve_principal_dep`, so every
query and write here is scoped to exactly the org the key authorises.
"""

import logging
import uuid
from typing import Any, Optional

import anyio
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from shared import brain
from shared.auth import KeyInfo
from shared.memory import retract_memory, supersede_with_revision
from sqlalchemy import text

from app.auth import resolve_principal_dep, verify_api_key
from app.db import SessionLocal
from app.identity import resolve_user_id

logger = logging.getLogger("agentguard.mcp.proposals")

router = APIRouter(prefix="/proposals", tags=["proposals"])

_DEFAULT_LIST_LIMIT = 50
_MAX_LIST_LIMIT = 200


def _validate_uuid(value: str) -> None:
    """Raise 404 for a non-UUID-shaped id — indistinguishable from an id
    that was never issued, matching :func:`shared.memory.retract_memory`'s
    ``invalid_id`` → 404 convention (see ``app.capture.capture_forget``).
    """
    try:
        uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=404, detail="not_found")


def _list_proposals(
    db: Any, *, org_id: str, status: Optional[str], limit: int
) -> list[dict[str, Any]]:
    query = (
        "SELECT id, org_id, scope, kind, detector, target_memory_id, diff, "
        "proposed_content, evidence, confidence, status, created_at, "
        "decided_at, decided_by "
        "FROM memory_proposals WHERE org_id = :org_id"
    )
    params: dict[str, Any] = {"org_id": org_id, "limit": limit}
    if status:
        query += " AND status = :status"
        params["status"] = status
    query += " ORDER BY created_at DESC LIMIT :limit"
    rows = db.execute(text(query), params).mappings().all()
    db.commit()
    return [dict(r) for r in rows]


def _load_proposal(db: Any, *, org_id: str, proposal_id: str) -> Optional[dict[str, Any]]:
    """Read-only lookup for the initial 404/409 checks and to gather what a
    decision needs to apply (target id, proposed content). NOT the
    concurrency guard — :func:`_claim_proposal` is.
    """
    row = (
        db.execute(
            text(
                """
                SELECT id, org_id, scope, kind, target_memory_id, diff,
                       proposed_content, status
                FROM memory_proposals
                WHERE id = :id AND org_id = :org_id
                """
            ),
            {"id": proposal_id, "org_id": org_id},
        )
        .mappings()
        .first()
    )
    db.commit()
    return dict(row) if row is not None else None


def _claim_proposal(
    db: Any, *, org_id: str, proposal_id: str, status: str, decided_by: str
) -> bool:
    """Atomically transition ``proposal_id`` from ``open`` to ``status``.

    Returns ``True`` if this call won the claim (rowcount 1), ``False`` if
    someone else already decided it (rowcount 0) — the caller's job at that
    point is to refuse with 409, not to proceed. This is the SAME
    CAS-before-apply discipline every write in this schema follows; a
    review decision is a write too.
    """
    result = db.execute(
        text(
            """
            UPDATE memory_proposals
            SET status = :status, decided_at = now(), decided_by = :decided_by
            WHERE id = :id AND org_id = :org_id AND status = 'open'
            """
        ),
        {"status": status, "decided_by": decided_by, "id": proposal_id, "org_id": org_id},
    )
    db.commit()
    return bool(result.rowcount)


def _revert_claim(db: Any, *, org_id: str, proposal_id: str) -> None:
    """Undo a won claim whose underlying apply then failed — the proposal
    goes back to ``open`` (decision cleared) rather than sitting decided
    with nothing actually written.

    Rolls back FIRST: the caller reaches this after an exception raised
    mid-apply (e.g. inside ``supersede_with_revision``'s own CAS write),
    which may have left this session's transaction holding a half-applied
    write or in a failed/aborted state. Reverting must start from a clean
    transaction, not build the revert UPDATE on top of whatever the failed
    apply left behind.

    The UPDATE itself is status-guarded (``AND status = 'approved'``) so a
    revert can only ever undo the claim IT won — it can never clobber a
    decision some other request already made (e.g. this proposal being
    re-claimed and re-decided between the failed apply and this revert).
    A guarded UPDATE that matches zero rows (someone else already decided
    it) is logged, not treated as an error.
    """
    db.rollback()
    result = db.execute(
        text(
            """
            UPDATE memory_proposals
            SET status = 'open', decided_at = NULL, decided_by = NULL
            WHERE id = :id AND org_id = :org_id AND status = 'approved'
            """
        ),
        {"id": proposal_id, "org_id": org_id},
    )
    db.commit()
    if not result.rowcount:
        logger.warning(
            "_revert_claim: no row reverted (already decided elsewhere) "
            "org=%s proposal=%s",
            org_id, proposal_id,
        )


def _target_row(db: Any, *, org_id: str, memory_id: str) -> Optional[dict[str, Any]]:
    """Includes ``project_id``: a ``scope='project'`` target's revise MUST
    forward it into ``supersede_with_revision`` (``_prepare_cas_write``
    raises ``ValueError`` for a project-scoped write with no
    ``project_id``, which the caller's ``except Exception`` turns into an
    opaque ``redaction_failed`` 409 — this SELECT closes that gap at the
    source rather than leaving the caller to guess).
    """
    row = (
        db.execute(
            text(
                """
                SELECT scope, revision, memory_type, project_id
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


@router.get("")
async def list_proposals(
    status: Optional[str] = Query("open"),
    limit: int = Query(_DEFAULT_LIST_LIMIT, ge=1, le=_MAX_LIST_LIMIT),
    principal: tuple[str, str] = Depends(resolve_principal_dep),
    key_info: KeyInfo = Depends(verify_api_key),
) -> dict[str, Any]:
    """``GET /proposals?status=open&limit=50`` — org-scoped, status-filtered,
    capped."""
    org_id, _agent_id = principal
    db = SessionLocal()
    try:
        rows = await anyio.to_thread.run_sync(
            lambda: _list_proposals(db, org_id=org_id, status=status, limit=limit)
        )
        return {"proposals": rows}
    finally:
        db.close()


@router.post("/{proposal_id}/approve")
async def approve_proposal(
    proposal_id: str,
    request: Request,
    principal: tuple[str, str] = Depends(resolve_principal_dep),
    key_info: KeyInfo = Depends(verify_api_key),
) -> dict[str, Any]:
    """Apply a proposal: ``add``/``revise`` write via ``proposed_content``,
    ``retire`` retracts. See module docstring for the full contract.
    """
    _validate_uuid(proposal_id)
    org_id, agent_id = principal
    user_id = resolve_user_id(request, key_info)
    if not user_id:
        raise HTTPException(status_code=403, detail="attribution_required")

    db = SessionLocal()
    try:
        proposal = await anyio.to_thread.run_sync(
            lambda: _load_proposal(db, org_id=org_id, proposal_id=proposal_id)
        )
        if proposal is None:
            raise HTTPException(status_code=404, detail="not_found")
        if proposal["status"] != "open":
            raise HTTPException(status_code=409, detail="already_decided")

        kind = proposal["kind"]
        if kind in ("add", "revise") and not proposal.get("proposed_content"):
            raise HTTPException(status_code=422, detail={"error": "needs_content"})

        target = None
        if kind in ("revise", "retire"):
            if not proposal["target_memory_id"]:
                raise HTTPException(status_code=422, detail="invalid_proposal")
            if kind == "revise":
                target = await anyio.to_thread.run_sync(
                    lambda: _target_row(
                        db, org_id=org_id,
                        memory_id=str(proposal["target_memory_id"]),
                    )
                )
                if target is None:
                    raise HTTPException(status_code=409, detail="target_not_found")
        elif kind != "add":
            raise HTTPException(status_code=422, detail="invalid_proposal")

        claimed = await anyio.to_thread.run_sync(
            lambda: _claim_proposal(
                db, org_id=org_id, proposal_id=proposal_id,
                status="approved", decided_by=user_id,
            )
        )
        if not claimed:
            raise HTTPException(status_code=409, detail="already_decided")

        if kind == "add":
            try:
                result = await brain.write_scoped_memory(
                    db,
                    org_id=org_id,
                    agent_id=agent_id,
                    content=proposal["proposed_content"],
                    memory_type="fact",
                    session_id=f"proposal-{proposal_id}",
                    scope=proposal["scope"],
                    user_id=user_id if proposal["scope"] == "private" else None,
                    provenance="INFERRED",
                )
            except Exception:
                logger.error(
                    "approve_proposal: write_scoped_memory raised org=%s "
                    "proposal=%s; reverting claim",
                    org_id, proposal_id, exc_info=True,
                )
                await anyio.to_thread.run_sync(
                    lambda: _revert_claim(db, org_id=org_id, proposal_id=proposal_id)
                )
                raise
            if not result.get("stored") and not result.get("deduplicated"):
                await anyio.to_thread.run_sync(
                    lambda: _revert_claim(db, org_id=org_id, proposal_id=proposal_id)
                )
                raise HTTPException(status_code=422, detail=result)
            return {"status": "approved", "kind": "add", "result": result}

        if kind == "revise":
            target_memory_id = str(proposal["target_memory_id"])
            try:
                apply_result = await anyio.to_thread.run_sync(
                    lambda: supersede_with_revision(
                        db,
                        org_id=org_id,
                        memory_id=target_memory_id,
                        base_revision=str(target["revision"]),
                        new_content=proposal["proposed_content"],
                        memory_type=target["memory_type"],
                        session_id=f"proposal-{proposal_id}",
                        user_id=user_id,
                        agent_id=agent_id,
                        scope=target["scope"],
                        project_id=target.get("project_id"),
                    )
                )
            except Exception:
                # An exception here happens AFTER _claim_proposal already
                # committed the proposal to 'approved' — without this revert
                # the proposal would be stuck permanently decided with
                # nothing actually written (unreviewable, unretriable).
                # Mirrors app.dreamer._try_auto_apply's own revert-on-failure
                # discipline for the identical failure mode.
                logger.error(
                    "approve_proposal: supersede_with_revision raised org=%s "
                    "proposal=%s memory=%s; reverting claim",
                    org_id, proposal_id, target_memory_id, exc_info=True,
                )
                await anyio.to_thread.run_sync(
                    lambda: _revert_claim(db, org_id=org_id, proposal_id=proposal_id)
                )
                raise
            if not apply_result.get("stored"):
                await anyio.to_thread.run_sync(
                    lambda: _revert_claim(db, org_id=org_id, proposal_id=proposal_id)
                )
                # Align with retire's error-code mapping: "forbidden" (the
                # scope-authorization ladder inside supersede_with_revision
                # refused the caller) is a 403, never a 409 — a 409 implies
                # a retriable conflict, but re-approving will not change
                # who the caller is.
                error = apply_result.get("error")
                status_code = 403 if error == "forbidden" else 409
                raise HTTPException(status_code=status_code, detail=apply_result)
            return {"status": "approved", "kind": "revise", "result": apply_result}

        # kind == "retire": retraction, never a supersede — see module docstring.
        target_memory_id = str(proposal["target_memory_id"])
        try:
            retract_result = await anyio.to_thread.run_sync(
                lambda: retract_memory(
                    db,
                    org_id=org_id,
                    agent_id=agent_id,
                    memory_id=target_memory_id,
                    via="dreamer-approved",
                    user_id=user_id,
                )
            )
        except Exception:
            logger.error(
                "approve_proposal: retract_memory raised org=%s proposal=%s "
                "memory=%s; reverting claim",
                org_id, proposal_id, target_memory_id, exc_info=True,
            )
            await anyio.to_thread.run_sync(
                lambda: _revert_claim(db, org_id=org_id, proposal_id=proposal_id)
            )
            raise
        if not retract_result.get("forgotten"):
            await anyio.to_thread.run_sync(
                lambda: _revert_claim(db, org_id=org_id, proposal_id=proposal_id)
            )
            error = retract_result.get("error")
            status_code = 403 if error == "forbidden" else 409
            raise HTTPException(status_code=status_code, detail=error)
        return {"status": "approved", "kind": "retire", "result": retract_result}
    finally:
        db.close()


@router.post("/{proposal_id}/reject")
async def reject_proposal(
    proposal_id: str,
    request: Request,
    principal: tuple[str, str] = Depends(resolve_principal_dep),
    key_info: KeyInfo = Depends(verify_api_key),
) -> dict[str, Any]:
    """Mark a proposal rejected. Same attribution/claim-before-decide
    contract as approve, but never writes to memory — a reject is purely a
    record.
    """
    _validate_uuid(proposal_id)
    org_id, _agent_id = principal
    user_id = resolve_user_id(request, key_info)
    if not user_id:
        raise HTTPException(status_code=403, detail="attribution_required")

    db = SessionLocal()
    try:
        proposal = await anyio.to_thread.run_sync(
            lambda: _load_proposal(db, org_id=org_id, proposal_id=proposal_id)
        )
        if proposal is None:
            raise HTTPException(status_code=404, detail="not_found")
        if proposal["status"] != "open":
            raise HTTPException(status_code=409, detail="already_decided")

        claimed = await anyio.to_thread.run_sync(
            lambda: _claim_proposal(
                db, org_id=org_id, proposal_id=proposal_id,
                status="rejected", decided_by=user_id,
            )
        )
        if not claimed:
            raise HTTPException(status_code=409, detail="already_decided")

        return {"status": "rejected"}
    finally:
        db.close()
