"""REST review surface for the dreamer's proposals (Build 5).

``app.dreamer`` (async-worker) files ``memory_proposals`` rows; this module
is where a human reviews them. Three endpoints, mounted under ``/proposals``:

* ``GET /proposals?status=open`` — list this org's proposals, filtered by
  status (default ``open``).
* ``POST /proposals/{id}/approve`` — apply the proposal:
  ``kind='add'`` writes a brand-new memory through the normal write path
  (:func:`shared.brain.write_scoped_memory`); ``kind='revise'``/``'retire'``
  with a ``target_memory_id`` apply through the same CAS path every other
  writer uses (:func:`shared.memory.supersede_with_revision`), so a human
  approving a proposal is subject to the identical authorization ladder and
  optimistic-concurrency guard as any other write — approving a proposal
  against a ``private`` row you cannot see still fails with ``not_found``,
  and a target that moved since the dreamer detected it still fails with a
  conflict rather than silently overwriting.
* ``POST /proposals/{id}/reject`` — marks the proposal decided, no write.

Both decision endpoints require an attributable caller
(:func:`app.identity.resolve_user_id`) and record it in ``decided_by`` — a
decision with no accountable human defeats the point of a review queue.
Auth mirrors ``app.capture``: the same ``X-Vex-Key`` (+ optional
``X-Vex-Agent``) bridge via :func:`app.auth.resolve_principal_dep`, so every
query and write here is scoped to exactly the org the key authorises.
Nothing here writes to ``session_memories`` directly — approval always goes
through :func:`shared.brain.write_scoped_memory` or
:func:`shared.memory.supersede_with_revision`, the same two primitives
every other writer in this service uses.
"""

import logging
import uuid
from typing import Any, Optional

import anyio
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from shared import brain
from shared.auth import KeyInfo
from shared.memory import supersede_with_revision
from sqlalchemy import text

from app.auth import resolve_principal_dep, verify_api_key
from app.db import SessionLocal
from app.identity import resolve_user_id

logger = logging.getLogger("agentguard.mcp.proposals")

router = APIRouter(prefix="/proposals", tags=["proposals"])

_APPLICABLE_KINDS = ("revise", "retire")


def _validate_uuid(value: str) -> None:
    """Raise 404 for a non-UUID-shaped id — indistinguishable from an id
    that was never issued, matching :func:`shared.memory.retract_memory`'s
    ``invalid_id`` → 404 convention (see ``app.capture.capture_forget``).
    """
    try:
        uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=404, detail="not_found")


def _list_proposals(db: Any, *, org_id: str, status: Optional[str]) -> list[dict[str, Any]]:
    query = (
        "SELECT id, org_id, scope, kind, target_memory_id, diff, evidence, "
        "confidence, status, created_at, decided_at, decided_by "
        "FROM memory_proposals WHERE org_id = :org_id"
    )
    params: dict[str, Any] = {"org_id": org_id}
    if status:
        query += " AND status = :status"
        params["status"] = status
    query += " ORDER BY created_at DESC"
    rows = db.execute(text(query), params).mappings().all()
    db.commit()
    return [dict(r) for r in rows]


def _load_proposal(db: Any, *, org_id: str, proposal_id: str) -> Optional[dict[str, Any]]:
    row = (
        db.execute(
            text(
                """
                SELECT id, org_id, scope, kind, target_memory_id, diff, status
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


def _target_row(db: Any, *, org_id: str, memory_id: str) -> Optional[dict[str, Any]]:
    row = (
        db.execute(
            text(
                """
                SELECT scope, revision FROM session_memories
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


def _mark_decided(
    db: Any, *, org_id: str, proposal_id: str, status: str, decided_by: str
) -> None:
    db.execute(
        text(
            """
            UPDATE memory_proposals
            SET status = :status, decided_at = now(), decided_by = :decided_by
            WHERE id = :id AND org_id = :org_id
            """
        ),
        {"status": status, "decided_by": decided_by, "id": proposal_id, "org_id": org_id},
    )
    db.commit()


@router.get("")
async def list_proposals(
    status: Optional[str] = Query("open"),
    principal: tuple[str, str] = Depends(resolve_principal_dep),
    key_info: KeyInfo = Depends(verify_api_key),
) -> dict[str, Any]:
    """``GET /proposals?status=open`` — org-scoped, optionally status-filtered."""
    org_id, _agent_id = principal
    db = SessionLocal()
    try:
        rows = await anyio.to_thread.run_sync(
            lambda: _list_proposals(db, org_id=org_id, status=status)
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
    """Apply a proposal: ``add`` writes a new memory, ``revise``/``retire``
    apply through the CAS path against ``target_memory_id``.

    404 for an unknown/cross-org/malformed id. 403 if the caller cannot be
    attributed to a person (``decided_by`` would otherwise be unattributable
    — the entire point of a review queue). 409 if the proposal was already
    decided, or if the CAS guard finds the target moved since the dreamer
    detected it (someone else's write raced this approval).
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

        if proposal["kind"] == "add":
            result = await brain.write_scoped_memory(
                db,
                org_id=org_id,
                agent_id=agent_id,
                content=proposal["diff"],
                memory_type="fact",
                session_id=f"dreamer:{proposal_id}",
                scope=proposal["scope"],
                user_id=user_id if proposal["scope"] == "private" else None,
                provenance="INFERRED",
            )
            if not result.get("stored") and not result.get("deduplicated"):
                raise HTTPException(status_code=422, detail=result)
            await anyio.to_thread.run_sync(
                lambda: _mark_decided(
                    db, org_id=org_id, proposal_id=proposal_id,
                    status="approved", decided_by=user_id,
                )
            )
            return {"status": "approved", "kind": "add", "result": result}

        if proposal["kind"] in _APPLICABLE_KINDS and proposal["target_memory_id"]:
            target_memory_id = str(proposal["target_memory_id"])
            target = await anyio.to_thread.run_sync(
                lambda: _target_row(db, org_id=org_id, memory_id=target_memory_id)
            )
            if target is None:
                raise HTTPException(status_code=409, detail="target_not_found")

            apply_result = await anyio.to_thread.run_sync(
                lambda: supersede_with_revision(
                    db,
                    org_id=org_id,
                    memory_id=target_memory_id,
                    base_revision=str(target["revision"]),
                    new_content=proposal["diff"],
                    user_id=user_id,
                    agent_id=agent_id,
                    scope=target["scope"],
                )
            )
            if not apply_result.get("stored"):
                raise HTTPException(status_code=409, detail=apply_result)

            await anyio.to_thread.run_sync(
                lambda: _mark_decided(
                    db, org_id=org_id, proposal_id=proposal_id,
                    status="approved", decided_by=user_id,
                )
            )
            return {"status": "approved", "kind": proposal["kind"], "result": apply_result}

        raise HTTPException(status_code=422, detail="invalid_proposal")
    finally:
        db.close()


@router.post("/{proposal_id}/reject")
async def reject_proposal(
    proposal_id: str,
    request: Request,
    principal: tuple[str, str] = Depends(resolve_principal_dep),
    key_info: KeyInfo = Depends(verify_api_key),
) -> dict[str, Any]:
    """Mark a proposal rejected. Same attribution/decided-once contract as
    approve, but never writes to memory — a reject is purely a record.
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

        await anyio.to_thread.run_sync(
            lambda: _mark_decided(
                db, org_id=org_id, proposal_id=proposal_id,
                status="rejected", decided_by=user_id,
            )
        )
        return {"status": "rejected"}
    finally:
        db.close()
