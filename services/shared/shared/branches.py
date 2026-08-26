"""The context-branch registry — a branch's lifecycle lives in a row.

A context branch is a project's draft space: work accumulates on the branch,
main stays clean, and merge/discard (Task C3) decide what main ever sees.
``main`` itself is deliberately NOT a row here: ``session_memories.branch_id
IS NULL`` **is** main, so a registry row named main would fork the meaning of
NULL — the name is reserved and refused.

The registry's one structural invariant is enforced by migration 050's
partial unique index ``uq_context_branches_open_name``: one OPEN branch per
``(org_id, project_id, name)``, while a merged/discarded name is reusable,
exactly like git. :func:`ensure_branch` upserts *through* that index —
``ON CONFLICT`` with the ``WHERE status = 'open'`` inference clause — so two
concurrent ensures land on the same row instead of one of them aborting.

Every query is filtered by ``org_id``. A ``branch_id`` or ``project_id`` is a
UUID a caller may have obtained from anywhere, so neither is ever trusted on
its own to identify a tenant's branch (the same rule as :mod:`shared.projects`).
Membership checks are the CALLER's job — this module is storage, and the
visibility ladder's project-membership arm is what gates who may name a branch.

Synchronous (blocking psycopg2 I/O), matching the house pattern in
:mod:`shared.projects`. Call from async handlers via
``anyio.to_thread.run_sync``.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger("agentguard.shared.branches")

#: Migration 050 sizes ``context_branches.name`` at 200.
MAX_BRANCH_NAME_LENGTH = 200

#: ``main`` is the absence of a branch (``branch_id IS NULL``); a registry row
#: by that name would give NULL two meanings. Compared case-insensitively.
RESERVED_BRANCH_NAME = "main"

#: Upsert through the partial unique index. The ``WHERE status = 'open'``
#: inference clause is REQUIRED: without it Postgres cannot match the partial
#: index as the conflict arbiter and refuses the statement. The DO UPDATE is
#: a deliberate no-op (name := same name) whose only purpose is RETURNING the
#: existing open row, so a repeated ensure is idempotent.
_ENSURE_BRANCH_SQL = text(
    """
    INSERT INTO context_branches
        (org_id, project_id, name, created_by, work_item_id)
    VALUES (:org_id, :project_id, :name, :created_by, :work_item_id)
    ON CONFLICT (org_id, project_id, name) WHERE status = 'open'
        DO UPDATE SET name = EXCLUDED.name
    RETURNING id, org_id, project_id, name, status, created_by,
              created_at, closed_at, work_item_id
    """
)

_GET_BRANCH_SQL = text(
    """
    SELECT id, org_id, project_id, name, status, created_by,
           created_at, closed_at, work_item_id
    FROM context_branches
    WHERE org_id = :org_id
      AND project_id = :project_id
      AND name = :name
      AND status = 'open'
    LIMIT 1
    """
)

_GET_BRANCH_BY_ID_SQL = text(
    """
    SELECT id, org_id, project_id, name, status, created_by,
           created_at, closed_at, work_item_id
    FROM context_branches
    WHERE org_id = :org_id
      AND id = :branch_id
    LIMIT 1
    """
)

#: ``unmerged_count`` (active memories on the branch) and ``last_activity``
#: (their max created_at) are the stale-branch surfacing from the spec,
#: computed here rather than stored — a stored counter would be a cache with
#: no invalidation. The join repeats ``org_id`` so a (theoretical) branch-id
#: collision across tenants can never cross-count.
_LIST_BRANCHES_SQL = text(
    """
    SELECT b.id, b.org_id, b.project_id, b.name, b.status, b.created_by,
           b.created_at, b.closed_at, b.work_item_id,
           COUNT(m.id) FILTER (WHERE m.status = 'active') AS unmerged_count,
           MAX(m.created_at) FILTER (WHERE m.status = 'active') AS last_activity
    FROM context_branches b
    LEFT JOIN session_memories m
        ON m.branch_id = b.id
       AND m.org_id = b.org_id
    WHERE b.org_id = :org_id
      AND b.project_id = :project_id
      AND (:work_item_id IS NULL OR b.work_item_id = CAST(:work_item_id AS uuid))
    GROUP BY b.id, b.org_id, b.project_id, b.name, b.status, b.created_by,
             b.created_at, b.closed_at, b.work_item_id
    ORDER BY b.created_at DESC, b.name
    """
)


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _validated_name(name: Optional[str]) -> str:
    """The stripped branch name, or a ValueError explaining the refusal."""
    cleaned = _clean(name)
    if cleaned is None:
        raise ValueError("a branch needs a non-blank name")
    if len(cleaned) > MAX_BRANCH_NAME_LENGTH:
        raise ValueError(f"branch name exceeds {MAX_BRANCH_NAME_LENGTH} characters")
    if cleaned.lower() == RESERVED_BRANCH_NAME:
        raise ValueError(
            "'main' is not a branch: it is the absence of one "
            "(branch_id IS NULL). Pick another name."
        )
    return cleaned


def _row_to_branch(row: Any) -> dict:
    return {
        "id": str(row.id),
        "org_id": row.org_id,
        "project_id": str(row.project_id),
        "name": row.name,
        "status": row.status,
        "created_by": row.created_by,
        "created_at": row.created_at,
        "closed_at": row.closed_at,
        "work_item_id": (
            str(row.work_item_id)
            if getattr(row, "work_item_id", None) is not None
            else None
        ),
    }


def ensure_branch(
    db_session: Any,
    *,
    org_id: str,
    project_id: str,
    name: str,
    created_by: Optional[str],
    work_item_id: Optional[str] = None,
) -> dict:
    """The open branch by this name, created if absent. Idempotent.

    Upserts through ``uq_context_branches_open_name`` (see the module
    docstring), so a repeated ensure — or two concurrent ones — returns the
    one existing open row rather than raising. A merged or discarded branch
    does NOT block the name: the partial index only covers open rows, so
    ensuring a previously-merged name mints a fresh branch, exactly like git.

    Args:
        db_session: A synchronous SQLAlchemy session/connection exposing
            ``.execute()`` and ``.commit()``.
        org_id: Owning org id (tenant scope). Always enforced.
        project_id: The project the branch belongs to. A branch always
            belongs to a project — the caller has already resolved one and
            checked the author may write there.
        name: The branch name. Stripped; blank, over-long (> 200) and
            ``main`` (case-insensitive) are refused.
        created_by: The resolved human who created the branch, or ``None``
            for an unattributable caller. Recorded on first creation only —
            an ensure that lands on an existing row never reassigns it.
        work_item_id: Optional work item this branch is the feature branch
            for (migration 051). Like ``created_by`` it is recorded on
            CREATION only: the ``DO UPDATE`` deliberately does not touch it,
            so an ensure that lands on an existing open branch never
            re-points it at a different work item. The branch still belongs
            to its PROJECT — merge promotes to project main, unchanged.

    Returns:
        The branch dict: ``{"id", "org_id", "project_id", "name", "status",
        "created_by", "created_at", "closed_at"}``.

    Raises:
        ValueError: If ``org_id`` or ``project_id`` is blank, or the name
            fails validation.
    """
    cleaned_org = _clean(org_id)
    cleaned_project = _clean(project_id)
    if cleaned_org is None or cleaned_project is None:
        raise ValueError("ensure_branch needs an org_id and a project_id")
    cleaned_name = _validated_name(name)

    row = db_session.execute(
        _ENSURE_BRANCH_SQL,
        {
            "org_id": cleaned_org,
            "project_id": cleaned_project,
            "name": cleaned_name,
            "created_by": _clean(created_by),
            "work_item_id": _clean(work_item_id),
        },
    ).first()
    db_session.commit()
    return _row_to_branch(row)


def get_branch(
    db_session: Any, *, org_id: str, project_id: str, name: str
) -> Optional[dict]:
    """The OPEN branch by this name, or ``None``.

    Open only, deliberately: a caller naming a branch means the live one, and
    resolving a merged/discarded row here would route new work onto a closed
    branch. Closed rows remain reachable by id via :func:`get_branch_by_id`.
    """
    cleaned_org = _clean(org_id)
    cleaned_project = _clean(project_id)
    cleaned_name = _clean(name)
    if cleaned_org is None or cleaned_project is None or cleaned_name is None:
        return None
    row = db_session.execute(
        _GET_BRANCH_SQL,
        {"org_id": cleaned_org, "project_id": cleaned_project, "name": cleaned_name},
    ).first()
    return _row_to_branch(row) if row is not None else None


def get_branch_by_id(db_session: Any, *, org_id: str, branch_id: str) -> Optional[dict]:
    """The branch with this id in this org, any status, or ``None``.

    Any status because the id path is how merge/discard load the row they are
    about to close (and how an idempotent re-merge discovers it already is).
    A malformed id yields ``None`` rather than a database cast error — the id
    is caller-supplied input, not a trusted reference.
    """
    cleaned_org = _clean(org_id)
    cleaned_id = _clean(branch_id)
    if cleaned_org is None or cleaned_id is None:
        return None
    try:
        uuid.UUID(cleaned_id)
    except ValueError:
        return None
    row = db_session.execute(
        _GET_BRANCH_BY_ID_SQL, {"org_id": cleaned_org, "branch_id": cleaned_id}
    ).first()
    return _row_to_branch(row) if row is not None else None


def list_branches(
    db_session: Any,
    *,
    org_id: str,
    project_id: str,
    work_item_id: Optional[str] = None,
) -> list[dict]:
    """Every branch of this project (all statuses), newest first.

    Each dict carries the registry columns plus two computed fields:

    * ``unmerged_count`` — active memories currently on the branch. This is
      the "is there work parked here?" number; retracted/superseded rows do
      not count.
    * ``last_activity`` — the newest active memory's ``created_at``, or
      ``None`` for a branch nothing has been written to. Together with
      ``unmerged_count`` this is the stale-branch surfacing from the spec.

    ``work_item_id`` narrows the list to one work item's feature branches.
    Omitted (the default) it is not a filter at all — every branch of the
    project answers, which is the pre-051 behaviour.
    """
    cleaned_org = _clean(org_id)
    cleaned_project = _clean(project_id)
    if cleaned_org is None or cleaned_project is None:
        return []
    rows = db_session.execute(
        _LIST_BRANCHES_SQL,
        {
            "org_id": cleaned_org,
            "project_id": cleaned_project,
            "work_item_id": _clean(work_item_id),
        },
    ).fetchall()
    branches = []
    for row in rows:
        branch = _row_to_branch(row)
        branch["unmerged_count"] = int(row.unmerged_count or 0)
        branch["last_activity"] = row.last_activity
        branches.append(branch)
    return branches


# ---------------------------------------------------------------------------
# Merge and discard (Task C3) — the two ways a branch's life ends
# ---------------------------------------------------------------------------

#: Lock the branch row for the duration of the state change. Any status is
#: loaded — idempotence needs to SEE 'merged'/'discarded' to answer
#: already_<status> instead of racing a second state change.
_LOCK_BRANCH_SQL = text(
    """
    SELECT id, status, project_id
    FROM context_branches
    WHERE org_id = :org_id
      AND id = :branch_id
    FOR UPDATE
    """
)

#: The similarity-vs-MAIN pass, entirely over STORED vectors (no embedding
#: service): for each active branch row that has an embedding, the single
#: nearest active MAIN row of the same project at or above the threshold.
#: This is the plan's "query directly" option — ``find_similar_memory`` is
#: shaped around one candidate text that still needs embedding, and merge has
#: N rows whose vectors already exist.
_MERGE_SIMILARITY_SQL = text(
    """
    SELECT bm.id AS branch_memory_id, nearest.id AS main_memory_id
    FROM session_memories bm
    JOIN LATERAL (
        SELECT mm.id
        FROM session_memories mm
        WHERE mm.org_id = :org_id
          AND mm.project_id = :project_id
          AND mm.branch_id IS NULL
          AND mm.status = 'active'
          AND mm.embedding IS NOT NULL
          AND 1 - (mm.embedding <=> bm.embedding) >= :threshold
        ORDER BY mm.embedding <=> bm.embedding
        LIMIT 1
    ) nearest ON TRUE
    WHERE bm.org_id = :org_id
      AND bm.branch_id = :branch_id
      AND bm.status = 'active'
      AND bm.embedding IS NOT NULL
    ORDER BY bm.created_at, bm.id
    """
)

_PROMOTE_ROWS_SQL = text(
    """
    UPDATE session_memories
    SET branch_id = NULL,
        updated_at = NOW()
    WHERE org_id = :org_id
      AND branch_id = :branch_id
    """
)

#: Mirrors :func:`shared.memory.retract_memory`'s metadata discipline —
#: ``retracted_by`` / ``retracted_at`` / ``retracted_via`` merged into the
#: existing metadata — with ``retracted_via='branch_discard'`` naming the path.
_RETRACT_BRANCH_ROWS_SQL = text(
    """
    UPDATE session_memories
    SET status = 'retracted',
        metadata = COALESCE(metadata, '{}'::jsonb)
                   || jsonb_build_object(
                       'retracted_by', CAST(:discarded_by AS text),
                       'retracted_at', NOW()::text,
                       'retracted_via', 'branch_discard'
                   ),
        updated_at = NOW()
    WHERE org_id = :org_id
      AND branch_id = :branch_id
      AND status = 'active'
    """
)

_CLOSE_BRANCH_SQL = text(
    """
    UPDATE context_branches
    SET status = :status,
        closed_at = NOW()
    WHERE org_id = :org_id
      AND id = :branch_id
    """
)


def _lock_branch(db_session: Any, *, org_id: str, branch_id: str) -> Optional[Any]:
    """The branch row FOR UPDATE, or ``None`` for unknown/cross-org/bad id."""
    cleaned_org = _clean(org_id)
    cleaned_id = _clean(branch_id)
    if cleaned_org is None or cleaned_id is None:
        return None
    try:
        uuid.UUID(cleaned_id)
    except ValueError:
        return None
    return db_session.execute(
        _LOCK_BRANCH_SQL, {"org_id": cleaned_org, "branch_id": cleaned_id}
    ).first()


def merge_branch(
    db_session: Any,
    *,
    org_id: str,
    project_id: str,
    branch_id: str,
    merged_by: Optional[str],
) -> dict:
    """Promote a branch's memories to main; near-duplicates supersede.

    The three steps, in order:

    1. Where an active branch row's stored embedding is within
       :data:`shared.memory.DEDUP_THRESHOLD` of an active MAIN row of the same
       project, the main row is retired through the existing
       :func:`shared.memory.supersede_memory` — the branch version is the
       newer belief — and the judgment is recorded as a ``supersedes`` edge,
       the same shape (and the same best-effort discipline) the curator's
       reconcile pass writes: the status flip is the contract, the edge is
       the explanation, and a graph problem never undoes a supersession.
    2. Every row still on the branch gets ``branch_id = NULL`` — the rows ARE
       main now, no copy, no re-write.
    3. The branch row closes as ``merged``.

    Contradiction detection is embedding similarity ONLY in v1 — no LLM judge
    (that is the curator's job on main, later, and is explicitly out of the
    Track C scope). ``merged_by`` is accepted for parity/audit at the call
    surface; membership checks are the CALLER's job, per the module contract.

    Idempotent: a branch that is not open answers
    ``{"merged": False, "reason": "already_<status>"}``; an unknown or
    cross-org id answers ``reason: "not_found"``. Neither raises — retrying a
    merge must always be safe.

    Returns:
        ``{"merged": True, "promoted": n, "superseded": m}`` on success.
    """
    # Imported lazily, mirroring shared.graph_store's treatment of
    # shared.memory: memory.py pulls in the embedding stack, which this
    # registry module must not load just to be imported.
    from shared import graph_store  # noqa: PLC0415
    from shared.memory import DEDUP_THRESHOLD, supersede_memory  # noqa: PLC0415

    row = _lock_branch(db_session, org_id=org_id, branch_id=branch_id)
    if row is None:
        db_session.rollback()
        return {"merged": False, "reason": "not_found"}
    if row.status != "open":
        db_session.rollback()
        return {"merged": False, "reason": f"already_{row.status}"}

    cleaned_org = _clean(org_id)
    target_project = str(row.project_id)

    pairs = db_session.execute(
        _MERGE_SIMILARITY_SQL,
        {
            "org_id": cleaned_org,
            "project_id": target_project,
            "branch_id": str(row.id),
            "threshold": DEDUP_THRESHOLD,
        },
    ).fetchall()

    superseded = 0
    for pair in pairs:
        # supersede_memory commits internally (and marks stale serves); the
        # branch-row lock is therefore released after the first pair. That is
        # accepted: the branch is still 'open' until step 3, and a concurrent
        # merge re-running this pass supersedes idempotently (same old, same
        # new) before one of the two closes the row.
        supersede_memory(
            db_session,
            uuid.UUID(str(pair.main_memory_id)),
            uuid.UUID(str(pair.branch_memory_id)),
            org_id=cleaned_org,
        )
        superseded += 1
        try:
            graph_store.add_edge(
                db_session,
                org_id=cleaned_org,
                src_id=str(pair.branch_memory_id),
                dst_kind="memory",
                dst_id=str(pair.main_memory_id),
                relation="supersedes",
                provenance="INFERRED",
                created_by=_clean(merged_by),
            )
            db_session.commit()
        except Exception:
            try:
                db_session.rollback()
            except Exception:
                logger.debug("merge_branch: rollback after edge failure also failed")
            logger.warning(
                "merge_branch: supersedes edge write failed for org %s "
                "(new=%s old=%s); the status flip stands",
                cleaned_org,
                pair.branch_memory_id,
                pair.main_memory_id,
                exc_info=True,
            )

    promoted = db_session.execute(
        _PROMOTE_ROWS_SQL, {"org_id": cleaned_org, "branch_id": str(row.id)}
    ).rowcount
    db_session.execute(
        _CLOSE_BRANCH_SQL,
        {"org_id": cleaned_org, "branch_id": str(row.id), "status": "merged"},
    )
    db_session.commit()
    return {"merged": True, "promoted": int(promoted or 0), "superseded": superseded}


def discard_branch(
    db_session: Any,
    *,
    org_id: str,
    branch_id: str,
    discarded_by: Optional[str],
) -> dict:
    """Retract a branch's memories and close the branch as ``discarded``.

    Every ACTIVE row on the branch becomes ``status='retracted'`` with the
    same audit metadata :func:`shared.memory.retract_memory` writes
    (``retracted_via='branch_discard'``), so the rows disappear from every
    read path at once — recall, the graph, and capture recency all exclude
    non-active rows already. Non-active rows (already retracted, or
    superseded) are left exactly as they are.

    Idempotent, mirroring :func:`merge_branch`: not open answers
    ``already_<status>``, unknown/cross-org answers ``not_found``, and
    neither raises.

    Returns:
        ``{"discarded": True, "retracted": n}`` on success.
    """
    row = _lock_branch(db_session, org_id=org_id, branch_id=branch_id)
    if row is None:
        db_session.rollback()
        return {"discarded": False, "reason": "not_found"}
    if row.status != "open":
        db_session.rollback()
        return {"discarded": False, "reason": f"already_{row.status}"}

    cleaned_org = _clean(org_id)
    retracted = db_session.execute(
        _RETRACT_BRANCH_ROWS_SQL,
        {
            "org_id": cleaned_org,
            "branch_id": str(row.id),
            "discarded_by": _clean(discarded_by),
        },
    ).rowcount
    db_session.execute(
        _CLOSE_BRANCH_SQL,
        {"org_id": cleaned_org, "branch_id": str(row.id), "status": "discarded"},
    )
    db_session.commit()
    return {"discarded": True, "retracted": int(retracted or 0)}
