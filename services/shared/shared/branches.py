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
    INSERT INTO context_branches (org_id, project_id, name, created_by)
    VALUES (:org_id, :project_id, :name, :created_by)
    ON CONFLICT (org_id, project_id, name) WHERE status = 'open'
        DO UPDATE SET name = EXCLUDED.name
    RETURNING id, org_id, project_id, name, status, created_by,
              created_at, closed_at
    """
)

_GET_BRANCH_SQL = text(
    """
    SELECT id, org_id, project_id, name, status, created_by,
           created_at, closed_at
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
           created_at, closed_at
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
           b.created_at, b.closed_at,
           COUNT(m.id) FILTER (WHERE m.status = 'active') AS unmerged_count,
           MAX(m.created_at) FILTER (WHERE m.status = 'active') AS last_activity
    FROM context_branches b
    LEFT JOIN session_memories m
        ON m.branch_id = b.id
       AND m.org_id = b.org_id
    WHERE b.org_id = :org_id
      AND b.project_id = :project_id
    GROUP BY b.id, b.org_id, b.project_id, b.name, b.status, b.created_by,
             b.created_at, b.closed_at
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
    }


def ensure_branch(
    db_session: Any,
    *,
    org_id: str,
    project_id: str,
    name: str,
    created_by: Optional[str],
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


def list_branches(db_session: Any, *, org_id: str, project_id: str) -> list[dict]:
    """Every branch of this project (all statuses), newest first.

    Each dict carries the registry columns plus two computed fields:

    * ``unmerged_count`` — active memories currently on the branch. This is
      the "is there work parked here?" number; retracted/superseded rows do
      not count.
    * ``last_activity`` — the newest active memory's ``created_at``, or
      ``None`` for a branch nothing has been written to. Together with
      ``unmerged_count`` this is the stale-branch surfacing from the spec.
    """
    cleaned_org = _clean(org_id)
    cleaned_project = _clean(project_id)
    if cleaned_org is None or cleaned_project is None:
        return []
    rows = db_session.execute(
        _LIST_BRANCHES_SQL,
        {"org_id": cleaned_org, "project_id": cleaned_project},
    ).fetchall()
    branches = []
    for row in rows:
        branch = _row_to_branch(row)
        branch["unmerged_count"] = int(row.unmerged_count or 0)
        branch["last_activity"] = row.last_activity
        branches.append(branch)
    return branches
