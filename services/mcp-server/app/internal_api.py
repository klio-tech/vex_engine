"""Read-only internal status endpoints for the workspace web app.

The workspace (Plane) backend proxies these to render the "Agent memory"
panel: is this project linked, how much memory does it hold, which context
branches are open. They are additive and read-only — nothing existing
changes because they exist.

Auth is a shared secret, not a tenant credential: the caller is the
workspace's Django backend holding ``INTERNAL_API_SECRET``, compared in
constant time (``hmac.compare_digest``). Every failure — wrong secret,
missing header, or the env var not being set at all — is the SAME 404
shape, so an unauthenticated probe cannot learn that the routes exist,
and an unset secret simply means the feature is off.

The secret is read from the environment PER REQUEST rather than captured
at import: the mcp app is built once at startup, and a module-level read
would freeze "feature off" into every test and every process whose env
is assembled after import.
"""

import hmac
import logging
import os
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import text

from app.db import get_db
from shared.branches import list_branches

logger = logging.getLogger("agentguard.mcp.internal_api")

INTERNAL_HEADER = "X-Klio-Internal"
INTERNAL_SECRET_ENV = "INTERNAL_API_SECRET"

#: The engine project linked to this workspace (Plane) project, if any.
_LINKED_PROJECT_SQL = text(
    """
    SELECT id, org_id, display_name
    FROM projects
    WHERE plane_project_id = CAST(:plane_project_id AS uuid)
    LIMIT 1
    """
)

#: Active memory rows for one project, by scope.
_PROJECT_MEMORY_COUNTS_SQL = text(
    """
    SELECT scope, COUNT(*) AS n
    FROM session_memories
    WHERE org_id = :org_id
      AND project_id = CAST(:project_id AS uuid)
      AND status = 'active'
      AND scope IN ('org', 'project', 'private')
    GROUP BY scope
    """
)

#: The org node linked to this workspace, if any.
_ORG_BY_WORKSPACE_SQL = text(
    """
    SELECT org_id
    FROM organizations
    WHERE workspace_id = CAST(:workspace_id AS uuid)
    LIMIT 1
    """
)

#: Active memory rows across the whole node.
_ORG_MEMORY_COUNT_SQL = text(
    """
    SELECT COUNT(*) AS n
    FROM session_memories
    WHERE org_id = :org_id
      AND status = 'active'
    """
)


#: Active rows by scope across the node.
_ORG_SCOPE_COUNTS_SQL = text(
    """
    SELECT scope, COUNT(*) AS n
    FROM session_memories
    WHERE org_id = :org_id
      AND status = 'active'
    GROUP BY scope
    """
)

#: The node's newest visible memories for the brain surface: org-wide rows,
#: plus the asking user's own private rows — never anyone else's.
_RECENT_MEMORIES_SQL = text(
    """
    SELECT id, LEFT(content, 280) AS content, scope, memory_type, created_at
    FROM session_memories
    WHERE org_id = :org_id
      AND status = 'active'
      AND (scope = 'org'
           OR (scope = 'private' AND CAST(:user_id AS text) IS NOT NULL
               AND CAST(user_id AS text) = CAST(:user_id AS text)))
    ORDER BY created_at DESC
    LIMIT :limit
    """
)

#: One branch's newest rows, for the branch view's preview.
_BRANCH_MEMORIES_SQL = text(
    """
    SELECT id, LEFT(content, 280) AS content, scope, memory_type, created_at
    FROM session_memories
    WHERE org_id = :org_id
      AND branch_id = CAST(:branch_id AS uuid)
      AND status = 'active'
    ORDER BY created_at DESC
    LIMIT 10
    """
)


def _not_found() -> HTTPException:
    """The one refusal shape. 404, never 401/403 — no oracle."""
    return HTTPException(status_code=404, detail="Not Found")


def require_internal_secret(request: Request) -> None:
    """Gate every internal route behind the shared secret.

    Unset env ⇒ the feature is off and every route 404s, whatever header
    arrives. The comparison is constant-time; the failure shape is the
    same 404 as an unknown path.
    """
    secret = os.environ.get(INTERNAL_SECRET_ENV)
    if not secret:
        raise _not_found()
    provided = request.headers.get(INTERNAL_HEADER) or ""
    if not hmac.compare_digest(provided.encode("utf-8"), secret.encode("utf-8")):
        raise _not_found()


router = APIRouter(
    prefix="/internal",
    include_in_schema=False,
    dependencies=[Depends(require_internal_secret)],
)


@router.get("/project-link")
def project_link_status(
    workspace_project: uuid.UUID, db: Any = Depends(get_db)
) -> dict:
    """Link status + memory counts + open branches for one Plane project."""
    empty_counts = {"org": 0, "project": 0, "private": 0}
    row = db.execute(
        _LINKED_PROJECT_SQL, {"plane_project_id": str(workspace_project)}
    ).fetchone()
    if row is None:
        return {
            "linked": False,
            "engine_project_id": None,
            "display_name": None,
            "memory_counts": empty_counts,
            "open_branches": [],
        }

    project_id = str(row.id)
    counts = dict(empty_counts)
    for count_row in db.execute(
        _PROJECT_MEMORY_COUNTS_SQL,
        {"org_id": row.org_id, "project_id": project_id},
    ).fetchall():
        counts[count_row.scope] = int(count_row.n)

    open_branches = [
        {
            "name": branch["name"],
            "unmerged_count": branch["unmerged_count"],
            "last_activity": branch["last_activity"],
        }
        for branch in list_branches(db, org_id=row.org_id, project_id=project_id)
        if branch["status"] == "open"
    ]

    return {
        "linked": True,
        "engine_project_id": project_id,
        "display_name": row.display_name,
        "memory_counts": counts,
        "open_branches": open_branches,
    }


@router.get("/workspace-node")
def workspace_node_status(workspace: uuid.UUID, db: Any = Depends(get_db)) -> dict:
    """Does this workspace have an engine node, and how much does it hold?"""
    row = db.execute(_ORG_BY_WORKSPACE_SQL, {"workspace_id": str(workspace)}).fetchone()
    if row is None:
        return {"org_id": None, "exists": False, "memory_count": 0}
    count_row = db.execute(_ORG_MEMORY_COUNT_SQL, {"org_id": row.org_id}).fetchone()
    return {
        "org_id": row.org_id,
        "exists": True,
        "memory_count": int(count_row.n) if count_row is not None else 0,
    }


@router.get("/node-brain")
def node_brain(
    workspace: uuid.UUID,
    user: str | None = None,
    limit: int = 20,
    db: Any = Depends(get_db),
) -> dict:
    """The brain surface: scope counts and recent memories for one node.

    ``user`` (a workspace user uuid) widens the recent list to include that
    user's own private rows; org rows are always included. The Django proxy
    authenticates the user — this parameter is attribution, not authz.
    """
    row = db.execute(_ORG_BY_WORKSPACE_SQL, {"workspace_id": str(workspace)}).fetchone()
    if row is None:
        return {"exists": False, "org_id": None, "counts": {}, "recent": []}

    counts = {
        count_row.scope: int(count_row.n)
        for count_row in db.execute(
            _ORG_SCOPE_COUNTS_SQL, {"org_id": row.org_id}
        ).fetchall()
    }
    try:
        user_id = str(uuid.UUID(user)) if user else None
    except (ValueError, TypeError):
        user_id = None
    recent = [
        {
            "id": str(memory.id),
            "content": memory.content,
            "scope": memory.scope,
            "memory_type": memory.memory_type,
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
        }
        for memory in db.execute(
            _RECENT_MEMORIES_SQL,
            {"org_id": row.org_id, "user_id": user_id, "limit": max(1, min(limit, 50))},
        ).fetchall()
    ]
    return {"exists": True, "org_id": row.org_id, "counts": counts, "recent": recent}


@router.get("/project-branches")
def project_branches(workspace_project: uuid.UUID, db: Any = Depends(get_db)) -> dict:
    """Every context branch of one linked project, with memory previews."""
    row = db.execute(
        _LINKED_PROJECT_SQL, {"plane_project_id": str(workspace_project)}
    ).fetchone()
    if row is None:
        return {"linked": False, "branches": []}

    branches = []
    for branch in list_branches(db, org_id=row.org_id, project_id=str(row.id)):
        memories = [
            {
                "id": str(memory.id),
                "content": memory.content,
                "scope": memory.scope,
                "memory_type": memory.memory_type,
                "created_at": memory.created_at.isoformat()
                if memory.created_at
                else None,
            }
            for memory in db.execute(
                _BRANCH_MEMORIES_SQL,
                {"org_id": row.org_id, "branch_id": branch["id"]},
            ).fetchall()
        ]
        branches.append(
            {
                "id": branch["id"],
                "name": branch["name"],
                "status": branch["status"],
                "unmerged_count": branch["unmerged_count"],
                "last_activity": branch["last_activity"],
                "created_at": branch["created_at"].isoformat()
                if branch["created_at"]
                else None,
                "memories": memories,
            }
        )
    return {"linked": True, "engine_project_id": str(row.id), "branches": branches}
