"""One project, two faces: a workspace (Plane) project linked to an engine one.

``projects.plane_project_id`` (migration 049) joins the two, and read
visibility — plus explicit-write permission — follows the WORKSPACE project's
membership: the GitHub-repo model. Add someone to the project in the UI, and
their agent can recall its memory.

**Two databases, joined in Python** — the :mod:`shared.membership` pattern.
Plane's ``projects`` / ``project_members`` live in the workspace DB
(``WORKSPACE_DATABASE_URL``); the engine's ``projects`` live in its own
database (``DATABASE_URL``). Both servers hold tables NAMED ``projects`` and
``project_members``, which is exactly why no statement here may name the other
side's tables: the same SQL that works against a test database holding both
fails in production with "relation does not exist" — or worse, silently
answers from the wrong server.

Plane roles are smallints: 20 admin, 15 member, 5 guest. Read reach begins at
guest (5); write reach begins at member (15). A link is NEVER re-pointed in
either direction — the ``6ad63ae`` lesson: identities do not transfer.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from shared import projects

logger = logging.getLogger(__name__)

#: Plane's ``project_members.role`` values.
PLANE_ROLE_ADMIN = 20
PLANE_ROLE_MEMBER = 15
PLANE_ROLE_GUEST = 5

#: The minimum plane role that grants read reach over a linked project.
PLANE_READ_ROLE = PLANE_ROLE_GUEST
#: The minimum plane role that grants explicit-write reach.
PLANE_WRITE_ROLE = PLANE_ROLE_MEMBER

#: Workspace DB side: which living plane projects is this user an active
#: member of, at what role? Every soft-delete/deactivation kill switch is a
#: WHERE clause here, mirroring ``membership._WORKSPACES_SQL``: revocation in
#: Plane is ``is_active = false`` or a ``deleted_at`` stamp, never a DELETE.
_PLANE_MEMBERSHIPS_SQL = text(
    """
    SELECT pm.project_id AS plane_project_id,
           pm.role AS role
    FROM project_members pm
    JOIN projects p ON p.id = pm.project_id
    JOIN workspaces w ON w.id = p.workspace_id
    WHERE pm.member_id = :user_id
      AND pm.is_active = true
      AND pm.deleted_at IS NULL
      AND p.deleted_at IS NULL
      AND w.deleted_at IS NULL
    """
)

#: Engine side: which of those plane projects are linked to THIS org's
#: projects? Tenant-scoped by ``org_id`` so a link in another org can never
#: enter this org's reach.
_LINKED_PROJECTS_SQL = text(
    """
    SELECT id, plane_project_id
    FROM projects
    WHERE org_id = :org_id
      AND plane_project_id = ANY(CAST(:plane_ids AS uuid[]))
    """
)

#: The guarded link write. ``plane_project_id IS NULL`` is repeated in SQL so
#: a concurrent link loses rather than being overwritten between a read and
#: this UPDATE — the same shape as ``membership._ADOPT_WORKSPACE_SQL``.
_LINK_SQL = text(
    """
    UPDATE projects
       SET plane_project_id = CAST(:plane_project_id AS uuid)
     WHERE id = :project_id
       AND org_id = :org_id
       AND plane_project_id IS NULL
    RETURNING id
    """
)

_LINK_STATE_SQL = text(
    """
    SELECT plane_project_id
    FROM projects
    WHERE id = :project_id AND org_id = :org_id
    LIMIT 1
    """
)


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def plane_project_memberships(
    workspace_db: Any, *, user_id: Optional[str]
) -> dict[str, int]:
    """Every living plane project this user is an active member of, by role.

    The workspace-DB half of the linked reach. Takes the WORKSPACE connection
    only: Plane's ``projects`` / ``project_members`` / ``workspaces`` do not
    exist in the engine's database (whatever their engine namesakes suggest).

    Args:
        workspace_db: A connection to the workspace database.
        user_id: The user UUID, or ``None``.

    Returns:
        ``{plane_project_id: role}`` with ids stringified and roles as ints
        (20 admin, 15 member, 5 guest). An unattributable caller gets ``{}``
        — the same fail-closed rule as :func:`shared.membership.reachable_orgs`.
    """
    cleaned = _clean(user_id)
    if cleaned is None:
        return {}
    rows = workspace_db.execute(_PLANE_MEMBERSHIPS_SQL, {"user_id": cleaned}).fetchall()
    return {str(r.plane_project_id): int(r.role) for r in rows}


def linked_engine_projects(
    tsdb: Any, *, org_id: str, plane_ids: list[str]
) -> dict[str, str]:
    """Which of ``plane_ids`` are linked to THIS org's engine projects?

    The engine-DB half. Tenant-scoped by ``org_id``: a linked project in
    another org never answers, however real the plane membership is.

    Args:
        tsdb: A connection to the engine's own database.
        org_id: The org whose projects may answer.
        plane_ids: Plane project UUIDs (typically the caller's memberships).

    Returns:
        ``{engine_project_id: plane_project_id}``, ids stringified. Empty
        input queries nothing and returns ``{}``.
    """
    if not plane_ids:
        return {}
    rows = tsdb.execute(
        _LINKED_PROJECTS_SQL,
        {"org_id": org_id, "plane_ids": [str(p) for p in plane_ids]},
    ).fetchall()
    return {str(r.id): str(r.plane_project_id) for r in rows}


def link_engine_project(
    tsdb: Any, *, org_id: str, project_id: str, plane_project_id: str
) -> dict[str, Any]:
    """Write the link — the guarded UPDATE only; the tool layer gates.

    Never re-points: the UPDATE carries ``plane_project_id IS NULL``, so an
    already-linked project refuses a different id, and the partial unique
    index ``uq_projects_plane_project_id`` refuses a plane id already linked
    to a different engine project. A lost race re-reads and wins only if the
    winner is us — linking the same pair twice is idempotent, not an error.

    Args:
        tsdb: A connection/session to the engine's own database. The caller
            owns the transaction and commits on success.
        org_id: Tenant scope; another org's project id is ``not_found`` here.
        project_id: The engine project to link.
        plane_project_id: The workspace (Plane) project UUID.

    Returns:
        ``{"linked": True, "project_id", "workspace_project"}`` on success
        (including the idempotent repeat), or ``{"linked": False, "error":
        "not_found" | "already_linked"}``. The error never names what the
        other side of an existing link is.
    """
    params = {
        "org_id": org_id,
        "project_id": project_id,
        "plane_project_id": plane_project_id,
    }
    try:
        row = tsdb.execute(_LINK_SQL, params).first()
    except IntegrityError:
        # The plane id was claimed by a different engine project between the
        # tool layer's pre-check and this write. The transaction is aborted,
        # so roll it back before answering; say nothing about the winner.
        tsdb.rollback()
        return {"linked": False, "error": "already_linked"}
    if row is not None:
        return {
            "linked": True,
            "project_id": str(row.id),
            "workspace_project": plane_project_id,
        }

    # The guarded UPDATE matched nothing: the project is missing, another
    # org's, or already linked. Re-read; winner-if-ours, else refuse.
    now = tsdb.execute(_LINK_STATE_SQL, params).first()
    if now is None:
        return {"linked": False, "error": "not_found"}
    if now.plane_project_id is not None and str(now.plane_project_id) == str(
        plane_project_id
    ):
        return {
            "linked": True,
            "project_id": project_id,
            "workspace_project": plane_project_id,
        }
    return {"linked": False, "error": "already_linked"}


def can_write_linked(
    workspace_db: Optional[Any],
    tsdb: Any,
    *,
    org_id: str,
    project_id: str,
    user_id: Optional[str],
) -> bool:
    """May ``user_id`` land an explicit ``project`` write on this project?

    Engine membership answers first (``projects.can_write``, unchanged); a
    caller who is not an engine member may still write when the project is
    LINKED and their plane role is member or above (≥ 15). Guests (5) read
    but never write. ``workspace_db=None`` keeps the engine-only answer —
    linked reach is a UNION on top of today's behaviour, never a replacement.
    """
    if projects.can_write(tsdb, org_id=org_id, project_id=project_id, user_id=user_id):
        return True
    cleaned = _clean(user_id)
    if workspace_db is None or cleaned is None:
        return False
    row = tsdb.execute(
        _LINK_STATE_SQL, {"org_id": org_id, "project_id": project_id}
    ).first()
    if row is None or row.plane_project_id is None:
        return False
    role = plane_project_memberships(workspace_db, user_id=cleaned).get(
        str(row.plane_project_id)
    )
    return role is not None and role >= PLANE_WRITE_ROLE
