"""Project ACL primitives — the storage layer behind ``scope='project'``.

``project_members`` (migration 031) is what turns a project from a *relevance
filter* into a *boundary*. Everything in this module exists to answer exactly
one of two questions:

* **May this user read this project?** — :func:`is_member`, and the equivalent
  ``project_id IN (SELECT ...)`` subquery inlined into
  :func:`shared.memory.retrieve_memories_scoped`. The recall path deliberately
  does NOT call :func:`is_member` per row: entitlement belongs in the same
  statement that returns the rows.
* **May this user write memories into it?** — :func:`can_write`.
* **May this user change who is on it?** — :func:`can_manage` (Read/Write
  grants) and :func:`is_admin` (any role, including Manage/Admin).

Two invariants hold throughout:

1. Every query is filtered by ``org_id``, via a join to ``projects``. A
   ``project_id`` is a UUID a caller may have obtained from anywhere, so it is
   never trusted on its own to identify a tenant's project.
2. Nothing here fails open. An unknown project, an unknown user, or a caller
   who is not an admin yields ``False`` / ``None`` / a refusal — never a
   widened result.

Synchronous (blocking psycopg2 I/O), matching the house pattern in
:mod:`shared.brain`. Call from async handlers via ``anyio.to_thread.run_sync``.
"""

import logging
import uuid
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger("agentguard.shared.projects")

ROLE_READ = "read"
ROLE_WRITE = "write"
ROLE_MANAGE = "manage"
ROLE_ADMIN = "admin"
# 031 stored "member" for "can see the project". After 040 that capability
# is ``write`` (existing members were not read-only). Accept the old name
# at the API edge so a client that has not updated still grants write.
ROLE_MEMBER = ROLE_WRITE

ROLES = (ROLE_READ, ROLE_WRITE, ROLE_MANAGE, ROLE_ADMIN)

# Who may persist into the project brain (and put/forget artifacts there).
WRITE_ROLES = frozenset({ROLE_WRITE, ROLE_MANAGE, ROLE_ADMIN})
# Who may change the member list at all.
MANAGE_ROLES = frozenset({ROLE_MANAGE, ROLE_ADMIN})
# What a Manage grantor may assign or revoke.
MANAGE_TOUCHABLE = frozenset({ROLE_READ, ROLE_WRITE})


def normalize_role(role: str) -> str:
    """Map a caller-supplied role onto a stored value.

    ``member`` is a synonym for ``write``. Anything else is returned as-is;
    :func:`add_member` still rejects unknown names.
    """
    if role == "member":
        return ROLE_WRITE
    return role


def can_assign_role(actor_role: Optional[str], new_role: str) -> bool:
    """May someone holding ``actor_role`` grant ``new_role``?"""
    actor = normalize_role(actor_role) if actor_role else None
    target = normalize_role(new_role)
    if target not in ROLES:
        return False
    if actor == ROLE_ADMIN:
        return True
    if actor == ROLE_MANAGE:
        return target in MANAGE_TOUCHABLE
    return False


def can_touch_member(actor_role: Optional[str], target_role: Optional[str]) -> bool:
    """May someone holding ``actor_role`` revoke or reassign ``target_role``?"""
    actor = normalize_role(actor_role) if actor_role else None
    target = normalize_role(target_role) if target_role else None
    if actor == ROLE_ADMIN:
        return True
    if actor == ROLE_MANAGE:
        return target in MANAGE_TOUCHABLE
    return False


# The surfaces a grant can arrive from. Recorded on every audit row so
# "which surface" in the spec's who/what/when/which-surface is answerable.
SURFACE_MCP = "mcp"
SURFACE_DASHBOARD = "dashboard"
#: The project-ownership-transfer sweep (``services/async-worker/app/
#: ownership_sweep.py``) — not a human-triggered surface, but audit rows it
#: writes need a "which surface" answer the same as any other.
SURFACE_SWEEP = "sweep"

# The two verbs ``project_grant_audit.action`` accepts (migration 033).
ACTION_GRANT = "grant"
ACTION_REVOKE = "revoke"


def _owned_project_id(
    db_session: Any, *, org_id: str, project_id: str
) -> Optional[str]:
    """Return ``project_id`` iff it names a project owned by ``org_id``.

    Every other function here routes through this. Without it a caller could
    pass a project UUID belonging to another tenant and have the membership
    table — which has no ``org_id`` column of its own — answer for it.
    """
    try:
        parsed = uuid.UUID(str(project_id))
    except (ValueError, AttributeError, TypeError):
        return None
    row = db_session.execute(
        text(
            """
            SELECT id FROM projects
            WHERE id = :project_id AND org_id = :org_id
            LIMIT 1
            """
        ),
        {"project_id": str(parsed), "org_id": org_id},
    ).fetchone()
    return str(row[0]) if row is not None else None


def is_member(
    db_session: Any, *, org_id: str, project_id: str, user_id: Optional[str]
) -> bool:
    """True iff ``user_id`` may read ``scope='project'`` rows of this project.

    ``None`` / empty ``user_id`` is ``False``: an unattributable caller is not
    a member of anything.
    """
    if not user_id:
        return False
    owned = _owned_project_id(db_session, org_id=org_id, project_id=project_id)
    if owned is None:
        return False
    row = db_session.execute(
        text(
            """
            SELECT 1 FROM project_members
            WHERE project_id = :project_id AND user_id = :user_id
            LIMIT 1
            """
        ),
        {"project_id": owned, "user_id": user_id},
    ).fetchone()
    return row is not None


def member_role(
    db_session: Any, *, org_id: str, project_id: str, user_id: Optional[str]
) -> Optional[str]:
    """The caller's stored role on this project, or ``None``.

    ``member`` is normalised to ``write`` so callers never have to know the
    pre-040 name.
    """
    if not user_id:
        return None
    owned = _owned_project_id(db_session, org_id=org_id, project_id=project_id)
    if owned is None:
        return None
    row = db_session.execute(
        text(
            """
            SELECT role FROM project_members
            WHERE project_id = :project_id AND user_id = :user_id
            LIMIT 1
            """
        ),
        {"project_id": owned, "user_id": user_id},
    ).fetchone()
    if row is None:
        return None
    return normalize_role(row.role)


def can_write(
    db_session: Any, *, org_id: str, project_id: str, user_id: Optional[str]
) -> bool:
    """True iff ``user_id`` may persist into this project's shared brain."""
    return (
        member_role(db_session, org_id=org_id, project_id=project_id, user_id=user_id)
        in WRITE_ROLES
    )


def can_manage(
    db_session: Any, *, org_id: str, project_id: str, user_id: Optional[str]
) -> bool:
    """True iff ``user_id`` may change Read/Write membership on this project."""
    return (
        member_role(db_session, org_id=org_id, project_id=project_id, user_id=user_id)
        in MANAGE_ROLES
    )


def is_admin(
    db_session: Any, *, org_id: str, project_id: str, user_id: Optional[str]
) -> bool:
    """True iff ``user_id`` is a project Admin (may grant any role).

    Narrower than :func:`can_manage`. ``project_grant`` is reachable by an
    LLM, so promoting someone to Manage/Admin has to stay on Admin.
    """
    return (
        member_role(db_session, org_id=org_id, project_id=project_id, user_id=user_id)
        == ROLE_ADMIN
    )


def admin_count(db_session: Any, *, org_id: str, project_id: str) -> Optional[int]:
    """Number of Admin rows, or ``None`` when the project is not this org's."""
    owned = _owned_project_id(db_session, org_id=org_id, project_id=project_id)
    if owned is None:
        return None
    row = db_session.execute(
        text(
            """
            SELECT COUNT(*) FROM project_members
            WHERE project_id = :project_id AND role = :role
            """
        ),
        {"project_id": owned, "role": ROLE_ADMIN},
    ).fetchone()
    return int(row[0])


def member_count(db_session: Any, *, org_id: str, project_id: str) -> Optional[int]:
    """Number of members, or ``None`` when the project is not this org's.

    Used to distinguish "a project nobody administers yet" (the 106 projects
    ``ensure_project`` auto-created before ACLs existed) from "a project
    someone else administers". The first may be claimed; the second may not.
    """
    owned = _owned_project_id(db_session, org_id=org_id, project_id=project_id)
    if owned is None:
        return None
    row = db_session.execute(
        text("SELECT COUNT(*) FROM project_members WHERE project_id = :project_id"),
        {"project_id": owned},
    ).fetchone()
    return int(row[0])


def add_member(
    db_session: Any,
    *,
    org_id: str,
    project_id: str,
    user_id: str,
    role: str = ROLE_WRITE,
    granted_by: Optional[str],
) -> bool:
    """Idempotently grant ``user_id`` access to a project. Returns success.

    **Authorization is the caller's job** — this is the storage primitive, and
    the only checks it performs are the structural ones (the project belongs to
    ``org_id``, the role is a legal value). :func:`can_assign_role` /
    :func:`can_touch_member` are the policy; see ``app.mcp_app``'s
    ``project_grant`` for the composed operation, which also writes the audit
    row.

    A repeat grant upgrades or demotes the role rather than erroring.
    ``member`` is accepted and stored as ``write``.
    """
    role = normalize_role(role)
    if role not in ROLES:
        raise ValueError(f"unsupported role: {role!r}")
    if not user_id:
        raise ValueError("user_id is required")
    owned = _owned_project_id(db_session, org_id=org_id, project_id=project_id)
    if owned is None:
        return False
    db_session.execute(
        text(
            """
            INSERT INTO project_members (
                project_id, user_id, role, granted_by
            ) VALUES (
                :project_id, :user_id, :role, :granted_by
            )
            ON CONFLICT (project_id, user_id)
              DO UPDATE SET role = EXCLUDED.role,
                            granted_by = EXCLUDED.granted_by,
                            granted_at = NOW()
            """
        ),
        {
            "project_id": owned,
            "user_id": user_id,
            "role": role,
            "granted_by": granted_by,
        },
    )
    db_session.commit()
    return True


def list_members(
    db_session: Any, *, org_id: str, project_id: str
) -> Optional[list[dict]]:
    """Members of a project, or ``None`` when it is not this org's project."""
    owned = _owned_project_id(db_session, org_id=org_id, project_id=project_id)
    if owned is None:
        return None
    rows = db_session.execute(
        text(
            """
            SELECT user_id, role, granted_by, granted_at
            FROM project_members
            WHERE project_id = :project_id
            ORDER BY granted_at ASC, user_id ASC
            """
        ),
        {"project_id": owned},
    ).fetchall()
    return [
        {
            "user_id": row.user_id,
            "role": row.role,
            "granted_by": row.granted_by,
            "granted_at": row.granted_at,
        }
        for row in rows
    ]


def list_projects(
    db_session: Any, *, org_id: str, user_id: Optional[str]
) -> list[dict]:
    """Projects ``user_id`` is a member of, newest-seen first.

    **Ruling, 2026-08-12 ("projects are visible only to their members"):** a
    project a caller does not belong to must not appear here at all — not
    with contents hidden, not even as a name. The previous contract (every
    project in the org, unconditionally) rested on "listing is not a
    disclosure of CONTENTS", which is true as far as it goes but did not
    survive contact with two things it did not anticipate: projects are
    auto-created from any write that carries a git remote or repo root, and
    the resulting project names and counts were themselves visible to every
    key holder in the org whether they had ever touched the project or not.

    ``user_id=None`` (an unattributed caller — a key with no resolvable
    person behind it) returns an empty list, not the org's full project set
    and not an error. There is no principal here to check membership for, and
    "we don't know who you are" must fail CLOSED for a listing that is now a
    boundary, not open wide because the check couldn't run. This mirrors
    :func:`is_member`, which already treats a missing ``user_id`` as "member
    of nothing".

    No org-admin bypass: the ruling is explicit that there is none in the
    engine. An org admin who wants to see a project must be a member of it,
    same as anyone else.
    """
    if not user_id:
        return []
    rows = db_session.execute(
        text(
            """
            SELECT p.id, p.display_name, p.git_remote, p.repo_root_path,
                   p.last_seen_at, p.created_at,
                   (SELECT COUNT(*) FROM project_members m
                     WHERE m.project_id = p.id) AS member_count
            FROM projects p
            WHERE p.org_id = :org_id
              AND EXISTS (
                SELECT 1 FROM project_members pm
                WHERE pm.project_id = p.id AND pm.user_id = :user_id
              )
            ORDER BY p.last_seen_at DESC NULLS LAST, p.created_at DESC, p.id ASC
            """
        ),
        {"org_id": org_id, "user_id": user_id},
    ).fetchall()
    return [
        {
            "id": str(row.id),
            "display_name": row.display_name,
            "git_remote": row.git_remote,
            "repo_root_path": row.repo_root_path,
            "last_seen_at": row.last_seen_at,
            "created_at": row.created_at,
            "member_count": int(row.member_count),
        }
        for row in rows
    ]


def record_grant_audit(
    db_session: Any,
    *,
    org_id: str,
    project_id: str,
    granted_to: str,
    granted_to_email: Optional[str],
    granted_by: Optional[str],
    role: str,
    surface: str,
    action: str = ACTION_GRANT,
) -> None:
    """Append one immutable row describing an ACL change.

    ``action`` defaults to ``'grant'`` (migration 033's column default, and
    the only verb every pre-existing call site ever wrote) so this stays a
    source-compatible addition — every caller that predates the ``action``
    parameter keeps writing exactly the row it always wrote. Pass
    ``ACTION_REVOKE`` to record a member's access being taken away instead of
    given; ``role`` should then be the role the member actually held at the
    moment of removal (migration 033's documented contract), not a role they
    are receiving.

    Best-effort by construction: the grant itself has already committed by the
    time this runs, and losing the log must not undo an access change the
    caller was told succeeded. A failure is logged loudly instead — with the
    same who/what/when/surface fields — so the record survives even when the
    table does not.
    """
    if action not in (ACTION_GRANT, ACTION_REVOKE):
        raise ValueError(f"unsupported action: {action!r}")
    try:
        db_session.execute(
            text(
                """
                INSERT INTO project_grant_audit (
                    id, org_id, project_id, granted_to, granted_to_email,
                    granted_by, role, surface, action
                ) VALUES (
                    :id, :org_id, :project_id, :granted_to, :granted_to_email,
                    :granted_by, :role, :surface, :action
                )
                """
            ),
            {
                "id": str(uuid.uuid4()),
                "org_id": org_id,
                "project_id": project_id,
                "granted_to": granted_to,
                "granted_to_email": granted_to_email,
                "granted_by": granted_by,
                "role": role,
                "surface": surface,
                "action": action,
            },
        )
        db_session.commit()
    except Exception:
        # The grant has already committed; this append has not. Roll back so
        # the caller's session is usable for the rest of the request —
        # otherwise the audit failure, which is explicitly not allowed to undo
        # the grant, takes down the response that reports it instead.
        try:
            db_session.rollback()
        except Exception:
            logger.warning(
                "rollback after a failed grant audit write also failed; "
                "org=%s project=%s",
                org_id,
                project_id,
                exc_info=True,
            )
        logger.error(
            "project grant audit write FAILED; grant stands. "
            "org=%s project=%s granted_to=%s granted_by=%s role=%s surface=%s",
            org_id,
            project_id,
            granted_to,
            granted_by,
            role,
            surface,
            exc_info=True,
        )
