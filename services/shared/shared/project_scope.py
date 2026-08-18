"""Where a passive capture lands: the project's default, narrowed by the member.

THE ONE RULE THIS MODULE ENFORCES: it can only ever WIDEN from ``private``, and
only when a project that already resolved says so. Every failure path — no
project, no user, an unreadable row, a bad value, a raised exception — returns
``private``. That is not defensive habit; it is the property that lets this ship
on top of the user silo without reopening the boundary the silo established.

The decision, in order:

1. No attributed user -> caller's fallback. Unattributable captures are not this
   module's business; ``capture._hook_write_scope`` already reasons about them
   and its fallback is deliberately NOT private (migration 031 rejects an
   owner-less private row, so writing one destroys data while reporting success).
2. No project -> ``private``. Unfiled context is whatever the person was doing
   outside any repo, and a project setting must never reach it. This is the
   branch most likely to be dropped by a careless refactor, so it is tested by
   name.
3. Member opted out -> ``private``, whatever the project says.
4. Otherwise -> the project's ``default_scope``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

SCOPE_PRIVATE = "private"
SCOPE_PROJECT = "project"
SCOPE_ORG = "org"

#: What a project may declare. Anything else is treated as ``private``.
DEFAULT_SCOPES = (SCOPE_PRIVATE, SCOPE_PROJECT, SCOPE_ORG)

#: What a member may set for themself. Narrowing only — see migration 047.
MEMBER_OVERRIDES = (SCOPE_PRIVATE,)


def normalize_default_scope(scope: Optional[str]) -> Optional[str]:
    """``scope`` if a project may declare it, else ``None``."""
    value = (scope or "").strip().lower()

    return value if value in DEFAULT_SCOPES else None


def resolve_capture_scope(
    db: Any,
    *,
    org_id: str,
    project_id: Optional[str],
    user_id: Optional[str],
) -> str:
    """The scope an attributable capture should land at.

    Returns ``private`` unless the resolved project declares otherwise and the
    member has not opted out. Never raises: a capture must not fail because a
    preference could not be read, and the safe answer is always available.

    The project row is fetched with ``org_id`` in the predicate rather than
    filtered afterwards, so a project id belonging to another tenant selects
    nothing instead of leaking its setting into this org's routing.
    """
    if not user_id or not project_id:
        return SCOPE_PRIVATE

    try:
        row = (
            db.execute(
                text(
                    """
                    SELECT p.default_scope AS default_scope,
                           m.scope_override AS scope_override
                    FROM projects p
                    LEFT JOIN project_members m
                           ON m.project_id = p.id AND m.user_id = :user_id
                    WHERE p.id = :project_id AND p.org_id = :org_id
                    """
                ),
                {
                    "project_id": project_id,
                    "org_id": org_id,
                    "user_id": user_id,
                },
            )
            .mappings()
            .first()
        )
    except Exception:
        logger.warning(
            "project-scope: could not read default for project=%s org=%s; "
            "capturing private",
            project_id,
            org_id,
            exc_info=True,
        )
        return SCOPE_PRIVATE

    if row is None:
        return SCOPE_PRIVATE

    # The member's own opt-out wins over the project, always. Narrowing only.
    if (row["scope_override"] or "").strip().lower() == SCOPE_PRIVATE:
        return SCOPE_PRIVATE

    return normalize_default_scope(row["default_scope"]) or SCOPE_PRIVATE


def set_project_default_scope(
    db: Any, *, org_id: str, project_id: str, scope: str
) -> Optional[str]:
    """Set a project's default capture scope. Returns the PREVIOUS value.

    Returns ``None`` when the project does not exist in this org, which callers
    must treat as a refusal rather than as "it was private". Authorisation is
    NOT checked here — the caller does it, at the tool boundary, where the
    acting user is known. This function is the write, not the gate.
    """
    normalized = normalize_default_scope(scope)
    if normalized is None:
        raise ValueError(f"invalid default scope: {scope!r}")

    row = (
        db.execute(
            text(
                """
                UPDATE projects AS p
                SET default_scope = :scope
                FROM projects AS prev
                WHERE prev.id = p.id
                  AND p.id = :project_id
                  AND p.org_id = :org_id
                RETURNING prev.default_scope AS previous
                """
            ),
            {"scope": normalized, "project_id": project_id, "org_id": org_id},
        )
        .mappings()
        .first()
    )
    db.commit()

    return row["previous"] if row else None


def set_member_scope_override(
    db: Any, *, org_id: str, project_id: str, user_id: str, override: Optional[str]
) -> bool:
    """Set (or clear, with ``None``) a member's own narrowing override.

    Only ever called for the ACTING user — see the tool boundary. An admin
    clearing somebody else's override could publish that person's context, which
    breaks the user silo's promise that private is a real boundary including
    from admins, so the caller must never pass another user's id.
    """
    if override is not None and override not in MEMBER_OVERRIDES:
        raise ValueError(f"invalid member override: {override!r}")

    result = db.execute(
        text(
            """
            UPDATE project_members AS m
            SET scope_override = :override
            FROM projects p
            WHERE p.id = m.project_id
              AND p.org_id = :org_id
              AND m.project_id = :project_id
              AND m.user_id = :user_id
            """
        ),
        {
            "override": override,
            "project_id": project_id,
            "org_id": org_id,
            "user_id": user_id,
        },
    )
    db.commit()

    return bool(result.rowcount)
