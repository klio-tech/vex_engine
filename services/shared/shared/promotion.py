"""Promotion — moving a private memory into a shared scope.

This is the one code path where something private becomes something the team
can read, so it is written to be boring and single-purpose.

Three rules, all enforced here rather than in the tool layer:

1. **Only the owner may promote.** ``user_id`` on the row must equal the
   resolved caller. Not an admin, not a teammate, not the agent that happens to
   hold the same API key under a different ``X-Vex-Agent`` — the owner.
2. **A non-owner sees ``not_found``, never ``forbidden``.** ``forbidden``
   confirms the row exists, which is a disclosure about a scope whose promise
   is that its shape is invisible from outside.
3. **The provenance label survives.** A curator-derived fact is ``INFERRED``
   before promotion and stays ``INFERRED`` after it. The moment an LLM's
   deduction becomes team truth is precisely when the label matters most; a
   promotion that laundered it into ``EXTRACTED`` would erase the distinction
   at the only point anyone was going to check.

``user_id`` is retained through the move, where it stops being visibility and
becomes authorship: "who taught the team this".

Synchronous (inline psycopg2 I/O), matching the house pattern.
"""

import logging
import uuid
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger("agentguard.shared.promotion")

PROMOTABLE_SCOPES = ("org", "project")


def promote_memory(
    db_session: Any,
    *,
    org_id: str,
    memory_id: str,
    user_id: Optional[str],
    to: str,
    via: str = "mcp",
) -> dict:
    """Move one of the caller's own private memories to ``org`` or ``project``.

    Args:
        db_session: Synchronous SQLAlchemy session/connection.
        org_id: Tenant scope. Always enforced.
        memory_id: The row to promote.
        user_id: The resolved caller. ``None`` can never promote anything —
            an unattributable request has no memories of its own to share.
        to: ``'org'`` or ``'project'``.
        via: Recorded in the row's metadata for the audit trail.

    Returns:
        ``{"shared": True, "id": ..., "scope": ..., "provenance": ...}`` on
        success, or ``{"shared": False, "error": ...}`` where ``error`` is one
        of ``invalid_id`` / ``not_found`` / ``already_shared`` /
        ``no_project`` / ``unsupported_scope``.
    """
    if to not in PROMOTABLE_SCOPES:
        return {"shared": False, "error": "unsupported_scope"}
    try:
        parsed = uuid.UUID(str(memory_id))
    except (ValueError, AttributeError, TypeError):
        return {"shared": False, "error": "invalid_id"}
    if not user_id:
        # Indistinguishable from an unknown id on purpose: an unattributable
        # caller must not be able to probe which ids exist.
        return {"shared": False, "error": "not_found"}

    # The ownership predicate is part of the SELECT, not a Python comparison
    # after a wider read. A row belonging to someone else is never fetched.
    row = db_session.execute(
        text(
            """
            SELECT scope, project_id, provenance
            FROM session_memories
            WHERE id = :id
              AND org_id = :org_id
              AND status = 'active'
              AND user_id = :user_id
            LIMIT 1
            """
        ),
        {"id": str(parsed), "org_id": org_id, "user_id": user_id},
    ).fetchone()
    if row is None:
        return {"shared": False, "error": "not_found"}

    if row.scope != "private":
        # Already shared. Promoting again is a no-op at best and a scope
        # DOWNGRADE at worst (project -> org widens it), so refuse and say so.
        return {"shared": False, "error": "already_shared", "scope": row.scope}

    if to == "project" and row.project_id is None:
        return {"shared": False, "error": "no_project"}

    # provenance is deliberately absent from the SET list. Not defaulted, not
    # recomputed — untouched. See rule 3 in the module docstring.
    db_session.execute(
        text(
            """
            UPDATE session_memories
            SET scope = :scope,
                metadata = COALESCE(metadata, '{}'::jsonb)
                           || jsonb_build_object(
                               'shared_by', CAST(:user_id AS text),
                               'shared_at', NOW()::text,
                               'shared_via', CAST(:via AS text),
                               'shared_from', 'private'
                           ),
                updated_at = NOW()
            WHERE id = :id AND org_id = :org_id AND user_id = :user_id
            """
        ),
        {
            "id": str(parsed),
            "org_id": org_id,
            "user_id": user_id,
            "scope": to,
            "via": via,
        },
    )
    db_session.commit()
    logger.info(
        "memory %s promoted from private to %s by user %s via %s",
        parsed,
        to,
        user_id,
        via,
    )
    return {
        "shared": True,
        "id": str(parsed),
        "scope": to,
        "provenance": row.provenance,
    }
