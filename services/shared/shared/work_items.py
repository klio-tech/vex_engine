"""Resolve a work item to something a memory may be filed under.

A work item is the finest grain a memory can carry (migration 051's
``session_memories.work_item_id``). It is a FILING dimension, never a
boundary: a work item's visibility in the workspace is exactly its project's
visibility, so the access predicate stays ``scope`` and the project ACL,
untouched. What this module decides is narrower and purely structural — *is
this work item a real one, in a project this org owns?*

**Two databases, joined in Python.** ``issues`` and ``projects`` (the
workspace's own) live in the workspace DB (``WORKSPACE_DATABASE_URL``);
``projects`` (the engine's) lives in ``DATABASE_URL``. They are separate
servers, so a single SQL statement naming both cannot run — however well it
works against a test database that happens to hold both. The join key is
``projects.plane_project_id``, the same link ``project_link`` writes.

Fails closed, always. An unknown work item, a soft-deleted one, a work item
whose project is not linked, and a work item in ANOTHER org all return
``None`` — the identical shape, so this resolver is never an existence
oracle for work items the caller cannot reach.

Synchronous (blocking psycopg2 I/O), matching the house pattern in
:mod:`shared.projects`. Call from async handlers via
``anyio.to_thread.run_sync``.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger("agentguard.shared.work_items")

#: ``PROJ-123``: a project's identifier, a hyphen, and the work item's
#: ``sequence_id``. This is the handle a human reads off the UI, so it is
#: accepted everywhere a raw uuid is. Case-insensitive on the identifier.
_HUMAN_REF = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)-(\d+)$")

#: Workspace DB: a work item by its uuid. ``deleted_at`` and ``is_draft`` are
#: both excluded — a draft is not yet a work item anyone else can see, and a
#: deleted one must not accept new memories.
_ITEM_BY_ID_SQL = text(
    """
    SELECT i.id, i.name, i.sequence_id, i.project_id, i.workspace_id,
           p.identifier AS project_identifier
    FROM issues i
    JOIN projects p ON p.id = i.project_id
    WHERE i.id = CAST(:item_id AS uuid)
      AND i.deleted_at IS NULL
      AND i.is_draft = false
      AND p.deleted_at IS NULL
    LIMIT 1
    """
)

#: Workspace DB: a work item by ``PROJ-123``. ``sequence_id`` is unique only
#: WITHIN a project, which is exactly why the identifier is half the key.
_ITEM_BY_REF_SQL = text(
    """
    SELECT i.id, i.name, i.sequence_id, i.project_id, i.workspace_id,
           p.identifier AS project_identifier
    FROM issues i
    JOIN projects p ON p.id = i.project_id
    WHERE UPPER(p.identifier) = UPPER(:identifier)
      AND i.sequence_id = :sequence_id
      AND i.deleted_at IS NULL
      AND i.is_draft = false
      AND p.deleted_at IS NULL
    LIMIT 1
    """
)

#: Engine DB: the linked engine project for a workspace project, in THIS org.
#: The org predicate is what stops a work item id borrowed from another tenant
#: resolving here.
_LINKED_PROJECT_SQL = text(
    """
    SELECT id
    FROM projects
    WHERE plane_project_id = CAST(:plane_project_id AS uuid)
      AND org_id = :org_id
    LIMIT 1
    """
)


def _clean(value: Any) -> str:
    return str(value).strip()


def resolve_work_item(
    db_session: Any,
    workspace_conn: Any,
    *,
    org_id: str,
    identifier: str,
) -> Optional[dict]:
    """Resolve ``identifier`` to a work item this org may file memories under.

    Args:
        db_session: The ENGINE database session (``DATABASE_URL``).
        workspace_conn: A connection to the WORKSPACE database. ``None`` is
            not "skip the check" — it means the work item cannot be
            established, and this returns ``None``.
        org_id: The caller's resolved tenant. Never caller input.
        identifier: A work item uuid, or a ``PROJ-123`` human reference.

    Returns:
        ``{"id", "name", "sequence_id", "reference", "project_id",
        "plane_project_id", "workspace_id"}`` where ``project_id`` is the
        ENGINE project id, or ``None`` if the work item does not resolve into
        this org.
    """
    if workspace_conn is None or not identifier or not org_id:
        return None
    text_id = _clean(identifier)
    if not text_id:
        return None

    row = None
    try:
        try:
            item_uuid = str(uuid.UUID(text_id))
        except (ValueError, AttributeError, TypeError):
            item_uuid = None

        if item_uuid is not None:
            row = workspace_conn.execute(
                _ITEM_BY_ID_SQL, {"item_id": item_uuid}
            ).fetchone()
        else:
            match = _HUMAN_REF.match(text_id)
            if match is None:
                return None
            row = workspace_conn.execute(
                _ITEM_BY_REF_SQL,
                {
                    "identifier": match.group(1),
                    "sequence_id": int(match.group(2)),
                },
            ).fetchone()
    except Exception:
        # Same degradation as every other workspace-DB read on this path: the
        # work item cannot be established, so it is not filed. Never raises
        # into a write.
        logger.warning(
            "work item lookup failed for %r in org %s", text_id, org_id, exc_info=True
        )
        return None

    if row is None:
        return None

    try:
        linked = db_session.execute(
            _LINKED_PROJECT_SQL,
            {"plane_project_id": str(row.project_id), "org_id": _clean(org_id)},
        ).fetchone()
    except Exception:
        logger.warning(
            "linked-project lookup failed for work item %s in org %s",
            text_id,
            org_id,
            exc_info=True,
        )
        return None

    if linked is None:
        # The work item is real, but its project was never linked into this
        # org — so there is no engine project to file the memory against, and
        # filing it against nothing would make it unreachable. Refuse.
        return None

    return {
        "id": str(row.id),
        "name": row.name,
        "sequence_id": int(row.sequence_id),
        "reference": f"{row.project_identifier}-{row.sequence_id}",
        "project_id": str(linked.id),
        "plane_project_id": str(row.project_id),
        "workspace_id": str(row.workspace_id),
    }
