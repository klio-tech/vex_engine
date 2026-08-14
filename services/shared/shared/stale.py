"""Retroactive stale-serve marking — the honest version.

A serve is stale ONLY if the superseding belief already existed when the old
one was served (curator lag). Serves before the new belief existed were
correct at the time and are never marked. This is the silent-failure counter;
its integrity is the point.
"""

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import text

logger = logging.getLogger("klio.stale")


def mark_stale_serves(
    db: Any,
    *,
    org_id: str,
    old_memory_id: str,
    superseding_created_at: datetime,
) -> int:
    """Mark past recall serves of ``old_memory_id`` stale where the
    superseding belief already existed when they were served.

    Best-effort: never raises. A ledger write failing must never take down
    the supersession it is describing, so any error here is logged and
    swallowed, returning ``0``.
    """
    try:
        result = db.execute(
            text(
                """
                UPDATE recall_outcomes ro
                SET served_stale = true
                FROM brain_recall_events bre
                WHERE ro.recall_event_id = bre.id
                  AND ro.org_id = :org_id
                  AND bre.org_id = :org_id
                  AND ro.memory_id = :old_memory_id
                  AND :old_memory_id = ANY(bre.memory_ids)
                  AND bre.created_at >= :superseding_created_at
                """
            ),
            {
                "org_id": org_id,
                "old_memory_id": old_memory_id,
                "superseding_created_at": superseding_created_at,
            },
        )
        marked = int(result.rowcount or 0)
        db.commit()
        return marked
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning(
            "stale marking failed org=%s memory=%s",
            org_id,
            old_memory_id,
            exc_info=True,
        )
        return 0
