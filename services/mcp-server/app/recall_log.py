"""Best-effort recall-event logging for the Klio brain.

Appends a row to brain_recall_events (migration 022) for each recall, so the
dashboard can show per-agent recall activity. SYNCHRONOUS psycopg2 I/O — call via
anyio.to_thread from async handlers (see app.capture). Every failure is swallowed
and logged: logging a recall must NEVER break the recall itself.
"""

import logging
import uuid
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger("agentguard.mcp.recall_log")


def log_recall_event(
    db: Any,
    *,
    org_id: str,
    agent_id: Optional[str],
    project_id: Optional[str],
    query_text: Optional[str],
    result_count: int,
    source: str,
) -> None:
    """Append one brain_recall_events row. Best-effort; never raises."""
    try:
        db.execute(
            text(
                """
                INSERT INTO brain_recall_events
                  (id, org_id, agent_id, project_id, query_text, result_count, source)
                VALUES (:id, :org_id, :agent_id, :project_id, :query_text,
                        :result_count, :source)
                """
            ),
            {
                "id": str(uuid.uuid4()),
                "org_id": org_id,
                "agent_id": agent_id,
                "project_id": project_id,
                "query_text": query_text,
                "result_count": result_count,
                "source": source,
            },
        )
        db.commit()
    except Exception:
        logger.warning("recall log failed for org %s; ignoring", org_id, exc_info=True)
