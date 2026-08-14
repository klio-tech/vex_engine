"""Trace persistence primitives (row upsert + fetch). Called by storage-worker.

Rows point at one S3 object holding the REDACTED structured payload
``{messages, tool_calls}`` (see migration 041, ``session_traces``). Both
functions here are best-effort: a DB hiccup degrades to "no row" / "no
trace found" rather than raising into the consumer loop that called them.
"""

import json
import logging
import uuid
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger("klio.traces")


def upsert_trace(
    db: Any,
    *,
    org_id: str,
    session_id: str,
    transcript_ref: str,
    message_count: int,
    tool_calls_count: int,
    redaction_version: str,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Optional[str]:
    """Insert or replace-if-richer. A second Stop hook with MORE messages wins;
    a shorter duplicate never clobbers a fuller trace. Best-effort."""
    trace_id = str(uuid.uuid4())
    try:
        db.execute(
            text(
                """
                INSERT INTO session_traces
                  (id, org_id, project_id, user_id, agent_id, session_id,
                   transcript_ref, message_count, tool_calls_count,
                   redaction_version)
                VALUES (:id, :org_id, :project_id, :user_id, :agent_id,
                        :session_id, :transcript_ref, :message_count,
                        :tool_calls_count, :redaction_version)
                ON CONFLICT (org_id, session_id) DO UPDATE SET
                  transcript_ref = EXCLUDED.transcript_ref,
                  message_count = EXCLUDED.message_count,
                  tool_calls_count = EXCLUDED.tool_calls_count,
                  redaction_version = EXCLUDED.redaction_version,
                  updated_at = now()
                WHERE EXCLUDED.message_count > session_traces.message_count
                """
            ),
            {
                "id": trace_id,
                "org_id": org_id,
                "project_id": project_id,
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "transcript_ref": transcript_ref,
                "message_count": message_count,
                "tool_calls_count": tool_calls_count,
                "redaction_version": redaction_version,
            },
        )
        db.commit()
        return trace_id
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning(
            "traces: upsert failed org=%s session=%s", org_id, session_id,
            exc_info=True,
        )
        return None


def fetch_trace(
    db: Any, s3: Any, *, org_id: str, session_id: str
) -> Optional[dict[str, Any]]:
    """Look up the trace row for (org_id, session_id) and load its S3 body.

    Always predicated on org_id — never on session_id alone — so a caller
    cannot fetch another org's trace by guessing/leaking a session_id.
    """
    try:
        row = db.execute(
            text(
                """
                SELECT id, transcript_ref, message_count, created_at
                FROM session_traces
                WHERE org_id = :org_id AND session_id = :session_id
                """
            ),
            {"org_id": org_id, "session_id": session_id},
        ).mappings().first()
        if row is None:
            return None
        bucket, key = row["transcript_ref"][len("s3://"):].split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return {
            "messages": data.get("messages", []),
            "tool_calls": data.get("tool_calls", []),
            "row": dict(row),
        }
    except Exception:
        logger.warning(
            "traces: fetch failed org=%s session=%s", org_id, session_id,
            exc_info=True,
        )
        return None
