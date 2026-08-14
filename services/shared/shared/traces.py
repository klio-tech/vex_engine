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
) -> Optional[dict[str, Any]]:
    """Insert or replace-if-richer. A second Stop hook with MORE messages wins;
    a shorter duplicate never clobbers a fuller trace. Best-effort — never
    raises.

    Returns ``{"id": <trace row id>, "previous_ref": <str | None>}`` when a
    row was actually written (fresh insert, or a conflict update that WON
    the richer-wins race) — ``previous_ref`` is whatever ``transcript_ref``
    the row pointed at immediately before this call, so the caller can clean
    up the S3 object it's no longer using.

    Returns ``None`` when nothing was written: either the conflict update
    LOST the richer-wins race (``ON CONFLICT ... WHERE`` matched zero rows —
    detected via ``RETURNING id``, so this function no longer reports a
    locally-generated id for a row that may not exist), or the write failed
    outright (exception, logged and swallowed). Either way, the caller's own
    freshly-written S3 object has nothing pointing at it and should be
    treated as orphaned by the caller (see ``trace_consumer.handle_captured``
    for the delete choreography and its bucket-lifecycle-rule backstop).
    """
    trace_id = str(uuid.uuid4())
    try:
        previous = db.execute(
            text(
                """
                SELECT transcript_ref FROM session_traces
                WHERE org_id = :org_id AND session_id = :session_id
                """
            ),
            {"org_id": org_id, "session_id": session_id},
        ).mappings().first()
        previous_ref = previous["transcript_ref"] if previous else None

        result = db.execute(
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
                RETURNING id
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
        row = result.first()
        db.commit()
        if row is None:
            return None
        return {"id": str(row[0]), "previous_ref": previous_ref}
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
