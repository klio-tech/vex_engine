"""sessions.captured -> S3 + session_traces row -> sessions.stored.

Payloads arrive ALREADY REDACTED (see the ``evidence_stream`` producer
contract in ``shared.evidence_stream``). ``KLIO_TRACE_BUCKET`` is required —
refusing to start beats silently writing retained transcripts to a default
bucket nobody chose.
"""

import json
import logging
import os
import uuid
from typing import Any

from shared import traces
from shared.evidence_stream import SESSIONS_STORED

logger = logging.getLogger("agentguard.trace-consumer")


def _bucket() -> str:
    bucket = os.environ.get("KLIO_TRACE_BUCKET")
    if not bucket:
        raise RuntimeError("KLIO_TRACE_BUCKET is required for trace retention")
    return bucket


async def handle_captured(db: Any, s3: Any, redis: Any, entry: dict) -> bool:
    """Persist one ``sessions.captured`` stream entry; emit ``sessions.stored``.

    Best-effort per entry: a malformed entry (bad JSON, missing required
    keys) is logged and skipped — it returns ``False`` but never raises, so
    one bad message never takes down the consumer loop.
    """
    try:
        payload = json.loads(entry["payload"])
        org_id, session_id = payload["org_id"], payload["session_id"]
        messages = payload["messages"]
        tool_calls = payload["tool_calls"]
    except Exception:
        logger.warning("trace-consumer: malformed entry skipped", exc_info=True)
        return False

    try:
        key = f"traces/{org_id}/{session_id}/{uuid.uuid4()}.json"
        body = json.dumps(
            {"messages": messages, "tool_calls": tool_calls}, ensure_ascii=False
        ).encode("utf-8")
        bucket = _bucket()
        s3.put_object(Bucket=bucket, Key=key, Body=body)
        traces.upsert_trace(
            db,
            org_id=org_id,
            session_id=session_id,
            transcript_ref=f"s3://{bucket}/{key}",
            message_count=len(messages),
            tool_calls_count=len(tool_calls),
            redaction_version=os.environ.get("REDACTION_VERSION", "1"),
            project_id=payload.get("project_id"),
            user_id=payload.get("user_id"),
            agent_id=payload.get("agent_id"),
        )
        await redis.xadd(
            SESSIONS_STORED,
            {"payload": json.dumps({"org_id": org_id, "session_id": session_id})},
        )
        return True
    except Exception:
        logger.warning(
            "trace-consumer: persist failed org=%s session=%s", org_id, session_id,
            exc_info=True,
        )
        return False
