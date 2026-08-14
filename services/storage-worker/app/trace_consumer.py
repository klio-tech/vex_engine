"""sessions.captured -> S3 + session_traces row -> sessions.stored.

Payloads arrive ALREADY REDACTED (see the ``evidence_stream`` producer
contract in ``shared.evidence_stream``). ``KLIO_TRACE_BUCKET`` is required —
refusing to start beats silently writing retained transcripts to a default
bucket nobody chose.

Every call writes a NEW S3 object before it knows whether
``traces.upsert_trace`` will actually point the row at it (richer-wins: a
second Stop hook with fewer messages loses the race). To avoid orphaning
objects nothing points to:

- If the upsert LOST the race (or failed outright), the object this call
  just wrote is deleted — a richer trace, written by someone else, already
  has the row; grading can proceed against it, so ``sessions.stored`` is
  still emitted.
- If the upsert WON and a ``previous_ref`` existed (a real replace, not a
  fresh insert), the object the row pointed at *before* this call is
  deleted.

Both deletes are best-effort: a delete failure is logged and swallowed,
never turned into a consumer failure. The S3 bucket's own lifecycle
expiry rule is the backstop for anything a failed delete leaves behind —
see the bucket's retention policy, not this module, for the hard guarantee.

That backstop also covers a narrower, non-failure gap: ``previous_ref`` is
read non-locking (see ``shared.traces.upsert_trace``), so two consumers
racing the same session can both "win" in sequence. The second winner's
``previous_ref`` can be stale — pointing at an object the first winner
already replaced — so its delete call succeeds against an already-deleted
key while the first winner's now-orphaned intermediate object is never
targeted at all. This is accepted as best-effort rather than fixed with
locking; the lifecycle rule is what actually reclaims that object. Session
data is never lost — only cleanup of an intermediate S3 object can miss.
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


def _parse_ref(ref: str) -> tuple[str, str]:
    """``"s3://bucket/key"`` -> ``(bucket, key)``."""
    bucket, key = ref[len("s3://") :].split("/", 1)
    return bucket, key


def _delete_object_best_effort(s3: Any, ref: str, *, reason: str) -> None:
    """Delete the S3 object at ``ref``, logging (never raising) on failure.

    Best-effort: a delete that fails here leaves an orphan the bucket's
    lifecycle expiry rule will eventually clean up — see the module
    docstring.
    """
    try:
        bucket, key = _parse_ref(ref)
        s3.delete_object(Bucket=bucket, Key=key)
    except Exception:
        logger.warning(
            "trace-consumer: failed to delete orphaned object %s (%s)",
            ref,
            reason,
            exc_info=True,
        )


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
        bucket = _bucket()
        key = f"traces/{org_id}/{session_id}/{uuid.uuid4()}.json"
        new_ref = f"s3://{bucket}/{key}"
        body = json.dumps(
            {"messages": messages, "tool_calls": tool_calls}, ensure_ascii=False
        ).encode("utf-8")
        s3.put_object(Bucket=bucket, Key=key, Body=body)
        result = traces.upsert_trace(
            db,
            org_id=org_id,
            session_id=session_id,
            transcript_ref=new_ref,
            message_count=len(messages),
            tool_calls_count=len(tool_calls),
            redaction_version=os.environ.get("REDACTION_VERSION", "1"),
            project_id=payload.get("project_id"),
            user_id=payload.get("user_id"),
            agent_id=payload.get("agent_id"),
        )
        if result is None:
            # Lost the richer-wins race (or the write failed outright): no
            # row points at the object we just wrote. A richer trace already
            # holds the row in the "lost" case, so evidence for this session
            # is still available — grading can proceed, hence we still emit
            # sessions.stored below rather than treating this as a failure.
            _delete_object_best_effort(s3, new_ref, reason="lost upsert race")
        else:
            previous_ref = result.get("previous_ref")
            if previous_ref and previous_ref != new_ref:
                _delete_object_best_effort(
                    s3, previous_ref, reason="superseded by richer trace"
                )
        await redis.xadd(
            SESSIONS_STORED,
            {"payload": json.dumps({"org_id": org_id, "session_id": session_id})},
        )
        return True
    except Exception:
        logger.warning(
            "trace-consumer: persist failed org=%s session=%s",
            org_id,
            session_id,
            exc_info=True,
        )
        return False
