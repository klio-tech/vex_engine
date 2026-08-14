"""Producer/consumer contract for the evidence pipeline.

mcp-server publishes REDACTED payloads to ``sessions.captured``;
storage-worker persists them (see the ``session_traces`` table, migration
041) and emits ``sessions.stored``; the grader consumes that. Redaction
failure means NO publish — unscreened text never leaves the process (the
``recall_log`` contract: see :mod:`app.recall_log` in mcp-server for the
established log-message style this module mirrors).

Evidence publishing is best-effort from the caller's point of view:
:func:`publish_captured` raises on failure (a Redis outage, an oversized
XADD, etc.) so the failure is observable, but the capture request itself
must never fail because of it — that is the caller's responsibility, not
this module's (see ``app.capture.capture_transcript``).
"""

import json
import logging
import os
from typing import Any, Optional

from shared.redaction import redact

logger = logging.getLogger("klio.evidence")

SESSIONS_CAPTURED = "sessions.captured"
SESSIONS_STORED = "sessions.stored"

#: Hard cap on the redacted payload, in bytes, enforced post-redaction. Kept
#: well under Redis's own limits and any downstream consumer's expectations.
#: A session that exceeds this is tail-trimmed (oldest messages dropped
#: first) rather than rejected outright — partial evidence beats none.
_MAX_PAYLOAD_BYTES = 256 * 1024

#: Default Redis URL for local development, matching async-worker's default
#: (see ``services/async-worker/app/main.py``).
_DEFAULT_REDIS_URL = "redis://localhost:6379"

#: Lazily created, process-wide aioredis client. Created on first use so
#: import-time never depends on Redis being reachable (mirrors the lazy-init
#: pattern already used for the DB engine elsewhere in this codebase).
_redis_client: Optional[Any] = None


def _get_redis() -> Any:
    """Return the lazy aioredis client used to publish evidence events.

    Points at ``REDIS_URL`` — the same Railway Redis instance async-worker
    already reads from (``services/async-worker/app/main.py``) — falling back
    to the local-dev default when unset. Created once per process and reused;
    a fresh client per call would defeat Redis connection pooling and drop
    the retry/health-check tuning in :data:`shared.redis_config.REDIS_CLIENT_OPTIONS`.
    """
    global _redis_client
    if _redis_client is None:
        import redis.asyncio as aioredis

        from shared.redis_config import REDIS_CLIENT_OPTIONS

        redis_url = os.environ.get("REDIS_URL", _DEFAULT_REDIS_URL)
        _redis_client = aioredis.from_url(redis_url, **REDIS_CLIENT_OPTIONS)
    return _redis_client


def _payload_size(payload: dict[str, Any]) -> int:
    """Size, in bytes, of ``payload`` as it will actually be serialized."""
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def build_captured_payload(
    *,
    org_id: str,
    session_id: str,
    messages: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Build the redacted ``sessions.captured`` payload for one transcript.

    Every message's ``content`` and every tool call's ``args`` are passed
    through :func:`shared.redaction.redact` BEFORE anything is assembled into
    the returned payload. If redaction raises for any reason, the entire
    payload is dropped — this function returns ``None`` and NOTHING is
    published — because the alternative is unscreened text leaving the
    process. This mirrors the ``recall_log`` contract (``app.recall_log`` in
    mcp-server): a broken redactor degrades to "no evidence event", never to
    "unredacted evidence event".

    The returned payload is capped at 256 KB post-redaction. When it would
    exceed the cap, messages are tail-trimmed — the OLDEST messages are
    dropped first — until it fits, so a long session degrades to "recent
    context" rather than being dropped entirely. If trimming every message
    still leaves the payload oversized (e.g. a small number of huge tool
    calls), ``tool_calls`` is trimmed the same way, oldest first, as a
    fallback — the newest tool calls are the most likely to matter for
    grading what an agent just did.
    """
    try:
        clean_messages = [
            {
                "role": m.get("role") or "user",
                "content": redact(str(m.get("content", ""))).clean_text,
            }
            for m in messages
        ]
        clean_tools = [
            {
                "tool": t.get("tool", ""),
                "args": redact(
                    json.dumps(t.get("args", {}), ensure_ascii=False)
                ).clean_text,
                "status": t.get("status"),
            }
            for t in tool_calls
        ]
    except Exception:
        logger.warning(
            "evidence: redaction failed for org %s session %s; payload dropped "
            "(content withheld — it is unscreened)",
            org_id,
            session_id,
            exc_info=True,
        )
        return None

    payload: dict[str, Any] = {
        "org_id": org_id,
        "session_id": session_id,
        "project_id": project_id,
        "user_id": user_id,
        "agent_id": agent_id,
        "messages": clean_messages,
        "tool_calls": clean_tools,
    }

    # Tail-trim: drop the oldest message first until the redacted payload
    # fits the cap. Bounded by len(messages) iterations so a pathological
    # single-message payload that is still oversized cannot loop forever —
    # it is published as-is (best-effort; the alternative is silent loss).
    while payload["messages"] and _payload_size(payload) > _MAX_PAYLOAD_BYTES:
        payload["messages"].pop(0)

    # Fallback: messages alone were not enough (e.g. one or two huge tool
    # calls). Trim tool_calls the same way — oldest first — so the newest
    # actions survive; those are what a grader is most likely to need.
    while payload["tool_calls"] and _payload_size(payload) > _MAX_PAYLOAD_BYTES:
        payload["tool_calls"].pop(0)

    return payload


async def publish_captured(*, payload: dict[str, Any]) -> None:
    """XADD the redacted ``payload`` to :data:`SESSIONS_CAPTURED`.

    Raises on failure. Evidence publishing is best-effort from the product's
    point of view — a Redis outage must never fail a capture request — but
    that trade-off belongs to the caller (``app.capture.capture_transcript``),
    which wraps this call in its own try/except. This function itself must
    not silently swallow errors, or a broken Redis connection would look
    identical to a healthy one with nothing to publish.
    """
    redis = _get_redis()
    await redis.xadd(
        SESSIONS_CAPTURED, {"payload": json.dumps(payload, ensure_ascii=False)}
    )
