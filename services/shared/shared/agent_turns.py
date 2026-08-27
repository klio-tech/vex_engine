"""Producer/consumer contract for agent turns.

``mcp-server`` publishes a turn when the workspace says an agent was
mentioned; ``async-worker`` consumes it and runs the turn. A stream with a
consumer group rather than a plain queue, matching the evidence pipeline:
a turn that kills its worker is redelivered rather than lost, and two
workers never run the same turn twice.

Publishing raises on failure. Whether a failed publish should be fatal is
the CALLER's judgement — for a mention it is not, because a message must
never fail to send just because an agent could not be woken — but this
module must not decide that silently, or a broken Redis would look
identical to a healthy one with nothing to run.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger("klio.agent_turns")

AGENT_TURNS = "agents.turns"

#: Consumer group name. One group, so adding workers spreads turns rather
#: than duplicating them.
AGENT_TURNS_GROUP = "agent-runner"

#: A turn payload is small by construction — it carries IDS, never content.
#: The worker re-reads the message itself, so a turn cannot act on a stale
#: copy of text that was edited or removed between publish and run.
MAX_PAYLOAD_BYTES = 4096

_redis: Optional[Any] = None


def _get_redis():
    global _redis
    if _redis is None:
        import redis.asyncio as aioredis  # noqa: PLC0415

        from shared.redis_config import REDIS_CLIENT_OPTIONS  # noqa: PLC0415

        url = os.environ.get("REDIS_URL")
        if not url:
            raise RuntimeError("REDIS_URL is not configured")
        _redis = aioredis.from_url(url, **REDIS_CLIENT_OPTIONS)
    return _redis


def build_turn(
    *,
    workspace_slug: str,
    channel_id: str,
    message_id: str,
    agent_id: str,
) -> dict[str, str]:
    """The turn payload: identifiers only, no content."""
    return {
        "workspace_slug": workspace_slug,
        "channel_id": channel_id,
        "message_id": message_id,
        "agent_id": agent_id,
    }


async def publish_turn(payload: dict[str, str]) -> None:
    """XADD one turn. Raises on failure; see the module docstring."""
    encoded = json.dumps(payload, ensure_ascii=False)
    if len(encoded.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        # Cannot happen with ids alone, so if it does the shape has drifted
        # and the loud failure is the point.
        raise ValueError("agent turn payload is unexpectedly large")
    redis = _get_redis()
    await redis.xadd(AGENT_TURNS, {"payload": encoded})
