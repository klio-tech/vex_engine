"""Nango proxy client for Slack API calls.

Uses Nango's REST proxy to make authenticated Slack API requests.
Nango handles OAuth token management and refresh automatically.

Requires NANGO_SECRET_KEY and optionally NANGO_BASE_URL environment variables.
"""

import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger("agentguard.alert-service.nango")

NANGO_BASE_URL = os.environ.get("NANGO_BASE_URL", "https://api.nango.dev")
NANGO_TIMEOUT_S = 10.0


def _get_secret_key() -> Optional[str]:
    """Read NANGO_SECRET_KEY from the environment at call time."""
    return os.environ.get("NANGO_SECRET_KEY")


async def post_slack_message(
    nango_connection_id: str,
    channel_id: str,
    text: str,
    blocks: list[dict[str, Any]],
) -> tuple[bool, Optional[str]]:
    """Post a message to Slack via Nango proxy.

    Args:
        nango_connection_id: The Nango connection ID for this org's Slack OAuth.
        channel_id: The Slack channel ID to post to.
        text: Fallback plain text for the message.
        blocks: Slack Block Kit blocks.

    Returns:
        Tuple of (success, error_message). error_message is None on success.
    """
    secret_key = _get_secret_key()
    if not secret_key:
        logger.warning("NANGO_SECRET_KEY not set, skipping Slack delivery")
        return False, "NANGO_SECRET_KEY not configured"

    payload = {
        "channel": channel_id,
        "text": text,
        "blocks": blocks,
    }

    try:
        async with httpx.AsyncClient(timeout=NANGO_TIMEOUT_S) as client:
            response = await client.post(
                f"{NANGO_BASE_URL}/proxy/chat.postMessage",
                json=payload,
                headers={
                    "Authorization": f"Bearer {secret_key}",
                    "Connection-Id": nango_connection_id,
                    "Provider-Config-Key": "slack",
                },
            )

            if response.status_code != 200:
                logger.error(
                    "Nango proxy returned %d for connection %s",
                    response.status_code,
                    nango_connection_id,
                )
                return False, f"HTTP {response.status_code}"

            data = response.json()
            if not data.get("ok"):
                error = data.get("error", "unknown")
                logger.error(
                    "Slack API error for connection %s: %s",
                    nango_connection_id,
                    error,
                )
                return False, error

            logger.info(
                "Slack message posted via Nango (connection=%s, channel=%s)",
                nango_connection_id,
                channel_id,
            )
            return True, None

    except Exception:
        logger.error(
            "Nango proxy request failed for connection %s",
            nango_connection_id,
            exc_info=True,
        )
        return False, "request_failed"
