"""Environment configuration for the dashboard API.

Read at call time rather than import time so tests can set the environment
around a request, and so a Railway variable change plus a restart is enough to
rotate the signing secret without a rebuild.
"""

import logging
import os

logger = logging.getLogger("agentguard.dashboard-api.config")

#: Shared HMAC secret used to verify the short-lived WebSocket tokens minted
#: by the dashboard web tier. The web tier holds the same value under
#: ``AGENTGUARD_WS_TOKEN_SECRET``.
WS_TOKEN_SECRET_ENV = "DASHBOARD_WS_TOKEN_SECRET"


def ws_token_secret() -> str:
    """Return the WebSocket token signing secret, or ``""`` if unset.

    An empty secret is not an error here and is not an error at startup
    either: ``/health`` must keep answering so the platform health check
    passes and the service is not flapped by a missing variable.

    It IS fatal to every handshake. ``ws_auth.verify_ws_token`` refuses an
    empty secret outright, so an unconfigured deployment serves nobody rather
    than serving everybody. That direction of failure is the whole point.
    """
    secret = os.environ.get(WS_TOKEN_SECRET_ENV, "")

    if not secret:
        logger.error(
            "%s is not set. Every WebSocket handshake will be refused until "
            "it is configured with the same value as the web tier's "
            "AGENTGUARD_WS_TOKEN_SECRET.",
            WS_TOKEN_SECRET_ENV,
        )

    return secret
