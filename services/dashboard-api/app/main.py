"""FastAPI application factory for the Vex Dashboard API.

Provides:
- ``GET /health`` -- service health check.
- ``WS /ws`` -- authenticated, org-scoped WebSocket feed of execution updates.

A background task subscribes to the ``executions.stored`` Redis Stream and
routes each event to the connections belonging to that event's organisation.

``/ws`` requires a short-lived signed token, offered as a
``Sec-WebSocket-Protocol`` value (preferred) or ``?token=``. See
``app.ws_auth`` for the format and the reasoning; the token is minted by the
dashboard web tier after it authenticates the session and checks account
membership. There is deliberately no CORS middleware: WebSocket handshakes are
not subject to the same-origin policy, so an allow-list of origins would be a
comfort rather than a control. The token is the control.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import WS_TOKEN_SECRET_ENV, ws_token_secret
from app.websocket import stream_updates, websocket_endpoint

logger = logging.getLogger("agentguard.dashboard-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle: start/stop the Redis Stream listener."""
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")

    # Surface a missing signing secret once, loudly, at boot -- rather than
    # only as a per-handshake refusal that looks like a client problem.
    # Deliberately not fatal: the service must stay up and keep answering
    # /health. It simply accepts no connections until the secret is set.
    if not ws_token_secret():
        logger.critical(
            "Dashboard API starting WITHOUT %s. The /ws feed will refuse "
            "every connection.",
            WS_TOKEN_SECRET_ENV,
        )

    # Start the background stream listener
    stream_task = asyncio.create_task(stream_updates(redis_url))
    logger.info("Dashboard API started, Redis stream listener active")

    yield

    # Shutdown: cancel the background task
    stream_task.cancel()

    try:
        await stream_task
    except asyncio.CancelledError:
        pass

    logger.info("Dashboard API shutting down")


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application instance."""
    app = FastAPI(
        title="Vex Dashboard API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health_check():
        """Return service health status."""
        return {"status": "healthy"}

    app.add_api_websocket_route("/ws", websocket_endpoint)

    return app


app = create_app()
