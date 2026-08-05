"""FastAPI application factory for the Vex MCP server.

This service exposes Vex's memory subsystem over the Model Context
Protocol (MCP). This module is the skeleton: it provides the app
factory and a health endpoint. Auth and MCP tools are layered on in
later tasks.
"""

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app import config, oauth_middleware
from app.auth import shutdown_validator, verify_api_key
from app.capture import router as capture_router
from app.observability import log_remote_write_state
from app.observability import router as observability_router
from app.db import SessionLocal
from app.mcp_app import build_mcp_server
from app.scheduler import (
    start_artifact_gc,
    start_curator,
    stop_artifact_gc,
    stop_curator,
)
from shared.auth import KeyInfo

# The MCP Streamable-HTTP ASGI app is mounted at the application root.
# Combined with the server's default ``streamable_http_path``, the
# transport endpoint is served at this path.  Mounting at root (rather
# than under a ``/mcp`` prefix) avoids a 307 trailing-slash redirect that
# the MCP client does not follow on POST.
MCP_ENDPOINT_PATH = "/mcp"
MCP_MOUNT_PATH = "/"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application-level resources.

    The MCP server talks to Postgres via a per-request synchronous
    session (see ``app.db.get_db``), so there are no async connections
    to open here.  The Streamable-HTTP transport, however, requires its
    session manager to be running for the lifetime of the app — without
    this context the mounted ``/mcp`` route accepts connections but
    cannot service MCP requests.

    The per-app MCP server is read from ``app.state`` (set in
    :func:`create_app`).  Its ``session_manager.run()`` is one-shot per
    instance, which is exactly why each app owns its own server rather
    than sharing a module-level singleton.

    On shutdown we cancel the background curator loop and the artifact GC
    loop, and flush/dispose the auth ``KeyValidator`` so its DB engine is
    released cleanly.

    The curator scheduler and the artifact GC sweep (both in
    ``app.scheduler``) are started here so the single-service deployment
    needs no external cron; each sleeps before its first pass, so entering
    and promptly exiting this context never triggers a real pass.
    """
    log_remote_write_state()
    start_curator(app)
    start_artifact_gc(app, SessionLocal)
    try:
        async with app.state.mcp_server.session_manager.run():
            yield
    finally:
        await stop_artifact_gc(app)
        await stop_curator(app)
        shutdown_validator()


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application instance.

    A fresh MCP server is built per app (its session manager's ``run()``
    can only be entered once per instance, so a shared singleton would
    break a second app), stashed on ``app.state.mcp_server``, and its
    Streamable-HTTP transport is mounted at the app root and served at
    :data:`MCP_ENDPOINT_PATH`.  The session manager is driven by
    :func:`lifespan`.
    """
    app = FastAPI(
        title="Vex MCP Server",
        version="0.1.0",
        lifespan=lifespan,
    )

    mcp_server = build_mcp_server()
    app.state.mcp_server = mcp_server

    @app.get("/healthz")
    def healthz():
        """Liveness. Deliberately touches nothing.

        This answers "is the process alive?", not "can it serve?". It must
        never query the database: a liveness probe that follows a dependency
        down makes the orchestrator restart every replica in a loop while the
        dependency — not the process — is what is broken.

        The readiness question lives at ``/readyz`` (see
        :mod:`app.observability`), which does reach the database, and which is
        what should go red during a database outage.
        """
        return {"status": "ok"}

    @app.get("/verify")
    def verify(key_info: KeyInfo = Depends(verify_api_key)):
        """Validate the caller's ``X-Vex-Key``.

        Returns 200 with the resolved org + scopes when the key is valid AND
        carries the required ``memory`` scope (the shared validator is built
        with ``required_scope="memory"``); otherwise the dependency raises 401
        (missing/invalid/expired) or 403 (key lacks the ``memory`` scope).

        This is the lightweight check ``klio init`` (cloud mode) calls to
        confirm a pasted API key before writing the agent MCP config — no MCP
        handshake required.
        """
        return {
            "valid": True,
            "org_id": key_info.org_id,
            "scopes": key_info.scopes,
        }

    # --- MCP OAuth (flag-gated; spec 2026-06-10-mcp-oauth-design.md) -------
    if config.MCP_OAUTH_ENABLED:
        # Fail fast: refuse to start with an incomplete OAuth configuration.
        # Without these three values the server will 500 on the first Bearer
        # request, which is harder to diagnose than an explicit startup error.
        missing = [
            name
            for name, val in (
                ("SUPABASE_AUTH_ISSUER", config.SUPABASE_AUTH_ISSUER),
                ("MCP_PUBLIC_URL", config.MCP_PUBLIC_URL),
                ("SUPABASE_DATABASE_URL", os.environ.get("SUPABASE_DATABASE_URL")),
            )
            if not val
        ]
        if missing:
            raise RuntimeError(
                f"MCP_OAUTH_ENABLED=1 but required config missing: {', '.join(missing)}"
            )

        def oauth_protected_resource():
            """RFC 9728 metadata: points MCP clients at the Supabase AS."""
            return oauth_middleware.protected_resource_metadata()

        # Serve at BOTH the bare well-known path and the RFC 9728 path-aware
        # location for the ``/mcp`` resource. Clients that follow the
        # WWW-Authenticate hint use the path-aware URL; others probe the bare
        # path. Both return identical metadata. Registered before the MCP mount
        # so the path-aware route is not swallowed by the catch-all transport.
        app.get(oauth_middleware.METADATA_PATH)(oauth_protected_resource)
        app.get(oauth_middleware.METADATA_PATH_FOR_RESOURCE)(oauth_protected_resource)

    # REST capture layer (passive memory for the thin `klio hook` client).
    # Registered BEFORE the root MCP mount so its ``/capture/*`` routes take
    # precedence over the catch-all mount below.
    app.include_router(capture_router)

    # ``/readyz`` and ``/metrics``. Registered before the root MCP mount for the
    # same reason as the capture routes: the mount is a catch-all and would
    # otherwise swallow them.
    app.include_router(observability_router)

    # Mount the MCP Streamable-HTTP ASGI app at the application root.
    # ``/healthz`` / ``/verify`` / ``/capture/*`` are registered above, so they
    # take precedence; all other paths fall through to the mounted MCP app,
    # whose transport serves the endpoint at ``MCP_ENDPOINT_PATH`` ("/mcp").
    mcp_asgi = mcp_server.streamable_http_app()
    if config.MCP_OAUTH_ENABLED:
        mcp_asgi = oauth_middleware.OAuthLadderMiddleware(
            mcp_asgi, mcp_path=MCP_ENDPOINT_PATH
        )
    app.mount(MCP_MOUNT_PATH, mcp_asgi)

    return app


app = create_app()
