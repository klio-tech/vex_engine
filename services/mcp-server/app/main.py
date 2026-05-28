"""FastAPI application factory for the Vex MCP server.

This service exposes Vex's memory subsystem over the Model Context
Protocol (MCP). This module is the skeleton: it provides the app
factory and a health endpoint. Auth and MCP tools are layered on in
later tasks.
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app.auth import shutdown_validator, verify_api_key
from app.mcp_app import build_mcp_server
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

    On shutdown we flush and dispose the auth ``KeyValidator`` so its DB
    engine is released cleanly.
    """
    try:
        async with app.state.mcp_server.session_manager.run():
            yield
    finally:
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

    # Mount the MCP Streamable-HTTP ASGI app at the application root.
    # ``/healthz`` is registered above, so it takes precedence; all other
    # paths fall through to the mounted MCP app, whose transport serves
    # the endpoint at ``MCP_ENDPOINT_PATH`` ("/mcp").
    app.mount(MCP_MOUNT_PATH, mcp_server.streamable_http_app())

    return app


app = create_app()
