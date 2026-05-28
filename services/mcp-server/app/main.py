"""FastAPI application factory for the Vex MCP server.

This service exposes Vex's memory subsystem over the Model Context
Protocol (MCP). This module is the skeleton: it provides the app
factory and a health endpoint. Auth and MCP tools are layered on in
later tasks.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.auth import shutdown_validator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application-level resources.

    The MCP server talks to Postgres via a per-request synchronous
    session (see ``app.db.get_db``), so there are no async connections
    to open here.  On shutdown we flush and dispose the auth
    ``KeyValidator`` so its DB engine is released cleanly.
    """
    try:
        yield
    finally:
        shutdown_validator()


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application instance."""
    app = FastAPI(
        title="Vex MCP Server",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    return app


app = create_app()
