"""FastAPI application factory for the Vex MCP server.

This service exposes Vex's memory subsystem over the Model Context
Protocol (MCP). This module is the skeleton: it provides the app
factory and a health endpoint. Auth and MCP tools are layered on in
later tasks.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application-level resources.

    Minimal for the skeleton: the MCP server talks to Postgres via a
    per-request synchronous session (see ``app.db.get_db``), so there
    are no async connections to open or close here yet.
    """
    yield


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
