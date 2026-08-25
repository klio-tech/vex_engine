"""The one pooled engine for the workspace database.

Membership resolution (:mod:`shared.membership`) reads ``workspaces`` and
``workspace_members``, which live in the Plane workspace database behind
``WORKSPACE_DATABASE_URL`` — a different server from the engine's own
``DATABASE_URL``. This module owns the only bounded, pooled connection to it
in the process, mirroring :mod:`app.directory`'s pattern for the Supabase
engine: a second ``create_engine`` for the same URL elsewhere would double the
connection footprint and give the boundary check its own, differently
configured timeouts.

``None`` is not "skip the check": every caller must fail closed on it. An
unconfigured workspace DB means membership cannot be established, and a node
selection that cannot be checked is a node selection that is refused.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

from shared.engine_bounds import engine_bounds
from sqlalchemy import create_engine

logger = logging.getLogger("agentguard.mcp.workspace_db")

WORKSPACE_DATABASE_URL = os.environ.get("WORKSPACE_DATABASE_URL")

_engine_lock = threading.Lock()
_workspace_engine: Optional[Any] = None


def workspace_engine() -> Optional[Any]:
    """The workspace DB engine, or ``None`` when none is configured.

    Lazily built and reused for the life of the process. Bounded via
    ``shared.engine_bounds``: this sits on the request path, so an unreachable
    host must fail rather than hang.
    """
    global _workspace_engine
    if not WORKSPACE_DATABASE_URL:
        return None
    with _engine_lock:
        if _workspace_engine is None:
            _workspace_engine = create_engine(
                WORKSPACE_DATABASE_URL,
                pool_pre_ping=True,
                pool_size=2,
                max_overflow=2,
                **engine_bounds(),
            )
    return _workspace_engine


def shutdown_workspace_db() -> None:
    """Dispose the engine. Called at app shutdown."""
    global _workspace_engine
    with _engine_lock:
        if _workspace_engine is not None:
            _workspace_engine.dispose()
        _workspace_engine = None
