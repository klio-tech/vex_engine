"""In-process background scheduler for the Klio brain curator.

The curator (:mod:`app.curator`) synthesises each org's RAW memories into
``summary`` / ``fact`` rows.  :func:`app.curator.run_curator_once` is a single
pass; this module drives it on a periodic loop started from the FastAPI
lifespan (:mod:`app.main`), so the single-service Railway deployment needs no
external cron.

Design notes:

* **Sleep-before-tick.** The loop sleeps the configured interval *before* the
  first pass.  This means a short-lived app (a test entering/exiting the
  lifespan, a quick restart) never triggers a real curation pass — the task is
  cancelled during its first sleep — so the scheduler is inert unless the
  process actually lives long enough to warrant a pass.
* **Crash-isolated.** A failure inside one pass is logged and swallowed; the
  loop keeps running so one bad tick never silently stops curation.
  ``CancelledError`` is re-raised so shutdown cancellation works promptly.
* **Single-replica assumption.** Concurrent passes across replicas are
  *tolerable* (the curator dedups and uses a keyset cursor) but wasteful; for
  multi-replica deployments disable the loop (``CURATOR_ENABLED=0``) and drive
  ``run_curator_once`` from one external scheduler.  See ``app.config``.
"""

import asyncio
import logging
from typing import Any, Callable, Optional

from app.config import CURATOR_ENABLED, CURATOR_INTERVAL_SECONDS
from app.curator import run_curator_once
from app.db import SessionLocal

logger = logging.getLogger("agentguard.mcp.scheduler")


async def _curator_loop(
    session_factory: Callable[[], Any], interval_seconds: float
) -> None:
    """Sleep ``interval_seconds``, run one curator pass, repeat until cancelled.

    Sleeps first (see module docstring) so a short-lived process never curates.
    Each pass is wrapped so an exception is logged without breaking the loop;
    ``CancelledError`` propagates so shutdown is prompt.
    """
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            result = await run_curator_once(session_factory)
            logger.info("curator pass complete: %s", result)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "curator pass failed; will retry next interval", exc_info=True
            )


def start_curator(
    app: Any,
    *,
    enabled: Optional[bool] = None,
    interval_seconds: Optional[float] = None,
    session_factory: Optional[Callable[[], Any]] = None,
) -> Optional[asyncio.Task]:
    """Start the background curator loop and stash the task on ``app.state``.

    Args are injectable for tests; production leaves them ``None`` and the values
    come from :mod:`app.config` / :data:`app.db.SessionLocal`.

    Returns the created :class:`asyncio.Task`, or ``None`` when disabled (in which
    case ``app.state.curator_task`` is also set to ``None``).
    """
    enabled = CURATOR_ENABLED if enabled is None else enabled
    if not enabled:
        logger.info("curator scheduler disabled")
        app.state.curator_task = None
        return None

    interval = (
        CURATOR_INTERVAL_SECONDS if interval_seconds is None else interval_seconds
    )
    factory = session_factory or SessionLocal
    task = asyncio.create_task(_curator_loop(factory, interval))
    app.state.curator_task = task
    logger.info("curator scheduler started (interval=%ss)", interval)
    return task


async def stop_curator(app: Any) -> None:
    """Cancel the background curator task (if any) and await its teardown.

    Idempotent: a no-op when no task is running.  Always clears
    ``app.state.curator_task`` afterwards.
    """
    task = getattr(app.state, "curator_task", None)
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        app.state.curator_task = None
