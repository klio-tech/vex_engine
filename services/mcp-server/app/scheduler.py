"""In-process background scheduler for the Klio brain curator and artifact GC.

The curator (:mod:`app.curator`) synthesises each org's RAW memories into
``summary`` / ``fact`` rows.  :func:`app.curator.run_curator_once` is a single
pass; this module drives it on a periodic loop started from the FastAPI
lifespan (:mod:`app.main`), so the single-service Railway deployment needs no
external cron.

The same module also drives the artifact GC sweep
(:func:`collect_artifacts_once`), which deletes ``pending`` artifact rows
whose upload was never confirmed — and their S3 objects, best-effort — after
``shared.artifacts.PENDING_TTL_HOURS``.  See :func:`_artifact_gc_loop` /
:func:`start_artifact_gc`, which mirror the curator loop's shape.

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

from sqlalchemy import text as sa_text

from app.config import CURATOR_ENABLED, CURATOR_INTERVAL_SECONDS
from app.curator import run_curator_once
from app.db import SessionLocal
from shared import artifact_storage, artifacts

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


async def collect_artifacts_once(db_factory: Callable[[], Any]) -> int:
    """Delete pending artifacts whose upload never arrived, and their bytes.

    An abandoned upload has no card, so nothing can reach it — it is invisible
    storage the org would otherwise be charged for. ``artifacts.collect_abandoned``
    is intentionally cross-org (see its docstring); the ``org_id`` used to build
    each object's key comes from the row itself, never from any other source.

    Object deletion is best-effort and never blocks row cleanup: the common
    case for an abandoned upload is that no object was ever put to S3 at all,
    so a missing-object failure here is expected, not exceptional, and must
    not stop the sweep or prevent other rows from being collected.
    """
    db = db_factory()
    try:
        stale = artifacts.collect_abandoned(db)
        if not stale:
            return 0

        s3 = artifact_storage.get_artifact_s3_client()
        for row in stale:
            key = artifact_storage.object_key(row["org_id"], row["id"], row["version"])
            try:
                artifact_storage.delete(s3, key)
            except Exception:
                logger.debug("artifact GC: no object at %s", key)

        db.execute(
            sa_text("DELETE FROM artifacts WHERE id = ANY(:ids)"),
            {"ids": [row["id"] for row in stale]},
        )
        db.commit()
        logger.info("artifact GC: collected %d abandoned uploads", len(stale))
        return len(stale)
    finally:
        db.close()


async def _artifact_gc_loop(db_factory: Callable[[], Any], interval_seconds: float) -> None:
    """Sleep ``interval_seconds``, run one sweep, repeat until cancelled.

    Mirrors :func:`_curator_loop`: sleeps before the first tick so a
    short-lived process never sweeps, and a failed tick is logged and
    swallowed so the loop survives to try again next interval.
    """
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            collected = await collect_artifacts_once(db_factory)
            if collected:
                logger.info("artifact GC pass complete: collected=%d", collected)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "artifact GC pass failed; will retry next interval", exc_info=True
            )


def start_artifact_gc(
    app: Any, db_factory: Callable[[], Any], interval_seconds: float = 3600
) -> asyncio.Task:
    """Start the hourly abandoned-upload sweep and stash the task on ``app.state``.

    Mirrors :func:`start_curator`'s shape (sleep-before-tick loop, task stored
    on ``app.state``), but the artifact sweep is not gated behind an enabled
    flag — it is cheap, idempotent, and safe to run in every environment.
    """
    task = asyncio.create_task(_artifact_gc_loop(db_factory, interval_seconds))
    app.state.artifact_gc_task = task
    logger.info("artifact GC scheduler started (interval=%ss)", interval_seconds)
    return task


async def stop_artifact_gc(app: Any) -> None:
    """Cancel the background artifact GC task (if any) and await its teardown.

    Idempotent: a no-op when no task is running. Always clears
    ``app.state.artifact_gc_task`` afterwards.
    """
    task = getattr(app.state, "artifact_gc_task", None)
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        app.state.artifact_gc_task = None
