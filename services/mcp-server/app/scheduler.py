"""In-process background scheduler for the Klio brain curator and artifact GC.

The curator (:mod:`app.curator`) synthesises each user's RAW private memories into
``summary`` / ``fact`` rows.  :func:`app.curator.run_curator_once` is a single
pass; this module drives it on a periodic loop started from the FastAPI
lifespan (:mod:`app.main`), so the single-service Railway deployment needs no
external cron.

The same module also drives the artifact GC sweep
(:func:`collect_artifacts_once`), which has two passes:

1. ``pending`` artifact rows whose upload was never confirmed are deleted —
   with their S3 objects, best-effort — after
   ``shared.artifacts.PENDING_TTL_HOURS``.
2. ``retracted`` / ``superseded`` rows older than
   ``shared.artifacts.PURGE_GRACE_HOURS`` have their bytes destroyed and their
   rows and cards removed (:func:`_purge_dead_artifacts`).  This is what makes
   ``forget`` a real deletion rather than a status change: nothing else in the
   system ever removes an artifact's object.

See :func:`_artifact_gc_loop` / :func:`start_artifact_gc`, which mirror the
curator loop's shape.

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

import anyio
from sqlalchemy import text as sa_text

from app.config import ARTIFACT_GC_ENABLED, CURATOR_ENABLED, CURATOR_INTERVAL_SECONDS
from app.curator import run_curator_once
from app.db import SessionLocal
from shared import artifact_storage, artifacts

logger = logging.getLogger("agentguard.mcp.scheduler")

ARTIFACT_GC_ROW_LIMIT = 500
"""Rows each GC selector may return in one tick.

A tick bound, not a memory bound. Each row costs a commit plus a boto3
``delete_object`` (5 s connect / 30 s read / 3 adaptive retries), so an
unbounded selector makes tick duration a function of how much work happened to
accumulate — and against a degraded object store that is ~105 s per row. 500
rows is far more than an hourly sweep normally sees while keeping the
worst-case tick finite; the selectors are ordered oldest-first, so successive
ticks drain any backlog rather than re-walking the same window."""


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


def _purge_dead_artifacts(db: Any, s3: Any, dead: list) -> int:
    """Second sweep pass: destroy the bytes of retracted/superseded artifacts.

    ``forget`` (``artifacts.retract``) and supersede only flip ``status``.
    Nothing else in the system ever deletes an object or clears
    ``inline_text``, so before this pass existed a forgotten artifact read as
    gone from every API — card hidden, ``artifact_get`` not_found, excluded
    from ``org_storage_bytes`` — while its object stayed in the bucket
    permanently. That is a deletion request over customer document content
    that could not be honoured, and a storage pool that under-counts what is
    actually stored.

    **Hard delete, deliberately.** The row is removed outright rather than
    given a terminal status with its bodies nulled. Two reasons, and this is
    the explicit choice the alternative was weighed against: ``artifacts``
    carries a CHECK constraint enumerating the four legal statuses (migration
    027), so a ``'purged'`` state would need its own migration for no
    behavioural gain; and the row itself holds customer-authored ``title`` and
    ``summary``, which a genuine deletion should not leave behind. The card is
    deleted in the same transaction for exactly that second reason — it is a
    copy of the same customer text — and deleting it is also what guarantees
    it can never become discoverable again, independent of whether its
    ``recall_hidden`` flag was ever set.

    **Ordering discipline, mirroring the pending sweep.** ``collect_purgeable``
    SELECTs rows dead as of that query, but the grace window exists precisely
    so a row could be restored to ``'active'`` before the sweep reaches it.
    Each row is therefore deleted with its own guarded, atomic
    ``DELETE ... WHERE id = :id AND status IN ('retracted','superseded')
    RETURNING id``, and the S3 object is deleted **only after** that DELETE
    confirms the row was actually removed. The sweep must never destroy bytes
    belonging to a row it did not also delete.

    **The key is derived, not read from ``storage_uri``.** ``artifact_get``
    with ``mode='url'`` on an INLINE artifact materialises an object at
    ``object_key(org_id, id, version)`` and leaves the row inline
    (``storage_uri`` NULL) — so a NULL ``storage_uri`` does not mean "no object
    exists". Keying the purge off that column would leave every such
    materialised object uncollected forever. Deleting at the derived key
    instead covers both cases and is harmless when nothing is there. That same
    determinism is what makes a materialised object safe while the artifact is
    alive: every ``mode='url'`` call re-puts identical bytes at the identical
    key, so a stale one is an idempotent overwrite, never an orphan with a
    second identity.

    Object deletion is best-effort per row: one failure logs and moves on so a
    single unreachable object cannot strand the rest of the batch.

    ``dead`` is passed in rather than selected here: the caller already has to
    know whether there is any work at all before it decides to build an S3
    client, and re-running ``collect_purgeable`` would be a second scan whose
    result could differ from the one that decision was made on.
    """
    if not dead:
        return 0

    purged = 0
    for row in dead:
        deleted = db.execute(
            sa_text(
                "DELETE FROM artifacts WHERE id = :id "
                "AND status IN ('retracted', 'superseded') "
                "RETURNING id"
            ),
            {"id": row["id"]},
        ).fetchone()

        if not deleted:
            # Restored (or already purged) between the SELECT and this DELETE.
            # The row was NOT removed, so its object must be left alone.
            db.commit()
            logger.info("artifact purge: skipping %s, no longer dead", row["id"])
            continue

        # Same transaction as the row delete: the card is a copy of the
        # customer's title and summary and must not outlive the artifact.
        # ``org_id`` comes from the row itself — this sweep is cross-org.
        db.execute(
            sa_text(
                "DELETE FROM session_memories "
                "WHERE org_id = :org_id AND memory_type = 'artifact' "
                "AND metadata->>'artifact_id' = :artifact_id"
            ),
            {"org_id": row["org_id"], "artifact_id": row["id"]},
        )
        db.commit()

        key = artifact_storage.object_key(row["org_id"], row["id"], row["version"])
        try:
            artifact_storage.delete(s3, key)
        except Exception:
            logger.warning(
                "artifact purge: object delete failed for %s; row already removed",
                key,
                exc_info=True,
            )
        purged += 1

    if purged:
        logger.info("artifact purge: destroyed bytes for %d dead artifacts", purged)
    return purged


async def collect_artifacts_once(db_factory: Callable[[], Any]) -> int:
    """Delete pending artifacts whose upload never arrived, and their bytes.

    An abandoned upload has no card, so nothing can reach it — it is invisible
    storage the org would otherwise be charged for. ``artifacts.collect_abandoned``
    is intentionally cross-org (see its docstring); the ``org_id`` used to build
    each object's key comes from the row itself, never from any other source.

    **TOCTOU guard.** ``collect_abandoned`` SELECTs rows that were ``pending``
    as of that query, but a late ``activate()`` call can flip a row to
    ``active`` in the (however small) window between that SELECT and this
    function's DELETE. Deleting unconditionally would destroy a live,
    just-confirmed artifact and the bytes a customer just uploaded — permanent
    data loss on a timer. So each row is deleted with its own conditional,
    atomic ``DELETE ... WHERE id = :id AND status = 'pending' RETURNING id``:
    the DELETE only removes the row if it is *still* pending at that instant.
    The S3 object for a row is deleted **only after** that DELETE confirms the
    row was actually removed — never before, and never for a row whose DELETE
    returned nothing. This ordering is the invariant: the sweep must never
    destroy bytes belonging to a row it did not also delete.

    Object deletion is best-effort and never blocks row cleanup: the common
    case for an abandoned upload is that no object was ever put to S3 at all,
    so a missing-object failure here is expected, not exceptional, and must
    not stop the sweep or prevent other rows from being collected.

    **Bounded, and off the event loop.** Everything below is blocking work:
    synchronous psycopg2, a commit per row, and boto3 ``delete_object`` with a
    5 s connect / 30 s read timeout and 3 adaptive retries. Run inline on the
    loop it cost ~2.5 s of wedged loop per hour in the normal case, and against
    a blackholed object store roughly 105 s **per row** — 50 rows would hold
    the loop, and therefore every request this process serves, for about an
    hour and a half. So the whole pass runs in a worker thread, and each tick
    takes at most :data:`ARTIFACT_GC_ROW_LIMIT` rows per selector.

    The bound is a tick bound, not a memory bound: pending rows are already
    capped at ``artifacts.MAX_PENDING_PER_ORG`` (32) per org. What was
    genuinely unbounded is ``collect_purgeable``, and what was unbounded in
    *duration* is the loop-blocking, which is what this fixes.
    """

    def _sweep() -> int:
        db = db_factory()
        try:
            return _collect_artifacts_sync(db)
        finally:
            db.close()

    return await anyio.to_thread.run_sync(_sweep)


def _collect_artifacts_sync(db: Any) -> int:
    """One bounded GC pass. Blocking; call via :func:`collect_artifacts_once`."""
    stale = artifacts.collect_abandoned(db, limit=ARTIFACT_GC_ROW_LIMIT)
    dead = artifacts.collect_purgeable(db, limit=ARTIFACT_GC_ROW_LIMIT)
    if not stale and not dead:
        # Nothing to collect means nothing to connect to. Building the
        # client reads endpoint and credentials from the environment and
        # raises when they are missing or wrong, so doing it
        # unconditionally turns every tick of an idle deployment into a
        # logged exception — noise that trains on-call to ignore this
        # sweep's alerts, on a job whose real failures delete data.
        return 0

    s3 = artifact_storage.get_artifact_s3_client()
    collected = 0
    for row in stale:
        deleted = db.execute(
            sa_text(
                "DELETE FROM artifacts WHERE id = :id AND status = 'pending' "
                "RETURNING id"
            ),
            {"id": row["id"]},
        ).fetchone()
        db.commit()

        if not deleted:
            # Confirmed between collect_abandoned's SELECT and this
            # DELETE: the row is no longer pending, so it was NOT removed
            # above. Its object must be left alone.
            logger.info("artifact GC: skipping %s, no longer pending", row["id"])
            continue

        key = artifact_storage.object_key(row["org_id"], row["id"], row["version"])
        try:
            artifact_storage.delete(s3, key)
        except Exception:
            logger.debug("artifact GC: no object at %s", key)
        collected += 1

    if collected:
        logger.info("artifact GC: collected %d abandoned uploads", collected)

    # Second pass: destroy the bytes of artifacts that are dead but whose
    # objects nothing else in the system would ever remove. See
    # ``_purge_dead_artifacts``.
    collected += _purge_dead_artifacts(db, s3, dead)
    return collected


async def _artifact_gc_loop(
    db_factory: Callable[[], Any], interval_seconds: float
) -> None:
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
    app: Any,
    db_factory: Callable[[], Any],
    interval_seconds: float = 3600,
    *,
    enabled: Optional[bool] = None,
) -> Optional[asyncio.Task]:
    """Start the hourly abandoned-upload sweep and stash the task on ``app.state``.

    Mirrors :func:`start_curator`'s shape exactly: same sleep-before-tick
    loop, same task-on-``app.state`` storage, and — since this sweep deletes
    DB rows and S3 objects — the same kill-switch convention
    (``ARTIFACT_GC_ENABLED``, read here exactly like ``CURATOR_ENABLED`` is
    read in :func:`start_curator`) so on-call can disable it with an env var
    instead of a code change and redeploy.

    Returns the created :class:`asyncio.Task`, or ``None`` when disabled (in
    which case ``app.state.artifact_gc_task`` is also set to ``None``).
    """
    enabled = ARTIFACT_GC_ENABLED if enabled is None else enabled
    if not enabled:
        logger.info("artifact GC scheduler disabled")
        app.state.artifact_gc_task = None
        return None

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
