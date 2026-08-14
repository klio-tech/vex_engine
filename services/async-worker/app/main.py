"""Redis Stream consumer loops for the async worker.

Runs two independent consumer loops concurrently (``asyncio.gather``,
mirroring ``services/storage-worker/app/main.py``'s multi-loop shape — one
coroutine per stream, each with its own consumer group so they never steal
each other's messages):

1. ``verification-workers`` on ``executions.raw`` — runs verification on
   each event and publishes to ``executions.verified``. This loop also
   drives the ownership-transfer sweep tick (unrelated to streaming, but it
   needs a recurring wakeup and this loop already has one).

2. ``outcome-grader`` on ``sessions.stored`` — grades a stored session
   (``app.outcome_grader.grade_stored_session``): one ``verify()`` call, one
   ``session_outcomes`` row, and per-memory attribution in
   ``recall_outcomes``. See ``app.outcome_grader`` for why the verdict is
   written exactly once per session rather than once per recall. This loop
   also drives the dreamer (``app.dreamer.dream_for_org``) per org, once
   ``DREAM_MIN_GRADED_SESSIONS`` new graded sessions have accumulated for
   that org or ``DREAM_INTERVAL_HOURS`` have elapsed since its last pass —
   see ``_maybe_dream_for_org``.

Both loops ACK on success; failures are logged but never crash the loop,
matching storage-worker's resilience contract for malformed/failing
messages.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

import anyio
import redis.asyncio as aioredis
from sqlalchemy import create_engine
from shared.evidence_stream import SESSIONS_STORED
from shared.models import IngestEvent
from shared.redis_config import REDIS_CLIENT_OPTIONS

from app import config
from app import dreamer
from app.db import SessionLocal
from app.outcome_grader import grade_stored_session
from app.ownership_sweep import run_ownership_sweep
from app.s3_client import get_s3_client
from app.worker import VERIFIED_STREAM_KEY, process_event

logger = logging.getLogger("agentguard.async-worker")

STREAM_KEY = "executions.raw"
CONSUMER_GROUP = "verification-workers"
CONSUMER_NAME = os.environ.get("CONSUMER_NAME", "async-worker-1")

SESSIONS_CONSUMER_GROUP = "outcome-grader"


def _run_ownership_sweep_tick(engine, supabase_engine) -> None:
    """One blocking sweep pass — run via ``asyncio.to_thread`` so real
    Postgres/Supabase I/O never wedges the event loop this consumer needs for
    the stream read on every other tick.
    """
    result = run_ownership_sweep(
        engine,
        supabase_engine=supabase_engine,
        dry_run=config.OWNERSHIP_SWEEP_DRY_RUN,
        max_transfers=config.OWNERSHIP_SWEEP_MAX_TRANSFERS_PER_RUN,
    )
    logger.info(
        "ownership sweep pass complete: dry_run=%s orgs_considered=%d "
        "transfers=%d removals=%d cap_tripped=%s "
        "orgs_skipped_lookup_failed=%d orgs_skipped_empty_membership=%d "
        "projects_left_alone_owner_unresolved=%d",
        result.dry_run,
        result.orgs_considered,
        result.transfer_count,
        len(result.removals),
        result.cap_tripped,
        len(result.orgs_skipped_lookup_failed),
        len(result.orgs_skipped_empty_membership),
        len(result.projects_left_alone_owner_unresolved),
    )
    if result.cap_tripped:
        logger.critical(
            "ownership sweep: transfer cap tripped this pass — see the "
            "preceding CRITICAL log line for which org/project stopped it"
        )


async def _ensure_consumer_group(
    redis_client: aioredis.Redis,
    stream_key: str,
    group_name: str,
) -> None:
    """Create a consumer group, ignoring if it already exists."""
    try:
        await redis_client.xgroup_create(
            stream_key,
            group_name,
            id="0",
            mkstream=True,
        )
        logger.info(
            "Created consumer group '%s' on stream '%s'", group_name, stream_key
        )
    except Exception:
        logger.debug(
            "Consumer group '%s' already exists on stream '%s'",
            group_name,
            stream_key,
        )


async def _consume_executions(redis_client: aioredis.Redis) -> None:
    """Consumer loop for ``executions.raw``.

    1. Reads batches of up to 10 messages from the stream.
    2. For each message, deserialises the event, runs verification,
       and ACKs the message on success.
    3. Publishes verified results to ``executions.verified``.
    4. On per-message failure, logs the error and continues.
    5. On stream-level failure, backs off for 1 second and retries.

    Also drives the ownership-transfer sweep tick: unrelated to the stream
    itself, but this loop already has the recurring wakeup (``block=5000``)
    the sweep needs, so it piggybacks here rather than opening a third loop
    just to sleep on an interval.
    """
    # --- project-ownership-transfer sweep: engines built once, reused every
    # tick, matching the module docstring's "one engine for a whole run"
    # convention. ``sweep_engine`` stays None when the sweep is disabled so a
    # deployment that never wants it never opens the connection at all.
    sweep_engine = (
        create_engine(config.DATABASE_URL) if config.OWNERSHIP_SWEEP_ENABLED else None
    )
    supabase_engine = (
        create_engine(config.SUPABASE_DATABASE_URL)
        if config.OWNERSHIP_SWEEP_ENABLED and config.SUPABASE_DATABASE_URL
        else None
    )
    if config.OWNERSHIP_SWEEP_ENABLED:
        logger.info(
            "ownership sweep enabled: interval=%ss dry_run=%s max_transfers=%d "
            "supabase_configured=%s",
            config.OWNERSHIP_SWEEP_INTERVAL_SECONDS,
            config.OWNERSHIP_SWEEP_DRY_RUN,
            config.OWNERSHIP_SWEEP_MAX_TRANSFERS_PER_RUN,
            supabase_engine is not None,
        )
    else:
        logger.info("ownership sweep disabled (OWNERSHIP_SWEEP_ENABLED=0)")
    # Sleep-before-tick, mirroring app/scheduler.py's curator loop in
    # mcp-server: a short-lived process (a quick restart) never runs a real
    # pass, and the first real pass on a fresh deploy waits a full interval
    # rather than firing immediately on startup.
    next_sweep_at = time.monotonic() + config.OWNERSHIP_SWEEP_INTERVAL_SECONDS

    await _ensure_consumer_group(redis_client, STREAM_KEY, CONSUMER_GROUP)

    logger.info("Async verification worker started. Listening on %s", STREAM_KEY)

    while True:
        try:
            messages = await redis_client.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=CONSUMER_NAME,
                streams={STREAM_KEY: ">"},
                count=10,
                block=5000,
            )

            # Checked every iteration — including the idle ones, which is the
            # common case at ``block=5000`` — so the sweep's cadence does not
            # depend on stream traffic. A failed pass is logged and swallowed
            # exactly like a failed curation tick in mcp-server's scheduler:
            # one bad pass must not take down the message consumer sharing
            # this loop.
            if sweep_engine is not None and time.monotonic() >= next_sweep_at:
                next_sweep_at = (
                    time.monotonic() + config.OWNERSHIP_SWEEP_INTERVAL_SECONDS
                )
                try:
                    await asyncio.to_thread(
                        _run_ownership_sweep_tick, sweep_engine, supabase_engine
                    )
                except Exception:
                    logger.error("ownership sweep pass failed", exc_info=True)

            if not messages:
                continue

            for _stream, entries in messages:
                for msg_id, data in entries:
                    try:
                        event = IngestEvent.model_validate_json(data["data"])

                        # Skip events already verified by the sync gateway
                        # (correction cascade path) to avoid overwriting
                        # corrected results.
                        if event.metadata and event.metadata.get("already_verified"):
                            await redis_client.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)
                            logger.info(
                                "Skipping already-verified event %s",
                                event.execution_id,
                            )
                            continue

                        verified_event = await process_event(event)

                        await redis_client.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)

                        # Publish verified result for storage worker and alert service
                        await redis_client.xadd(
                            VERIFIED_STREAM_KEY,
                            {"data": json.dumps(verified_event)},
                        )
                    except Exception as exc:
                        logger.error(
                            "Failed to process message %s: %s",
                            msg_id,
                            exc,
                            exc_info=True,
                            extra={
                                "msg_id": msg_id,
                                "execution_id": event.execution_id
                                if "event" in locals()
                                else None,
                            },
                        )
        except Exception as exc:
            logger.error(
                "Stream read error: %s",
                exc,
                exc_info=True,
                extra={"stream": STREAM_KEY},
            )
            await asyncio.sleep(1)


async def _maybe_dream_for_org(
    s3_client: object, org_id: str, dream_state: dict[str, dict[str, Any]]
) -> None:
    """Run the dreamer for ``org_id`` once this org has crossed either dream
    threshold, then reset its counters.

    ``dream_state`` is process-local (``{org_id: {"sessions_since": int,
    "last_run": float}}``, keyed by ``time.monotonic()``) — reset on
    restart, matching the ownership sweep's own no-persisted-cursor
    contract for the analogous per-org cadence. A short-lived process (a
    quick restart) never runs a real dream pass for an org it barely saw;
    the first real pass after a fresh deploy waits for a real threshold
    crossing rather than firing immediately, same idiom as the ownership
    sweep's ``next_sweep_at`` above.

    A failed dream pass is logged and swallowed — it must never take down
    the grading loop sharing this coroutine — and its org's counters are
    reset regardless of outcome, so a persistently-failing org (e.g. a bad
    LLM config) does not retry every single graded session; it gets one
    attempt per threshold crossing.
    """
    state = dream_state.setdefault(
        org_id, {"sessions_since": 0, "last_run": time.monotonic()}
    )
    state["sessions_since"] += 1
    elapsed_hours = (time.monotonic() - state["last_run"]) / 3600.0
    if (
        state["sessions_since"] < config.DREAM_MIN_GRADED_SESSIONS
        and elapsed_hours < config.DREAM_INTERVAL_HOURS
    ):
        return

    dream_db = SessionLocal()
    try:
        # dream_for_org is a plain (synchronous) function on purpose — it does
        # blocking DB I/O, up to DEFAULT_SESSION_LOOKBACK synchronous S3 GETs,
        # and a synchronous LLM call. Running it via anyio.to_thread.run_sync
        # (rather than awaiting an async wrapper) means all of that happens on
        # a worker thread, never on this event loop — which the verification
        # consumer (_consume_executions) shares via asyncio.gather, so a slow
        # dream pass must not stall grading or verification traffic.
        written = await anyio.to_thread.run_sync(
            lambda: dreamer.dream_for_org(dream_db, s3_client, org_id=org_id)
        )
        logger.info(
            "dreamer: org=%s wrote %d proposals (sessions_since=%d "
            "elapsed_hours=%.1f)",
            org_id, written, state["sessions_since"], elapsed_hours,
        )
    except Exception:
        logger.error("dreamer: pass failed org=%s", org_id, exc_info=True)
    finally:
        dream_db.close()
        state["sessions_since"] = 0
        state["last_run"] = time.monotonic()


async def _consume_sessions_stored(redis_client: aioredis.Redis, s3_client: object) -> None:
    """Consumer loop for ``sessions.stored`` — grades each stored session.

    Same consumer-group/ack shape as ``_consume_executions``: read a batch,
    process each entry, ACK on success, log and continue on per-message
    failure, back off a second on stream-level failure. Grading itself
    (``app.outcome_grader.grade_stored_session``) is already best-effort
    internally (a ``verify()``/embedding failure degrades the verdict to
    unknown rather than raising), so a failure surfacing here means the
    entry itself was malformed or the DB write failed outright.

    Also drives the dreamer (``app.dreamer.dream_for_org``): every
    successfully-graded session increments that org's counter
    (``_maybe_dream_for_org``), which fires a dream pass once the org
    crosses ``DREAM_MIN_GRADED_SESSIONS`` new graded sessions or
    ``DREAM_INTERVAL_HOURS`` have elapsed since its last pass — the same
    "piggyback on a loop that already has the recurring wakeup" shape
    ``_consume_executions`` uses for the ownership sweep, except keyed off
    real grading events rather than a fixed timer alone, since the
    threshold is explicitly "N new graded sessions OR M hours".
    """
    await _ensure_consumer_group(redis_client, SESSIONS_STORED, SESSIONS_CONSUMER_GROUP)

    logger.info("Outcome grader started. Listening on %s", SESSIONS_STORED)

    dream_state: dict[str, dict[str, Any]] = {}

    while True:
        try:
            messages = await redis_client.xreadgroup(
                groupname=SESSIONS_CONSUMER_GROUP,
                consumername=CONSUMER_NAME,
                streams={SESSIONS_STORED: ">"},
                count=10,
                block=5000,
            )
            if not messages:
                continue

            for _stream, entries in messages:
                for msg_id, data in entries:
                    try:
                        payload = json.loads(data["payload"])
                        org_id = payload["org_id"]
                        session_id = payload["session_id"]

                        db_session = SessionLocal()
                        try:
                            result = await grade_stored_session(
                                db_session,
                                s3_client,
                                org_id=org_id,
                                session_id=session_id,
                            )
                        except Exception:
                            db_session.rollback()
                            raise
                        finally:
                            db_session.close()

                        await redis_client.xack(
                            SESSIONS_STORED, SESSIONS_CONSUMER_GROUP, msg_id
                        )
                        logger.info(
                            "graded session org=%s session=%s outcome=%s "
                            "attributions=%d",
                            org_id,
                            session_id,
                            result.get("session_outcome_id"),
                            result.get("attributions", 0),
                        )
                        # Only a REAL grade counts toward the dream threshold —
                        # a None session_outcome_id means nothing-to-grade or a
                        # duplicate re-delivery (see grade_stored_session's
                        # docstring), neither of which is new evidence.
                        if result.get("session_outcome_id") is not None:
                            await _maybe_dream_for_org(
                                s3_client, org_id, dream_state
                            )
                    except Exception:
                        logger.error(
                            "Failed to grade session message %s",
                            msg_id,
                            exc_info=True,
                            extra={"msg_id": msg_id},
                        )
        except Exception as exc:
            logger.error(
                "Sessions-stored stream read error: %s",
                exc,
                exc_info=True,
                extra={"stream": SESSIONS_STORED},
            )
            await asyncio.sleep(1)


_S3_ENDPOINT_DEV_DEFAULT = "http://localhost:9000"


def _warn_if_s3_env_missing() -> None:
    """M5: the grader (``app.outcome_grader``, via ``shared.traces.fetch_trace``)
    reads trace objects from S3, using the SAME generic ``S3_*`` env vars as
    ``app.s3_client.get_s3_client`` (see that module's docstring). If those
    are unset in a real deployment, every ``fetch_trace`` call silently
    points at a local MinIO that does not exist there — nothing raises,
    every session simply grades ``verdict='unknown'`` forever, and there is
    no other signal anything is wrong.

    There is no environment flag in ``app.config`` to gate this on (no
    ``ENVIRONMENT``/``APP_ENV`` distinguishing dev from prod exists anywhere
    in this service), so this logs unconditionally whenever the endpoint is
    still the localhost dev default or credentials are unset — a real dev
    setup logs one harmless error line on every start; a real deployment
    that forgot to set these gets a loud, findable signal instead of silent
    all-unknown grading. Never fails startup (no fail-fast) — grading
    degrading to unknown is bad, but it must not be worse than not starting
    at all.
    """
    endpoint = os.environ.get("S3_ENDPOINT")
    access_key = os.environ.get("S3_ACCESS_KEY")
    secret_key = os.environ.get("S3_SECRET_KEY")
    if not endpoint or endpoint == _S3_ENDPOINT_DEV_DEFAULT or not access_key or not secret_key:
        logger.error(
            "S3 env not fully configured (S3_ENDPOINT=%s, S3_ACCESS_KEY=%s, "
            "S3_SECRET_KEY=%s) — if this is a real deployment, not local "
            "dev, every session will grade verdict='unknown' because "
            "fetch_trace cannot reach the S3 bucket traces were written to. "
            "Set S3_ENDPOINT/S3_ACCESS_KEY/S3_SECRET_KEY.",
            endpoint or "<unset>",
            "<set>" if access_key else "<unset>",
            "<set>" if secret_key else "<unset>",
        )


async def run() -> None:
    """Start all consumer loops concurrently."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    _warn_if_s3_env_missing()

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    redis_client = aioredis.from_url(redis_url, **REDIS_CLIENT_OPTIONS)
    s3_client = get_s3_client()

    await asyncio.gather(
        _consume_executions(redis_client),
        _consume_sessions_stored(redis_client, s3_client),
    )


if __name__ == "__main__":
    asyncio.run(run())
