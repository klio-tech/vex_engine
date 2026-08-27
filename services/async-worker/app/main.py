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
   also drives the dreamer (``app.dreamer.dream_for_org``) per org on two
   independent triggers, both of which read PERSISTED state
   (``dream_state``, migration 046) rather than process memory:
   ``DREAM_MIN_GRADED_SESSIONS`` new graded sessions accumulated for that org
   (``_maybe_dream_for_org``, driven by grading events), or
   ``DREAM_INTERVAL_HOURS`` elapsed since its last pass (``_run_dream_tick``,
   driven by this loop's idle wakeup every ``DREAM_TICK_SECONDS``).

   Both of those used to live in a process-local dict, so every redeploy
   reset the counter and the interval clock and neither trigger was ever
   reached — the dreamer produced zero proposals for the entire life of the
   feature. See ``app.dream_state``.

Both loops ACK on success; failures are logged but never crash the loop,
matching storage-worker's resilience contract for malformed/failing
messages.
"""

import asyncio
import json
import logging
import os
import time

import anyio
import redis.asyncio as aioredis
from sqlalchemy import create_engine, text
from shared.evidence_stream import SESSIONS_STORED
from shared.models import IngestEvent
from shared.redis_config import REDIS_CLIENT_OPTIONS

from app import config
from app import dream_state
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


async def _run_claimed_dream_pass(
    s3_client: object, claim: dream_state.DreamClaim
) -> None:
    """Run one ALREADY-CLAIMED dream pass and record its result.

    The caller has won the claim (``app.dream_state.claim_dream_pass``),
    which already reset this org's counters and interval clock — so a pass
    that fails here is not retried until the next threshold crossing. That is
    deliberate and unchanged from the original in-memory behaviour: a
    persistently-failing org (a bad LLM config, unreachable S3) must get one
    attempt per threshold crossing, not one per graded session.

    A failed dream pass is logged and swallowed — it must never take down the
    grading loop sharing this coroutine.
    """
    org_id = claim.org_id
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
        await anyio.to_thread.run_sync(
            lambda: dream_state.record_dream_result(
                dream_db, org_id=org_id, written=written
            )
        )
        logger.info(
            "dreamer: org=%s wrote %d proposals (trigger=%s sessions_since=%d "
            "elapsed_hours=%.1f)",
            org_id,
            written,
            claim.reason,
            claim.sessions_since,
            claim.elapsed_hours,
        )
    except Exception:
        logger.error("dreamer: pass failed org=%s", org_id, exc_info=True)
    finally:
        dream_db.close()


async def _maybe_dream_for_org(s3_client: object, org_id: str) -> None:
    """Count one graded session for ``org_id`` and run a dream pass if that
    crossed a threshold.

    The counter and the interval clock live in ``dream_state`` (migration
    046), NOT in this process. That is the fix for the original bug: both
    triggers used to be tracked in a process-local dict keyed by
    ``time.monotonic()``, so every ``async-worker`` redeploy — which happens
    routinely, sometimes several times a day — reset the graded-session count
    to zero AND restarted the interval clock. Neither threshold was ever
    reached in production and ``memory_proposals`` held zero rows for the
    entire life of the feature.

    Both statements run on a worker thread: they are synchronous
    psycopg2 calls and this coroutine shares its event loop with the
    verification consumer.
    """
    db = SessionLocal()
    try:
        await anyio.to_thread.run_sync(
            lambda: dream_state.record_graded_session(db, org_id=org_id)
        )
        claim = await anyio.to_thread.run_sync(
            lambda: dream_state.claim_dream_pass(
                db,
                org_id=org_id,
                min_graded_sessions=config.DREAM_MIN_GRADED_SESSIONS,
                interval_hours=config.DREAM_INTERVAL_HOURS,
            )
        )
    except Exception:
        logger.error(
            "dreamer: trigger bookkeeping failed org=%s", org_id, exc_info=True
        )
        return
    finally:
        db.close()

    if claim is None:
        return
    await _run_claimed_dream_pass(s3_client, claim)


async def _run_dream_tick(s3_client: object) -> None:
    """One periodic pass of the INTERVAL trigger.

    This exists because ``DREAM_INTERVAL_HOURS`` was never actually a timer.
    It was a condition evaluated only inside ``_maybe_dream_for_org``, which
    only runs when a session is graded — so an org with a real trace corpus
    but few graded sessions could reach NEITHER trigger: not enough grading
    events to cross the count threshold, and no grading events with which to
    evaluate the interval. Persisting the state fixes the redeploy reset;
    this tick is what gives the interval a wakeup of its own.

    Piggybacks on the sessions-stored consumer's existing idle wakeup
    (``block=5000``), the same "reuse a loop that already ticks" shape
    ``_consume_executions`` uses for the ownership sweep, rather than opening
    a third loop just to sleep on an interval.

    Every org found here is claimed through the SAME atomic
    ``claim_dream_pass`` statement the grading path uses, so this tick and a
    concurrent graded session can never both dream for one org.
    """
    db = SessionLocal()
    try:
        due = await anyio.to_thread.run_sync(
            lambda: dream_state.orgs_due_for_dream(
                db,
                interval_hours=config.DREAM_INTERVAL_HOURS,
                limit=config.DREAM_MAX_ORGS_PER_TICK,
            )
        )
        claims: list[dream_state.DreamClaim] = []
        for org_id in due:
            # An org can be due on the interval while having no state row at
            # all — it has produced traces but never had a graded session, so
            # nothing ever created its row, and claim_dream_pass's UPDATE
            # cannot claim a row that does not exist.
            await anyio.to_thread.run_sync(
                lambda oid=org_id: dream_state.ensure_state_row(db, org_id=oid)
            )
            claim = await anyio.to_thread.run_sync(
                lambda oid=org_id: dream_state.claim_dream_pass(
                    db,
                    org_id=oid,
                    min_graded_sessions=config.DREAM_MIN_GRADED_SESSIONS,
                    interval_hours=config.DREAM_INTERVAL_HOURS,
                )
            )
            if claim is not None:
                claims.append(claim)
    except Exception:
        logger.error("dreamer: periodic tick failed to select orgs", exc_info=True)
        return
    finally:
        db.close()

    for claim in claims:
        await _run_claimed_dream_pass(s3_client, claim)


async def _consume_sessions_stored(
    redis_client: aioredis.Redis, s3_client: object
) -> None:
    """Consumer loop for ``sessions.stored`` — grades each stored session.

    Same consumer-group/ack shape as ``_consume_executions``: read a batch,
    process each entry, ACK on success, log and continue on per-message
    failure, back off a second on stream-level failure. Grading itself
    (``app.outcome_grader.grade_stored_session``) is already best-effort
    internally (a ``verify()``/embedding failure degrades the verdict to
    unknown rather than raising), so a failure surfacing here means the
    entry itself was malformed or the DB write failed outright.

    Also drives the dreamer (``app.dreamer.dream_for_org``) on BOTH of its
    triggers, which are now genuinely independent:

    * count — every successfully-graded session increments that org's
      PERSISTED counter (``_maybe_dream_for_org`` ->
      ``dream_state.record_graded_session``), firing a pass once the org
      crosses ``DREAM_MIN_GRADED_SESSIONS``;
    * interval — a periodic tick (``_run_dream_tick``, every
      ``DREAM_TICK_SECONDS``) asks the database which orgs are past
      ``DREAM_INTERVAL_HOURS`` and dreams for them, regardless of whether any
      session was graded. Before this, the interval was only ever evaluated
      as a side condition of a grading event, so an org that graded slowly
      reached neither trigger.

    The tick uses this loop's existing recurring wakeup (``block=5000``), the
    same "piggyback on a loop that already ticks" shape
    ``_consume_executions`` uses for the ownership sweep.
    """
    await _ensure_consumer_group(redis_client, SESSIONS_STORED, SESSIONS_CONSUMER_GROUP)

    logger.info(
        "Outcome grader started. Listening on %s (dreamer: min_graded_sessions=%d "
        "interval_hours=%d tick=%ss max_orgs_per_tick=%d)",
        SESSIONS_STORED,
        config.DREAM_MIN_GRADED_SESSIONS,
        config.DREAM_INTERVAL_HOURS,
        config.DREAM_TICK_SECONDS,
        config.DREAM_MAX_ORGS_PER_TICK,
    )

    # Sleep-before-tick, same idiom as the ownership sweep's ``next_sweep_at``:
    # a process that restarts immediately does not fire a dream tick on the
    # way up. Unlike the old in-memory dream state, this delays only the
    # POLL — the interval clock it polls lives in ``dream_state`` and is
    # unaffected by the restart, so nothing is lost by waiting one tick.
    next_dream_tick_at = time.monotonic() + config.DREAM_TICK_SECONDS

    while True:
        try:
            messages = await redis_client.xreadgroup(
                groupname=SESSIONS_CONSUMER_GROUP,
                consumername=CONSUMER_NAME,
                streams={SESSIONS_STORED: ">"},
                count=10,
                block=5000,
            )

            # Checked every iteration — including the idle ones, which are the
            # common case at ``block=5000`` — so the interval trigger's cadence
            # does not depend on stream traffic. That independence is the whole
            # point: the previous design could only evaluate the interval when
            # traffic arrived.
            if time.monotonic() >= next_dream_tick_at:
                next_dream_tick_at = time.monotonic() + config.DREAM_TICK_SECONDS
                await _run_dream_tick(s3_client)

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
                            await _maybe_dream_for_org(s3_client, org_id)
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
    if (
        not endpoint
        or endpoint == _S3_ENDPOINT_DEV_DEFAULT
        or not access_key
        or not secret_key
    ):
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


async def _consume_agent_turns(redis_client: aioredis.Redis) -> None:
    """Run agent turns as they are queued.

    Turns run in a worker THREAD: a turn is minutes of blocking HTTP (a
    model call, then tool calls) and running it on this loop would stall
    every other consumer in the process for its duration.

    A turn that fails is acked anyway. Redelivering it would re-run a model
    call that already cost money and may already have posted, and the
    failure is visible in the thread the agent was replying to — which is a
    better place for a person to see it than a redelivery loop.
    """
    from shared.agent_turns import AGENT_TURNS, AGENT_TURNS_GROUP  # noqa: PLC0415

    from app.agent_turn import execute_turn, load_turn  # noqa: PLC0415
    from app.agent_tools import build_tools  # noqa: PLC0415

    workspace_url = os.environ.get("WORKSPACE_DATABASE_URL")
    if not workspace_url:
        logger.warning(
            "agent turns disabled: WORKSPACE_DATABASE_URL is not configured"
        )
        return

    workspace_engine = create_engine(workspace_url, pool_pre_ping=True, pool_size=2)
    engine_db = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True, pool_size=2)

    await _ensure_consumer_group(redis_client, AGENT_TURNS, AGENT_TURNS_GROUP)
    logger.info("agent turn consumer ready")

    def _run_one(payload: dict) -> str:
        with workspace_engine.connect() as workspace_conn, engine_db.connect() as engine_conn:
            loaded = load_turn(
                workspace_conn,
                engine_conn,
                agent_id=payload["agent_id"],
                message_id=payload["message_id"],
            )
        if loaded is None:
            # Disabled, deleted, removed, or the agent's own words. A turn
            # that should not happen is not an error.
            return "skipped"
        return execute_turn(loaded, build_tools=build_tools)

    while True:
        try:
            messages = await redis_client.xreadgroup(
                groupname=AGENT_TURNS_GROUP,
                consumername=CONSUMER_NAME,
                streams={AGENT_TURNS: ">"},
                count=1,
                block=5000,
            )
            if not messages:
                continue
            for _stream, entries in messages:
                for msg_id, data in entries:
                    try:
                        payload = json.loads(data["payload"])
                        outcome = await asyncio.to_thread(_run_one, payload)
                        logger.info(
                            "agent turn %s: agent=%s %s",
                            msg_id,
                            payload.get("agent_id"),
                            outcome,
                        )
                    except Exception:
                        logger.exception("agent turn %s failed", msg_id)
                    finally:
                        await redis_client.xack(AGENT_TURNS, AGENT_TURNS_GROUP, msg_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("agent turn consumer error; continuing")
            await asyncio.sleep(1)


async def _consume_capture_jobs(redis_client: aioredis.Redis) -> None:
    """Distil captured transcripts as they are queued.

    This work used to run inside ``POST /capture/transcript`` under a 10s
    timeout, because the Stop hook that posted the transcript was waiting on
    the response. Here nothing is waiting, so the timeout is sized for the
    work -- see ``CAPTURE_DISTIL_TIMEOUT_S``.

    A job is acked on EVERY outcome, including failure. The durable record is
    the ``capture_jobs`` row (``status``, ``attempts``, ``error``), not the
    stream: redelivering here would just re-spend two LLM calls on a job whose
    own row already says it failed, and ``claim_job`` would refuse it anyway
    once it is out of attempts.

    A pending job whose publish never landed (dead Redis at request time) is
    picked up by the sweep below rather than lost.
    """
    from shared.capture_jobs import (  # noqa: PLC0415
        CAPTURE_JOBS,
        CAPTURE_JOBS_GROUP,
        publish_job,
    )

    from app.capture_distill import distill_job  # noqa: PLC0415

    await _ensure_consumer_group(redis_client, CAPTURE_JOBS, CAPTURE_JOBS_GROUP)
    logger.info(
        "capture distiller ready (timeout=%ss)", config.CAPTURE_DISTIL_TIMEOUT_S
    )

    # The distiller's own timeout, not the hook's. Set on this process only,
    # so the request path's shorter ceiling elsewhere is untouched.
    os.environ["VERIFICATION_TIMEOUT_S"] = str(config.CAPTURE_DISTIL_TIMEOUT_S)

    last_sweep = 0.0
    while True:
        try:
            messages = await redis_client.xreadgroup(
                groupname=CAPTURE_JOBS_GROUP,
                consumername=CONSUMER_NAME,
                streams={CAPTURE_JOBS: ">"},
                count=1,
                block=5000,
            )
            now = time.monotonic()
            if now - last_sweep > config.CAPTURE_SWEEP_SECONDS:
                last_sweep = now
                # The DB half runs in a thread; the publish is async and
                # belongs on this loop, so the sweep RETURNS ids rather than
                # publishing them from a thread it does not own.
                for orphan in await asyncio.to_thread(_sweep_orphaned_capture_jobs):
                    try:
                        await publish_job(orphan)
                    except Exception:
                        logger.warning(
                            "re-publish of capture job %s failed", orphan, exc_info=True
                        )

            if not messages:
                continue
            for _stream, entries in messages:
                for msg_id, data in entries:
                    try:
                        job_id = json.loads(data["payload"])["job_id"]
                        db = SessionLocal()
                        try:
                            outcome = await distill_job(db, job_id)
                        finally:
                            db.close()
                        logger.info("capture job %s: %s", job_id, outcome)
                    except Exception:
                        logger.exception("capture job %s failed", msg_id)
                    finally:
                        await redis_client.xack(
                            CAPTURE_JOBS, CAPTURE_JOBS_GROUP, msg_id
                        )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("capture distiller error; continuing")
            await asyncio.sleep(1)


def _sweep_orphaned_capture_jobs() -> list[str]:
    """Re-publish jobs the stream never got, or that died mid-flight.

    Two ways a durable job can end up with nobody coming for it: the request
    persisted the row but the XADD failed, or a worker claimed it and was
    killed before finishing. Neither is visible to the stream, so without this
    the transcript is silently lost -- which is precisely the failure mode this
    whole change exists to remove.
    """
    from shared.capture_jobs import MAX_ATTEMPTS  # noqa: PLC0415

    db = SessionLocal()
    try:
        rows = db.execute(
            text(
                """
                SELECT id FROM capture_jobs
                 WHERE attempts < :max_attempts
                   AND (
                        (status = 'pending' AND created_at < NOW() - INTERVAL '5 minutes')
                     OR (status = 'running' AND started_at < NOW() - INTERVAL '30 minutes')
                   )
                 ORDER BY created_at
                 LIMIT 100
                """
            ),
            {"max_attempts": MAX_ATTEMPTS},
        ).scalars().all()
        if not rows:
            return []
        # A 'running' row whose worker died must become claimable again.
        db.execute(
            text(
                "UPDATE capture_jobs SET status = 'pending' "
                " WHERE status = 'running' AND started_at < NOW() - INTERVAL '30 minutes'"
            )
        )
        db.commit()
        logger.info("capture sweep re-publishing %d orphaned job(s)", len(rows))
        return [str(job_id) for job_id in rows]
    except Exception:
        logger.warning("capture job sweep failed; will retry", exc_info=True)
        return []
    finally:
        db.close()


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
        _consume_agent_turns(redis_client),
        _consume_capture_jobs(redis_client),
    )


if __name__ == "__main__":
    asyncio.run(run())
