"""Producer/consumer contract for transcript distillation.

``mcp-server`` accepts a transcript, resolves everything that must fail loudly
at the caller (identity, scope, project, branch, quota), writes the redacted
text to ``capture_jobs`` and returns. ``async-worker`` consumes the job id and
does the two LLM calls with a timeout sized for the work.

Why the transcript lives in a TABLE and only its id on the stream: the house
contract for ``agents.turns`` is ids only, never content, because a worker can
re-read content from the database. A transcript had no such home -- it existed
only in the HTTP request -- so this table gives it one, and the stream stays
honest. The table also carries the thing the old inline path most
conspicuously lacked: a distiller failure is a ``status='failed'`` row with an
``error`` on it, not a warning line in a log nobody reads.

Publishing raises on failure. Whether that is fatal is the CALLER's judgement
-- for ``/capture/transcript`` it is not, because the row is already durable
and a sweeper can pick it up -- but this module must not decide it silently,
or a broken Redis would look identical to a healthy one with nothing to do.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger("klio.capture_jobs")

CAPTURE_JOBS = "capture.jobs"

#: One group, so adding workers spreads jobs rather than duplicating them.
CAPTURE_JOBS_GROUP = "capture-distiller"

#: The payload is one id. See the module docstring.
MAX_PAYLOAD_BYTES = 512

#: Failure reasons are for a human reading a row, not a stack trace archive.
#: Bounded so a pathological exception string cannot bloat the table.
MAX_ERROR_CHARS = 2_000

#: A job that has failed this many times is left ``failed`` for good. Two LLM
#: calls per attempt means an infinitely retried poison transcript is a
#: money leak, not just noise.
MAX_ATTEMPTS = 3

_CLAIMABLE = ("pending", "failed")

_redis: Optional[Any] = None


def _get_redis():
    global _redis
    if _redis is None:
        import redis.asyncio as aioredis  # noqa: PLC0415

        from shared.redis_config import REDIS_CLIENT_OPTIONS  # noqa: PLC0415

        url = os.environ.get("REDIS_URL")
        if not url:
            raise RuntimeError("REDIS_URL is not configured")
        _redis = aioredis.from_url(url, **REDIS_CLIENT_OPTIONS)
    return _redis


def create_job(
    db_session: Any,
    *,
    org_id: str,
    agent_id: str,
    session_id: str,
    transcript: str,
    scope: str,
    user_id: Optional[str] = None,
    project_id: Optional[str] = None,
    branch_id: Optional[str] = None,
    work_item_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    over_cap: bool = False,
) -> str:
    """Persist one pending distillation job and return its id.

    ``transcript`` MUST already have been through
    :func:`shared.redaction.redact`. This function is a writer, not an egress
    guard; the guard is at ``_combine_messages``, before the text ever reaches
    a caller of this module.
    """
    row = db_session.execute(
        text(
            """
            INSERT INTO capture_jobs (
                org_id, agent_id, session_id, user_id, scope,
                project_id, branch_id, work_item_id,
                transcript, metadata, over_cap
            ) VALUES (
                :org_id, :agent_id, :session_id, :user_id, :scope,
                :project_id, :branch_id, :work_item_id,
                :transcript, CAST(:metadata AS JSONB), :over_cap
            )
            RETURNING id
            """
        ),
        {
            "org_id": org_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "user_id": user_id,
            "scope": scope,
            "project_id": project_id,
            "branch_id": branch_id,
            "work_item_id": work_item_id,
            "transcript": transcript,
            "metadata": json.dumps(metadata) if metadata is not None else None,
            "over_cap": over_cap,
        },
    ).scalar()
    db_session.commit()
    return str(row)


def claim_job(db_session: Any, job_id: str) -> Optional[dict]:
    """Atomically take ownership of a job, or return ``None``.

    The compare-and-set is the whole point. A Redis consumer group redelivers
    on worker death, and one distillation is two LLM calls plus memory writes,
    so a second delivery of a job that is already ``running`` or ``done`` must
    do nothing. ``failed`` IS claimable, up to :data:`MAX_ATTEMPTS`: a
    transient model failure should not bury a transcript forever, but a poison
    one should stop costing money.

    Returns the full job row as a dict on success.
    """
    try:
        row = (
            db_session.execute(
                text(
                    """
                UPDATE capture_jobs
                   SET status = 'running',
                       attempts = attempts + 1,
                       started_at = NOW(),
                       error = NULL
                 WHERE id = CAST(:job_id AS uuid)
                   AND status IN ('pending', 'failed')
                   AND attempts < :max_attempts
             RETURNING id, org_id, agent_id, session_id, user_id, scope,
                       project_id, branch_id, work_item_id, transcript,
                       metadata, over_cap, attempts
                """
                ),
                {"job_id": job_id, "max_attempts": MAX_ATTEMPTS},
            )
            .mappings()
            .one_or_none()
        )
    except Exception:
        # A malformed id is a stream entry that outlived its row, not an
        # outage. Roll back so the caller's connection stays usable.
        db_session.rollback()
        logger.warning("capture job claim failed for %s", job_id, exc_info=True)
        return None
    db_session.commit()
    if row is None:
        return None
    return dict(row)


def finish_job(
    db_session: Any,
    job_id: str,
    *,
    summary_written: bool,
    memories_written: int,
) -> None:
    """Mark a job done, recording what it actually produced.

    ``memories_written=0`` with ``status='done'`` is a legitimate and common
    outcome -- plenty of turns contain nothing durable. That is exactly why it
    must be distinguishable from a failure, which the old inline path could
    not do.
    """
    db_session.execute(
        text(
            """
            UPDATE capture_jobs
               SET status = 'done',
                   summary_written = :summary_written,
                   memories_written = :memories_written,
                   error = NULL,
                   finished_at = NOW()
             WHERE id = CAST(:job_id AS uuid)
            """
        ),
        {
            "job_id": job_id,
            "summary_written": summary_written,
            "memories_written": memories_written,
        },
    )
    db_session.commit()


def fail_job(db_session: Any, job_id: str, *, error: str) -> None:
    """Record a distillation failure against the job."""
    db_session.execute(
        text(
            """
            UPDATE capture_jobs
               SET status = 'failed',
                   error = :error,
                   finished_at = NOW()
             WHERE id = CAST(:job_id AS uuid)
            """
        ),
        {"job_id": job_id, "error": (error or "")[:MAX_ERROR_CHARS]},
    )
    db_session.commit()


async def publish_job(job_id: str) -> None:
    """XADD one job id. Raises on failure; see the module docstring."""
    encoded = json.dumps({"job_id": job_id})
    if len(encoded.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        # Cannot happen with one uuid, so if it does the shape has drifted
        # and the loud failure is the point.
        raise ValueError("capture job payload is unexpectedly large")
    redis = _get_redis()
    await redis.xadd(CAPTURE_JOBS, {"payload": encoded})
