"""Distil one captured transcript into a summary and typed memories.

This is the work that used to run INSIDE ``POST /capture/transcript``, where
the Stop hook that posted the transcript waited for it. Two LLM calls under a
10s ceiling meant roughly a third of captured sessions produced nothing at
all, and produced it silently -- ``generate_summary`` returns ``None`` and
``extract_typed_memories`` returns ``[]`` on timeout, which is exactly what
they return for a turn that genuinely held nothing worth keeping.

Moving it here changes three things and deliberately nothing else:

* the timeout is sized for the work (``VERIFICATION_TIMEOUT_S`` on this
  service) rather than for a hook's patience;
* "the model failed" is now recorded as ``capture_jobs.status='failed'`` with
  a reason, distinct from a legitimate empty result (``'done'``, zero
  memories);
* a redelivered job does no model work at all, because ``claim_job`` is a
  compare-and-set.

Everything about WHAT gets written is unchanged and must stay unchanged. The
scope, user_id, project, branch and work item all come from the job row
exactly as the request resolved them. This process never re-derives a
boundary: re-deriving scope in a different process from the one that
authorised the write is how a private capture silently becomes an org one.
"""

from __future__ import annotations

import logging
from typing import Any

from shared.brain import write_scoped_memory
from shared.capture_jobs import claim_job, fail_job, finish_job
from shared.memory import extract_typed_memories, generate_summary
from shared.usage import record_usage

logger = logging.getLogger("agentguard.async-worker.capture")


async def distill_job(db_session: Any, job_id: str) -> str:
    """Claim, distil and record one capture job.

    Returns ``"done"``, ``"skipped"`` (already claimed or exhausted) or
    ``"failed"``. Never raises: the consumer loop acks on every outcome, and
    the durable record of what happened is the job row itself.
    """
    job = claim_job(db_session, job_id)
    if job is None:
        # Already running, already done, or out of attempts. A redelivery is
        # not an error, and re-running it would re-spend two LLM calls.
        return "skipped"

    try:
        return await _distil(db_session, job)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "capture distillation failed for job %s (org %s): %s",
            job_id,
            job.get("org_id"),
            reason,
            exc_info=True,
        )
        fail_job(db_session, job_id, error=reason)
        return "failed"


async def _distil(db_session: Any, job: dict) -> str:
    transcript = job["transcript"]
    # Every one of these is a value the REQUEST resolved. Read, never re-derive.
    common = {
        "org_id": job["org_id"],
        "agent_id": job["agent_id"],
        "session_id": job["session_id"],
        "project_id": job["project_id"],
        "branch_id": job["branch_id"],
        "work_item_id": job["work_item_id"],
        "scope": job["scope"],
        "user_id": job["user_id"],
        "metadata": job["metadata"] or {"source": "hook-transcript"},
    }
    over_cap = bool(job["over_cap"])
    stored = 0

    # Summary and distillation are tolerated INDEPENDENTLY, exactly as the
    # endpoint documented: a failed summary still allows the memories, and
    # vice versa. Only an unexpected exception fails the job.
    summary = await generate_summary(transcript)
    summary_written = False
    if summary:
        result = await write_scoped_memory(
            db_session, content=summary, memory_type="summary", **common
        )
        if result.get("stored"):
            summary_written = True
            stored += 1
            record_usage(db_session, job["org_id"], "memory", over_cap=over_cap)

    memories = await extract_typed_memories(transcript)
    written = 0
    for item in memories:
        if not item.content:
            continue
        result = await write_scoped_memory(
            db_session,
            content=item.content,
            memory_type=item.memory_type,
            **common,
        )
        if result.get("stored"):
            written += 1
            stored += 1
            record_usage(db_session, job["org_id"], "memory", over_cap=over_cap)

    finish_job(
        db_session,
        job["id"],
        summary_written=summary_written,
        memories_written=written,
    )
    logger.info(
        "capture job %s distilled for org %s: summary=%s memories=%d/%d",
        job["id"],
        job["org_id"],
        summary_written,
        written,
        len(memories),
    )
    return "done"
