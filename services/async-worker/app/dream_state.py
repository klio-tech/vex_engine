"""Persisted dreamer trigger state (migration 046, ``dream_state``).

This module owns the answer to one question: "is a dream pass due for this
org, and did I win the right to run it?" Everything here is a single SQL
statement, because the trigger test and the counter reset MUST be the same
statement — see :func:`claim_dream_pass`.

Why the state is in Postgres rather than in the process: ``async-worker``
redeploys routinely, and the previous in-memory dict reset both the graded
session counter and the interval clock on every deploy, so in production
neither dream trigger was ever reached and ``memory_proposals`` stayed
empty. Migration 046's module docstring has the full rationale.

Every function is synchronous and takes a plain SQLAlchemy ``Session``,
matching :mod:`app.dreamer` — the caller runs these on a worker thread via
``anyio.to_thread.run_sync``, never on the event loop the consumers share.
Each function commits its own single statement and holds no transaction open
across a network call.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger("agentguard.dream-state")


@dataclass(frozen=True)
class DreamClaim:
    """A WON claim on a dream pass for one org, with the trigger snapshot
    that justified it — reported in the caller's log line so an operator can
    tell which trigger fired without re-deriving it from the table.

    ``sessions_since`` and ``elapsed_hours`` are the PRE-reset values (the
    claim statement resets them as it wins), which is why they are captured
    here rather than re-read afterwards: after the claim they are gone.
    """

    org_id: str
    sessions_since: int
    elapsed_hours: float
    reason: str


def record_graded_session(db: Any, *, org_id: str) -> None:
    """Increment ``org_id``'s graded-session counter by one, creating the row
    on first sight.

    "Graded" means exactly what ``app.main._consume_sessions_stored`` counts:
    a ``grade_stored_session`` that returned a non-NULL
    ``session_outcome_id``. That is NOT a verdict — an ``unknown`` verdict
    still counts, because the trigger asks "did new evidence arrive", not
    "was the evidence conclusive". There is no verdict column on
    ``recall_outcomes`` at all; verdicts live on ``session_outcomes`` and the
    dreamer's own detectors read them, not this counter.

    On INSERT, ``last_run_at`` is seeded from this org's OLDEST
    ``session_traces`` row rather than ``now()``. The interval trigger is
    meant to express "this org has had un-dreamed-over evidence for M
    hours"; seeding from ``now()`` would instead express "the worker noticed
    this org M hours ago", which is what made the in-memory version push
    every first pass an interval into the future on every redeploy. An org
    with three days of traces is therefore due immediately on the first
    tick after this migration lands; an org whose first trace is a minute
    old is not. ``COALESCE`` to ``now()`` covers the (unusual) org that is
    being graded with no trace row of its own.

    Never raises: a failed counter bump must not fail the grading that
    triggered it, and the interval timer (:func:`orgs_due_for_dream`) is an
    independent path to the same pass.
    """
    try:
        db.execute(
            text(
                """
                INSERT INTO dream_state (org_id, sessions_since, last_run_at)
                VALUES (
                    :org_id,
                    1,
                    COALESCE(
                        (SELECT MIN(created_at) FROM session_traces
                          WHERE org_id = :org_id),
                        now()
                    )
                )
                ON CONFLICT (org_id) DO UPDATE
                SET sessions_since = dream_state.sessions_since + 1,
                    updated_at = now()
                """
            ),
            {"org_id": org_id},
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.warning(
            "dream-state: failed to record graded session org=%s", org_id, exc_info=True
        )


def claim_dream_pass(
    db: Any, *, org_id: str, min_graded_sessions: int, interval_hours: int
) -> Optional[DreamClaim]:
    """Atomically test both dream triggers for ``org_id`` and, if either has
    fired, CLAIM the pass by resetting the counters in the same statement.

    Returns a :class:`DreamClaim` if this caller won, ``None`` if the org is
    not due or another replica won first.

    The ``WHERE`` clause IS the trigger test, so there is no read-then-write
    window: two ``async-worker`` replicas grading the same org concurrently
    both run this statement, row-level locking serialises them, and the
    second one re-evaluates the predicate against the first one's already
    reset ``sessions_since = 0`` / ``last_run_at = now()`` and matches
    nothing. Exactly one pass runs. That is why no advisory lock and no
    ``work_claims`` row is needed here.

    ``last_run_at`` is reset on CLAIM, not on successful completion — a
    persistently failing org (bad LLM config, unreachable S3) therefore
    retries once per interval instead of on every graded session, preserving
    the one-attempt-per-threshold-crossing property the in-memory version
    had. Losing a claimed pass to a crash costs one interval of latency, and
    that is the right side of the trade: the alternative is a hot retry loop
    against a broken dependency.

    ``FROM dream_state prev`` reads the row's PRE-update snapshot in the same
    statement, which is the only way to report the counter values that
    justified the claim — ``RETURNING`` on the updated table yields the
    post-reset zeros.
    """
    row = (
        db.execute(
            text(
                """
                UPDATE dream_state AS d
                SET sessions_since = 0,
                    last_run_at = now(),
                    dream_passes = d.dream_passes + 1,
                    updated_at = now()
                FROM dream_state AS prev
                WHERE prev.org_id = d.org_id
                  AND d.org_id = :org_id
                  AND (
                        d.sessions_since >= :min_sessions
                     OR d.last_run_at <= now()
                          - make_interval(hours => :interval_hours)
                  )
                RETURNING prev.sessions_since AS sessions_since,
                          EXTRACT(
                              EPOCH FROM (now() - prev.last_run_at)
                          ) / 3600.0 AS elapsed_hours
                """
            ),
            {
                "org_id": org_id,
                "min_sessions": min_graded_sessions,
                "interval_hours": interval_hours,
            },
        )
        .mappings()
        .first()
    )
    db.commit()
    if row is None:
        return None

    sessions_since = int(row["sessions_since"] or 0)
    elapsed_hours = float(row["elapsed_hours"] or 0.0)
    if sessions_since >= min_graded_sessions:
        reason = "graded_sessions"
    else:
        reason = "interval"
    return DreamClaim(
        org_id=org_id,
        sessions_since=sessions_since,
        elapsed_hours=elapsed_hours,
        reason=reason,
    )


def orgs_due_for_dream(db: Any, *, interval_hours: int, limit: int) -> list[str]:
    """Orgs whose interval has elapsed, for the periodic dream tick.

    THIS IS THE OTHER HALF OF THE BUG. The interval trigger was never a
    timer — it was a gate evaluated only inside ``_maybe_dream_for_org``,
    which only ran on a graded session. An org with a real trace corpus but
    few graded sessions (production's larger org has 8 in its lifetime) could
    therefore never reach EITHER trigger: not enough grading events to cross
    the count threshold, and no grading events to evaluate the interval
    against. Persisting the clock alone would not have fixed that. This
    query gives the interval a wakeup of its own, driven by the consumer
    loop's existing idle tick.

    Orgs are discovered from ``session_traces`` — the dreamer's actual
    input — LEFT JOINed to ``dream_state``, so an org that has produced
    evidence but has never had a graded session (and therefore has no state
    row) is still found. A missing row is treated as "due if the org's
    oldest trace is older than the interval", the same clock semantics
    :func:`record_graded_session` seeds a new row with.

    ``limit`` bounds how many orgs one tick will dream for, matching the
    ownership sweep's per-pass ceiling convention: a backlog drains over
    several ticks rather than blocking one worker thread on every org at
    once.
    """
    rows = db.execute(
        text(
            """
            SELECT t.org_id
            FROM (
                SELECT org_id, MIN(created_at) AS first_trace_at
                FROM session_traces
                GROUP BY org_id
            ) AS t
            LEFT JOIN dream_state AS d ON d.org_id = t.org_id
            WHERE COALESCE(d.last_run_at, t.first_trace_at)
                  <= now() - make_interval(hours => :interval_hours)
            ORDER BY COALESCE(d.last_run_at, t.first_trace_at) ASC
            LIMIT :limit
            """
        ),
        {"interval_hours": interval_hours, "limit": limit},
    ).all()
    db.commit()
    return [r[0] for r in rows]


def ensure_state_row(db: Any, *, org_id: str) -> None:
    """Create ``org_id``'s state row if absent, seeded exactly as
    :func:`record_graded_session` seeds it, WITHOUT incrementing the counter.

    Needed by the interval timer path: :func:`orgs_due_for_dream` can name an
    org that has never had a graded session and therefore has no row, and
    :func:`claim_dream_pass`'s ``UPDATE`` cannot claim a row that does not
    exist. ``ON CONFLICT DO NOTHING`` keeps this safe to call on every tick
    and safe against two replicas racing the same first sighting.

    Never raises, for the same reason :func:`record_graded_session` does not.
    """
    try:
        db.execute(
            text(
                """
                INSERT INTO dream_state (org_id, sessions_since, last_run_at)
                VALUES (
                    :org_id,
                    0,
                    COALESCE(
                        (SELECT MIN(created_at) FROM session_traces
                          WHERE org_id = :org_id),
                        now()
                    )
                )
                ON CONFLICT (org_id) DO NOTHING
                """
            ),
            {"org_id": org_id},
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.warning(
            "dream-state: failed to ensure state row org=%s", org_id, exc_info=True
        )


def record_dream_result(db: Any, *, org_id: str, written: int) -> None:
    """Record that a claimed pass COMPLETED, and how many proposals it filed.

    Observability only — no trigger reads these columns; the claim already
    reset the clock. ``last_dreamed_at IS NULL`` is what tells an operator
    "this org has never actually been dreamed for", and
    ``last_proposals_written = 0`` is a meaningfully different answer from
    ``NULL``: the pass ran and found nothing, versus never ran. Those two
    were indistinguishable while the state lived in process memory, which is
    a large part of why this bug went unnoticed.

    Never raises: losing the bookkeeping must not turn a successful dream
    pass into a logged failure.
    """
    try:
        db.execute(
            text(
                """
                UPDATE dream_state
                SET last_dreamed_at = now(),
                    last_proposals_written = :written,
                    updated_at = now()
                WHERE org_id = :org_id
                """
            ),
            {"org_id": org_id, "written": written},
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.warning(
            "dream-state: failed to record dream result org=%s", org_id, exc_info=True
        )
