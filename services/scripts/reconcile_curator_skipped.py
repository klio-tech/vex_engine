"""Measure — and optionally re-queue — the rows the curator abandoned.

The defect (fixed in WP-D, ``curator.py``)
------------------------------------------
``curator_tick`` partitions a batch by ``(space_id, project_id)``. A partition
below ``min_inputs`` was skipped with a bare ``continue`` — and then the cursor
was advanced unconditionally to the end of the *whole batch*. The read query
looks only ahead of the cursor, so a skipped partition's rows were never
reconsidered by any later tick.

Why the 7-day TTL is the whole story
------------------------------------
Those rows did not sit patiently waiting to be re-read. ``run_retention.py``
deletes ``recall_hidden`` RAW observations older than
``RAW_OBSERVATION_TTL_DAYS`` (default 7) — and migration 026 backfilled
``recall_hidden = TRUE`` onto every passive hook capture, which is nearly the
entire RAW stream. Advancing past a skipped row put it on a one-week fuse.

So this script reports **two populations and never one**:

``recoverable``
    Candidate rows still present. These can be re-queued.

``expiring``
    Candidate rows still present but already past the TTL horizon and
    ``recall_hidden`` — readable this minute, on the next retention sweep's
    list. Recoverable only until that cron runs.

``unrecoverable_upper_bound``
    Rows the cursor advanced over that no longer exist at all. They cannot be
    read back — that is what unrecoverable means — so they are measured
    indirectly: ``brain_curator_state.processed_count`` counts what the cursor
    passed, and what is still behind it is what survived. The difference is
    what is gone. It is an upper bound on the permanent loss because some of
    those rows were distilled before they were deleted, and nothing on disk
    now distinguishes the two. It is reported anyway, because a number with a
    stated direction of error is worth more than a silence.

Reporting these as one figure would hide the only fact the user actually needs
in order to decide anything.

How a skipped row is recognised
-------------------------------
A distilled partition leaves curator output (``metadata->>'source' =
'curator'``, ``memory_type`` in ``summary``/``fact``) in that same partition at
or after the input row's timestamp, because the curator writes at tick time. A
RAW private row behind the cursor whose partition holds no such output is a
**candidate**. Where the detector errs it errs low: a later tick's output in
the same partition masks an earlier skipped row. That is the right direction
for a number whose purpose is to justify re-running an LLM over somebody's
private memories.

What ``--execute`` does
-----------------------
Rewinds the ``(org, user)`` cursor to just before the earliest recoverable
candidate. That is the entire remediation: one UPDATE against
``brain_curator_state``, never a write to ``session_memories``. The curator's
next tick re-reads from there, and the WP-D fix now holds the cursor at
undistilled rows instead of stepping over them. Rows in between that *were*
distilled get re-read too; that is absorbed by dedup at the cost of an LLM
call, exactly the trade ``curator._cursor_stop`` already documents.

Usage:
    python -m scripts.reconcile_curator_skipped                    # dry run, fleet
    python -m scripts.reconcile_curator_skipped --org <org_id>     # dry run, one org
    python -m scripts.reconcile_curator_skipped --org <org_id> \\
        --execute --confirm <token-from-the-dry-run>

Environment:
    DATABASE_URL: TimescaleDB connection string.
    RAW_OBSERVATION_TTL_DAYS: read only to locate the TTL horizon. This script
        never deletes anything; it reads the same knob ``run_retention.py``
        deletes by so the two agree on where the horizon is.
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from sqlalchemy import create_engine, text

# Both the sibling module and the shared package must resolve whether this is
# run as ``python -m scripts.<name>`` from ``services/`` (which puts
# ``services/`` on the path, not ``services/scripts/``) or executed directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))

import reconcile_common as common  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("reconcile.curator_skipped")

SCRIPT_NAME = "reconcile_curator_skipped"

#: The curator's input types, mirroring ``curator.RAW_MEMORY_TYPES``. Its own
#: outputs (``summary``/``fact``) are deliberately absent: they are evidence of
#: distillation, not candidates for it.
RAW_MEMORY_TYPES = ("observation", "note", "memory", "plan", "decision")

_RAW_TYPES_SQL = ", ".join(f"'{t}'" for t in RAW_MEMORY_TYPES)

#: Mirrors ``run_retention.DEFAULT_RAW_OBSERVATION_TTL_DAYS``.
DEFAULT_RAW_OBSERVATION_TTL_DAYS = 7


@dataclass(frozen=True)
class SkippedPopulation:
    """One ``(org, user)`` pair's residue. Ids and counts only, by design."""

    org_id: str
    user_id: str
    recoverable_ids: list
    expiring_ids: list
    surviving_behind_cursor: int
    processed_count: int
    unrecoverable_upper_bound: int


def ttl_days() -> int:
    """Where the retention sweep's horizon currently sits, in days.

    Reads the same variable ``run_retention.py`` deletes by. A value this
    script cannot parse falls back to the documented default rather than
    raising: being wrong about the horizon mis-labels a population, whereas
    refusing to run produces no measurement at all — and this script deletes
    nothing, so an over-optimistic horizon cannot destroy anything.
    """
    raw = os.environ.get("RAW_OBSERVATION_TTL_DAYS")
    if raw is None:
        return DEFAULT_RAW_OBSERVATION_TTL_DAYS
    try:
        days = int(raw.strip())
    except (AttributeError, ValueError):
        logger.warning(
            "RAW_OBSERVATION_TTL_DAYS=%r is unparseable; reporting the TTL "
            "horizon at the %d-day default",
            raw,
            DEFAULT_RAW_OBSERVATION_TTL_DAYS,
        )
        return DEFAULT_RAW_OBSERVATION_TTL_DAYS
    if days < 1:
        return DEFAULT_RAW_OBSERVATION_TTL_DAYS
    return days


#: Rows behind the cursor whose partition never produced curator output.
#:
#: Scoped by ``org_id`` on both the outer select and the NOT EXISTS. The
#: partition match is NULL-safe (``IS NOT DISTINCT FROM``): a NULL
#: ``space_id``/``project_id`` is a genuine org-wide partition in the curator,
#: not a wildcard, and plain ``=`` would silently make every such row a
#: candidate.
_CANDIDATES_SQL = """
    SELECT m.id,
           m.created_at,
           (m.recall_hidden AND m.created_at < now() - (:ttl_days || ' days')::interval)
               AS past_ttl
    FROM session_memories m
    WHERE m.org_id = :org_id
      AND m.user_id = :user_id
      AND m.scope = 'private'
      AND m.status = 'active'
      AND m.memory_type IN ({raw_types})
      AND (m.created_at, m.id) <= (:cursor_at, :cursor_id)
      AND NOT EXISTS (
            SELECT 1
            FROM session_memories d
            WHERE d.org_id = :org_id
              AND d.user_id = :user_id
              AND d.scope = 'private'
              AND d.memory_type IN ('summary', 'fact')
              AND d.metadata ->> 'source' = 'curator'
              AND d.space_id IS NOT DISTINCT FROM m.space_id
              AND d.project_id IS NOT DISTINCT FROM m.project_id
              AND d.created_at >= m.created_at
      )
    ORDER BY m.created_at ASC, m.id ASC
""".format(raw_types=_RAW_TYPES_SQL)


#: Everything still sitting behind the cursor, whatever its status. Used only
#: as the subtrahend for the gone-row estimate, so it must NOT filter on
#: ``status = 'active'``: a retracted row still exists, and excluding it would
#: inflate the count of rows reported as destroyed.
_SURVIVORS_SQL = """
    SELECT count(*) AS n
    FROM session_memories
    WHERE org_id = :org_id
      AND user_id = :user_id
      AND scope = 'private'
      AND memory_type IN ({raw_types})
      AND (created_at, id) <= (:cursor_at, :cursor_id)
""".format(raw_types=_RAW_TYPES_SQL)


def _cursor_rows(db: Any, org_id: Optional[str]) -> list:
    """Every ``(org, user)`` whose cursor has actually moved.

    A pair with a NULL cursor has never advanced past anything, so it has no
    residue by construction and is excluded in SQL rather than skipped later.
    """
    where = ["last_processed_at IS NOT NULL", "last_processed_id IS NOT NULL"]
    params: dict[str, Any] = {}
    if org_id:
        where.append("org_id = :org_id")
        params["org_id"] = org_id
    return list(
        db.execute(
            text(
                f"""
                SELECT org_id, user_id, last_processed_at, last_processed_id,
                       processed_count
                FROM brain_curator_state
                WHERE {" AND ".join(where)}
                ORDER BY org_id ASC, user_id ASC
                """
            ),
            params,
        ).fetchall()
    )


def find_skipped(db: Any, *, org_id: Optional[str] = None) -> list[SkippedPopulation]:
    """Measure the abandonment residue, one entry per ``(org, user)``.

    Pairs with nothing to report are omitted entirely, so an empty list is the
    healthy answer and the caller never has to distinguish "no residue" from
    "no rows examined".
    """
    days = ttl_days()
    populations: list[SkippedPopulation] = []

    for state in _cursor_rows(db, org_id):
        params = {
            "org_id": state.org_id,
            "user_id": state.user_id,
            "cursor_at": state.last_processed_at,
            "cursor_id": state.last_processed_id,
        }
        candidates = db.execute(
            text(_CANDIDATES_SQL), {**params, "ttl_days": days}
        ).fetchall()
        surviving = int(db.execute(text(_SURVIVORS_SQL), params).fetchone().n or 0)
        processed = int(state.processed_count or 0)

        recoverable = [str(r.id) for r in candidates if not r.past_ttl]
        expiring = [str(r.id) for r in candidates if r.past_ttl]
        # ``processed_count`` is bookkeeping, not a constraint: an earlier
        # rewind by this very script leaves more rows behind the cursor than
        # it claims were passed. Clamped at zero — "-6 rows destroyed" is
        # worse than no number at all.
        gone = max(processed - surviving, 0)

        if not recoverable and not expiring and gone == 0:
            continue
        populations.append(
            SkippedPopulation(
                org_id=state.org_id,
                user_id=state.user_id,
                recoverable_ids=recoverable,
                expiring_ids=expiring,
                surviving_behind_cursor=surviving,
                processed_count=processed,
                unrecoverable_upper_bound=gone,
            )
        )
    return populations


def candidate_ids(populations: Sequence[SkippedPopulation]) -> list[str]:
    """The exact set the confirmation token is computed over.

    Only ``recoverable`` ids: they are the only rows ``--execute`` would act
    on, and a token that moved because an unrelated row aged into ``expiring``
    would train operators to re-copy tokens without reading them.
    """
    ids: list[str] = []
    for population in populations:
        ids.extend(population.recoverable_ids)
    return ids


def rewind_cursor(db: Any, population: SkippedPopulation) -> None:
    """Move one pair's cursor to just before its earliest recoverable row.

    The target is computed in SQL from the row ids, so the rewind cannot land
    somewhere the measurement did not justify. One microsecond before the
    earliest candidate is strictly earlier than it at Postgres timestamp
    resolution, which is what the curator's ``created_at > cursor`` predicate
    needs in order to read that row again.

    Guarded so it can only ever move the cursor BACK. A rewind is a re-read; a
    fast-forward would be exactly the original defect, issued deliberately.
    """
    db.execute(
        text(
            """
            UPDATE brain_curator_state s
            SET last_processed_at = target.at - interval '1 microsecond',
                last_processed_id = '00000000-0000-0000-0000-000000000000'::uuid
            FROM (
                SELECT min(created_at) AS at
                FROM session_memories
                WHERE org_id = :org_id
                  AND user_id = :user_id
                  AND id = ANY(CAST(:ids AS uuid[]))
            ) AS target
            WHERE s.org_id = :org_id
              AND s.user_id = :user_id
              AND target.at IS NOT NULL
              AND s.last_processed_at > target.at - interval '1 microsecond'
            """
        ),
        {
            "org_id": population.org_id,
            "user_id": population.user_id,
            "ids": population.recoverable_ids,
        },
    )
    db.commit()


def report(populations: Sequence[SkippedPopulation], *, limit: int) -> None:
    """Log the measurement. Ids, counts and scopes; never content."""
    days = ttl_days()
    if not populations:
        logger.info("[DRY-RUN] no curator abandonment residue found")
        return

    totals = {"recoverable": 0, "expiring": 0, "gone": 0}
    for p in populations:
        totals["recoverable"] += len(p.recoverable_ids)
        totals["expiring"] += len(p.expiring_ids)
        totals["gone"] += p.unrecoverable_upper_bound
        logger.info(
            "org=%s user=%s recoverable=%d expiring=%d unrecoverable<=%d "
            "(cursor passed %d, %d still present)",
            p.org_id,
            p.user_id,
            len(p.recoverable_ids),
            len(p.expiring_ids),
            p.unrecoverable_upper_bound,
            p.processed_count,
            p.surviving_behind_cursor,
        )
        logger.info(
            "  recoverable ids: %s", common.format_sample(p.recoverable_ids, limit)
        )
        logger.info(
            "  expiring ids:    %s", common.format_sample(p.expiring_ids, limit)
        )

    logger.info(
        "TOTAL recoverable=%d expiring=%d unrecoverable<=%d",
        totals["recoverable"],
        totals["expiring"],
        totals["gone"],
    )
    logger.info(
        "WOULD DO: rewind %d curator cursor(s) to just before their earliest "
        "recoverable row, so the next tick re-reads %d row(s). No memory row "
        "is created, modified or deleted.",
        len(populations),
        totals["recoverable"],
    )
    logger.info(
        "WOULD NOT DO: recover the %d row(s) already destroyed by the %d-day "
        "RAW observation TTL, nor the %d row(s) the next retention sweep will "
        "destroy. Nothing can.",
        totals["gone"],
        days,
        totals["expiring"],
    )


def run(db: Any, args: Any) -> list[SkippedPopulation]:
    """Measure, report, and — only when fully confirmed — re-queue."""
    populations = find_skipped(db, org_id=args.org)
    report(populations, limit=args.sample_limit)

    ids = candidate_ids(populations)
    execute = common.resolve_execute(args, script=SCRIPT_NAME, candidate_ids=ids)
    if not execute:
        logger.info(
            "DRY RUN — nothing was written. To re-queue, review the counts "
            "above and re-run with: --org <org_id> --execute --confirm %s",
            common.confirmation_token(SCRIPT_NAME, args.org, ids),
        )
        return populations

    for population in populations:
        if not population.recoverable_ids:
            continue
        rewind_cursor(db, population)
        logger.info(
            "[EXECUTED] rewound cursor org=%s user=%s to re-queue %d row(s)",
            population.org_id,
            population.user_id,
            len(population.recoverable_ids),
        )
    return populations


def main() -> None:
    parser = common.build_parser(__doc__)
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable is required")
        sys.exit(1)

    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as db:
            run(db, args)
    except common.ConfirmationError as exc:
        logger.error("REFUSED: %s", exc)
        sys.exit(2)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
