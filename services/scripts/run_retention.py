"""Run plan-based data retention enforcement.

Reads plan data from Supabase (source of truth), then enforces
retention in TimescaleDB per organization.

Usage:
    python -m scripts.run_retention

    Or schedule via cron:
    0 3 * * * cd /path/to/services && python -m scripts.run_retention

Environment:
    DATABASE_URL: TimescaleDB connection string
    SUPABASE_DATABASE_URL: Supabase Postgres connection string
    RAW_OBSERVATION_TTL_DAYS: Days to retain recall_hidden raw observations
        before they are swept (default: 7). These rows are curator fuel with
        a short shelf life and are deleted on a fixed TTL independent of
        per-plan retention settings. Must be a whole number of days in
        [1, 3650], or one of "off" / "never" / "disabled" to skip the sweep
        entirely. Anything else is fatal — see
        :func:`parse_raw_observation_ttl_days`.
"""

import json
import logging
import os
import sys
import time

from sqlalchemy import create_engine, text

# When the shared package is not installed, add its source root to sys.path
# so that `from shared.plan_limits import ...` resolves to
# services/shared/shared/plan_limits.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from shared import graph_store
from shared.plan_limits import get_plan_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("retention")


# --- RAW_OBSERVATION_TTL_DAYS ------------------------------------------
#
# This is the only unbounded-blast-radius value in the retention job. The
# hidden-raw sweep below is the one DELETE in this module with NO ``org_id``
# predicate — it deliberately sits outside the per-org loop and is bounded by
# age alone — so the parsed TTL is the only thing standing between a cron run
# and every tenant's ``recall_hidden`` rows.
#
# Executed against Postgres 16, ``NOW() - (:days || ' days')::interval`` gives:
#
#      7 -> cutoff 7 days ago -> a one-second-old row survives
#      0 -> cutoff NOW()      -> a one-second-old row is deleted
#     -1 -> cutoff TOMORROW   -> everything is deleted
#
# and a seeded run with ``days = 0`` removed 3 of 3 hidden rows spanning two
# orgs. The input that triggers it is not adversarial, it is the obvious thing
# an operator types when they want the sweep to stop. So ``0`` and negatives
# are fatal, and there is a real "off" token for them to reach for instead.

DEFAULT_RAW_OBSERVATION_TTL_DAYS = 7

#: Below this the cutoff is at or ahead of NOW() and the sweep stops being a
#: TTL at all.
MIN_RAW_OBSERVATION_TTL_DAYS = 1

#: Ten years. Not a safety property — a sanity bound, so a value that is
#: obviously a typo (a timestamp, a byte count) is reported rather than
#: silently turning the sweep into a no-op nobody notices for a decade.
MAX_RAW_OBSERVATION_TTL_DAYS = 3650

#: Tokens that mean "do not run the hidden-raw sweep at all".
_TTL_DISABLED_TOKENS = frozenset({"off", "never", "disabled"})


class RetentionConfigError(ValueError):
    """A retention environment variable is unusable; the run must not start."""


def parse_raw_observation_ttl_days(raw: str | None) -> int | None:
    """Parse and validate ``RAW_OBSERVATION_TTL_DAYS``.

    Args:
        raw: The raw environment value, or ``None`` when unset.

    Returns:
        The TTL in days, or ``None`` when the sweep is explicitly disabled.

    Raises:
        RetentionConfigError: On any value that is not a whole number of days
            in ``[MIN, MAX]`` and not a disable token. Deliberately fatal
            rather than clamped: an operator who typed ``0`` meant "stop
            deleting", and substituting the 7-day default would delete a week
            of curator fuel while appearing to obey.
    """
    if raw is None:
        return DEFAULT_RAW_OBSERVATION_TTL_DAYS

    token = raw.strip()
    if token.lower() in _TTL_DISABLED_TOKENS:
        return None

    try:
        days = int(token)
    except ValueError:
        raise RetentionConfigError(
            f"RAW_OBSERVATION_TTL_DAYS={raw!r} is not a whole number of days. "
            f"Use an integer in [{MIN_RAW_OBSERVATION_TTL_DAYS}, "
            f"{MAX_RAW_OBSERVATION_TTL_DAYS}], or 'off' to skip the sweep."
        ) from None

    if days < MIN_RAW_OBSERVATION_TTL_DAYS:
        raise RetentionConfigError(
            f"RAW_OBSERVATION_TTL_DAYS={days} would put the delete cutoff at or "
            f"after NOW(), deleting every recall_hidden row in EVERY org "
            f"(the sweep has no org_id predicate by design). Refusing to run. "
            f"To turn the sweep off, set RAW_OBSERVATION_TTL_DAYS=off."
        )

    if days > MAX_RAW_OBSERVATION_TTL_DAYS:
        raise RetentionConfigError(
            f"RAW_OBSERVATION_TTL_DAYS={days} exceeds the "
            f"{MAX_RAW_OBSERVATION_TTL_DAYS}-day sanity bound and is almost "
            f"certainly a typo. Use a value in "
            f"[{MIN_RAW_OBSERVATION_TTL_DAYS}, {MAX_RAW_OBSERVATION_TTL_DAYS}], "
            f"or 'off' to skip the sweep."
        )

    return days


def sweep_expired_originals(db, *, org_id: str) -> int:
    """Delete this org's compression originals whose TTL has passed.

    Migration 034 gave ``compression_cache`` an ``expires_at`` column, an
    index created explicitly for "the expiry sweep", and a docstring promising
    24-hour retention — and then nothing ever deleted a row. ``expires_at``
    was read only as a predicate in ``load_original`` / ``handle_is_expired``,
    so an expired original became unreachable to its owner while staying on
    disk forever. The rows are whatever the user compressed (logs, source
    files, API responses); content a user can neither retrieve nor be rid of
    is a retention-policy hole, and the unbounded growth is the lesser half
    of it.

    Expiry is absolute, not plan-derived: the TTL is hours, far inside any
    plan's retention window, so there is nothing for a plan to lengthen.

    Returns the number of originals removed.
    """
    result = db.execute(
        text(
            """
            DELETE FROM compression_cache
            WHERE org_id = :org_id
              AND expires_at <= NOW()
            """
        ),
        {"org_id": org_id},
    )
    db.commit()
    return int(result.rowcount or 0)


def sweep_evidence_for_org(conn, *, org_id: str, retention_days: int) -> dict[str, int]:
    """Delete this org's expired evidence: traces and outcome grades.

    Evidence, not decisions. Three tables:

    - ``session_traces`` (migration 041): the redacted transcript pointer
      the dreamer reads to explain a proposal.
    - ``session_outcomes`` (migration 042): the ONE verdict per graded
      session.
    - ``recall_outcomes`` (migration 042): per-memory attribution rows
      joined to a ``session_outcomes`` row.

    ``memory_proposals`` (migration 045) is deliberately NOT swept here —
    a proposal is a decision made FROM evidence, not evidence itself, and
    it outlives the sessions that justified it (an approved/rejected
    proposal is an audit record). ``claims`` are also untouched — they
    carry their own TTL and expire on their own schedule, independent of
    plan retention.

    ``brain_recall_events`` is untouched too, on purpose: nothing in this
    codebase deletes those rows today (see ``run_retention.py``'s module
    history), and adding that here would silently widen this function's
    blast radius past what the three evidence tables above call for. If
    that gap is ever closed, it should be its own sweep with its own
    tests, not folded into this one.

    ``recall_outcomes`` rows also disappear via ``ON DELETE CASCADE`` when
    their parent ``session_outcomes`` row is deleted (migration 042). This
    function deletes ``recall_outcomes`` directly anyway, on its own
    ``created_at`` cutoff, rather than relying solely on that cascade:
    ``recall_outcomes.created_at`` and its parent's ``graded_at`` are not
    guaranteed to march in lockstep, and a caller of this function (or of
    the DB-gated tests) should be able to reason about each table's own
    expiry rule without tracing a foreign key to find out why a row went.
    The direct delete and the cascade agree on every real row; the direct
    delete is just the one that is not an implementation detail.

    Guard: ``retention_days <= 0`` means "keep forever" (the same -1
    convention every other quota field on ``PlanConfig`` uses — see
    ``shared.plan_limits``) and is refused here exactly as ``run_retention``
    refuses it before calling ``enforce_plan_retention`` /
    ``enforce_memory_retention``: passed through, ``NOW() - (:days ||
    ' days')::interval`` lands at or after NOW(), deleting every row that
    exists. Skipped, not clamped — the caller may pass ``-1`` for
    "unlimited" and this function must not turn that into "today".

    Returns ``{"session_traces": n, "session_outcomes": n,
    "recall_outcomes": n}`` — each count is rows actually removed for
    ``org_id``, or all zeros when the guard above skips the sweep.
    """
    if retention_days <= 0:
        return {"session_traces": 0, "session_outcomes": 0, "recall_outcomes": 0}

    cutoff_interval = {"days": retention_days}

    recall_outcomes_removed = (
        conn.execute(
            text(
                """
            DELETE FROM recall_outcomes
            WHERE org_id = :org_id
              AND created_at < NOW() - (:days || ' days')::interval
            """
            ),
            {"org_id": org_id, **cutoff_interval},
        ).rowcount
        or 0
    )

    session_outcomes_removed = (
        conn.execute(
            text(
                """
            DELETE FROM session_outcomes
            WHERE org_id = :org_id
              AND graded_at < NOW() - (:days || ' days')::interval
            """
            ),
            {"org_id": org_id, **cutoff_interval},
        ).rowcount
        or 0
    )

    session_traces_removed = (
        conn.execute(
            text(
                """
            DELETE FROM session_traces
            WHERE org_id = :org_id
              AND created_at < NOW() - (:days || ' days')::interval
            """
            ),
            {"org_id": org_id, **cutoff_interval},
        ).rowcount
        or 0
    )

    conn.commit()

    return {
        "session_traces": int(session_traces_removed),
        "session_outcomes": int(session_outcomes_removed),
        "recall_outcomes": int(recall_outcomes_removed),
    }


def _contained(sweep, conn, *, org_id: str, what: str, dry_run: bool = False) -> None:
    """Run one per-org sweep, containing its failure to that org.

    ``dry_run`` skips the sweep entirely rather than attempting a read-only
    variant: these sweeps are DELETEs behind a seam and have no counting form,
    so the honest thing under --dry-run is to run none of them and say so. A
    flag named "dry run" that still deletes four things is worse than no flag.
    """
    if dry_run:
        logger.info("DRY RUN: skipping %s for org %s", what, org_id)
        return

    return _contained_impl(sweep, conn, org_id=org_id, what=what)


def _contained_impl(sweep, conn, *, org_id: str, what: str) -> None:
    """Run one org's sweep so that its failure stays its own.

    Hygiene, not the job: a sweep that blows up must not leave every later
    org's retention unenforced. The rollback matters as much as the except —
    a failed statement aborts the transaction, and without it every
    subsequent statement on this connection would raise
    ``InFailedSqlTransaction``, converting one org's bad sweep into a
    silently empty retention run.
    """
    try:
        removed = sweep(conn, org_id=org_id)
        logger.info("Swept %s: %s row(s) for org=%s", what, removed, org_id)
    except Exception:
        logger.exception("%s sweep failed for org=%s", what, org_id)
        try:
            conn.rollback()
        except Exception:
            logger.exception("rollback after the %s sweep also failed", what)


def main() -> None:
    # --dry-run reports what the observation sweep would remove and what it is
    # holding back, and issues no DELETE. A fleet-wide, no-org-predicate DELETE
    # that cannot be rehearsed is a job nobody can safely schedule for the
    # first time, which is part of why this one never was.
    dry_run = "--dry-run" in sys.argv[1:]
    if dry_run:
        logger.info("DRY RUN: no rows will be deleted")

    database_url = os.environ.get("DATABASE_URL")
    supabase_url = os.environ.get("SUPABASE_DATABASE_URL")

    if not database_url:
        logger.error("DATABASE_URL environment variable is required")
        sys.exit(1)
    if not supabase_url:
        logger.error("SUPABASE_DATABASE_URL environment variable is required")
        sys.exit(1)

    # Validated here, at the top, and not where it is used: everything below
    # this point issues DELETEs (the per-org edge, entity and compression
    # sweeps each commit as they go). Validating at the point of use would
    # leave a run that dies on a bad TTL having already committed deletions
    # for every org.
    try:
        raw_ttl_days = parse_raw_observation_ttl_days(
            os.environ.get("RAW_OBSERVATION_TTL_DAYS")
        )
    except RetentionConfigError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    tsdb_engine = create_engine(database_url, pool_pre_ping=True)
    supa_engine = create_engine(supabase_url, pool_pre_ping=True)

    logger.info("Starting plan-based data retention enforcement...")
    start = time.monotonic()

    try:
        # 1. Read all team accounts with plan data from Supabase
        with supa_engine.connect() as supa_conn:
            accounts = supa_conn.execute(
                text(
                    "SELECT slug, vex_plan, vex_plan_overrides FROM accounts "
                    "WHERE is_personal_account = false AND slug IS NOT NULL"
                )
            ).fetchall()

        # 2. For each account, resolve retention days and enforce in TimescaleDB
        with tsdb_engine.connect() as tsdb_conn:
            for row in accounts:
                slug, plan, overrides_raw = row[0], row[1] or "free", row[2]
                overrides = (
                    json.loads(overrides_raw)
                    if isinstance(overrides_raw, str)
                    else overrides_raw
                )
                config = get_plan_config(plan, overrides)

                # Find org_id in TimescaleDB by account_slug
                org_row = tsdb_conn.execute(
                    text("SELECT org_id FROM organizations WHERE account_slug = :slug"),
                    {"slug": slug},
                ).fetchone()

                if org_row is None:
                    continue

                org_id = org_row[0]

                # A non-positive day count means "keep forever" — it is the
                # -1 convention every other quota field on PlanConfig uses, and
                # Klio's Free and Team tiers are published as "retention
                # forever". Passed through, it is the most destructive value
                # available: NOW() - ('-1 days')::interval lands in the FUTURE,
                # so every row matches and the org's whole history goes. Zero is
                # the same story with the cutoff at NOW().
                #
                # Migration 038 guards both SQL functions too. This check is
                # here as well because skipping the round trip is clearer than
                # relying on the callee to decline, and because a reader of this
                # loop should not have to go and check.
                if config.retention_days > 0:
                    tsdb_conn.execute(
                        text("SELECT enforce_plan_retention(:org_id, :days)"),
                        {"org_id": org_id, "days": config.retention_days},
                    )
                    logger.info(
                        "Enforced %d-day retention for org=%s plan=%s",
                        config.retention_days,
                        org_id,
                        plan,
                    )
                else:
                    logger.info(
                        "Execution retention is unlimited for org=%s plan=%s; "
                        "skipping the sweep",
                        org_id,
                        plan,
                    )

                if config.memory_retention_days > 0:
                    tsdb_conn.execute(
                        text("SELECT enforce_memory_retention(:org_id, :days)"),
                        {"org_id": org_id, "days": config.memory_retention_days},
                    )
                    logger.info(
                        "Enforced %d-day MEMORY retention for org=%s plan=%s",
                        config.memory_retention_days,
                        org_id,
                        plan,
                    )
                else:
                    logger.info(
                        "Memory retention is unlimited for org=%s plan=%s; "
                        "skipping the sweep",
                        org_id,
                        plan,
                    )

                # Retention deletes from session_memories only. Everything the
                # memory row anchored has to be collected here, each sweep
                # contained per-org so one failure cannot silently empty the
                # rest of the run.
                #
                # 1. Edges. The edge table carries no foreign keys BY DESIGN
                #    (a forget must never fail because of an edge), so nothing
                #    else collects the edges whose endpoints just vanished.
                # 2. Entities. An entity's evidence IS its edges, so this must
                #    run AFTER the edge sweep: until the dead edges are gone
                #    they still count as evidence and the entity is spared.
                #
                # Both routed through the GraphStore seam, the only module
                # allowed to name the graph tables.
                _contained(
                    graph_store.sweep_orphan_edges,
                    tsdb_conn,
                    org_id=org_id,
                    what="orphan graph edges",
                    dry_run=dry_run,
                )
                _contained(
                    graph_store.sweep_orphan_entities,
                    tsdb_conn,
                    org_id=org_id,
                    what="orphan graph entities",
                    dry_run=dry_run,
                )
                # 3. Compression originals. Not anchored to a memory at all —
                #    these expire on their own short TTL and, until now, were
                #    never deleted by anything.
                _contained(
                    sweep_expired_originals,
                    tsdb_conn,
                    org_id=org_id,
                    what="expired compression originals",
                    dry_run=dry_run,
                )
                # 4. Evidence: session_traces / session_outcomes /
                #    recall_outcomes. These are the dreamer's raw material,
                #    not the memories themselves, so they follow the memory
                #    retention window (the same value gating
                #    enforce_memory_retention above) rather than the shorter
                #    Vex execution-record window — a free-tier org that just
                #    had its retention_days=1 execution log swept must not
                #    also lose the transcript its own proposals cite as
                #    evidence a day later. sweep_evidence_for_org carries its
                #    own <= 0 guard, so passing -1/"unlimited" through here is
                #    safe.
                _contained(
                    lambda conn, *, org_id: sweep_evidence_for_org(
                        conn, org_id=org_id, retention_days=config.memory_retention_days
                    ),
                    tsdb_conn,
                    org_id=org_id,
                    what="expired evidence (traces/outcomes)",
                    dry_run=dry_run,
                )

            # Hidden raw captures are curator fuel with a short shelf life:
            # sweep them on a fixed TTL regardless of plan retention. This is
            # the one DELETE here with no org_id predicate — global by design,
            # age its only bound — which is why ``raw_ttl_days`` was validated
            # before the first statement of the run.
            #
            # AGE ALONE IS NOT A SAFE BOUND, and measuring it proved that.
            # These rows are fuel: the curator distils them into facts,
            # decisions and plans, and until it has read a row that row is the
            # only copy of what happened. The curator can fall behind the TTL —
            # it is a periodic job over a per-user cursor, and on 2026-08-18 it
            # was six days behind for the fleet's largest org. Sweeping on age
            # alone that day would have deleted 10,580 observations nobody had
            # ever distilled, 9,848 of them from one org.
            #
            # So the sweep is bounded by BOTH: older than the TTL, and already
            # passed by the curator's cursor for that row's own (org, user).
            # A row whose owner has no cursor at all is never swept — the
            # curator has not run for them even once, so every row is still
            # fuel.
            #
            # The cost of being wrong in each direction is asymmetric, which is
            # what settles the design. Keeping a row too long costs storage.
            # Deleting one early destroys the only record of a session, with no
            # way to know what was lost.
            if raw_ttl_days is None:
                logger.info(
                    "RAW_OBSERVATION_TTL_DAYS is off; skipping the hidden raw "
                    "observation sweep"
                )
            else:
                sweep_sql = """
                        DELETE FROM session_memories m
                        WHERE m.recall_hidden = TRUE
                          AND m.created_at < NOW() - (:days || ' days')::interval
                          AND EXISTS (
                                SELECT 1 FROM brain_curator_state c
                                WHERE c.org_id = m.org_id
                                  AND c.user_id IS NOT DISTINCT FROM m.user_id
                                  AND c.last_processed_at IS NOT NULL
                                  AND c.last_processed_at >= m.created_at
                          )
                """
                if dry_run:
                    held = tsdb_conn.execute(
                        text(
                            """
                            SELECT count(*) FROM session_memories m
                            WHERE m.recall_hidden = TRUE
                              AND m.created_at
                                  < NOW() - (:days || ' days')::interval
                              AND NOT EXISTS (
                                    SELECT 1 FROM brain_curator_state c
                                    WHERE c.org_id = m.org_id
                                      AND c.user_id IS NOT DISTINCT FROM m.user_id
                                      AND c.last_processed_at IS NOT NULL
                                      AND c.last_processed_at >= m.created_at
                              )
                            """
                        ),
                        {"days": raw_ttl_days},
                    ).scalar()
                    would = tsdb_conn.execute(
                        text(
                            sweep_sql.replace(
                                "DELETE FROM session_memories m",
                                "SELECT count(*) FROM session_memories m",
                            )
                        ),
                        {"days": raw_ttl_days},
                    ).scalar()
                    logger.info(
                        "DRY RUN: would sweep %s hidden raw observations older "
                        "than %d days; holding %s the curator has not read yet",
                        would,
                        raw_ttl_days,
                        held,
                    )
                else:
                    deleted = tsdb_conn.execute(text(sweep_sql), {"days": raw_ttl_days})
                    logger.info(
                        "Swept hidden raw observations older than %d days "
                        "and already curated (%s rows)",
                        raw_ttl_days,
                        getattr(deleted, "rowcount", "?"),
                    )

            tsdb_conn.commit()

    except Exception:
        logger.exception("Retention enforcement failed")
        sys.exit(1)
    finally:
        tsdb_engine.dispose()
        supa_engine.dispose()

    elapsed = time.monotonic() - start
    logger.info("Retention enforcement completed in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
