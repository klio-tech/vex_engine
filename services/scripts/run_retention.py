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


def _contained(sweep, conn, *, org_id: str, what: str) -> None:
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
                )
                _contained(
                    graph_store.sweep_orphan_entities,
                    tsdb_conn,
                    org_id=org_id,
                    what="orphan graph entities",
                )
                # 3. Compression originals. Not anchored to a memory at all —
                #    these expire on their own short TTL and, until now, were
                #    never deleted by anything.
                _contained(
                    sweep_expired_originals,
                    tsdb_conn,
                    org_id=org_id,
                    what="expired compression originals",
                )

            # Hidden raw captures are curator fuel with a short shelf life:
            # sweep them on a fixed TTL regardless of plan retention. This is
            # the one DELETE here with no org_id predicate — global by design,
            # age its only bound — which is why ``raw_ttl_days`` was validated
            # before the first statement of the run.
            if raw_ttl_days is None:
                logger.info(
                    "RAW_OBSERVATION_TTL_DAYS is off; skipping the hidden raw "
                    "observation sweep"
                )
            else:
                deleted = tsdb_conn.execute(
                    text(
                        """
                        DELETE FROM session_memories
                        WHERE recall_hidden = TRUE
                          AND created_at < NOW() - (:days || ' days')::interval
                        """
                    ),
                    {"days": raw_ttl_days},
                )
                logger.info(
                    "Swept hidden raw observations older than %d days (%s rows)",
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
