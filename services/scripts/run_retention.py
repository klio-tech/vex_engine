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
        per-plan retention settings.
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
            # sweep them on a fixed TTL regardless of plan retention.
            raw_ttl_days = int(os.environ.get("RAW_OBSERVATION_TTL_DAYS", "7"))
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
