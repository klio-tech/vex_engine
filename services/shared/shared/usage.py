"""Klio memory metering: monthly usage counters + quota enforcement.

Mirrors the best-effort, SYNCHRONOUS psycopg2 style of app.recall_log. Reads one
``memory_usage_counters`` row per check (cheap PK lookup) and UPSERT-increments
it on record. ``check_memory_quota`` raises ``MemoryQuotaExceeded`` for a hard
block (free/starter at cap) and returns an ``over_cap`` bool otherwise; it FAILS
OPEN on any DB error (a metering hiccup must never block a real operation).
``record_usage`` is best-effort and never raises.

Threading: synchronous. Call directly from MCP tool bodies; from async FastAPI
handlers call via ``anyio.to_thread.run_sync`` (see app.capture).
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import text

from shared.plan_limits import PlanConfig

logger = logging.getLogger("shared.usage")

Kind = Literal["memory", "recall"]

# Upgrade ladder: maps a plan name to the next tier up. Used to surface a
# machine-readable upgrade target on a quota block so clients can deep-link to
# the right plan. ``enterprise`` is terminal (already the top tier) and maps to
# itself; any unknown plan also resolves to ``enterprise`` via ``.get`` default.
NEXT_TIER: dict[str, str] = {
    "free": "starter",
    "starter": "pro",
    "pro": "team",
    "team": "enterprise",
    "enterprise": "enterprise",
}


class MemoryQuotaExceeded(Exception):
    """Raised when a hard-capped plan is at/over its monthly memory quota."""

    def __init__(
        self, *, metric: Kind, limit: int, plan_used: int, suggested_plan: str
    ) -> None:
        self.metric = metric
        self.limit = limit
        self.used = plan_used
        self.suggested_plan = suggested_plan
        self.message = (
            f"Monthly {metric} limit reached ({limit:,}). "
            f"Upgrade to the {suggested_plan} plan at https://app.klio.tech "
            "to continue."
        )
        super().__init__(self.message)


@dataclass(frozen=True)
class Counters:
    memories_captured: int = 0
    recalls: int = 0


def current_period() -> dt.date:
    """First day of the current UTC month (the counter's period key)."""
    today = dt.datetime.now(dt.timezone.utc).date()
    return today.replace(day=1)


def get_counters(db: Any, org_id: str) -> Counters:
    """Read the current-period counter row (zeros if none)."""
    row = db.execute(
        text(
            """
            SELECT memories_captured, recalls
            FROM memory_usage_counters
            WHERE org_id = :org_id AND period_month = :period
            """
        ),
        {"org_id": org_id, "period": current_period()},
    ).fetchone()
    if row is None:
        return Counters()
    return Counters(
        memories_captured=int(row.memories_captured),
        recalls=int(row.recalls),
    )


def _safe_rollback(db: Any, org_id: str) -> None:
    """Clear an aborted transaction so the caller's session stays usable.

    The rollback itself is guarded: this only ever runs inside a handler that
    has already decided to degrade, and a session so broken that even ROLLBACK
    fails must not turn a best-effort path into a raising one.
    """
    try:
        db.rollback()
    except Exception:
        logger.warning(
            "rollback after a swallowed metering error failed for org %s",
            org_id,
            exc_info=True,
        )


def check_memory_quota(
    db: Any, org_id: str, plan_config: PlanConfig, kind: Kind, plan_name: str = "free"
) -> bool:
    """Check the org's quota for *kind* this month.

    Returns ``over_cap`` (True when at/over the limit but allowed via overage).
    Raises ``MemoryQuotaExceeded`` when at/over the limit and overage is not
    allowed. Fails OPEN (returns False) on any DB/unexpected error.

    ``plan_name`` is the resolved plan tier; it only drives the
    ``suggested_plan`` upgrade target on a hard block (the quota math itself
    comes from ``plan_config``). It defaults to ``"free"`` so existing positional
    callers stay valid while the upgrade hint degrades gracefully.
    """
    limit = (
        plan_config.memories_per_month
        if kind == "memory"
        else plan_config.recalls_per_month
    )
    if limit == -1:
        return False
    try:
        counters = get_counters(db, org_id)
    except Exception:
        # Roll back before failing open. Postgres aborts the whole transaction
        # on the first failed statement, so swallowing the error without a
        # ROLLBACK hands the caller a session that refuses everything — and
        # this check runs BEFORE the memory write, on the same session. The
        # caller would then be told their memory was not stored, because of a
        # fault in metering they never asked about. ``record_usage`` below has
        # always done this; this call site is its sibling, twenty lines up.
        _safe_rollback(db, org_id)
        logger.warning(
            "memory quota read failed for org %s; allowing", org_id, exc_info=True
        )
        return False
    used = counters.memories_captured if kind == "memory" else counters.recalls
    if used >= limit:
        if plan_config.overage_allowed:
            return True
        raise MemoryQuotaExceeded(
            metric=kind,
            limit=limit,
            plan_used=used,
            suggested_plan=NEXT_TIER.get(plan_name, "enterprise"),
        )
    return False


def record_usage(db: Any, org_id: str, kind: Kind, *, over_cap: bool) -> None:
    """Increment the monthly counter for *kind*. Best-effort; never raises."""
    mem = 1 if kind == "memory" else 0
    rec = 1 if kind == "recall" else 0
    mem_over = mem if over_cap else 0
    rec_over = rec if over_cap else 0
    try:
        db.execute(
            text(
                """
                INSERT INTO memory_usage_counters
                  (org_id, period_month, memories_captured, recalls,
                   memories_overage, recalls_overage, updated_at)
                VALUES (:org_id, :period, :mem, :rec, :mem_over, :rec_over, now())
                ON CONFLICT (org_id, period_month) DO UPDATE SET
                  memories_captured = memory_usage_counters.memories_captured + :mem,
                  recalls           = memory_usage_counters.recalls + :rec,
                  memories_overage  = memory_usage_counters.memories_overage + :mem_over,
                  recalls_overage   = memory_usage_counters.recalls_overage + :rec_over,
                  updated_at        = now()
                """
            ),
            {
                "org_id": org_id,
                "period": current_period(),
                "mem": mem,
                "rec": rec,
                "mem_over": mem_over,
                "rec_over": rec_over,
            },
        )
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning(
            "usage record failed for org %s; ignoring", org_id, exc_info=True
        )
