# Klio Memory Metering, Entitlements & Enforcement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Meter Klio memory usage (memories captured + recalls per month), enforce per-plan caps + retention, and gate the cross-agent-sync flag — by extending the engine's existing entitlements machinery.

**Architecture:** Add memory fields to the existing `PlanConfig`; add a monthly counter table + a `shared/usage.py` quota module (raw SQL, mirrors `recall_log.py`); enforce check-before / record-after at the write and recall boundaries (MCP tools + REST `/capture/*`); extend the retention cron to prune `session_memories`. Additive — reliability enforcement is untouched.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy Core (raw `text()` SQL, no ORM), Alembic, pytest (mock-based), Postgres + pgvector.

**Spec:** `docs/plans/2026-06-01-klio-memory-metering-design.md`

**Branch:** `feat/klio-memory-metering` (already created). Local commits only.

**Test conventions:** `shared` tests run from `services/shared/` with `pytest` (`pythonpath=["."]`, `asyncio_mode=auto`). mcp-server tests run from `services/mcp-server/`. Mock-based — no real Postgres/LLM. Lint: `ruff check` from repo root.

---

## File map
- Modify: `services/shared/shared/plan_limits.py` — add memory fields + values (Task 1)
- Create: `services/migrations/alembic/versions/024_memory_usage_counters.py` — counter table (Task 2)
- Create: `services/shared/shared/usage.py` — counters + quota + exception (Task 3)
- Create: `services/shared/tests/test_plan_memory_limits.py`, `services/shared/tests/test_usage.py`
- Modify: `services/mcp-server/app/auth.py` — `resolve_principal_plan` (Task 4)
- Modify: `services/mcp-server/app/mcp_app.py` — enforce on `_perform_write` / `_perform_recall` (Task 5)
- Modify: `services/mcp-server/app/capture.py` — enforce on `/capture/event|transcript|recall` (Task 6)
- Create: `services/migrations/alembic/versions/025_enforce_memory_retention.py`; Modify: `services/scripts/run_retention.py` (Task 7)
- Modify: `services/shared/shared/plan_limits.py` (cross-ref comment), final verify (Task 8)

---

## Task 0: Validate the environment

- [ ] **Step 1: Confirm Postgres is up and migrations apply**

Run (from `services/migrations/`):
```bash
DATABASE_URL="postgresql://agentguard:agentguard_dev@localhost:5432/agentguard" alembic upgrade head
```
Expected: ends at `023` with no error. (If Postgres isn't running, start the dev compose stack first.)

- [ ] **Step 2: Confirm the shared test suite runs green at baseline**

Run (from `services/shared/`):
```bash
pytest -q
```
Expected: existing suite PASSES (this is the regression baseline).

---

## Task 1: Add memory fields to PlanConfig

**Files:**
- Modify: `services/shared/shared/plan_limits.py`
- Test: `services/shared/tests/test_plan_memory_limits.py`

- [ ] **Step 1: Write the failing test**

Create `services/shared/tests/test_plan_memory_limits.py`:
```python
"""Memory-tier fields on PlanConfig (Klio memory metering)."""

from shared.plan_limits import PLAN_LIMITS, get_plan_config


def test_free_memory_limits():
    p = PLAN_LIMITS["free"]
    assert p.memories_per_month == 1_000
    assert p.recalls_per_month == 10_000
    assert p.memory_retention_days == 1
    assert p.memory_sync_enabled is False


def test_paid_memory_limits():
    assert PLAN_LIMITS["starter"].recalls_per_month == 100_000
    assert PLAN_LIMITS["pro"].memories_per_month == 150_000
    assert PLAN_LIMITS["team"].recalls_per_month == 10_000_000
    assert PLAN_LIMITS["starter"].memory_sync_enabled is True


def test_overage_alignment():
    # Free/Starter hard-cap; Pro/Team allow overage.
    assert PLAN_LIMITS["free"].overage_allowed is False
    assert PLAN_LIMITS["starter"].overage_allowed is False
    assert PLAN_LIMITS["pro"].overage_allowed is True
    assert PLAN_LIMITS["team"].overage_allowed is True


def test_defaults_keep_old_constructors_working():
    # A PlanConfig built without memory fields must still construct.
    from shared.plan_limits import PlanConfig

    cfg = get_plan_config("free", {"observations_per_month": 5})
    assert cfg.observations_per_month == 5
    assert cfg.memories_per_month == 1_000  # unchanged by the override
    assert isinstance(PlanConfig, type)
```

- [ ] **Step 2: Run it to verify it fails**

Run (from `services/shared/`): `pytest tests/test_plan_memory_limits.py -q`
Expected: FAIL — `AttributeError: 'PlanConfig' object has no attribute 'memories_per_month'`.

- [ ] **Step 3: Add the fields to the dataclass**

In `services/shared/shared/plan_limits.py`, inside the `PlanConfig` dataclass, after the `max_dataset_items: int = 0` line (the last field), add:
```python
    # Memory (Klio) — value metric for the memory product. See the marketing
    # mirror at Dashboard/apps/landing/lib/pricing.ts (keep the two in sync).
    # memories_captured = non-deduped writes/mo; recalls = read queries/mo.
    # -1 = unlimited, 0 = disabled.
    memories_per_month: int = -1
    recalls_per_month: int = -1
    # Days a session_memories row is retained before the retention sweep prunes it.
    memory_retention_days: int = 30
    # Flag only in this sub-project: real-time cross-agent push (Redis pub/sub)
    # is a later sub-project. Memory is already cross-agent via shared Postgres.
    memory_sync_enabled: bool = False
```

- [ ] **Step 4: Set the values in each plan**

In the same file, add these keyword args to each entry of `PLAN_LIMITS` (place them after `overage_allowed=...` in each):
```python
# free:
        memories_per_month=1_000,
        recalls_per_month=10_000,
        memory_retention_days=1,
        memory_sync_enabled=False,
# starter:
        memories_per_month=25_000,
        recalls_per_month=100_000,
        memory_retention_days=7,
        memory_sync_enabled=True,
# pro:
        memories_per_month=150_000,
        recalls_per_month=1_000_000,
        memory_retention_days=30,
        memory_sync_enabled=True,
# team:
        memories_per_month=1_500_000,
        recalls_per_month=10_000_000,
        memory_retention_days=90,
        memory_sync_enabled=True,
# enterprise:
        memories_per_month=-1,
        recalls_per_month=-1,
        memory_retention_days=365,
        memory_sync_enabled=True,
```
(`overage_allowed` already matches the spec — free/starter `False`, pro/team/enterprise `True` — leave it.)

- [ ] **Step 5: Run the test to verify it passes**

Run (from `services/shared/`): `pytest tests/test_plan_memory_limits.py -q`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add services/shared/shared/plan_limits.py services/shared/tests/test_plan_memory_limits.py
git commit -m "feat(plans): add memory tier fields (memories/recalls/retention/sync) to PlanConfig"
```

---

## Task 2: Migration 024 — memory_usage_counters table

**Files:**
- Create: `services/migrations/alembic/versions/024_memory_usage_counters.py`

- [ ] **Step 1: Write the migration**

Create `services/migrations/alembic/versions/024_memory_usage_counters.py`:
```python
"""Klio memory metering: monthly usage counters.

ADDITIVE ONLY. Introduces ``memory_usage_counters``, a per-(org, month) rollup
of billable memory usage (memories captured + recalls), incremented on the write
and recall paths. Enforcement reads one row per request; the counter survives
retention deletes (unlike COUNT over session_memories). Nothing existing is
touched.

Revision ID: 024
Revises: 023
Create Date: 2026-06-01
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "024"
down_revision: Union[str, None] = "023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_usage_counters",
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("period_month", sa.Date(), nullable=False),
        sa.Column("memories_captured", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("recalls", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("memories_overage", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("recalls_overage", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("org_id", "period_month"),
    )


def downgrade() -> None:
    op.drop_table("memory_usage_counters")
```

- [ ] **Step 2: Apply the migration**

Run (from `services/migrations/`):
```bash
DATABASE_URL="postgresql://agentguard:agentguard_dev@localhost:5432/agentguard" alembic upgrade head
```
Expected: applies `024`, no error.

- [ ] **Step 3: Verify the table exists**

Run:
```bash
psql "$DATABASE_URL" -c "\d memory_usage_counters"
```
Expected: shows the 7 columns and PK `(org_id, period_month)`.

- [ ] **Step 4: Commit**

```bash
git add services/migrations/alembic/versions/024_memory_usage_counters.py
git commit -m "feat(db): migration 024 — memory_usage_counters table"
```

---

## Task 3: usage.py — counters, quota check, exception

**Files:**
- Create: `services/shared/shared/usage.py`
- Test: `services/shared/tests/test_usage.py`

- [ ] **Step 1: Write the failing test**

Create `services/shared/tests/test_usage.py`:
```python
"""Memory usage metering + quota enforcement."""

import datetime as dt
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from shared.plan_limits import get_plan_config
from shared.usage import (
    MemoryQuotaExceeded,
    check_memory_quota,
    current_period,
    record_usage,
)


def _db_with_counters(memories: int, recalls: int):
    """A mock db whose SELECT returns one counter row."""
    db = MagicMock()
    row = SimpleNamespace(memories_captured=memories, recalls=recalls)
    db.execute.return_value.fetchone.return_value = row
    return db


def _db_no_row():
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = None
    return db


def test_current_period_is_first_of_month():
    p = current_period()
    assert isinstance(p, dt.date)
    assert p.day == 1


def test_under_cap_allows_no_overage():
    db = _db_with_counters(memories=10, recalls=0)
    over = check_memory_quota(db, "org-1", get_plan_config("free"), "memory")
    assert over is False


def test_at_cap_free_raises():
    db = _db_with_counters(memories=1_000, recalls=0)
    with pytest.raises(MemoryQuotaExceeded) as exc:
        check_memory_quota(db, "org-1", get_plan_config("free"), "memory")
    assert exc.value.metric == "memory"
    assert exc.value.limit == 1_000


def test_at_cap_pro_allows_overage():
    db = _db_with_counters(memories=150_000, recalls=0)
    over = check_memory_quota(db, "org-1", get_plan_config("pro"), "memory")
    assert over is True  # allowed, but flagged as overage


def test_unlimited_never_blocks():
    db = _db_with_counters(memories=99_999_999, recalls=0)
    over = check_memory_quota(db, "org-1", get_plan_config("enterprise"), "memory")
    assert over is False


def test_recall_metric_uses_recall_limit():
    db = _db_with_counters(memories=0, recalls=10_000)
    with pytest.raises(MemoryQuotaExceeded):
        check_memory_quota(db, "org-1", get_plan_config("free"), "recall")


def test_check_fails_open_on_db_error():
    db = MagicMock()
    db.execute.side_effect = RuntimeError("db down")
    # A metering DB error must never block a real operation.
    over = check_memory_quota(db, "org-1", get_plan_config("free"), "memory")
    assert over is False


def test_record_usage_upserts_memory():
    db = _db_no_row()
    record_usage(db, "org-1", "memory", over_cap=False)
    sql = db.execute.call_args[0][0].text
    assert "INSERT INTO memory_usage_counters" in sql
    assert "ON CONFLICT" in sql
    params = db.execute.call_args[0][1]
    assert params["mem"] == 1 and params["rec"] == 0
    db.commit.assert_called_once()


def test_record_usage_flags_overage():
    db = _db_no_row()
    record_usage(db, "org-1", "recall", over_cap=True)
    params = db.execute.call_args[0][1]
    assert params["rec"] == 1 and params["rec_over"] == 1
```

- [ ] **Step 2: Run it to verify it fails**

Run (from `services/shared/`): `pytest tests/test_usage.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.usage'`.

- [ ] **Step 3: Write the module**

Create `services/shared/shared/usage.py`:
```python
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


class MemoryQuotaExceeded(Exception):
    """Raised when a hard-capped plan is at/over its monthly memory quota."""

    def __init__(self, *, metric: Kind, limit: int, plan_used: int) -> None:
        self.metric = metric
        self.limit = limit
        self.used = plan_used
        self.message = (
            f"Monthly {metric} limit reached ({limit:,}). "
            "Upgrade your plan at https://app.klio.tech to continue."
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


def check_memory_quota(
    db: Any, org_id: str, plan_config: PlanConfig, kind: Kind
) -> bool:
    """Check the org's quota for *kind* this month.

    Returns ``over_cap`` (True when at/over the limit but allowed via overage).
    Raises ``MemoryQuotaExceeded`` when at/over the limit and overage is not
    allowed. Fails OPEN (returns False) on any DB/unexpected error.
    """
    limit = (
        plan_config.memories_per_month
        if kind == "memory"
        else plan_config.recalls_per_month
    )
    if limit == -1:  # unlimited
        return False
    try:
        counters = get_counters(db, org_id)
    except Exception:
        logger.warning("memory quota read failed for org %s; allowing", org_id, exc_info=True)
        return False
    used = counters.memories_captured if kind == "memory" else counters.recalls
    if used >= limit:
        if plan_config.overage_allowed:
            return True
        raise MemoryQuotaExceeded(metric=kind, limit=limit, plan_used=used)
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
        logger.warning("usage record failed for org %s; ignoring", org_id, exc_info=True)
```

- [ ] **Step 4: Run the test to verify it passes**

Run (from `services/shared/`): `pytest tests/test_usage.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add services/shared/shared/usage.py services/shared/tests/test_usage.py
git commit -m "feat(usage): memory usage counters + quota check/record (shared.usage)"
```

---

## Task 4: Thread the resolved plan to the boundary

**Files:**
- Modify: `services/mcp-server/app/auth.py`
- Test: `services/mcp-server/tests/test_auth_plan.py`

- [ ] **Step 1: Write the failing test**

Create `services/mcp-server/tests/test_auth_plan.py`:
```python
"""resolve_principal_plan returns (org_id, agent_id, plan)."""

from unittest.mock import MagicMock, patch

from shared.auth import KeyInfo

from app import auth


def _request(headers: dict):
    req = MagicMock()
    req.headers.get = lambda k, default=None: headers.get(k, default)
    return req


def test_resolve_principal_plan_returns_plan():
    info = KeyInfo(org_id="org-1", key_id="k1", scopes=["memory"], plan="pro")
    with patch.object(auth, "verify_api_key", return_value=info):
        org_id, agent_id, plan = auth.resolve_principal_plan(
            _request({"X-Vex-Agent": "claude-code"})
        )
    assert (org_id, agent_id, plan) == ("org-1", "claude-code", "pro")


def test_resolve_principal_plan_defaults_agent():
    info = KeyInfo(org_id="org-1", key_id="k1", scopes=["memory"], plan="free")
    with patch.object(auth, "verify_api_key", return_value=info):
        _, agent_id, _ = auth.resolve_principal_plan(_request({}))
    assert agent_id == auth.DEFAULT_AGENT_ID
```

- [ ] **Step 2: Run it to verify it fails**

Run (from `services/mcp-server/`): `pytest tests/test_auth_plan.py -q`
Expected: FAIL — `AttributeError: module 'app.auth' has no attribute 'resolve_principal_plan'`.

- [ ] **Step 3: Add the function**

In `services/mcp-server/app/auth.py`, after `resolve_principal` (ends at line 114), add:
```python
def resolve_principal_plan(request: Request) -> tuple[str, str, str]:
    """Like :func:`resolve_principal` but also returns the resolved plan name.

    Used by the memory-metering checks, which need the plan to look up the
    per-plan memory quota. Validates the key once (no extra round trip beyond
    what :func:`resolve_principal` already does).

    Returns:
        A ``(org_id, agent_id, plan)`` tuple.
    """
    key_info = verify_api_key(request)
    agent_id = request.headers.get("X-Vex-Agent") or DEFAULT_AGENT_ID
    return key_info.org_id, agent_id, key_info.plan
```

- [ ] **Step 4: Run the test to verify it passes**

Run (from `services/mcp-server/`): `pytest tests/test_auth_plan.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add services/mcp-server/app/auth.py services/mcp-server/tests/test_auth_plan.py
git commit -m "feat(auth): resolve_principal_plan exposes plan for memory metering"
```

---

## Task 5: Enforce on the MCP write + recall tools

**Files:**
- Modify: `services/mcp-server/app/mcp_app.py`
- Test: `services/mcp-server/tests/test_memory_enforcement_mcp.py`

- [ ] **Step 1: Write the failing test**

Create `services/mcp-server/tests/test_memory_enforcement_mcp.py`:
```python
"""MCP write/recall enforce memory quota."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.usage import MemoryQuotaExceeded

from app import mcp_app


def _ctx():
    ctx = MagicMock()
    ctx.request_context.request = MagicMock()
    return ctx


@pytest.mark.asyncio
async def test_write_blocked_returns_error_dict():
    with patch.object(
        mcp_app, "resolve_principal_plan", return_value=("org-1", "a1", "free")
    ), patch.object(
        mcp_app,
        "check_memory_quota",
        side_effect=MemoryQuotaExceeded(metric="memory", limit=1000, plan_used=1000),
    ):
        result = await mcp_app._perform_write(
            _ctx(), content="x", memory_type="memory"
        )
    assert result["stored"] is False
    assert result["error"] == "quota_exceeded"
    assert "limit" in result["detail"].lower() or "upgrade" in result["detail"].lower()


@pytest.mark.asyncio
async def test_write_records_only_when_stored():
    with patch.object(
        mcp_app, "resolve_principal_plan", return_value=("org-1", "a1", "free")
    ), patch.object(mcp_app, "check_memory_quota", return_value=False), patch.object(
        mcp_app, "record_usage"
    ) as rec, patch.object(
        mcp_app.brain,
        "write_scoped_memory",
        new=AsyncMock(return_value={"stored": True, "id": "m1"}),
    ), patch.object(mcp_app, "SessionLocal", return_value=MagicMock()):
        await mcp_app._perform_write(_ctx(), content="x", memory_type="memory")
    rec.assert_called_once()
    assert rec.call_args.kwargs.get("over_cap") is False or rec.call_args[0]


@pytest.mark.asyncio
async def test_write_skips_record_when_deduped():
    with patch.object(
        mcp_app, "resolve_principal_plan", return_value=("org-1", "a1", "free")
    ), patch.object(mcp_app, "check_memory_quota", return_value=False), patch.object(
        mcp_app, "record_usage"
    ) as rec, patch.object(
        mcp_app.brain,
        "write_scoped_memory",
        new=AsyncMock(return_value={"stored": False, "deduplicated": True}),
    ), patch.object(mcp_app, "SessionLocal", return_value=MagicMock()):
        await mcp_app._perform_write(_ctx(), content="x", memory_type="memory")
    rec.assert_not_called()


@pytest.mark.asyncio
async def test_recall_blocked_returns_error_dict():
    with patch.object(
        mcp_app, "resolve_principal_plan", return_value=("org-1", "a1", "free")
    ), patch.object(
        mcp_app,
        "check_memory_quota",
        side_effect=MemoryQuotaExceeded(metric="recall", limit=10000, plan_used=10000),
    ):
        result = await mcp_app._perform_recall(
            _ctx(), query="q", scope="org", space=None, project=None
        )
    assert result["error"] == "quota_exceeded"
```

- [ ] **Step 2: Run it to verify it fails**

Run (from `services/mcp-server/`): `pytest tests/test_memory_enforcement_mcp.py -q`
Expected: FAIL — `check_memory_quota`/`record_usage`/`resolve_principal_plan` not importable in `app.mcp_app`.

- [ ] **Step 3: Add imports**

In `services/mcp-server/app/mcp_app.py`, find the existing auth import (`from app.auth import resolve_principal`) and replace it with:
```python
from app.auth import resolve_principal, resolve_principal_plan
```
Add near the other `shared`/`app` imports:
```python
from shared.usage import MemoryQuotaExceeded, check_memory_quota, record_usage
```

- [ ] **Step 4: Enforce in `_perform_write`**

In `_perform_write`, replace the principal-resolution + DB block (lines 139–162, from `try:` through the `finally: db.close()`) with:
```python
    try:
        org_id, agent_id, plan = resolve_principal_plan(request)
    except HTTPException as exc:
        return {"stored": False, "error": "unauthorized", "detail": str(exc.detail)}

    from shared.plan_limits import get_plan_config

    session_id = brain_session_id(request, org_id, agent_id)

    db = SessionLocal()
    try:
        try:
            over_cap = check_memory_quota(db, org_id, get_plan_config(plan), "memory")
        except MemoryQuotaExceeded as exc:
            return {"stored": False, "error": "quota_exceeded", "detail": exc.message}

        result = await brain.write_scoped_memory(
            db,
            org_id=org_id,
            agent_id=agent_id,
            content=content,
            memory_type=memory_type,
            session_id=session_id,
            metadata=metadata,
            scope=scope,
        )
        if result.get("stored"):
            record_usage(db, org_id, "memory", over_cap=over_cap)
        return result
    except Exception:
        logger.exception("brain write failed (memory_type=%s)", memory_type)
        return {"stored": False, "error": "write_failed"}
    finally:
        db.close()
```
(Note: the old code computed `session_id` before opening the DB; this keeps that order. Remove the now-duplicate `session_id = brain_session_id(...)` line that preceded the old `db = SessionLocal()`.)

- [ ] **Step 5: Enforce in `_perform_recall`**

In `_perform_recall`, replace the principal-resolution line (387) block:
```python
    try:
        org_id, agent_id = resolve_principal(request)
    except HTTPException as exc:
        return {"error": "unauthorized", "detail": str(exc.detail)}
```
with:
```python
    try:
        org_id, agent_id, plan = resolve_principal_plan(request)
    except HTTPException as exc:
        return {"error": "unauthorized", "detail": str(exc.detail)}
```
Then, immediately after `db = SessionLocal()` and inside its `try:` (before the space-slug resolution), add:
```python
        from shared.plan_limits import get_plan_config

        try:
            recall_over_cap = check_memory_quota(
                db, org_id, get_plan_config(plan), "recall"
            )
        except MemoryQuotaExceeded as exc:
            return {"error": "quota_exceeded", "detail": exc.message}
```
And immediately after the existing `log_recall_event(...)` call (line 432–440), add:
```python
        record_usage(db, org_id, "recall", over_cap=recall_over_cap)
```

- [ ] **Step 6: Run the test to verify it passes**

Run (from `services/mcp-server/`): `pytest tests/test_memory_enforcement_mcp.py -q`
Expected: PASS (4 tests).

- [ ] **Step 7: Run the existing MCP tool tests (regression)**

Run: `pytest tests/test_write_tools.py tests/test_recall_tool.py -q`
Expected: PASS (existing behavior preserved; quota checks default-allow under mocks).

- [ ] **Step 8: Commit**

```bash
git add services/mcp-server/app/mcp_app.py services/mcp-server/tests/test_memory_enforcement_mcp.py
git commit -m "feat(mcp): enforce memory quota on write/recall tools"
```

---

## Task 6: Enforce on the REST capture endpoints

**Files:**
- Modify: `services/mcp-server/app/capture.py`
- Test: `services/mcp-server/tests/test_memory_enforcement_capture.py`

- [ ] **Step 1: Write the failing test**

Create `services/mcp-server/tests/test_memory_enforcement_capture.py`:
```python
"""REST capture endpoints enforce memory quota → HTTP 402."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from shared.usage import MemoryQuotaExceeded

from app import capture


@pytest.mark.asyncio
async def test_capture_event_quota_exceeded_raises_402():
    req = MagicMock()
    req.memory_type = "observation"
    req.metadata = None
    info = MagicMock(plan="free")
    with patch.object(
        capture, "check_memory_quota",
        side_effect=MemoryQuotaExceeded(metric="memory", limit=1000, plan_used=1000),
    ), patch.object(capture, "SessionLocal", return_value=MagicMock()), patch.object(
        capture, "_ensure_project_id", return_value=None
    ):
        with pytest.raises(HTTPException) as exc:
            await capture.capture_event(req, principal=("org-1", "a1"), key_info=info)
    assert exc.value.status_code == 402


@pytest.mark.asyncio
async def test_capture_event_records_when_stored():
    req = MagicMock()
    req.memory_type = "observation"
    req.metadata = None
    info = MagicMock(plan="free")
    with patch.object(capture, "check_memory_quota", return_value=False), patch.object(
        capture, "record_usage"
    ) as rec, patch.object(
        capture.brain, "write_scoped_memory",
        new=AsyncMock(return_value={"stored": True, "id": "m1"}),
    ), patch.object(capture, "SessionLocal", return_value=MagicMock()), patch.object(
        capture, "_ensure_project_id", return_value=None
    ):
        await capture.capture_event(req, principal=("org-1", "a1"), key_info=info)
    rec.assert_called_once()
```

- [ ] **Step 2: Run it to verify it fails**

Run (from `services/mcp-server/`): `pytest tests/test_memory_enforcement_capture.py -q`
Expected: FAIL — `capture_event` has no `key_info` param / `check_memory_quota` not importable.

- [ ] **Step 3: Add imports + a shared helper**

In `services/mcp-server/app/capture.py`, add to the imports:
```python
from fastapi import HTTPException
from shared.auth import KeyInfo
from shared.plan_limits import get_plan_config
from shared.usage import MemoryQuotaExceeded, check_memory_quota, record_usage
from app.auth import verify_api_key
```
(Some of `fastapi`/`shared.auth` symbols may already be imported — merge, don't duplicate.)

Add a small helper near the top of the routes section (after the `router = APIRouter(...)` line):
```python
def _check_memory_quota_402(db, org_id: str, plan: str, kind: str) -> bool:
    """Run the memory quota check, translating a hard block to HTTP 402.

    Returns ``over_cap``. Synchronous; call via anyio.to_thread from handlers.
    """
    try:
        return check_memory_quota(db, org_id, get_plan_config(plan), kind)
    except MemoryQuotaExceeded as exc:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "quota_exceeded",
                "metric": exc.metric,
                "limit": exc.limit,
                "message": exc.message,
            },
        ) from exc
```

- [ ] **Step 4: Enforce in `/capture/event`**

Change the `capture_event` signature to also receive the validated key:
```python
@router.post("/event")
async def capture_event(
    req: CaptureEventRequest,
    principal: tuple[str, str] = Depends(resolve_principal_dep),
    key_info: KeyInfo = Depends(verify_api_key),
) -> dict[str, Any]:
```
Inside the `try:`, after `project_id = await anyio.to_thread.run_sync(...)` and before `return await brain.write_scoped_memory(...)`, replace the bare `return await brain.write_scoped_memory(...)` with:
```python
        over_cap = await anyio.to_thread.run_sync(
            lambda: _check_memory_quota_402(db, org_id, key_info.plan, "memory")
        )
        result = await brain.write_scoped_memory(
            db,
            org_id=org_id,
            agent_id=agent_id,
            content=req.content,
            memory_type=memory_type,
            session_id=_capture_session_id(org_id, agent_id, req.session_id),
            project_id=project_id,
            metadata=metadata,
            scope=req.scope,
        )
        if result.get("stored"):
            await anyio.to_thread.run_sync(
                lambda: record_usage(db, org_id, "memory", over_cap=over_cap)
            )
        return result
```
IMPORTANT: the existing `except Exception:` handler returns `{"stored": False, "error": "capture_failed"}`, which would swallow the 402. Re-raise `HTTPException` before that generic catch — change the handler to:
```python
    except HTTPException:
        raise
    except Exception:
        logger.exception("capture event failed (memory_type=%s)", memory_type)
        return {"stored": False, "error": "capture_failed"}
```

- [ ] **Step 5: Enforce in `/capture/transcript`**

Add the same `key_info: KeyInfo = Depends(verify_api_key)` param. After `session_id = ...` / before the summary block, add a single entry check + record per stored row. Insert after `stored = 0`:
```python
        over_cap = await anyio.to_thread.run_sync(
            lambda: _check_memory_quota_402(db, org_id, key_info.plan, "memory")
        )
```
After each `if result.get("stored"): stored += 1`, add the record (two places — summary and the fact loop):
```python
            if result.get("stored"):
                stored += 1
                await anyio.to_thread.run_sync(
                    lambda: record_usage(db, org_id, "memory", over_cap=over_cap)
                )
```
And add the `except HTTPException: raise` ahead of the generic `except Exception:` (same as Step 4).

- [ ] **Step 6: Enforce in `/capture/recall`**

Add the `key_info: KeyInfo = Depends(verify_api_key)` param. Inside the `try:`, right after `query = req.query.strip()`... actually before the recency/vector branch, add:
```python
        recall_over_cap = await anyio.to_thread.run_sync(
            lambda: _check_memory_quota_402(db, org_id, key_info.plan, "recall")
        )
```
After the existing `log_recall_event(...)` `anyio.to_thread.run_sync` block, add:
```python
        await anyio.to_thread.run_sync(
            lambda: record_usage(db, org_id, "recall", over_cap=recall_over_cap)
        )
```
And add `except HTTPException: raise` ahead of the generic `except Exception:` handler.

- [ ] **Step 7: Run the test to verify it passes**

Run (from `services/mcp-server/`): `pytest tests/test_memory_enforcement_capture.py -q`
Expected: PASS (2 tests).

- [ ] **Step 8: Regression — existing capture tests**

Run: `pytest tests/test_capture.py -q`
Expected: PASS. (If `test_capture.py` calls the handlers directly, it must pass a `key_info` mock; update those call sites to add `key_info=MagicMock(plan="free")` — do so if they fail.)

- [ ] **Step 9: Commit**

```bash
git add services/mcp-server/app/capture.py services/mcp-server/tests/test_memory_enforcement_capture.py
git commit -m "feat(capture): enforce memory quota on REST capture endpoints (402)"
```

---

## Task 7: Memory retention sweep

**Files:**
- Create: `services/migrations/alembic/versions/025_enforce_memory_retention.py`
- Modify: `services/scripts/run_retention.py`
- Test: `services/shared/tests/test_run_memory_retention.py` (logic-level)

- [ ] **Step 1: Write the migration (SQL function)**

Create `services/migrations/alembic/versions/025_enforce_memory_retention.py`:
```python
"""Klio memory retention: per-org session_memories pruning function.

ADDITIVE ONLY. Adds an ``enforce_memory_retention(org_id, days)`` SQL function
that deletes ``session_memories`` rows older than the plan's window. Mirrors
``enforce_plan_retention`` (migration 008) but targets the memory table. The
retention cron (run_retention.py) calls it per org.

Revision ID: 025
Revises: 024
Create Date: 2026-06-01
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

revision: str = "025"
down_revision: Union[str, None] = "024"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION enforce_memory_retention(
            p_org_id TEXT, p_retention_days INT
        ) RETURNS void AS $$
        BEGIN
            DELETE FROM session_memories
            WHERE org_id = p_org_id
              AND created_at < NOW() - (p_retention_days || ' days')::interval;
        END;
        $$ LANGUAGE plpgsql;
        """
    )


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS enforce_memory_retention(TEXT, INT);")
```

- [ ] **Step 2: Apply + verify the function**

Run (from `services/migrations/`):
```bash
DATABASE_URL="postgresql://agentguard:agentguard_dev@localhost:5432/agentguard" alembic upgrade head
psql "$DATABASE_URL" -c "SELECT proname FROM pg_proc WHERE proname='enforce_memory_retention';"
```
Expected: applies `025`; the SELECT returns one row.

- [ ] **Step 3: Write the failing test for the script extension**

Create `services/shared/tests/test_run_memory_retention.py`:
```python
"""run_retention also enforces memory retention per org."""

import importlib
import sys
from unittest.mock import MagicMock, patch


def _load_run_retention():
    # The script lives in services/scripts; import it by path.
    import os

    sys.path.insert(
        0,
        os.path.join(os.path.dirname(__file__), "..", "..", "scripts"),
    )
    return importlib.import_module("run_retention")


def test_calls_enforce_memory_retention_with_memory_days():
    mod = _load_run_retention()
    calls = []

    class FakeConn:
        def execute(self, stmt, params=None):
            text_sql = getattr(stmt, "text", str(stmt))
            if params:
                calls.append((text_sql, params))
            result = MagicMock()
            result.fetchall.return_value = [("acct", "pro", None)]
            result.fetchone.return_value = ("org-1",)
            return result

        def commit(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake_engine = MagicMock()
    fake_engine.connect.return_value = FakeConn()

    with patch.dict("os.environ", {"DATABASE_URL": "x", "SUPABASE_DATABASE_URL": "y"}), \
         patch.object(mod, "create_engine", return_value=fake_engine):
        mod.main()

    # pro plan → memory_retention_days == 30 must be passed to the memory function
    assert any(
        "enforce_memory_retention" in sql and params.get("days") == 30
        for sql, params in calls
    )
```

- [ ] **Step 4: Run it to verify it fails**

Run (from `services/shared/`): `pytest tests/test_run_memory_retention.py -q`
Expected: FAIL — no `enforce_memory_retention` call is made.

- [ ] **Step 5: Extend `run_retention.py`**

In `services/scripts/run_retention.py`, inside the `for row in accounts:` loop, after the existing `enforce_plan_retention` call + its `logger.info(...)` (after line 95), add:
```python
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
```

- [ ] **Step 6: Run the test to verify it passes**

Run (from `services/shared/`): `pytest tests/test_run_memory_retention.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add services/migrations/alembic/versions/025_enforce_memory_retention.py services/scripts/run_retention.py services/shared/tests/test_run_memory_retention.py
git commit -m "feat(retention): prune session_memories per plan window (migration 025 + cron)"
```

---

## Task 8: Sync note + full verification

**Files:**
- Modify: `services/shared/shared/plan_limits.py` (cross-ref comment only)

- [ ] **Step 1: Add the marketing-sync cross-reference**

At the top of `services/shared/shared/plan_limits.py`, in the module docstring, add a line:
```
The memory-tier fields (memories/recalls/retention/sync) are mirrored on the
marketing site at Dashboard/apps/landing/lib/pricing.ts — keep the two in sync.
```
(Also add the reverse comment in `lib/pricing.ts` — separate repo, tracked as a follow-up; note in the commit.)

- [ ] **Step 2: Run the full shared suite + ruff**

Run (from `services/shared/`): `pytest -q`
Expected: PASS (all, including existing reliability/brain/recall/curator tests — regression guard).

Run (from `services/mcp-server/`): `pytest -q`
Expected: PASS.

Run (from repo root): `ruff check services/shared services/mcp-server services/migrations services/scripts`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add services/shared/shared/plan_limits.py
git commit -m "docs(plans): cross-reference marketing pricing mirror"
```

---

## Self-Review

**Spec coverage:**
- Unit A (PlanConfig memory fields + overage align) → Task 1. ✓
- Unit B (counter table + usage.py + MemoryQuotaExceeded) → Tasks 2, 3. ✓
- Unit C (enforce write/recall at MCP + REST, count non-deduped only, 402/tool-error, fail-open) → Tasks 4, 5, 6. ✓
- Unit D (memory retention sweep) → Task 7. ✓
- Marketing↔engine sync note → Task 8. ✓
- Deferred items (Stripe, B2B2C, UI, real-time sync) — correctly NOT in this plan. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. The one cross-repo follow-up (lib/pricing.ts reverse comment) is explicitly flagged, not a silent gap.

**Type consistency:** `check_memory_quota(db, org_id, plan_config, kind) -> bool` and `record_usage(db, org_id, kind, *, over_cap)` and `MemoryQuotaExceeded(metric, limit, plan_used)` are used identically in Tasks 3, 5, 6. `resolve_principal_plan -> (org_id, agent_id, plan)` (Task 4) is consumed in Task 5. `get_plan_config(plan)` returns a `PlanConfig` with the new fields (Task 1) read by `check_memory_quota` (Task 3). `Kind ∈ {"memory","recall"}` consistent throughout.
