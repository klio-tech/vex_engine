# Klio Memory Metering, Entitlements & Enforcement — Design

**Date:** 2026-06-01
**Status:** Approved direction; design for review → implementation plan.
**Repo:** `vex_engine` (this repo). Branch `feat/klio-memory-metering`.
**Pricing source of truth (marketing):** `Dashboard/apps/landing/lib/pricing.ts`.

## Goal

Make the Klio landing's **memory pricing** real in the engine: meter the use of
the shared memory (memories captured + recalls per month), enforce per-plan caps
and retention, and gate the cross-agent-sync flag — by **extending the
entitlements machinery that already exists**, not building billing from scratch.

This spec covers **sub-project 1 of 4** (the engine foundation). It produces
working, testable software on its own: the engine meters memory usage and
hard-caps it per plan, with overage recorded for later billing.

## What already exists (extend, don't rebuild)

- **`services/shared/shared/plan_limits.py`** — `PlanConfig` (frozen dataclass) +
  `PLAN_LIMITS` dict (`free`/`starter`/`pro`/`team`/`enterprise`) +
  `get_plan_config(plan, overrides)`. Today it carries *reliability* limits
  (`observations_per_month`, `verifications_per_month`, `corrections_per_month`,
  `max_rpm`, `max_agents`, `retention_days`, `overage_allowed`, feature flags).
- **`services/shared/shared/auth.py`** — `KeyValidator.validate(api_key, agent_id)`
  already enforces, on every request: rate limit (`_check_rate_limit`), monthly
  quota (`_check_quota`, counted from `executions`), and agent cap
  (`_check_agent_limit`). Resolves `plan` from the Supabase `accounts` table
  (`vex_plan`, `vex_plan_overrides`). Returns `KeyInfo(org_id, key_id, scopes, plan)`.
  Raises `AuthError(status, message, retry_after_seconds)`.
- **Recall logging** — `brain_recall_events` table (migration `022`) +
  `services/mcp-server/app/recall_log.py::log_recall_event(...)`, called from the
  MCP `recall` tool (`_perform_recall`) and `POST /capture/recall`. Append-only;
  the basis for counting recalls.
- **Write path** — `shared/memory.py::write_memory(...)` (raw INSERT into
  `session_memories`) wrapped by `shared/brain.py::write_scoped_memory(...)`
  (dedup via `find_similar_memory` → write → promote scope). Returns
  `{"stored": bool, "deduplicated": bool, "id": ..., "scope": ...}`. Call sites:
  MCP write tools (`mcp_app.py::_perform_write`) and `POST /capture/event` /
  `POST /capture/transcript`.
- **Retention** — `enforce_plan_retention(org_id, days)` SQL function (migration
  `008`) + `services/scripts/run_retention.py` cron. **Prunes `executions` only —
  not `session_memories`.** (The gap this spec fills for memory.)
- **Background jobs** — curator runs as an in-process lifespan loop
  (`services/mcp-server/app/scheduler.py::_curator_loop` started from
  `main.py` lifespan). Retention runs as an external cron script.
- **DB access** — raw SQL via `sqlalchemy.text()`; `SessionLocal` in
  `services/mcp-server/app/db.py`; sync calls wrapped in
  `anyio.to_thread.run_sync(...)` from async handlers. **No ORM models.**
- **Latest migration:** `023`. Next is **`024`**.

## Decisions (confirmed)

1. **Additive alongside reliability.** Same account, same tiers. The engine
   enforces memory quotas *in addition to* the existing reliability quotas. No
   reliability enforcement is removed.
2. **Real-time cross-agent sync is descoped from this spec.** Memory is already
   cross-agent (every agent reads the same org memory in Postgres); real-time
   *push* via Redis pub/sub is **not built** and becomes a later sub-project. This
   spec ships the `memory_sync_enabled` *flag* only.
3. **Counter table for metering** (not live `COUNT`). Live-counting
   `session_memories` is wrong because retention deletes rows; recalls could be
   counted from `brain_recall_events` but we keep one authoritative source for
   enforcement.

## Architecture

Three units, each independently testable:

### Unit A — Plan limits (config)
Extend `PlanConfig` (`shared/plan_limits.py`) with memory fields, **with defaults**
so existing constructors keep working:

```python
memories_per_month: int = -1     # -1 = unlimited, 0 = disabled
recalls_per_month: int = -1
memory_retention_days: int = 30
memory_sync_enabled: bool = False  # flag only; feature deferred
```

Set them in `PLAN_LIMITS` to **mirror the landing** (`lib/pricing.ts`):

| plan | memories/mo | recalls/mo | retention | sync |
|---|---|---|---|---|
| free | 1,000 | 10,000 | 1 day | off |
| starter | 25,000 | 100,000 | 7 days | on |
| pro | 150,000 | 1,000,000 | 30 days | on |
| team | 1,500,000 | 10,000,000 | 90 days | on |
| enterprise | -1 | -1 | 365 days | on |

Also align the existing `overage_allowed` flag to the landing's "Overage" rows so
`check_memory_quota` behaves correctly: **free=False, starter=False, pro=True,
team=True** (verify current values; fix any that differ). `max_agents` already
`-1` (unlimited) on every tier — matches "Connected agents: Unlimited"; no change.

`PLAN_LIMITS` is the **enforcement source of truth**; `lib/pricing.ts` mirrors it
(documented manual sync — a comment in both files pointing at the other).

### Unit B — Usage metering + quota (`shared/usage.py`, new)
A small, dependency-light module (raw SQL, mirrors `recall_log.py` style):

- **Counter table** (migration `024`): `memory_usage_counters`
  - `org_id VARCHAR(64) NOT NULL`
  - `period_month DATE NOT NULL` (= `date_trunc('month', now())::date`)
  - `memories_captured BIGINT NOT NULL DEFAULT 0`
  - `recalls BIGINT NOT NULL DEFAULT 0`
  - `memories_overage BIGINT NOT NULL DEFAULT 0`
  - `recalls_overage BIGINT NOT NULL DEFAULT 0`
  - `updated_at TIMESTAMPTZ NOT NULL DEFAULT now()`
  - `PRIMARY KEY (org_id, period_month)`
- **Functions:**
  - `current_period() -> date`
  - `get_counters(db, org_id) -> Counters` — single PK read (zeros if absent).
  - `check_memory_quota(db, org_id, kind, plan_config) -> None` — reads the
    counter; if the relevant count `>=` the plan limit (and limit `!= -1`):
    - `overage_allowed` (pro/team) → allow, mark `over_cap=True`.
    - else (free/starter) → raise `MemoryQuotaExceeded(metric, limit, suggested_plan)`.
  - `record_usage(db, org_id, kind, *, over_cap: bool) -> None` — atomic UPSERT:
    `INSERT ... ON CONFLICT (org_id, period_month) DO UPDATE SET memories_captured = memories_captured + 1 [, memories_overage = + 1 if over_cap]`.
  - `kind ∈ {"memory", "recall"}`.
- **Exception:** `MemoryQuotaExceeded(Exception)` with `.metric`, `.limit`,
  `.suggested_plan`, `.message` — defined here, mapped to transport errors by the
  boundary (Unit C). No reads are cached (single indexed PK lookup is cheap);
  accept negligible over-allow under burst.

### Unit C — Enforcement at the boundary
Wire check-before / record-after into the two operations, at both entry styles.
The resolved **plan** is already available (`KeyInfo.plan` from `verify_api_key`);
thread it to the checks. (If `resolve_principal` doesn't currently expose the
plan, extend it to return it — small, local change.)

- **Write** (`mcp_app.py::_perform_write`, `capture.py::/capture/event` and
  `/capture/transcript`): `check_memory_quota(kind="memory")` before writing;
  call `record_usage(kind="memory", over_cap=…)` **only when
  `write_scoped_memory(...)` returns `stored=True`** (deduped writes are not
  billable). For transcript (N memories), check once at entry; record per stored
  row.
- **Recall** (`mcp_app.py::_perform_recall`, `capture.py::/capture/recall`):
  `check_memory_quota(kind="recall")` before retrieving; `record_usage(kind="recall")`
  after, alongside the existing `log_recall_event(...)`.
- **Transport mapping of `MemoryQuotaExceeded`:** REST capture → `HTTPException(402,
  detail={message, metric, suggested_plan})`; MCP tools → a clear error result
  string ("You've hit your monthly {metric} limit on the {plan} plan — upgrade at
  app.klio.tech"). MCP has no HTTP status, so a guiding text error matches how
  existing tools surface failures.
- **Rate limit (RPM)** needs no new work: memory requests already pass through
  `KeyValidator` via `verify_api_key`, which enforces `max_rpm`.

### Unit D — Memory retention sweep
Extend retention to memory. Add `enforce_memory_retention(org_id, days)` (mirror
the `008` SQL function) in migration `024`, and extend
`services/scripts/run_retention.py` to also call it per org using
`plan_config.memory_retention_days` — deleting `session_memories` rows where
`created_at < now() - (days || ' days')::interval`. One cron, both sweeps.

## Data flow

```
write:  client → verify_api_key (KeyValidator: rate-limit, plan) →
        check_memory_quota("memory", plan) → write_scoped_memory →
        if stored: record_usage("memory")            [+ existing dedup logic]

recall: client → verify_api_key → check_memory_quota("recall", plan) →
        retrieve_memories_scoped → record_usage("recall") + log_recall_event

retention (cron): per team account → resolve plan → memory_retention_days →
        enforce_memory_retention(org_id, days)   [+ existing executions sweep]
```

## Error handling
- Quota over cap (free/starter): hard-block — 402 (REST) / guiding error (MCP).
- Pro/team over cap: allow, increment `*_overage` (for later Stripe reporting).
- Counter UPSERT is atomic (`ON CONFLICT DO UPDATE`); safe under concurrency.
- Metering must never silently corrupt the user operation: `record_usage` failures
  are logged and swallowed *after* a successful write/recall (mirror
  `log_recall_event`'s swallow-and-log), so a counter glitch never loses a memory.
  `check_memory_quota` failures (DB error) **fail open** (allow the op, log) — we
  never block a paying user's memory on a metering hiccup.
- Month rollover handled by `period_month` in the PK (new row per month).

## Testing
Follow the existing mock-based pattern (`services/shared/tests`, pytest with
`pythonpath=["."]`, `asyncio_mode=auto`; run with `PYTHONPATH=services/shared`):
- `usage.py`: counter UPSERT increments; `check_memory_quota` allows under cap,
  raises at cap for free/starter, allows + flags overage for pro/team; unlimited
  (`-1`) never blocks; period boundary creates a new row. (mock DB)
- `plan_limits.py`: memory fields present + defaults; values match the table.
- Enforcement: `_perform_write` / `_perform_recall` and capture endpoints call
  check + record; deduped write does **not** record; quota exceeded surfaces 402 /
  tool error. (mock the quota helper)
- Retention: `enforce_memory_retention` deletes only rows older than the window
  for the given org.
- Regression guard: existing reliability quota/recall/dedup/curator tests stay green.
- `ruff` clean.

## Risks & mitigations
- **Double counting** (transcript writes N rows) → count per `stored=True` row only.
- **Dedup** → never count deduped writes.
- **Retention deleting wanted memory** → it's the plan's stated window (matches the
  landing); deletes by `created_at` regardless of type. *Open refinement:* consider
  exempting curator-derived `summary`/`fact` rows — deferred, noted not decided.
- **Marketing/engine drift** → cross-reference comments in `plan_limits.py` ↔
  `lib/pricing.ts`; the recall caps are a *starting point* to revisit with real
  usage data (instrument first, finalize later).

## Out of scope (later sub-projects)
1. **Stripe billing** — products/prices for the 4 tiers + metered overage; checkout;
   webhooks → entitlements; usage reporting from the `*_overage` counters.
2. **B2B2C per-end-user** — count distinct active end-users per embedding org; bill
   per end-user (likely manual invoicing first).
3. **Usage UI** (app.klio.tech) — plan, usage-vs-caps, upgrade/downgrade, invoices.
4. **Real-time cross-agent sync feature** — the Redis pub/sub push behind the
   `memory_sync_enabled` flag.
