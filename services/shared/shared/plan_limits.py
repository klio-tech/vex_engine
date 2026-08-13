"""Plan-level limits for Vex and Klio pricing tiers.

These constants define the quotas, rate limits, and feature flags for each plan.
The ``PLAN_LIMITS`` dict is keyed by the ``plan`` column value in the
``organizations`` table.

**One ladder, two products.** An org has a single ``plan`` value, and each tier
carries both Vex's fields (observations, verifications, corrections, alerts) and
Klio's memory fields. They are not separate ladders and must not be split into
two: the plan column is what billing writes, and there is one of it.

* Vex is sold on ``starter`` / ``pro`` / ``team`` and is currently hidden from
  the marketing site rather than withdrawn. Its fields stay exactly as they are.
* Klio is sold on ``free`` (one person) and ``team`` ($20 per seat), with
  Platform handled as a sales conversation rather than a self-serve tier.

The memory-tier fields (memories_per_month / recalls_per_month /
memory_retention_days / memory_sync_enabled / artifact_*) are mirrored on the
marketing site at Dashboard/apps/landing/lib/pricing.ts — keep the two in sync.
That mirror had drifted: the site promised Free unlimited memories and retention
forever while this file gave 1,000 per month and a one-day window, so anyone who
signed up on the published terms lost their history after a day.

``-1`` means unlimited, for quotas and for retention. For retention that is only
safe because two guards exist: ``run_retention.py`` skips the sweep for a
non-positive value, and migration 038 makes both SQL functions return early.
Without those, ``-1`` reads as ``NOW() + 1 day`` and deletes the org's entire
history. Do not remove either guard while any tier declares -1.

``memory_sync_enabled`` currently gates nothing in code. It is declarative, so
it exists to keep this file honest against the published table.

**2026-08-13 repricing.** ``free.memory_retention_days`` moved from ``-1`` to
``30`` — a positive value, which is new territory: this is the first tier to
ever activate the retention sweep for real, rather than relying on the -1
guards above to make it a no-op. The guards protect ``-1``; they do nothing
for ``30``, so a real free-tier org will really lose memories older than 30
days the first time ``run_retention.py`` runs against it. Existing orgs are
grandfathered onto ``memory_retention_days: -1`` via ``vex_plan_overrides``
(a data operation, done separately from this file) so nobody's existing
history is deleted by this change landing. ``enterprise.memory_retention_days``
also moved, from ``365`` to ``-1``: it was previously *worse* than free and
team (both sold as forever) despite being the top tier — a bug, not a choice.
``team.max_rpm`` moved from ``5_000`` to ``1_000`` to match what the site has
always advertised. ``max_projects`` / ``max_users`` / ``extractions_per_month``
/ ``intelligence_enabled`` are new fields, added additively with defaults so
existing direct ``PlanConfig(...)`` constructors elsewhere in the codebase
keep building unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from typing import Any

logger = logging.getLogger("agentguard.shared.plan_limits")


@dataclass(frozen=True)
class PlanConfig:
    """Immutable configuration for a single pricing plan."""

    # Monthly quotas
    # ──────────────
    # Observations: async fire-and-forget events sent via POST /v1/ingest
    # or the SDK's async mode. These are telemetry records of agent executions
    # (input, output, latency, steps) stored for dashboards and analytics.
    # No LLM scoring is performed — just data collection.
    observations_per_month: int

    # Verifications: synchronous POST /v1/verify calls where the output is
    # scored in real-time by the verification engine (schema validation,
    # hallucination detection, drift scoring, coherence checking). Each call
    # invokes 2-4 LLM checks and returns a confidence score + pass/flag/block
    # action. Significantly more expensive than observations due to LLM costs.
    verifications_per_month: int

    # Corrections: monthly quota for auto-correction cascade invocations.
    # When a verification fails and correction=cascade is requested, this
    # counter is decremented. -1 means unlimited (full cascade available).
    # 0 means corrections are not available on this plan.
    corrections_per_month: int  # -1 = unlimited (full cascade)

    # Rate limit
    # ──────────
    # Maximum requests per minute across all endpoints for this org.
    # Applied as a sliding 60-second window. If a per-key RPM is also set,
    # the effective limit is min(per_key_rpm, plan_max_rpm).
    # Exceeding this returns HTTP 429 with a Retry-After header.
    max_rpm: int

    # Resource limits
    # ───────────────
    # Max agents: the number of distinct agent_id values this org can use.
    # Each unique agent_id seen in ingest/verify requests counts toward this
    # limit. -1 means unlimited. Exceeding this returns HTTP 403.
    max_agents: int

    # Feature flags
    # ─────────────
    # Corrections: when enabled, the verification engine can auto-correct
    # failing outputs using a 3-layer cascade (L1 Repair → L2 Constrained
    # Regen → L3 Full Reprompt). Each layer uses progressively stronger LLM
    # models. When disabled, correction=cascade requests are silently skipped
    # and the response includes correction_skipped=true with
    # correction_skipped_reason="upgrade_required".
    corrections_enabled: bool

    # Webhook alerts: when enabled, the alert service can deliver
    # notifications to user-configured HTTP webhook endpoints for events
    # like block actions, confidence drops, or anomaly detection.
    webhook_alerts: bool

    # Slack alerts: when enabled, the alert service can post notifications
    # directly to Slack channels via incoming webhooks for real-time
    # team visibility into agent health issues.
    slack_alerts: bool

    # Data retention
    # ──────────────
    # Number of days execution records, check results, and correction
    # attempts are retained in the database before automatic deletion.
    # After this period, data is permanently purged. Longer retention
    # enables deeper historical analysis and trend tracking.
    retention_days: int

    # Overage handling
    # ────────────────
    # When True: requests beyond the monthly quota are allowed to proceed
    # and the excess usage is billed at the per-unit overage rate.
    # When False: requests beyond the monthly quota are hard-rejected with
    # HTTP 429 and a message to upgrade. No overage charges are incurred.
    overage_allowed: bool

    # Experiments
    # ───────────
    # A/B testing experiments allow comparing different agent configurations
    # (prompts, models, parameters) against evaluation datasets to measure
    # quality and performance. -1 = unlimited.
    max_experiments: int = 0
    # Datasets are reusable collections of test cases used to evaluate
    # experiment variants. Each dataset contains input/expected-output pairs.
    max_datasets: int = 0
    # Maximum number of items (test cases) allowed per dataset.
    max_dataset_items: int = 0

    # Memory (Klio) — value metric for the memory product. Mirror at
    # Dashboard/apps/landing/lib/pricing.ts (keep in sync). -1 = unlimited,
    # 0 = disabled.
    memories_per_month: int = -1
    recalls_per_month: int = -1
    memory_retention_days: int = 30
    memory_sync_enabled: bool = False

    # Artifacts (Klio) — durable files, documents and records. Capacity is the
    # paid axis; the feature is available on every plan, because handing work
    # between agents is the wedge (same reason max_agents is -1 everywhere).
    # 0 disables artifacts entirely. Mirror at
    # Dashboard/apps/landing/lib/pricing.ts (keep in sync).
    artifact_max_file_bytes: int = 10 * 1024 * 1024
    artifact_storage_bytes: int = 100 * 1024 * 1024

    # Klio pricing (2026-08-13) — added additively, with defaults, so the
    # existing direct `PlanConfig(...)` constructors in ingestion-api and
    # sync-gateway (which predate these fields) keep building without being
    # touched. Mirror at Dashboard/apps/landing/lib/pricing.ts (keep in sync).

    # Max projects: the number of distinct projects an org may create.
    # -1 means unlimited. "Just you" vs "everyone you invite" is enforced by
    # max_users below; this is the secondary lever — real developers work
    # across many repos, so a low ceiling is confining for sustained use
    # without blocking evaluation.
    max_projects: int = -1

    # Max users: the number of distinct members an org may have. -1 means
    # unlimited. Before this field existed nothing in the engine limited org
    # membership at all, so "Just you" was decoration — a ten-person team
    # could sit on Free forever. This is the field that makes the seat model
    # real.
    max_users: int = -1

    # Extractions per month: a fair-use ceiling on LLM extraction runs,
    # pooled across the org (per-seat allowances multiply into the pool, they
    # are not enforced per user). -1 means unlimited. This bounds a runaway
    # account, not typical usage — sized at several multiples of a typical
    # user's monthly writes. Past the ceiling, degrade: keep storing the
    # memory verbatim and skip LLM extraction on it. Never refuse the write;
    # a write that fails is a broken product, a write that stores plainly
    # without enrichment is a nudge to upgrade.
    extractions_per_month: int = -1

    # Intelligence layer: gates hybrid lexical search, the knowledge graph,
    # the curator, and compression. When False, an org gets plain vector
    # store-and-retrieve only — no reasoning over what was stored. This is
    # the paid value story: the intelligence layer is also the actual COGS,
    # so it is the honest place to draw the free/paid line.
    intelligence_enabled: bool = True


PLAN_LIMITS: dict[str, PlanConfig] = {
    "free": PlanConfig(
        observations_per_month=1_000,
        verifications_per_month=50,
        corrections_per_month=0,
        max_rpm=100,
        max_agents=-1,
        corrections_enabled=False,
        webhook_alerts=False,
        slack_alerts=True,
        retention_days=1,
        overage_allowed=False,
        max_experiments=1,
        max_datasets=1,
        max_dataset_items=20,
        # Klio Free (2026-08-13 repricing): unlimited memories, but a real
        # 30-day retention window instead of forever — this is the primary
        # upgrade lever and the first time the retention sweep runs for real
        # on this tier. See the module docstring's -1 guard note before ever
        # changing this back to a non-positive value casually.
        memories_per_month=-1,
        recalls_per_month=-1,
        memory_retention_days=30,
        memory_sync_enabled=True,
        artifact_max_file_bytes=10 * 1024 * 1024,
        artifact_storage_bytes=100 * 1024 * 1024,
        max_projects=3,
        max_users=1,
        extractions_per_month=3_000,
        intelligence_enabled=False,
    ),
    "starter": PlanConfig(
        observations_per_month=25_000,
        verifications_per_month=1_000,
        corrections_per_month=100,
        max_rpm=500,
        max_agents=-1,
        corrections_enabled=True,
        webhook_alerts=False,
        slack_alerts=True,
        retention_days=7,
        overage_allowed=False,
        max_experiments=5,
        max_datasets=5,
        max_dataset_items=100,
        memories_per_month=25_000,
        recalls_per_month=100_000,
        memory_retention_days=7,
        memory_sync_enabled=True,
        artifact_max_file_bytes=50 * 1024 * 1024,
        artifact_storage_bytes=5 * 1024 * 1024 * 1024,
    ),
    "pro": PlanConfig(
        observations_per_month=150_000,
        verifications_per_month=15_000,
        corrections_per_month=-1,
        max_rpm=1_000,
        max_agents=-1,
        corrections_enabled=True,
        webhook_alerts=True,
        slack_alerts=True,
        retention_days=30,
        overage_allowed=True,
        max_experiments=-1,
        max_datasets=25,
        max_dataset_items=500,
        memories_per_month=150_000,
        recalls_per_month=1_000_000,
        memory_retention_days=30,
        memory_sync_enabled=True,
        artifact_max_file_bytes=50 * 1024 * 1024,
        artifact_storage_bytes=5 * 1024 * 1024 * 1024,
    ),
    "team": PlanConfig(
        observations_per_month=1_500_000,
        verifications_per_month=150_000,
        corrections_per_month=-1,
        # 1,000 RPM (2026-08-13 repricing), down from 5,000. This is a fair-
        # use ceiling, not a Vex regression: it was previously more generous
        # than published, and the site (apps/landing/lib/pricing.ts) has
        # always said 1,000 — this closes that drift rather than opening one.
        max_rpm=1_000,
        max_agents=-1,
        corrections_enabled=True,
        webhook_alerts=True,
        slack_alerts=True,
        retention_days=90,
        overage_allowed=True,
        max_experiments=-1,
        max_datasets=-1,
        max_dataset_items=-1,
        # Klio Team, as published: unlimited memories, retention forever,
        # 5 GB artifact pool. Vex's fields on this tier are untouched.
        memories_per_month=-1,
        recalls_per_month=-1,
        memory_retention_days=-1,
        memory_sync_enabled=True,
        artifact_max_file_bytes=50 * 1024 * 1024,
        artifact_storage_bytes=5 * 1024 * 1024 * 1024,
        max_projects=-1,
        max_users=-1,
        extractions_per_month=10_000,
        intelligence_enabled=True,
    ),
    "max": PlanConfig(
        observations_per_month=1_500_000,
        verifications_per_month=150_000,
        corrections_per_month=-1,
        max_rpm=5_000,
        max_agents=-1,
        corrections_enabled=True,
        webhook_alerts=True,
        slack_alerts=True,
        retention_days=90,
        overage_allowed=True,
        max_experiments=-1,
        max_datasets=-1,
        max_dataset_items=-1,
        memories_per_month=1_500_000,
        recalls_per_month=10_000_000,
        memory_retention_days=90,
        memory_sync_enabled=True,
        artifact_max_file_bytes=1024 * 1024 * 1024,
        artifact_storage_bytes=1024 * 1024 * 1024 * 1024,
    ),
    "enterprise": PlanConfig(
        observations_per_month=10_000_000,
        verifications_per_month=1_000_000,
        corrections_per_month=-1,
        max_rpm=10_000,
        max_agents=-1,
        corrections_enabled=True,
        webhook_alerts=True,
        slack_alerts=True,
        retention_days=365,
        overage_allowed=True,
        max_experiments=-1,
        max_datasets=-1,
        max_dataset_items=-1,
        memories_per_month=-1,
        recalls_per_month=-1,
        # Bug fix (2026-08-13): this was 365, which made enterprise memory
        # retention WORSE than free and team (both -1 / forever) while the
        # retention sweep genuinely deletes. A year is not "forever," so the
        # first enterprise customer to cross it would have silently lost
        # history against a plan that is supposed to be the most generous
        # tier. -1 matches what enterprise is actually sold as.
        memory_retention_days=-1,
        memory_sync_enabled=True,
        artifact_max_file_bytes=1024 * 1024 * 1024,
        artifact_storage_bytes=1024 * 1024 * 1024 * 1024,
    ),
}


#: Type-name -> type. ``from __future__ import annotations`` makes every
#: ``dataclasses.Field.type`` a *string*, so the declared annotation is
#: resolved through this table rather than by evaluating it.
_TYPE_NAMES: dict[str, type] = {
    "int": int,
    "bool": bool,
    "float": float,
    "str": str,
}


def _declared_type(annotation: Any) -> type:
    """Resolve a dataclass field annotation to a concrete type."""
    if isinstance(annotation, type):
        return annotation
    return _TYPE_NAMES.get(str(annotation), object)


#: Field name -> declared type, resolved once at import. ``PlanConfig`` is
#: frozen and never gains fields at runtime, so this is a constant.
_PLAN_FIELD_TYPES: dict[str, type] = {
    f.name: _declared_type(f.type) for f in fields(PlanConfig)
}


def _coerce_override(name: str, value: Any) -> Any:
    """Return *value* typed for field *name*, or raise ``ValueError``.

    ``vex_plan_overrides`` is an operator-editable JSON column, so every value
    arriving here is untrusted input at a system boundary. Two coercions are
    real and safe; everything else is a typo and must be refused:

    * ``int`` -> ``float`` for a declared float field (JSON has one number
      type, so ``3`` for a float field is the operator writing 3.0);
    * nothing else. In particular ``bool`` is NOT accepted for an ``int``
      field even though ``bool`` subclasses ``int`` — ``True`` in
      ``memories_per_month`` means a quota of 1, which is not what anybody
      typed — and ``int`` is not accepted for a ``bool`` field.

    Strings are never parsed. ``"1000"`` for ``max_rpm`` is what produced the
    original outage, and silently coercing it would mean the column's type
    discipline lives nowhere at all.
    """
    expected = _PLAN_FIELD_TYPES.get(name)
    if expected is None:
        raise ValueError(f"unknown field {name!r}")

    if expected is bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"{name} expects bool, got {type(value).__name__}")

    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} expects int, got {type(value).__name__}")
        return value

    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} expects float, got {type(value).__name__}")
        return float(value)

    if isinstance(value, expected):
        return value
    raise ValueError(f"{name} expects {expected.__name__}, got {type(value).__name__}")


def get_plan_config(plan: str, overrides: Any = None) -> PlanConfig:
    """Return the PlanConfig for the given plan name.

    Falls back to ``"free"`` for unknown plan values.
    If *overrides* is provided (from ``accounts.vex_plan_overrides``),
    the values that are **known fields of the correct type** are merged on top
    of the plan defaults.

    Degradation, not failure. This function sits on the hot path of every
    authenticated request — ``verify_api_key`` calls it, and so does the OAuth
    middleware — and neither caller guards against anything but ``AuthError``.
    Before this filtering, one malformed key in one account's JSON blob raised
    ``TypeError`` out of *every* request for that tenant, and because the
    failure happened before the plan cache was populated it repeated
    identically forever rather than self-healing. An override that cannot be
    applied is therefore dropped with a WARNING naming the field, and the
    caller gets a usable config built from the plan defaults plus whatever
    survived. Never a tenant-wide failure over a config typo.

    A non-mapping blob (a JSON list or string in the column) is ignored whole,
    for the same reason.
    """
    base = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])
    if not overrides:
        return base

    if not isinstance(overrides, dict):
        logger.warning(
            "plan override for plan=%s is %s, not an object; ignoring it entirely",
            plan,
            type(overrides).__name__,
        )
        return base

    accepted: dict[str, Any] = {}
    for key, value in overrides.items():
        # ``None`` has always meant "not set", not "set to null"; keep that.
        if value is None:
            continue
        try:
            accepted[key] = _coerce_override(key, value)
        except ValueError as exc:
            logger.warning(
                "ignoring plan override %s=%r for plan=%s: %s",
                key,
                value,
                plan,
                exc,
            )

    if not accepted:
        return base
    return replace(base, **accepted)
