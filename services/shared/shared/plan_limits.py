"""Plan-level limits for Vex pricing tiers.

These constants define the quotas, rate limits, and feature flags
for each plan. The ``PLAN_LIMITS`` dict is keyed by the ``plan``
column value in the ``organizations`` table.

The memory-tier fields (memories_per_month / recalls_per_month /
memory_retention_days / memory_sync_enabled) are mirrored on the marketing
site at Dashboard/apps/landing/lib/pricing.ts — keep the two in sync.
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
        memories_per_month=1_000,
        recalls_per_month=10_000,
        memory_retention_days=1,
        memory_sync_enabled=False,
        artifact_max_file_bytes=10 * 1024 * 1024,
        artifact_storage_bytes=100 * 1024 * 1024,
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
        artifact_max_file_bytes=50 * 1024 * 1024,
        artifact_storage_bytes=5 * 1024 * 1024 * 1024,
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
        memory_retention_days=365,
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
