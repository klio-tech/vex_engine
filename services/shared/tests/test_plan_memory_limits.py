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
    assert PLAN_LIMITS["free"].overage_allowed is False
    assert PLAN_LIMITS["starter"].overage_allowed is False
    assert PLAN_LIMITS["pro"].overage_allowed is True
    assert PLAN_LIMITS["team"].overage_allowed is True


def test_defaults_keep_old_constructors_working():
    from shared.plan_limits import PlanConfig

    cfg = get_plan_config("free", {"observations_per_month": 5})
    assert cfg.observations_per_month == 5
    assert cfg.memories_per_month == 1_000
    assert isinstance(PlanConfig, type)
