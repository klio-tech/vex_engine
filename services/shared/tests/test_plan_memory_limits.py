"""Memory-tier fields on PlanConfig, and the promise they have to keep.

The published pricing table at Dashboard/apps/landing/lib/pricing.ts is the
contract. These tests exist because it drifted from this file once already:
the site sold Free as unlimited memories kept forever while the engine enforced
1,000 a month and a one-day window, so a free signup lost their history after a
day. Asserting the published numbers here is what makes that drift a test
failure rather than a support ticket.
"""

import pathlib

import pytest
from shared.plan_limits import PLAN_LIMITS, PlanConfig, get_plan_config

#: The tiers Klio is sold on. Vex is sold on starter/pro/team.
KLIO_TIERS = ("free", "team")

#: Klio tiers whose published retention is still "forever" (-1) after the
#: 2026-08-13 repricing. Free moved to a real 30-day window; only team is
#: still unlimited here.
KLIO_FOREVER_RETENTION_TIERS = ("team",)


@pytest.mark.parametrize("tier", KLIO_TIERS)
def test_klio_tiers_are_unlimited_writes_with_no_recall_cap(tier):
    """Both published Klio tiers say "Memories: Unlimited" and have no recall cap.

    Retention is no longer identical across the two tiers as of the
    2026-08-13 repricing — see test_free_retention_is_30_days and
    test_team_retention_is_forever below — so it is asserted separately.
    """
    p = PLAN_LIMITS[tier]

    assert p.memories_per_month == -1, f"{tier} is sold as unlimited memories"
    assert p.recalls_per_month == -1, f"{tier} is sold without a recall cap"
    assert p.memory_sync_enabled is True


@pytest.mark.parametrize("tier", KLIO_FOREVER_RETENTION_TIERS)
def test_unlimited_retention_is_only_safe_because_it_is_guarded(tier):
    """A -1 retention is load bearing on two guards, and this says so.

    ``NOW() - ('-1 days')::interval`` is ``NOW() + 1 day``, so an unguarded
    sweep with -1 deletes the org's entire history. The guards live in
    ``run_retention.py`` (skips a non-positive value) and migration 038 (both
    SQL functions return early). If someone declares -1 on a tier without them,
    this is the test that should have stopped it.
    """
    assert PLAN_LIMITS[tier].memory_retention_days <= 0

    # services/shared/tests/ -> services/
    retention = (
        pathlib.Path(__file__).parents[2].joinpath("scripts", "run_retention.py")
    )
    assert retention.is_file(), f"expected the retention script at {retention}"
    guard = retention.read_text()
    assert "if config.memory_retention_days > 0:" in guard, (
        "run_retention.py must skip the memory sweep for a non-positive "
        "retention, or -1 deletes everything it was meant to preserve"
    )
    assert "if config.retention_days > 0:" in guard, (
        "the execution sweep needs the same guard"
    )


def test_free_retention_is_30_days():
    """2026-08-13 repricing: free moved off "forever" onto a real 30-day
    window — the primary upgrade lever. This is new territory: it is the
    first positive (sweep-activating) value Free has ever declared, so the
    -1 guards in run_retention.py / migration 038 do not apply to it. See the
    plan_limits module docstring's "2026-08-13 repricing" note."""
    assert PLAN_LIMITS["free"].memory_retention_days == 30


def test_team_retention_is_forever():
    assert PLAN_LIMITS["team"].memory_retention_days == -1


def test_enterprise_retention_is_forever_not_worse_than_free():
    """Regression guard: enterprise.memory_retention_days was 365, which made
    the top tier's retention WORSE than free and team (both forever) while
    the retention sweep genuinely deletes. Fixed to -1 in the same change
    that gave free a real 30-day window."""
    assert PLAN_LIMITS["enterprise"].memory_retention_days == -1


def test_grandfathering_override_beats_a_free_orgs_30_day_default():
    """Existing Free orgs are grandfathered onto unlimited retention via
    ``vex_plan_overrides`` rather than the engine deleting their history the
    day this change ships. This is the mechanism test: an override of -1 must
    win over Free's new 30-day default."""
    cfg = get_plan_config("free", {"memory_retention_days": -1})
    assert cfg.memory_retention_days == -1
    # Everything else about free stays free — the override is surgical.
    assert cfg.max_projects == 3
    assert cfg.max_users == 1


def test_free_artifact_allowance_matches_the_published_table():
    p = PLAN_LIMITS["free"]
    assert p.artifact_storage_bytes == 100 * 1024 * 1024
    assert p.max_rpm == 100


def test_team_artifact_allowance_matches_the_published_table():
    p = PLAN_LIMITS["team"]
    assert p.artifact_storage_bytes == 5 * 1024 * 1024 * 1024
    # 2026-08-13 repricing closed the drift: the site has always advertised
    # 1,000 RPM and the engine used to over-deliver at 5,000. Now both agree
    # exactly, so this is an equality, not a floor.
    assert p.max_rpm == 1_000


def test_vex_tiers_are_untouched():
    """Vex is hidden from the marketing site, not withdrawn.

    starter and pro keep their metered memory allowances. If a Klio change ever
    rewrites these, an existing Vex subscriber's entitlements move underneath
    them, which is the one thing this change was told not to do.
    """
    assert PLAN_LIMITS["starter"].memories_per_month == 25_000
    assert PLAN_LIMITS["starter"].recalls_per_month == 100_000
    assert PLAN_LIMITS["starter"].memory_retention_days == 7
    assert PLAN_LIMITS["pro"].memories_per_month == 150_000
    assert PLAN_LIMITS["pro"].memory_retention_days == 30
    assert PLAN_LIMITS["starter"].memory_sync_enabled is True


def test_vex_only_fields_survive_on_the_klio_tiers():
    """The two products share one ladder, so Klio's tiers still carry Vex's
    quotas. Zeroing them to "clean up" would silently disable Vex for anyone on
    free or team."""
    assert PLAN_LIMITS["free"].observations_per_month == 1_000
    assert PLAN_LIMITS["free"].verifications_per_month == 50
    assert PLAN_LIMITS["team"].observations_per_month == 1_500_000
    assert PLAN_LIMITS["team"].verifications_per_month == 150_000
    assert PLAN_LIMITS["team"].corrections_enabled is True


def test_overage_alignment():
    assert PLAN_LIMITS["free"].overage_allowed is False
    assert PLAN_LIMITS["starter"].overage_allowed is False
    assert PLAN_LIMITS["pro"].overage_allowed is True
    assert PLAN_LIMITS["team"].overage_allowed is True


def test_defaults_keep_old_constructors_working():
    cfg = get_plan_config("free", {"observations_per_month": 5})
    assert cfg.observations_per_month == 5
    assert cfg.memories_per_month == -1
    assert isinstance(PlanConfig, type)
