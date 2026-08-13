"""New PlanConfig fields from the 2026-08-13 Klio repricing.

``max_projects`` / ``max_users`` / ``extractions_per_month`` /
``intelligence_enabled`` were added additively, with defaults, so that
existing direct ``PlanConfig(...)`` constructors elsewhere in the codebase
(see test_plan_override_hardening.py) keep building without modification.
These tests pin the four new fields' defaults and their published values on
free / team / enterprise, per the 2026-08-13 Klio pricing spec's
"Configuration changes -> 1. plan_limits.py" section.
"""

import dataclasses

from shared.plan_limits import PlanConfig, get_plan_config


def test_new_field_defaults_are_permissive():
    """Every new field defaults to "no limit" / "on", so a plan dict that
    predates these fields (or a direct constructor that omits them) behaves
    exactly as it did before this change — nothing new is silently
    restricted."""
    cfg = PlanConfig(
        observations_per_month=1,
        verifications_per_month=1,
        corrections_per_month=1,
        max_rpm=1,
        max_agents=1,
        corrections_enabled=True,
        webhook_alerts=True,
        slack_alerts=True,
        retention_days=1,
        overage_allowed=True,
    )
    assert cfg.max_projects == -1
    assert cfg.max_users == -1
    assert cfg.extractions_per_month == -1
    assert cfg.intelligence_enabled is True


def test_free_pricing_limits_match_the_spec():
    p = get_plan_config("free")
    assert p.max_projects == 3
    assert p.max_users == 1
    assert p.extractions_per_month == 3_000
    assert p.intelligence_enabled is False


def test_team_pricing_limits_match_the_spec():
    p = get_plan_config("team")
    assert p.max_projects == -1
    assert p.max_users == -1
    assert p.extractions_per_month == 10_000
    assert p.intelligence_enabled is True


def test_free_and_team_new_fields_actually_differ():
    """A mutation that copy-pastes free's block onto team (or vice versa)
    would pass field-by-field equality checks against a single expected
    tuple; comparing the two tiers directly closes that gap."""
    free = get_plan_config("free")
    team = get_plan_config("team")
    assert free.max_projects != team.max_projects
    assert free.max_users != team.max_users
    assert free.extractions_per_month != team.extractions_per_month
    assert free.intelligence_enabled != team.intelligence_enabled


def test_existing_constructors_without_the_new_fields_still_build():
    """The additive-with-defaults contract: a caller that only knows the
    pre-2026-08-13 fields (mirrors ingestion-api / sync-gateway's direct
    PlanConfig(...) call sites) must still construct a valid, usable
    PlanConfig."""
    cfg = PlanConfig(
        observations_per_month=100,
        verifications_per_month=10,
        corrections_per_month=0,
        max_rpm=60,
        max_agents=5,
        corrections_enabled=False,
        webhook_alerts=False,
        slack_alerts=False,
        retention_days=7,
        overage_allowed=False,
    )
    assert dataclasses.is_dataclass(cfg)
    assert cfg.max_projects == -1
    assert cfg.max_users == -1
    assert cfg.extractions_per_month == -1
    assert cfg.intelligence_enabled is True
