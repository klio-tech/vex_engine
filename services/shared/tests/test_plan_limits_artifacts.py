"""Artifact capacity per plan.

Capacity is the paid axis; the feature itself is on every tier. A plan whose
``artifact_max_file_bytes`` is 0 has artifacts disabled entirely, which no
shipped plan does.
"""

from shared.plan_limits import PLAN_LIMITS, get_plan_config

MB = 1024 * 1024
GB = 1024 * MB

EXPECTED = {
    "free": (10 * MB, 100 * MB),
    "starter": (50 * MB, 5 * GB),
    "max": (1 * GB, 1024 * GB),
}


def test_artifact_limits_per_plan():
    for plan, (per_file, pool) in EXPECTED.items():
        cfg = get_plan_config(plan)
        assert cfg.artifact_max_file_bytes == per_file, plan
        assert cfg.artifact_storage_bytes == pool, plan


def test_max_plan_is_registered():
    # get_plan_config falls back to free for an unknown plan, so a missing
    # "max" key would silently give a $499 customer 10MB files.
    assert "max" in PLAN_LIMITS
    assert get_plan_config("max").artifact_max_file_bytes > (
        get_plan_config("free").artifact_max_file_bytes
    )


def test_every_plan_has_artifacts_enabled():
    for plan in PLAN_LIMITS:
        cfg = get_plan_config(plan)
        assert cfg.artifact_max_file_bytes > 0, plan
        assert cfg.artifact_storage_bytes > 0, plan


def test_overrides_apply_to_artifact_limits():
    cfg = get_plan_config("free", {"artifact_storage_bytes": 999})
    assert cfg.artifact_storage_bytes == 999
