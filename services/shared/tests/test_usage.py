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
    over = check_memory_quota(db, "org-1", get_plan_config("starter"), "memory")
    assert over is False


def test_the_klio_tiers_never_block():
    """Free and Team are published as unlimited, so no counter blocks them.

    This is the assertion that would fail if someone reintroduced metering on a
    tier the pricing page sells as unlimited.
    """
    db = _db_with_counters(memories=99_999_999, recalls=99_999_999)
    for tier in ("free", "team"):
        for metric in ("memory", "recall"):
            assert (
                check_memory_quota(db, "org-1", get_plan_config(tier), metric) is False
            ), f"{tier} blocked on {metric} but is sold as unlimited"


def test_at_cap_metered_plan_raises():
    """Klio's free and team tiers are unlimited, so a metered plan is needed to
    exercise blocking at all. `starter` is a Vex tier and still metered."""
    db = _db_with_counters(memories=25_000, recalls=0)
    with pytest.raises(MemoryQuotaExceeded) as exc:
        check_memory_quota(
            db, "org-1", get_plan_config("starter"), "memory", plan_name="starter"
        )
    assert exc.value.metric == "memory"
    assert exc.value.limit == 25_000
    # The block carries the machine-readable upgrade target one tier up.
    assert exc.value.suggested_plan == "pro"


def test_override_cap_is_enforced():
    # An account override that lowers the cap must be honored in real-time
    # enforcement, exactly as the retention sweep honors it. A counter at the
    # overridden cap (5) blocks even though the base free cap is 1,000.
    db = _db_with_counters(memories=5, recalls=0)
    plan_config = get_plan_config("free", {"memories_per_month": 5})
    with pytest.raises(MemoryQuotaExceeded) as exc:
        check_memory_quota(db, "org-1", plan_config, "memory", plan_name="free")
    assert exc.value.limit == 5
    assert exc.value.suggested_plan == "starter"


def test_at_cap_pro_allows_overage():
    db = _db_with_counters(memories=150_000, recalls=0)
    over = check_memory_quota(db, "org-1", get_plan_config("pro"), "memory")
    assert over is True


def test_unlimited_never_blocks():
    db = _db_with_counters(memories=99_999_999, recalls=0)
    over = check_memory_quota(db, "org-1", get_plan_config("enterprise"), "memory")
    assert over is False


def test_recall_metric_uses_recall_limit():
    """Reads recalls_per_month rather than memories_per_month. Uses a metered
    plan because the Klio tiers no longer cap either."""
    db = _db_with_counters(memories=0, recalls=100_000)
    with pytest.raises(MemoryQuotaExceeded):
        check_memory_quota(db, "org-1", get_plan_config("starter"), "recall")


def test_check_fails_open_on_db_error():
    """Fail-open only. This mock CANNOT see the transaction half of the bug.

    A ``MagicMock`` has no transaction state: it answers the statement after a
    raised one exactly as happily as the first, so this test passed identically
    with and without the ``rollback()`` that the real defect needed. The
    ``rollback`` assertion below is the cheap half of the guard; the real one
    is ``tests/test_rollback_on_swallowed_error.py``, which poisons a live
    Postgres connection and then asks whether the caller can still use it.
    """
    db = MagicMock()
    db.execute.side_effect = RuntimeError("db down")
    # A METERED plan, deliberately. On an unlimited tier the function returns
    # before it ever queries, so `free` would make this pass without the
    # database being touched and the fail-open path never running.
    over = check_memory_quota(db, "org-1", get_plan_config("starter"), "memory")
    assert over is False
    db.rollback.assert_called_once()


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
