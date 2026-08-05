"""A malformed ``vex_plan_overrides`` blob must not hard-down a tenant.

``accounts.vex_plan_overrides`` is an operator-editable JSON column. Before
this module's fix, :func:`shared.plan_limits.get_plan_config` splatted it
straight into ``PlanConfig(**merged)``:

* an unknown key raised ``TypeError: __init__() got an unexpected keyword
  argument`` at construction time;
* a wrong-typed value (``"1000"`` for ``max_rpm``) constructed fine and then
  raised ``TypeError: '<' not supported between instances of 'int' and 'str'``
  one line later, in the quota comparison.

Either shape escapes ``shared.auth.verify_api_key`` (which catches only
``AuthError``) and the OAuth middleware (which has no guard at all), so *every*
request for that account failed identically — and because the failure happens
before the plan cache is written, it never self-healed.

The contract these tests pin: an override that cannot be applied is **ignored
with a WARNING**, and the caller gets a usable, correctly-typed
:class:`PlanConfig` built from the plan defaults plus whatever overrides were
valid.
"""

import hashlib
import logging
from unittest.mock import MagicMock, patch

import pytest
from shared.auth import KeyInfo, KeyValidator
from shared.plan_limits import PLAN_LIMITS, PlanConfig, get_plan_config


# ── Positive controls: the fixture is not vacuous ──────────────────────


def test_a_valid_override_is_still_applied():
    """Positive control: hardening must not stop honouring real overrides."""
    cfg = get_plan_config("free", {"memories_per_month": 5})
    assert cfg.memories_per_month == 5
    # Untouched fields keep the plan default.
    assert cfg.max_rpm == PLAN_LIMITS["free"].max_rpm


def test_no_override_returns_the_plan_default_object():
    assert get_plan_config("pro") is PLAN_LIMITS["pro"]


# ── The failures that hard-downed a tenant ────────────────────────────


def test_an_unknown_override_key_is_ignored_not_fatal(caplog):
    """``TypeError: unexpected keyword argument`` used to escape to the caller."""
    with caplog.at_level(logging.WARNING, logger="agentguard.shared.plan_limits"):
        cfg = get_plan_config("free", {"max_rpmm": 1000})

    assert isinstance(cfg, PlanConfig)
    assert cfg.max_rpm == PLAN_LIMITS["free"].max_rpm
    assert any("max_rpmm" in r.getMessage() for r in caplog.records)


def test_a_wrong_typed_override_is_ignored_not_fatal(caplog):
    """``"1000"`` for an int field used to blow up in the quota comparison."""
    with caplog.at_level(logging.WARNING, logger="agentguard.shared.plan_limits"):
        cfg = get_plan_config("free", {"max_rpm": "1000"})

    assert cfg.max_rpm == PLAN_LIMITS["free"].max_rpm
    assert isinstance(cfg.max_rpm, int)
    # The comparison that used to raise now works.
    assert (cfg.max_rpm < 10) in (True, False)


def test_a_bool_is_not_smuggled_into_an_int_field():
    """``bool`` is an ``int`` subclass; a flag in a quota field is a typo."""
    cfg = get_plan_config("free", {"memories_per_month": True})
    assert cfg.memories_per_month == PLAN_LIMITS["free"].memories_per_month


def test_a_non_bool_is_not_smuggled_into_a_bool_field():
    cfg = get_plan_config("free", {"memory_sync_enabled": 1})
    assert cfg.memory_sync_enabled is PLAN_LIMITS["free"].memory_sync_enabled


def test_the_valid_half_of_a_mixed_blob_still_applies(caplog):
    """One bad key must not discard the operator's good keys."""
    with caplog.at_level(logging.WARNING, logger="agentguard.shared.plan_limits"):
        cfg = get_plan_config(
            "free", {"memories_per_month": 42, "nonsense": 1, "max_rpm": "x"}
        )
    assert cfg.memories_per_month == 42
    assert cfg.max_rpm == PLAN_LIMITS["free"].max_rpm


def test_a_non_dict_override_blob_is_ignored(caplog):
    """A JSON list/string in the column is not a mapping and must not crash."""
    with caplog.at_level(logging.WARNING, logger="agentguard.shared.plan_limits"):
        assert get_plan_config("free", ["max_rpm", 10]) is PLAN_LIMITS["free"]
        assert get_plan_config("free", "max_rpm=10") is PLAN_LIMITS["free"]


def test_an_int_is_accepted_for_a_float_field_if_any_exist():
    """Widening int->float is a real coercion, not a type error."""
    float_fields = [
        name
        for name, ftype in PlanConfig.__annotations__.items()
        if ftype in ("float", float)
    ]
    if not float_fields:
        pytest.skip("PlanConfig currently declares no float fields")
    cfg = get_plan_config("free", {float_fields[0]: 3})
    assert getattr(cfg, float_fields[0]) == 3.0


def test_a_none_valued_override_still_means_leave_it_alone(caplog):
    """Pre-existing behaviour: ``None`` is 'unset', not 'set to null'.

    And it is *silent*. A JSON ``null`` is how the dashboard clears a custom
    limit, so it is the normal case, not an operator error — warning about it
    would train on-call to ignore the very log line that flags a real typo.
    """
    with caplog.at_level(logging.WARNING, logger="agentguard.shared.plan_limits"):
        cfg = get_plan_config("free", {"max_rpm": None})
    assert cfg.max_rpm == PLAN_LIMITS["free"].max_rpm
    assert caplog.records == []


# ── The reachable version: the auth path a real request takes ─────────


_RAW_KEY = "ag_live_testkey1234567890abcdefgh"


def _validator_with_overrides(overrides):
    """A ``KeyValidator`` whose account carries *overrides* verbatim."""
    key_entry = {
        "id": "key-001",
        "prefix": _RAW_KEY[:12],
        "key_hash": hashlib.sha256(_RAW_KEY.encode("utf-8")).hexdigest(),
        "name": "Test Key",
        "scopes": ["ingest", "verify", "read"],
        "rate_limit_rpm": 1000,
        "expires_at": None,
        "created_at": "2026-02-12T00:00:00Z",
        "created_by": "user-001",
        "last_used_at": None,
        "revoked": False,
    }
    with patch("shared.auth.create_engine") as mock_create:
        engine = MagicMock()
        mock_create.return_value = engine
        validator = KeyValidator(
            database_url="postgresql://test:test@localhost/test",
            required_scope="verify",
        )

    conn = MagicMock()
    result = MagicMock()
    result.fetchone.return_value = ("test-org", [key_entry], "test-org")
    result.scalar.return_value = 0
    conn.execute.return_value = result
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    engine.connect.return_value = conn

    validator._query_plan_from_supabase = lambda slug, org: ("free", overrides)
    return validator


def test_a_sane_override_still_authenticates():
    """Positive control: the harness authenticates when the blob is fine."""
    validator = _validator_with_overrides({"max_rpm": 4242})
    assert isinstance(validator.validate(_RAW_KEY), KeyInfo)


def test_an_unknown_override_key_does_not_down_the_tenant():
    """``TypeError`` at ``PlanConfig(**merged)`` used to escape ``validate``."""
    validator = _validator_with_overrides({"max_rpmm": 1000})
    assert isinstance(validator.validate(_RAW_KEY), KeyInfo)


def test_a_string_rpm_does_not_down_the_tenant():
    """``min(int, str)`` one line after construction used to escape ``validate``."""
    validator = _validator_with_overrides({"max_rpm": "1000"})
    info = validator.validate(_RAW_KEY)
    assert isinstance(info, KeyInfo)
    # The raw blob is still carried through to the caller verbatim — the
    # filtering happens where the value is *used*, so no data is rewritten.
    assert info.plan_overrides == {"max_rpm": "1000"}
