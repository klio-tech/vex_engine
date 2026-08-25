"""Tests for the shared KeyValidator authentication module."""

import hashlib
import logging
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from shared.auth import VALID_SCOPES, AuthError, KeyInfo, KeyValidator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_key_entry(
    key_id="key-001",
    raw_key="ag_live_testkey1234567890abcdefgh",
    name="Test Key",
    scopes=None,
    rate_limit_rpm=1000,
    expires_at=None,
    revoked=False,
    created_by="user-001",
):
    """Build an api_keys JSONB entry for test fixtures.

    ``created_by`` defaults to ``"user-001"``, deliberately not a UUID --
    the membership-check machinery in ``shared.auth`` treats a non-UUID
    ``user_id`` the same as an absent one (skip the check), so every
    existing fixture built from this default is unaffected by that check.
    Tests that need a real membership check pass a UUID explicitly.
    """
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    entry = {
        "id": key_id,
        "prefix": raw_key[:12],
        "key_hash": key_hash,
        "name": name,
        "scopes": scopes or ["ingest", "verify", "read"],
        "rate_limit_rpm": rate_limit_rpm,
        "expires_at": expires_at,
        "created_at": "2026-02-12T00:00:00Z",
        "created_by": created_by,
        "last_used_at": None,
        "revoked": revoked,
    }
    return entry, raw_key, key_hash


RAW_KEY = "ag_live_testkey1234567890abcdefgh"
KEY_HASH = hashlib.sha256(RAW_KEY.encode("utf-8")).hexdigest()


def _make_db_row(api_keys, org_id="test-org", account_slug="test-org"):
    """Create a mock DB row tuple (org_id, api_keys, account_slug)."""
    return (org_id, api_keys, account_slug)


def _build_validator(required_scope="verify", cache_ttl_s=60.0):
    """Build a KeyValidator with a mocked database engine."""
    with patch("shared.auth.create_engine") as mock_create:
        mock_engine = MagicMock()
        mock_create.return_value = mock_engine
        validator = KeyValidator(
            database_url="postgresql://test:test@localhost/test",
            required_scope=required_scope,
            cache_ttl_s=cache_ttl_s,
        )
    return validator, mock_engine


def _mock_db_query(mock_engine, row):
    """Configure the mocked engine to return a specific row."""
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchone.return_value = row
    mock_result.scalar.return_value = 0  # quota usage = 0
    mock_conn.execute.return_value = mock_result
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_engine.connect.return_value = mock_conn
    return mock_conn


# ---------------------------------------------------------------------------
# Tests: Valid key
# ---------------------------------------------------------------------------


class TestValidKey:
    def test_valid_key_returns_key_info(self):
        entry, raw_key, _ = _make_key_entry()
        validator, engine = _build_validator(required_scope="verify")
        row = _make_db_row([entry])
        _mock_db_query(engine, row)

        result = validator.validate(raw_key)

        assert isinstance(result, KeyInfo)
        assert result.org_id == "test-org"
        assert result.key_id == "key-001"
        assert "verify" in result.scopes

    def test_valid_key_with_all_scopes(self):
        entry, raw_key, _ = _make_key_entry(scopes=["ingest", "verify", "read"])
        validator, engine = _build_validator(required_scope="ingest")
        _mock_db_query(engine, _make_db_row([entry]))

        result = validator.validate(raw_key)
        assert result.scopes == ["ingest", "verify", "read"]


# ---------------------------------------------------------------------------
# Tests: Invalid key
# ---------------------------------------------------------------------------


class TestInvalidKey:
    def test_missing_key_raises_401(self):
        validator, engine = _build_validator()
        _mock_db_query(engine, None)

        with pytest.raises(AuthError) as exc_info:
            validator.validate("ag_live_nonexistent_key_abcdefg")

        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.detail

    def test_empty_api_keys_array(self):
        validator, engine = _build_validator()
        _mock_db_query(engine, None)

        with pytest.raises(AuthError) as exc_info:
            validator.validate("ag_live_bogus_key_0000000000000")

        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Tests: Revoked key
# ---------------------------------------------------------------------------


class TestRevokedKey:
    def test_revoked_key_raises_401(self):
        entry, raw_key, _ = _make_key_entry(revoked=True)
        validator, engine = _build_validator()
        _mock_db_query(engine, _make_db_row([entry]))

        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key)

        assert exc_info.value.status_code == 401
        assert "revoked" in exc_info.value.detail


# ---------------------------------------------------------------------------
# Tests: Expired key
# ---------------------------------------------------------------------------


class TestExpiredKey:
    def test_expired_key_raises_401(self):
        yesterday = (datetime.now(tz=timezone.utc) - timedelta(days=1)).isoformat()
        entry, raw_key, _ = _make_key_entry(expires_at=yesterday)
        validator, engine = _build_validator()
        _mock_db_query(engine, _make_db_row([entry]))

        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key)

        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail

    def test_non_expired_key_passes(self):
        tomorrow = (datetime.now(tz=timezone.utc) + timedelta(days=1)).isoformat()
        entry, raw_key, _ = _make_key_entry(expires_at=tomorrow)
        validator, engine = _build_validator()
        _mock_db_query(engine, _make_db_row([entry]))

        result = validator.validate(raw_key)
        assert result.org_id == "test-org"

    def test_null_expiry_passes(self):
        entry, raw_key, _ = _make_key_entry(expires_at=None)
        validator, engine = _build_validator()
        _mock_db_query(engine, _make_db_row([entry]))

        result = validator.validate(raw_key)
        assert result.org_id == "test-org"


# ---------------------------------------------------------------------------
# Tests: Scope enforcement
# ---------------------------------------------------------------------------


class TestScopeEnforcement:
    def test_wrong_scope_raises_403(self):
        entry, raw_key, _ = _make_key_entry(scopes=["ingest"])
        validator, engine = _build_validator(required_scope="verify")
        _mock_db_query(engine, _make_db_row([entry]))

        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key)

        assert exc_info.value.status_code == 403
        assert "verify" in exc_info.value.detail

    def test_ingest_scope_accepted_by_ingest_service(self):
        entry, raw_key, _ = _make_key_entry(scopes=["ingest"])
        validator, engine = _build_validator(required_scope="ingest")
        _mock_db_query(engine, _make_db_row([entry]))

        result = validator.validate(raw_key)
        assert result.org_id == "test-org"

    def test_read_scope_rejected_by_verify_service(self):
        entry, raw_key, _ = _make_key_entry(scopes=["read"])
        validator, engine = _build_validator(required_scope="verify")
        _mock_db_query(engine, _make_db_row([entry]))

        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key)

        assert exc_info.value.status_code == 403

    def test_invalid_required_scope_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid scope"):
            with patch("shared.auth.create_engine"):
                KeyValidator(
                    database_url="postgresql://x:x@localhost/x",
                    required_scope="admin",
                )

    def test_memory_scope_accepted_by_constructor(self):
        """The dedicated ``memory`` scope (Klio brain) is a valid scope."""
        assert "memory" in VALID_SCOPES
        # Constructing a validator with it must not raise.
        validator, _ = _build_validator(required_scope="memory")
        assert validator._required_scope == "memory"

    def test_key_without_memory_scope_rejected_by_memory_service(self):
        """A key lacking ``memory`` is 403'd by a memory-scoped validator.

        The scope check (validate step 5) runs before rate-limit/quota, so
        a wrong-scope key raises 403 without those steps interfering.
        """
        entry, raw_key, _ = _make_key_entry(scopes=["ingest", "verify", "read"])
        validator, engine = _build_validator(required_scope="memory")
        _mock_db_query(engine, _make_db_row([entry]))

        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key)

        assert exc_info.value.status_code == 403
        assert "memory" in exc_info.value.detail

    def test_key_with_memory_scope_accepted_by_memory_service(self):
        """A key carrying ``memory`` validates against a memory validator."""
        entry, raw_key, _ = _make_key_entry(
            scopes=["ingest", "verify", "read", "memory"]
        )
        validator, engine = _build_validator(required_scope="memory")
        _mock_db_query(engine, _make_db_row([entry]))

        result = validator.validate(raw_key)
        assert isinstance(result, KeyInfo)
        assert result.org_id == "test-org"
        assert "memory" in result.scopes


# ---------------------------------------------------------------------------
# Tests: Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_rate_limit_exceeded_raises_429(self):
        entry, raw_key, _ = _make_key_entry(rate_limit_rpm=3)
        validator, engine = _build_validator()
        _mock_db_query(engine, _make_db_row([entry]))

        # First 3 requests succeed
        for _ in range(3):
            validator.validate(raw_key)

        # 4th request should be rate limited
        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key)

        assert exc_info.value.status_code == 429
        assert "Rate limit" in exc_info.value.detail
        assert exc_info.value.retry_after_seconds is not None
        assert exc_info.value.retry_after_seconds > 0

    def test_rate_limit_resets_after_window(self):
        entry, raw_key, _ = _make_key_entry(rate_limit_rpm=2)
        validator, engine = _build_validator()
        _mock_db_query(engine, _make_db_row([entry]))

        # Use up the rate limit
        validator.validate(raw_key)
        validator.validate(raw_key)

        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key)
        assert exc_info.value.status_code == 429

        # Simulate window expiry by clearing the deque
        validator._rate_counters[entry["id"]].clear()

        # Should work again
        result = validator.validate(raw_key)
        assert result.org_id == "test-org"


# ---------------------------------------------------------------------------
# Tests: Cache behavior
# ---------------------------------------------------------------------------


class TestCacheBehavior:
    def test_cache_hit_skips_db_query(self):
        entry, raw_key, _ = _make_key_entry()
        validator, engine = _build_validator()
        mock_conn = _mock_db_query(engine, _make_db_row([entry]))

        # First call queries DB (key lookup + quota check)
        validator.validate(raw_key)
        call_count_after_first = mock_conn.execute.call_count

        # Second call should use cache for both key lookup and quota check
        # So no additional DB calls occur
        validator.validate(raw_key)
        assert mock_conn.execute.call_count == call_count_after_first

    def test_cache_expiry_re_queries_db(self):
        entry, raw_key, _ = _make_key_entry()
        validator, engine = _build_validator(cache_ttl_s=0.1)
        mock_conn = _mock_db_query(engine, _make_db_row([entry]))

        # First call queries DB
        validator.validate(raw_key)
        call_count_after_first = mock_conn.execute.call_count

        # Wait for cache to expire
        time.sleep(0.15)

        # Second call should query DB again
        validator.validate(raw_key)
        assert mock_conn.execute.call_count > call_count_after_first

    def test_revoked_key_in_cache_rejected(self):
        """If a key is cached then revoked, the cached entry should still reject."""
        entry, raw_key, _ = _make_key_entry(revoked=False)
        validator, engine = _build_validator()
        _mock_db_query(engine, _make_db_row([entry]))

        # First call succeeds
        result = validator.validate(raw_key)
        assert result.org_id == "test-org"

        # Manually mark the cached entry as revoked (simulates cache refresh)
        key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        validator._cache[key_hash].revoked = True

        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key)
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Tests: Batched last_used_at
# ---------------------------------------------------------------------------


class TestLastUsedAt:
    def test_usage_tracked_in_buffer(self):
        entry, raw_key, _ = _make_key_entry()
        validator, engine = _build_validator()
        _mock_db_query(engine, _make_db_row([entry]))

        validator.validate(raw_key)

        assert "key-001" in validator._usage_buffer

    def test_flush_clears_buffer(self):
        entry, raw_key, _ = _make_key_entry()
        validator, engine = _build_validator()
        _mock_db_query(engine, _make_db_row([entry]))

        validator.validate(raw_key)
        assert len(validator._usage_buffer) > 0

        validator.flush_usage()
        assert len(validator._usage_buffer) == 0


# ---------------------------------------------------------------------------
# Tests: Multiple keys per org
# ---------------------------------------------------------------------------


class TestMultipleKeys:
    def test_org_with_multiple_keys(self):
        entry1, raw_key1, _ = _make_key_entry(
            key_id="key-001",
            raw_key="ag_live_firstkey_000000000000000",
            scopes=["ingest"],
        )
        entry2, raw_key2, _ = _make_key_entry(
            key_id="key-002",
            raw_key="ag_live_secondkey_00000000000000",
            scopes=["verify"],
        )

        validator, engine = _build_validator(required_scope="verify")

        # Key 1 (ingest only) should fail on verify service
        _mock_db_query(engine, _make_db_row([entry1, entry2]))
        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key1)
        assert exc_info.value.status_code == 403

        # Key 2 (verify) should succeed
        _mock_db_query(engine, _make_db_row([entry1, entry2]))
        result = validator.validate(raw_key2)
        assert result.key_id == "key-002"


# ---------------------------------------------------------------------------
# Tests: DB error handling
# ---------------------------------------------------------------------------


class TestDBErrors:
    def test_db_connection_error_raises_500(self):
        validator, engine = _build_validator()
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("connection refused")
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        engine.connect.return_value = mock_conn

        with pytest.raises(AuthError) as exc_info:
            validator.validate(RAW_KEY)

        assert exc_info.value.status_code == 500
        assert "Internal" in exc_info.value.detail


# ---------------------------------------------------------------------------
# Tests: Quota enforcement
# ---------------------------------------------------------------------------


class TestQuotaEnforcement:
    """Tests for monthly quota checks against the executions table."""

    def _make_validator_with_quota(
        self,
        scope="verify",
        plan="free",
        usage_count=0,
        plan_overrides=None,
    ):
        """Build a validator with configurable quota usage from DB."""
        entry, raw_key, _ = _make_key_entry(scopes=["ingest", "verify", "read"])
        validator, engine = _build_validator(required_scope=scope)

        # Mock the DB to return key data on first call, usage on second
        mock_conn = MagicMock()
        key_row = _make_db_row([entry])
        usage_result = MagicMock()
        usage_result.scalar.return_value = usage_count

        key_result = MagicMock()
        key_result.fetchone.return_value = key_row

        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            # First call is key lookup, subsequent are quota check
            if call_count["n"] == 1:
                return key_result
            return usage_result

        mock_conn.execute.side_effect = side_effect
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        engine.connect.return_value = mock_conn

        # Directly set the plan on the validator's supabase query
        validator._query_plan_from_supabase = MagicMock(
            return_value=(plan, plan_overrides, None),
        )

        return validator, raw_key, mock_conn

    def test_free_plan_under_verification_quota_passes(self):
        """Free plan with 10/50 verifications should pass."""
        validator, raw_key, _ = self._make_validator_with_quota(
            scope="verify",
            plan="free",
            usage_count=10,
        )
        result = validator.validate(raw_key)
        assert result.org_id == "test-org"

    def test_free_plan_at_verification_quota_blocks(self):
        """Free plan with 50/50 verifications should return 429."""
        validator, raw_key, _ = self._make_validator_with_quota(
            scope="verify",
            plan="free",
            usage_count=50,
        )
        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key)
        assert exc_info.value.status_code == 429
        assert "quota exceeded" in exc_info.value.detail.lower()
        assert exc_info.value.retry_after_seconds == 3600

    def test_free_plan_over_verification_quota_blocks(self):
        """Free plan with 589/50 verifications should return 429."""
        validator, raw_key, _ = self._make_validator_with_quota(
            scope="verify",
            plan="free",
            usage_count=589,
        )
        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key)
        assert exc_info.value.status_code == 429

    def test_free_plan_under_observation_quota_passes(self):
        """Free plan with 500/1000 observations should pass."""
        validator, raw_key, _ = self._make_validator_with_quota(
            scope="ingest",
            plan="free",
            usage_count=500,
        )
        result = validator.validate(raw_key)
        assert result.org_id == "test-org"

    def test_free_plan_at_observation_quota_blocks(self):
        """Free plan with 1000/1000 observations should return 429."""
        validator, raw_key, _ = self._make_validator_with_quota(
            scope="ingest",
            plan="free",
            usage_count=1000,
        )
        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key)
        assert exc_info.value.status_code == 429

    def test_pro_plan_overage_allowed(self):
        """Pro plan allows overage — should log but not block."""
        validator, raw_key, _ = self._make_validator_with_quota(
            scope="verify",
            plan="pro",
            usage_count=20_000,
        )
        result = validator.validate(raw_key)
        assert result.org_id == "test-org"

    def test_starter_plan_no_overage(self):
        """Starter plan does NOT allow overage — should block at limit."""
        validator, raw_key, _ = self._make_validator_with_quota(
            scope="verify",
            plan="starter",
            usage_count=1000,
        )
        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key)
        assert exc_info.value.status_code == 429

    def test_db_error_returns_503_fail_closed(self):
        """If quota query fails, return 503 (not silently allow)."""
        entry, raw_key, _ = _make_key_entry()
        validator, engine = _build_validator(required_scope="verify")

        # First call succeeds (key lookup), second call fails (quota)
        mock_conn = MagicMock()
        key_result = MagicMock()
        key_result.fetchone.return_value = _make_db_row([entry])

        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return key_result
            raise Exception("connection lost")

        mock_conn.execute.side_effect = side_effect
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        engine.connect.return_value = mock_conn

        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key)
        assert exc_info.value.status_code == 503
        assert exc_info.value.retry_after_seconds == 30

    def test_quota_cache_avoids_repeated_db_queries(self):
        """Second call within 30s should use cached quota, not query DB."""
        validator, raw_key, mock_conn = self._make_validator_with_quota(
            scope="verify",
            plan="free",
            usage_count=10,
        )

        # First call
        validator.validate(raw_key)
        first_call_count = mock_conn.execute.call_count

        # Second call — should use quota cache
        validator.validate(raw_key)
        second_call_count = mock_conn.execute.call_count

        # No additional DB calls (key is cached, quota is cached)
        assert second_call_count == first_call_count

    def test_verify_scope_query_filters_by_confidence(self):
        """Verify scope should query with 'confidence IS NOT NULL'."""
        validator, raw_key, mock_conn = self._make_validator_with_quota(
            scope="verify",
            plan="free",
            usage_count=10,
        )
        validator.validate(raw_key)

        # Find the quota query call (second execute call)
        calls = mock_conn.execute.call_args_list
        assert len(calls) >= 2
        quota_query = str(calls[1][0][0])
        assert "confidence IS NOT NULL" in quota_query


# ---------------------------------------------------------------------------
# Helpers: engine routers that record the SQL a validator actually issues
# ---------------------------------------------------------------------------

ACCOUNT_UUID = "3f2a9b1c-7d64-4c18-9a55-1e0b6d2f8c43"


def _build_validator_with_supabase(required_scope="verify", cache_ttl_s=60.0):
    """Build a KeyValidator with mocked primary *and* Supabase engines."""
    main_engine = MagicMock()
    supabase_engine = MagicMock()
    with patch("shared.auth.create_engine") as mock_create:
        mock_create.side_effect = [main_engine, supabase_engine]
        validator = KeyValidator(
            database_url="postgresql://test:test@localhost/test",
            required_scope=required_scope,
            supabase_database_url="postgresql://test:test@localhost/supabase",
            cache_ttl_s=cache_ttl_s,
        )
    return validator, main_engine, supabase_engine


def _route_main_engine(engine, *, key_row, usage_count=0, usage_error=False):
    """Route primary-engine queries by statement and record every SQL string.

    Returns the recording list so a test can assert *which* queries ran --
    notably whether the reliability ``executions`` COUNT was issued at all.
    """
    executed: list[str] = []
    mock_conn = MagicMock()

    def _execute(statement, params=None, *args, **kwargs):
        sql = str(statement)
        executed.append(sql)
        result = MagicMock()
        if "FROM organizations" in sql:
            result.fetchone.return_value = key_row
            return result
        if "FROM executions" in sql:
            if usage_error:
                raise Exception("executions hypertable unavailable")
            result.scalar.return_value = usage_count
            return result
        result.fetchone.return_value = None
        result.scalar.return_value = 0
        return result

    mock_conn.execute.side_effect = _execute
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    engine.connect.return_value = mock_conn
    return executed


def _route_supabase_engine(engine, *, by_id=None, by_slug=None, error=False):
    """Route Supabase plan lookups and record ``(sql, params)`` per call.

    ``by_id`` is the row returned for the immutable account-UUID join and
    ``by_slug`` the row for the legacy mutable-slug join (``None`` = no row).
    """
    executed: list[tuple[str, dict]] = []
    mock_conn = MagicMock()

    def _execute(statement, params=None, *args, **kwargs):
        bound = params or {}
        executed.append((str(statement), bound))
        if error:
            raise Exception("supabase unavailable")
        result = MagicMock()
        if "account_id" in bound:
            result.fetchone.return_value = by_id
        else:
            result.fetchone.return_value = by_slug
        return result

    mock_conn.execute.side_effect = _execute
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    engine.connect.return_value = mock_conn
    return executed


# ---------------------------------------------------------------------------
# Tests: plan lookup joins Supabase on the immutable account UUID
# ---------------------------------------------------------------------------


class TestPlanLookupJoin:
    """The plan join must survive a workspace rename.

    ``organizations.account_slug`` mirrors a user-editable Supabase slug, so
    it goes stale the moment somebody renames their workspace. Joining on the
    immutable ``accounts.id`` keeps paid tenants on their paid plan instead of
    silently downgrading every API key to ``free``.
    """

    def test_stale_slug_still_resolves_paid_plan_via_account_uuid(self):
        """Renamed workspace: stale slug misses, UUID join still finds 'pro'."""
        entry, raw_key, _ = _make_key_entry()
        validator, main_engine, supabase_engine = _build_validator_with_supabase(
            required_scope="verify"
        )
        _route_main_engine(
            main_engine,
            key_row=_make_db_row(
                [entry], org_id=ACCOUNT_UUID, account_slug="old-workspace-name"
            ),
            usage_count=10,
        )
        _route_supabase_engine(
            supabase_engine,
            by_id=("pro", None),
            by_slug=None,  # slug was renamed -> legacy lookup misses
        )

        result = validator.validate(raw_key)

        assert result.plan == "pro"
        assert result.org_id == ACCOUNT_UUID

    def test_uuid_join_binds_org_id_and_skips_slug_query(self):
        """A UUID hit short-circuits: no second query on the mutable slug."""
        entry, raw_key, _ = _make_key_entry()
        validator, main_engine, supabase_engine = _build_validator_with_supabase(
            required_scope="verify"
        )
        _route_main_engine(
            main_engine,
            key_row=_make_db_row(
                [entry], org_id=ACCOUNT_UUID, account_slug="old-workspace-name"
            ),
        )
        supabase_calls = _route_supabase_engine(
            supabase_engine,
            by_id=("team", {"observations_per_month": 999}),
            by_slug=("free", None),
        )

        result = validator.validate(raw_key)

        assert result.plan == "team"
        assert result.plan_overrides == {"observations_per_month": 999}
        assert len(supabase_calls) == 1
        sql, params = supabase_calls[0]
        # user_id is always bound (even as None) so the membership EXISTS
        # subquery can live in the same SELECT -- see
        # _query_plan_from_supabase. This fixture's key entry carries
        # created_by="user-001", not a UUID, so the membership check itself
        # is skipped (user_id bound as None).
        assert params == {"account_id": ACCOUNT_UUID, "user_id": None}
        assert ACCOUNT_UUID not in sql  # bound, never interpolated

    def test_overrides_json_string_is_parsed(self):
        """Drivers that hand back JSONB as text still yield a dict."""
        entry, raw_key, _ = _make_key_entry()
        validator, main_engine, supabase_engine = _build_validator_with_supabase(
            required_scope="verify"
        )
        _route_main_engine(
            main_engine,
            key_row=_make_db_row([entry], org_id=ACCOUNT_UUID, account_slug="acme"),
        )
        _route_supabase_engine(
            supabase_engine, by_id=("enterprise", '{"max_agents": 42}')
        )

        result = validator.validate(raw_key)

        assert result.plan == "enterprise"
        assert result.plan_overrides == {"max_agents": 42}

    def test_legacy_non_uuid_org_id_falls_back_to_slug_lookup(self):
        """Legacy orgs (org_id='vex') keep working via the slug join."""
        entry, raw_key, _ = _make_key_entry()
        validator, main_engine, supabase_engine = _build_validator_with_supabase(
            required_scope="verify"
        )
        _route_main_engine(
            main_engine,
            key_row=_make_db_row([entry], org_id="vex", account_slug="vex"),
        )
        supabase_calls = _route_supabase_engine(
            supabase_engine, by_id=None, by_slug=("pro", None)
        )

        result = validator.validate(raw_key)

        assert result.plan == "pro"
        # A non-UUID org_id must not be sent to a uuid-typed column at all.
        assert len(supabase_calls) == 1
        assert supabase_calls[0][1] == {"slug": "vex", "user_id": None}

    def test_uuid_org_without_account_row_falls_back_to_slug(self):
        """UUID miss still tries the slug before giving up."""
        entry, raw_key, _ = _make_key_entry()
        validator, main_engine, supabase_engine = _build_validator_with_supabase(
            required_scope="verify"
        )
        _route_main_engine(
            main_engine,
            key_row=_make_db_row([entry], org_id=ACCOUNT_UUID, account_slug="acme"),
        )
        supabase_calls = _route_supabase_engine(
            supabase_engine, by_id=None, by_slug=("starter", None)
        )

        result = validator.validate(raw_key)

        assert result.plan == "starter"
        assert len(supabase_calls) == 2
        assert supabase_calls[0][1] == {"account_id": ACCOUNT_UUID, "user_id": None}
        assert supabase_calls[1][1] == {"slug": "acme", "user_id": None}

    def test_unknown_org_falls_back_to_free(self):
        """Neither join matches -> safe ('free', None) fallback."""
        entry, raw_key, _ = _make_key_entry()
        validator, main_engine, supabase_engine = _build_validator_with_supabase(
            required_scope="verify"
        )
        _route_main_engine(
            main_engine,
            key_row=_make_db_row([entry], org_id=ACCOUNT_UUID, account_slug="ghost"),
        )
        _route_supabase_engine(supabase_engine, by_id=None, by_slug=None)

        result = validator.validate(raw_key)

        assert result.plan == "free"
        assert result.plan_overrides is None

    def test_supabase_error_falls_back_to_free(self):
        """A genuine Supabase failure keeps the existing 'free' posture."""
        entry, raw_key, _ = _make_key_entry()
        validator, main_engine, supabase_engine = _build_validator_with_supabase(
            required_scope="verify"
        )
        _route_main_engine(
            main_engine,
            key_row=_make_db_row([entry], org_id=ACCOUNT_UUID, account_slug="acme"),
        )
        _route_supabase_engine(supabase_engine, error=True)

        result = validator.validate(raw_key)

        assert result.plan == "free"
        assert result.plan_overrides is None

    def test_no_supabase_engine_returns_free(self):
        """Unconfigured Supabase engine short-circuits to the free plan."""
        validator, _ = _build_validator(required_scope="verify")
        assert validator._query_plan_from_supabase("acme", ACCOUNT_UUID) == (
            "free",
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Tests: memory scope is decoupled from the reliability executions table
# ---------------------------------------------------------------------------


class TestMemoryQuotaDecoupling:
    """Memory availability must not ride on the reliability hypertable.

    Memory usage is metered by ``shared.usage`` (``memory_usage_counters``)
    at the MCP/capture layer, so a memory-scoped validation has no business
    running -- let alone failing closed on -- a COUNT over ``executions``.
    """

    MEMORY_SCOPES = ["memory"]
    BOTH_SCOPES = ["ingest", "verify", "read", "memory"]

    def _validate_with(self, *, required_scope, key_scopes, **route_kwargs):
        entry, raw_key, _ = _make_key_entry(scopes=key_scopes)
        validator, engine = _build_validator(required_scope=required_scope)
        executed = _route_main_engine(
            engine, key_row=_make_db_row([entry]), **route_kwargs
        )
        validator._query_plan_from_supabase = MagicMock(
            return_value=("free", None, None)
        )
        return validator, raw_key, executed

    def test_memory_scope_does_not_count_executions(self):
        validator, raw_key, executed = self._validate_with(
            required_scope="memory",
            key_scopes=self.MEMORY_SCOPES,
        )

        result = validator.validate(raw_key)

        assert result.org_id == "test-org"
        assert not any("FROM executions" in sql for sql in executed)

    def test_memory_scope_survives_broken_executions_table(self):
        """A sick reliability hypertable must not 503 a memory request."""
        validator, raw_key, executed = self._validate_with(
            required_scope="memory",
            key_scopes=self.MEMORY_SCOPES,
            usage_error=True,
        )

        result = validator.validate(raw_key)

        assert result.org_id == "test-org"
        assert not any("FROM executions" in sql for sql in executed)

    def test_memory_scope_not_blocked_by_reliability_overage(self):
        """Reliability usage far past the free cap must not gate memory."""
        validator, raw_key, _ = self._validate_with(
            required_scope="memory",
            key_scopes=self.MEMORY_SCOPES,
            usage_count=10_000_000,
        )

        assert validator.validate(raw_key).org_id == "test-org"

    def test_dual_scope_key_still_metered_on_reliability_surface(self):
        """A key with both scopes is unchanged on the ingest surface."""
        validator, raw_key, executed = self._validate_with(
            required_scope="ingest",
            key_scopes=self.BOTH_SCOPES,
            usage_count=1000,  # free plan observations cap
        )

        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key)

        assert exc_info.value.status_code == 429
        assert any("FROM executions" in sql for sql in executed)

    @pytest.mark.parametrize("scope", ["ingest", "verify", "read"])
    def test_reliability_scopes_still_count_executions(self, scope):
        validator, raw_key, executed = self._validate_with(
            required_scope=scope,
            key_scopes=self.BOTH_SCOPES,
            usage_count=1,
        )

        validator.validate(raw_key)

        assert any("FROM executions" in sql for sql in executed)

    @pytest.mark.parametrize("scope", ["ingest", "verify", "read"])
    def test_reliability_scopes_still_fail_closed(self, scope):
        """Regression guard: the 503 fail-closed path is untouched."""
        validator, raw_key, _ = self._validate_with(
            required_scope=scope,
            key_scopes=self.BOTH_SCOPES,
            usage_error=True,
        )

        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key)

        assert exc_info.value.status_code == 503
        assert exc_info.value.retry_after_seconds == 30


# ---------------------------------------------------------------------------
# Key-path identity
#
# Every key entry already stores `created_by` (verified in production: 55/55
# keys, 36-char UUIDs, 14 distinct creators). It has never been propagated to
# the caller, so the engine has had a user identity available on every keyed
# request and thrown it away.
# ---------------------------------------------------------------------------


def test_keyinfo_carries_created_by_as_user_id():
    from shared.auth import KeyInfo

    info = KeyInfo(
        org_id="acme",
        key_id="k1",
        scopes=["memory:write"],
        user_id="11111111-2222-3333-4444-555555555555",
    )
    assert info.user_id == "11111111-2222-3333-4444-555555555555"


def test_keyinfo_user_id_defaults_to_none():
    """Additive: ingestion-api and sync-gateway construct KeyInfo without it."""
    from shared.auth import KeyInfo

    info = KeyInfo(org_id="acme", key_id="k1", scopes=[])
    assert info.user_id is None


def test_cached_key_reads_created_by_from_the_key_entry():
    from shared.auth import _CachedKey

    entry = _CachedKey(
        org_id="acme",
        key_id="k1",
        scopes=[],
        rate_limit_rpm=100,
        expires_at=None,
        revoked=False,
        user_id="11111111-2222-3333-4444-555555555555",
    )
    assert entry.user_id == "11111111-2222-3333-4444-555555555555"


def test_cached_key_user_id_defaults_to_none_for_legacy_entries():
    """A key entry written before created_by existed must not crash the path."""
    from shared.auth import _CachedKey

    entry = _CachedKey(
        org_id="acme",
        key_id="k1",
        scopes=[],
        rate_limit_rpm=100,
        expires_at=None,
        revoked=False,
    )
    assert entry.user_id is None


# ---------------------------------------------------------------------------
# Key-path identity — the producer, end to end.
#
# The tests above construct KeyInfo/_CachedKey directly; they prove the
# dataclasses can carry a user_id but not that the DB-backed path actually
# populates one. These two exercise the real ``_query_db`` -> ``validate``
# pipeline against a JSONB ``api_keys`` element, which is where the field
# would silently stay None if the SELECT ever stopped reading ``created_by``.
# ---------------------------------------------------------------------------


def test_validate_populates_user_id_from_the_jsonb_key_entry():
    """The full producer path: JSONB ``created_by`` reaches ``KeyInfo.user_id``.

    ``_make_key_entry`` stamps every fixture with ``created_by: "user-001"``
    (see the docstring above it), so this is the byte-for-byte end-to-end
    check that ``_query_db`` selects the field out of the element and
    ``validate`` carries it through onto the returned ``KeyInfo`` rather than
    leaving it stuck on the internal ``_CachedKey``.
    """
    entry, raw_key, _ = _make_key_entry()
    validator, engine = _build_validator(required_scope="verify")
    _mock_db_query(engine, _make_db_row([entry]))

    result = validator.validate(raw_key)

    assert result.user_id == "user-001"


def test_validate_leaves_user_id_none_for_a_key_entry_with_no_created_by():
    """Back-compat contract: a key entry predating ``created_by`` must not
    crash the lookup and must resolve to an unattributed (service) key.

    Unlike the legacy-entry unit tests above, this builds the JSONB element
    by hand with no ``created_by`` key at all (not even ``None``), which is
    what a pre-existing production row actually looks like, and drives it
    through ``validate`` rather than constructing ``_CachedKey`` directly.
    """
    raw_key = "ag_live_legacykey1234567890abcdefg"
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    entry = {
        "id": "key-legacy",
        "prefix": raw_key[:12],
        "key_hash": key_hash,
        "name": "Legacy Key",
        "scopes": ["ingest", "verify", "read"],
        "rate_limit_rpm": 1000,
        "expires_at": None,
        "created_at": "2025-01-01T00:00:00Z",
        "last_used_at": None,
        "revoked": False,
        # deliberately no "created_by" key
    }
    validator, engine = _build_validator(required_scope="verify")
    _mock_db_query(engine, _make_db_row([entry]))

    result = validator.validate(raw_key)
    assert result.user_id is None


# ---------------------------------------------------------------------------
# Tests: a key stops working when its owner leaves the workspace
#
# Mirrors app/oauth.py's GrantResolver._lookup, which re-queries
# accounts_memberships on every grant resolution rather than trusting the
# grant row's account_slug alone -- "deliberately redundant", per its own
# comment, because a removed membership must not stay usable indefinitely.
# KeyValidator._query_plan_from_supabase folds the same membership predicate
# into the Supabase plan query it already issues per cache miss.
# ---------------------------------------------------------------------------

OWNER_UUID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


class TestOwnerMembershipGate:
    def test_owner_is_a_member_validates_unchanged(self):
        """Positive control: the suite can't pass vacuously -- a real member
        must still authenticate normally, distinct from every rejection
        test below."""
        entry, raw_key, _ = _make_key_entry(created_by=OWNER_UUID)
        validator, main_engine, supabase_engine = _build_validator_with_supabase(
            required_scope="verify"
        )
        _route_main_engine(main_engine, key_row=_make_db_row([entry]))
        _route_supabase_engine(supabase_engine, by_slug=("free", None, True))

        result = validator.validate(raw_key)

        assert result.org_id == "test-org"
        assert result.user_id == OWNER_UUID

    def test_owner_not_a_member_is_rejected_with_a_distinct_error(self):
        """Removed-member rejection must not be the generic 401 invalid-key
        path -- support needs to tell the two apart."""
        entry, raw_key, _ = _make_key_entry(created_by=OWNER_UUID)
        validator, main_engine, supabase_engine = _build_validator_with_supabase(
            required_scope="verify"
        )
        _route_main_engine(main_engine, key_row=_make_db_row([entry]))
        _route_supabase_engine(supabase_engine, by_slug=("free", None, False))

        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key)

        assert exc_info.value.status_code == 403
        assert exc_info.value.status_code != 401
        assert "member" in exc_info.value.detail.lower()
        assert "invalid api key" not in exc_info.value.detail.lower()

    def test_owner_member_of_a_different_account_is_rejected(self):
        """Adversarial case mirroring OAuth's 'grant repointed at someone
        else's workspace': membership is scoped to the resolved account's
        id, so being a member elsewhere buys nothing here."""
        entry, raw_key, _ = _make_key_entry(created_by=OWNER_UUID)
        validator, main_engine, supabase_engine = _build_validator_with_supabase(
            required_scope="verify"
        )
        _route_main_engine(
            main_engine,
            key_row=_make_db_row([entry], org_id=ACCOUNT_UUID, account_slug="acme"),
        )
        # The account resolves fine (a real plan comes back) but this
        # owner's EXISTS check against *this* account's memberships is
        # false -- they belong to some other workspace instead.
        _route_supabase_engine(supabase_engine, by_id=("pro", None, False))

        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key)

        assert exc_info.value.status_code == 403

    def test_unattributed_key_ignores_the_membership_answer_entirely(self):
        """No user_id -> the membership question isn't asked, and even a
        stray falsy value in that column must never be examined. Proves
        'no membership query issued' at the parameter level: user_id is
        bound as None, so the EXISTS subquery can never match anything by
        construction, not merely by convention."""
        entry, raw_key, _ = _make_key_entry(created_by=None)
        validator, main_engine, supabase_engine = _build_validator_with_supabase(
            required_scope="verify"
        )
        _route_main_engine(main_engine, key_row=_make_db_row([entry]))
        supabase_calls = _route_supabase_engine(
            supabase_engine, by_slug=("free", None, False)
        )

        result = validator.validate(raw_key)

        assert result.org_id == "test-org"
        assert result.user_id is None
        assert supabase_calls[0][1]["user_id"] is None

    def test_supabase_transient_failure_is_retryable_not_an_allow(self):
        """decision 4: a DB blip must not become an open door, and must not
        be indistinguishable from a genuine 'you were removed' 403."""
        entry, raw_key, _ = _make_key_entry(created_by=OWNER_UUID)
        validator, main_engine, supabase_engine = _build_validator_with_supabase(
            required_scope="verify"
        )
        _route_main_engine(main_engine, key_row=_make_db_row([entry]))
        _route_supabase_engine(supabase_engine, error=True)

        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key)

        assert exc_info.value.status_code == 503
        assert exc_info.value.status_code != 403
        assert exc_info.value.retry_after_seconds == 30

    def test_supabase_unconfigured_validates_with_a_warning(self, caplog):
        """decision 3: no Supabase engine configured at all -> skip the
        check and keep today's behaviour, but log once so a misconfigured
        deployment doesn't silently run with the gate disabled forever."""
        entry, raw_key, _ = _make_key_entry(created_by=OWNER_UUID)
        validator, engine = _build_validator(required_scope="verify")
        _mock_db_query(engine, _make_db_row([entry]))

        with caplog.at_level(logging.WARNING, logger="agentguard.auth"):
            result = validator.validate(raw_key)

        assert result.org_id == "test-org"
        assert result.user_id == OWNER_UUID
        assert any(
            "membership" in r.getMessage().lower() and "not" in r.getMessage().lower()
            for r in caplog.records
        )

    def test_within_ttl_the_prior_membership_answer_is_reused(self):
        """The membership answer rides the existing key cache -- no second
        cache, no extra query on a cache hit."""
        entry, raw_key, _ = _make_key_entry(created_by=OWNER_UUID)
        validator, main_engine, supabase_engine = _build_validator_with_supabase(
            required_scope="verify", cache_ttl_s=60.0
        )
        _route_main_engine(main_engine, key_row=_make_db_row([entry]))
        supabase_calls = _route_supabase_engine(
            supabase_engine, by_slug=("free", None, True)
        )

        validator.validate(raw_key)
        validator.validate(raw_key)

        assert len(supabase_calls) == 1

    def test_membership_answer_is_none_when_not_asked_even_if_the_row_says_true(self):
        """Unit-level check of _query_plan_from_supabase's own contract.

        validate() independently gates on entry.user_id is not None before
        ever consulting is_member, so a wrong answer here would not
        currently escape to an authentication decision -- but the method's
        own documented contract ('None means not evaluated, never treat it
        as an answer') must hold on its own, not merely as an accident of
        how its one caller happens to use it today."""
        validator, main_engine, supabase_engine = _build_validator_with_supabase(
            required_scope="verify"
        )
        _route_supabase_engine(supabase_engine, by_slug=("free", None, True))

        _plan, _overrides, is_member = validator._query_plan_from_supabase(
            "acme",
            None,
            None,  # no user_id -> membership not requested
        )

        assert is_member is None

    def test_after_ttl_expiry_a_revoked_membership_is_rejected(self):
        """decision 5: a removed member's key stops working within one
        cache_ttl_s window -- not immediately, not 'eventually'."""
        entry, raw_key, _ = _make_key_entry(created_by=OWNER_UUID)
        validator, main_engine, supabase_engine = _build_validator_with_supabase(
            required_scope="verify", cache_ttl_s=0.05
        )
        _route_main_engine(main_engine, key_row=_make_db_row([entry]))

        _route_supabase_engine(supabase_engine, by_slug=("free", None, True))
        first = validator.validate(raw_key)
        assert first.org_id == "test-org"

        time.sleep(0.1)  # let the cache entry expire
        _route_supabase_engine(supabase_engine, by_slug=("free", None, False))

        with pytest.raises(AuthError) as exc_info:
            validator.validate(raw_key)

        assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# check_credential — authentication only, no metering
# ---------------------------------------------------------------------------
#
# The transport layer calls this in front of a `validate` that will run
# anyway. If it ever acquires `validate`'s side effects, every request is
# charged twice and everyone's effective RPM halves — silently. These pin
# both halves: it must reject dead credentials, and it must touch nothing
# that meters.


def _validator_with_key(monkeypatch, **entry_kwargs):
    """A KeyValidator whose cache is pre-seeded with one key."""
    import hashlib

    from shared.auth import KeyValidator, _CachedKey

    validator = KeyValidator.__new__(KeyValidator)
    validator._required_scope = "memory"

    defaults = dict(
        org_id="org-1",
        key_id="key-1",
        scopes=["memory"],
        rate_limit_rpm=1000,
        expires_at=None,
        revoked=False,
        plan="free",
        plan_overrides=None,
        user_id=None,
        is_member=None,
    )
    defaults.update(entry_kwargs)
    entry = _CachedKey(**defaults)

    key = "ag_live_testkey"
    key_hash = hashlib.sha256(key.encode()).hexdigest()

    monkeypatch.setattr(
        validator, "_cache_get", lambda h: entry if h == key_hash else None
    )
    monkeypatch.setattr(validator, "_cache_put", lambda h, e: None)
    monkeypatch.setattr(
        validator, "_query_db", lambda h: (_ for _ in ()).throw(AssertionError("db"))
    )

    return validator, key


def test_check_credential_accepts_a_live_key(monkeypatch):
    validator, key = _validator_with_key(monkeypatch)

    validator.check_credential(key)  # must not raise


def test_check_credential_rejects_a_revoked_key(monkeypatch):
    from shared.auth import AuthError

    validator, key = _validator_with_key(monkeypatch, revoked=True)

    with pytest.raises(AuthError) as exc:
        validator.check_credential(key)

    assert exc.value.status_code == 401


def test_check_credential_rejects_an_expired_key(monkeypatch):
    from datetime import datetime, timedelta, timezone

    from shared.auth import AuthError

    validator, key = _validator_with_key(
        monkeypatch,
        expires_at=datetime.now(tz=timezone.utc) - timedelta(days=1),
    )

    with pytest.raises(AuthError) as exc:
        validator.check_credential(key)

    assert exc.value.status_code == 401


def test_check_credential_rejects_an_unknown_key(monkeypatch):
    from shared.auth import AuthError, KeyValidator

    validator = KeyValidator.__new__(KeyValidator)
    validator._required_scope = "memory"
    monkeypatch.setattr(validator, "_cache_get", lambda h: None)
    monkeypatch.setattr(validator, "_query_db", lambda h: None)

    with pytest.raises(AuthError) as exc:
        validator.check_credential("ag_live_nosuchkey")

    assert exc.value.status_code == 401


def test_check_credential_does_not_meter(monkeypatch):
    """The whole reason this method exists instead of reusing `validate`."""
    validator, key = _validator_with_key(monkeypatch)

    called: list[str] = []
    for name in (
        "_check_rate_limit",
        "_check_quota",
        "_check_agent_limit",
        "_track_usage",
    ):
        monkeypatch.setattr(validator, name, lambda *a, _n=name, **k: called.append(_n))

    validator.check_credential(key)

    assert called == [], f"check_credential metered: {called}"


def test_validate_still_meters(monkeypatch):
    """Positive control: if metering moved out of `validate` too, the cap and
    quota would stop being enforced anywhere and the test above would still
    pass."""
    validator, key = _validator_with_key(monkeypatch)

    called: list[str] = []
    for name in (
        "_check_rate_limit",
        "_check_quota",
        "_check_agent_limit",
        "_track_usage",
    ):
        monkeypatch.setattr(validator, name, lambda *a, _n=name, **k: called.append(_n))

    validator.validate(key)

    assert set(called) == {
        "_check_rate_limit",
        "_check_quota",
        "_check_agent_limit",
        "_track_usage",
    }
