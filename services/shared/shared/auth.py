"""API key validation for Vex backend services.

Provides ``KeyValidator``, a shared authentication module used by the
sync gateway and ingestion API to validate incoming API keys against
the ``organizations.api_keys`` JSONB column in TimescaleDB.

Features:
- SHA-256 hash-based key lookup (keys are never stored in plaintext)
- In-memory cache with configurable TTL (default 60 s)
- Per-key scope enforcement (``ingest``, ``verify``, ``read``,
  ``memory`` for Klio brain tools)
- Per-key rate limiting (sliding window, requests per minute)
- Batched ``last_used_at`` updates (avoids write pressure on hot path)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from shared.engine_bounds import engine_bounds
from shared.plan_limits import PlanConfig, get_plan_config

logger = logging.getLogger("agentguard.auth")

VALID_SCOPES = frozenset({"ingest", "verify", "read", "memory"})

# Scopes whose monthly usage is metered by the reliability ``executions``
# hypertable.  The memory product meters itself through
# ``shared.usage`` (``memory_usage_counters``, enforced at the MCP/capture
# layer), so memory-scoped requests must neither pay for nor fail closed on a
# COUNT over a reliability table their tenant may never write to.
RELIABILITY_SCOPES = frozenset({"ingest", "verify", "read"})


# --- agent_id, the one unbounded client string on the write path ------
#
# ``agent_id`` is client input: it comes verbatim from the ``X-Vex-Agent``
# header, or on the OAuth path is derived from ``clientInfo.name``, which the
# client also supplies. It ends up in ``session_memories.agent_id``, declared
# ``String(128)`` since migration 001. Nothing bounded it. Executed against
# Postgres 16:
#
#   X-Vex-Agent header, 500 chars           -> agent_id len 500
#   OAuth slug path, clientInfo.name 500 ch -> agent_id len 515  (the wrapper
#                                              "oauth:<user8>/" adds 15)
#   INSERT -> (psycopg2.errors.StringDataRightTruncation)
#             value too long for type character varying(128)
#
# So an over-long header did not degrade, it turned every write into a 500.

#: Matches ``session_memories.agent_id``, the narrowest column this value
#: reaches. ``agents.agent_id`` and ``tool_calls.agent_id`` are also 128.
MAX_AGENT_ID_LEN = 128

#: Hex characters of SHA-256 kept when an over-long id has to be shortened.
#: Truncating alone would merge two distinct principals into one — and two
#: agents sharing an ``agent_id`` share an ``agent``-scoped memory bucket,
#: which is worse than the error it replaces. 12 hex chars is 48 bits; these
#: are not adversarial-collision-resistant identifiers, they are a
#: disambiguator on a value that is already tenant-scoped by ``org_id``.
_AGENT_ID_DIGEST_LEN = 12

_AGENT_ID_SEPARATOR = "~"


def normalize_agent_id(value: str | None, *, default: str) -> str:
    """Bound a client-supplied ``agent_id`` to what the column can hold.

    Args:
        value: The raw ``X-Vex-Agent`` header or derived OAuth agent id.
        default: The agent principal to use when the value is absent or blank.

    Returns:
        ``value`` stripped, unchanged when it already fits; otherwise a
        deterministic ``<prefix>~<sha256-prefix>`` of at most
        :data:`MAX_AGENT_ID_LEN` characters. Absent or blank yields ``default``.

    Deterministic on purpose: dedup and agent-scoped recall both key on this
    string, so the same header must map to the same principal on every request
    and across restarts.
    """
    if value is None:
        return default

    cleaned = value.strip()
    if not cleaned:
        return default

    if len(cleaned) <= MAX_AGENT_ID_LEN:
        return cleaned

    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:_AGENT_ID_DIGEST_LEN]
    keep = MAX_AGENT_ID_LEN - len(_AGENT_ID_SEPARATOR) - _AGENT_ID_DIGEST_LEN
    truncated = f"{cleaned[:keep]}{_AGENT_ID_SEPARATOR}{digest}"
    logger.warning(
        "agent_id of %d characters exceeds the %d-character column; bounded to %r",
        len(cleaned),
        MAX_AGENT_ID_LEN,
        truncated,
    )
    return truncated


def _is_uuid(value: str | None) -> bool:
    """Return True when ``value`` is a well-formed UUID string.

    Guards the ``accounts.id`` join: ``org_id`` is a free-form ``varchar(64)``
    and legacy rows hold slug-like values, which Postgres would reject when
    compared against a ``uuid`` column.
    """
    if not value:
        return False
    try:
        uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return False
    return True


@dataclass
class KeyInfo:
    """Validated API key information returned to the caller."""

    org_id: str
    key_id: str
    scopes: list[str]
    plan: str = "free"
    # Per-account plan overrides (from ``accounts.vex_plan_overrides``), if any.
    # Additive with a ``None`` default so existing callers that construct a
    # ``KeyInfo`` without it (ingestion-api / sync-gateway) stay valid. Callers
    # that enforce per-plan quotas must pass this into ``get_plan_config`` so a
    # custom cap is honored at the boundary, exactly as the retention sweep does.
    plan_overrides: dict | None = None
    # The Supabase user UUID of whoever created this key, copied from the key
    # entry's ``created_by``. Every key already carries it; it simply never
    # reached the caller. Optional because ingestion-api and sync-gateway
    # construct KeyInfo directly, and because a key written before the field
    # existed must not crash the auth path.
    #
    # This is the ONLY identity source on the key path. Never derive it from
    # X-Vex-Agent, which the client declares and can therefore forge.
    user_id: str | None = None


@dataclass
class _CachedKey:
    """Internal cached representation of a validated key."""

    org_id: str
    key_id: str
    scopes: list[str]
    rate_limit_rpm: int
    expires_at: datetime | None
    revoked: bool
    plan: str = "free"
    plan_overrides: dict | None = None
    user_id: str | None = None
    cached_at: float = field(default_factory=time.monotonic)


class KeyValidator:
    """Validates API keys against the organizations table.

    Each backend service creates a ``KeyValidator`` instance at startup,
    specifying the required scope for that service.  The validator
    maintains an in-memory cache to avoid hitting the database on every
    request.

    Args:
        database_url: PostgreSQL connection string.
        required_scope: The scope that incoming keys must have
            (e.g. ``"ingest"`` or ``"verify"``).
        cache_ttl_s: How long cached key entries remain valid (seconds).
        flush_interval_s: How often ``last_used_at`` is flushed to the DB.
    """

    def __init__(
        self,
        database_url: str,
        required_scope: str,
        supabase_database_url: str | None = None,
        cache_ttl_s: float = 60.0,
        flush_interval_s: float = 60.0,
    ) -> None:
        if required_scope not in VALID_SCOPES:
            raise ValueError(
                f"Invalid scope {required_scope!r}; must be one of {VALID_SCOPES}"
            )

        # Both engines are on the request path, so an unreachable database used
        # to present as a hung request rather than an error: `pool_pre_ping`
        # detects a dead connection but does not bound how long opening one may
        # take. See :mod:`shared.engine_bounds`.
        self._engine: Engine = create_engine(
            database_url,
            pool_size=2,
            max_overflow=3,
            pool_pre_ping=True,
            **engine_bounds(),
        )
        self._supabase_engine: Engine | None = None
        if supabase_database_url:
            self._supabase_engine = create_engine(
                supabase_database_url,
                pool_size=1,
                max_overflow=2,
                pool_pre_ping=True,
                **engine_bounds(),
            )
        self._required_scope = required_scope
        self._cache_ttl_s = cache_ttl_s
        self._flush_interval_s = flush_interval_s

        # key_hash -> _CachedKey
        self._cache: dict[str, _CachedKey] = {}
        self._cache_lock = Lock()

        # Rate limit counters: key_id -> deque of request timestamps
        self._rate_counters: dict[str, deque[float]] = {}
        self._rate_lock = Lock()

        # Batched last_used_at updates: key_id -> timestamp
        self._usage_buffer: dict[str, str] = {}
        self._usage_lock = Lock()
        self._last_flush = time.monotonic()

        # Known agent IDs per org (avoids DB query on every request)
        self._known_agents: dict[str, set] = {}
        self._known_agents_lock = Lock()

        # Quota usage cache: org_id -> (usage_count, cached_at)
        self._quota_cache: dict[str, tuple[int, float]] = {}
        self._quota_cache_lock = Lock()
        self._quota_cache_ttl_s = 30.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, api_key: str, agent_id: str | None = None) -> KeyInfo:
        """Validate an API key and return org information.

        This method is synchronous.  When used as a FastAPI dependency
        in an async route, FastAPI automatically runs it in a thread
        pool so it does not block the event loop.

        Args:
            api_key: The raw API key from the ``X-Vex-Key`` header
                (falls back to ``X-AgentGuard-Key`` for backward compatibility).

        Returns:
            A ``KeyInfo`` with the resolved org_id, key_id, and scopes.

        Raises:
            AuthError: On any validation failure (invalid, revoked,
                expired, wrong scope, rate limited).
        """
        key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()

        # 1. Cache lookup
        entry = self._cache_get(key_hash)

        # 2. Cache miss -> DB query
        if entry is None:
            entry = self._query_db(key_hash)
            if entry is None:
                raise AuthError(401, "Invalid API key")
            self._cache_put(key_hash, entry)

        # 3. Check revoked
        if entry.revoked:
            raise AuthError(401, "API key has been revoked")

        # 4. Check expired
        if entry.expires_at is not None:
            now = datetime.now(tz=timezone.utc)
            if now > entry.expires_at:
                raise AuthError(401, "API key has expired")

        # 5. Check scope
        if self._required_scope not in entry.scopes:
            raise AuthError(
                403,
                f"API key missing required scope: {self._required_scope}",
            )

        # 6. Check rate limit
        self._check_rate_limit(entry)

        # 7. Check monthly quota
        self._check_quota(entry)

        # 8. Check agent limit
        self._check_agent_limit(entry, agent_id)

        # 9. Track usage (batched)
        self._track_usage(entry.key_id)

        return KeyInfo(
            org_id=entry.org_id,
            key_id=entry.key_id,
            scopes=entry.scopes,
            plan=entry.plan,
            plan_overrides=entry.plan_overrides,
            user_id=entry.user_id,
        )

    def flush_usage(self) -> None:
        """Flush buffered ``last_used_at`` updates to the database.

        Called automatically by ``_track_usage`` when the flush interval
        elapses.  Can also be called explicitly at shutdown.
        """
        with self._usage_lock:
            if not self._usage_buffer:
                return
            buffer = dict(self._usage_buffer)
            self._usage_buffer.clear()
            self._last_flush = time.monotonic()

        try:
            with self._engine.connect() as conn:
                for key_id, used_at in buffer.items():
                    conn.execute(
                        text(
                            """
                            UPDATE organizations
                            SET api_keys = (
                                SELECT jsonb_agg(
                                    CASE
                                        WHEN elem->>'id' = :key_id
                                        THEN jsonb_set(elem, '{last_used_at}', to_jsonb(CAST(:used_at AS text)))
                                        ELSE elem
                                    END
                                )
                                FROM jsonb_array_elements(api_keys) AS elem
                            )
                            WHERE api_keys @> CAST(:filter AS jsonb)
                            """
                        ),
                        {
                            "key_id": key_id,
                            "used_at": used_at,
                            "filter": json.dumps([{"id": key_id}]),
                        },
                    )
                conn.commit()
        except Exception:
            logger.warning("Failed to flush last_used_at updates", exc_info=True)

    def close(self) -> None:
        """Flush pending usage and dispose of the DB engine(s)."""
        self.flush_usage()
        self._engine.dispose()
        if self._supabase_engine:
            self._supabase_engine.dispose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_plan_from_supabase(
        self,
        account_slug: str,
        org_id: str | None = None,
    ) -> tuple:
        """Query plan data from the Supabase ``accounts`` table.

        The join prefers the **immutable** account UUID (``accounts.id``),
        which the dashboard writes into ``organizations.org_id`` when it
        provisions an org.  Account slugs are user-editable: renaming a
        workspace leaves the engine's cached ``organizations.account_slug``
        stale, and a slug-only join would then miss and silently downgrade a
        paying tenant to ``free`` until a dashboard page load re-heals the row.

        Legacy orgs predate that convention and carry a non-UUID ``org_id``
        (migration 003 backfilled ``account_slug = org_id`` for them), so the
        mutable-slug join is retained as a fallback.

        Returns a ``(plan, plan_overrides)`` tuple.  Falls back to
        ``("free", None)`` if the Supabase engine is not configured, neither
        join matches, or any error occurs.
        """
        if self._supabase_engine is None:
            return ("free", None)
        try:
            with self._supabase_engine.connect() as conn:
                row = None
                if _is_uuid(org_id):
                    row = conn.execute(
                        text(
                            "SELECT vex_plan, vex_plan_overrides FROM accounts "
                            "WHERE id = CAST(:account_id AS uuid) "
                            "AND is_personal_account = false LIMIT 1"
                        ),
                        {"account_id": org_id},
                    ).fetchone()

                if row is None:
                    row = conn.execute(
                        text(
                            "SELECT vex_plan, vex_plan_overrides FROM accounts "
                            "WHERE slug = :slug AND is_personal_account = false LIMIT 1"
                        ),
                        {"slug": account_slug},
                    ).fetchone()

                if row is None:
                    return ("free", None)
                plan = row[0] or "free"
                overrides = (
                    row[1]
                    if isinstance(row[1], dict)
                    else (json.loads(row[1]) if row[1] else None)
                )
                return (plan, overrides)
        except Exception:
            logger.warning(
                "Failed to query plan from Supabase, defaulting to free",
                exc_info=True,
            )
            return ("free", None)

    def _cache_get(self, key_hash: str) -> _CachedKey | None:
        with self._cache_lock:
            entry = self._cache.get(key_hash)
            if entry is None:
                return None
            if (time.monotonic() - entry.cached_at) > self._cache_ttl_s:
                del self._cache[key_hash]
                return None
            return entry

    def _cache_put(self, key_hash: str, entry: _CachedKey) -> None:
        with self._cache_lock:
            self._cache[key_hash] = entry

    def _query_db(self, key_hash: str) -> _CachedKey | None:
        """Query the organizations table for a matching key hash."""
        try:
            with self._engine.connect() as conn:
                result = conn.execute(
                    text(
                        """
                        SELECT org_id, api_keys, account_slug
                        FROM organizations
                        WHERE api_keys @> CAST(:filter AS jsonb)
                        """
                    ),
                    {"filter": json.dumps([{"key_hash": key_hash}])},
                )
                row = result.fetchone()
        except Exception:
            logger.error("Database error during key lookup", exc_info=True)
            raise AuthError(500, "Internal authentication error") from None

        if row is None:
            return None

        org_id = row[0]
        api_keys = row[1] if isinstance(row[1], list) else json.loads(row[1])
        account_slug = row[2] if row[2] else org_id
        plan_val, plan_overrides_val = self._query_plan_from_supabase(
            account_slug, org_id
        )
        plan_config = get_plan_config(plan_val, plan_overrides_val)

        # Find the matching key entry in the JSONB array
        for key_entry in api_keys:
            if key_entry.get("key_hash") == key_hash:
                expires_at = None
                if key_entry.get("expires_at"):
                    expires_at = datetime.fromisoformat(
                        key_entry["expires_at"].replace("Z", "+00:00")
                    )

                per_key_rpm = key_entry.get("rate_limit_rpm", 1000)
                effective_rpm = min(per_key_rpm, plan_config.max_rpm)

                return _CachedKey(
                    org_id=org_id,
                    key_id=key_entry["id"],
                    scopes=key_entry.get("scopes", []),
                    rate_limit_rpm=effective_rpm,
                    expires_at=expires_at,
                    revoked=key_entry.get("revoked", False),
                    plan=plan_val,
                    plan_overrides=plan_overrides_val,
                    user_id=key_entry.get("created_by"),
                )

        return None

    def _check_agent_limit(self, entry: _CachedKey, agent_id: str | None) -> None:
        """Enforce per-plan agent limit."""
        if agent_id is None:
            return

        plan_config = get_plan_config(entry.plan, entry.plan_overrides)
        if plan_config.max_agents == -1:
            return

        with self._known_agents_lock:
            known = self._known_agents.get(entry.org_id)
            if known and agent_id in known:
                return  # Already known, skip DB check

        try:
            with self._engine.connect() as conn:
                agent_count = (
                    conn.execute(
                        text(
                            "SELECT COUNT(DISTINCT agent_id) FROM agents WHERE org_id = :org_id"
                        ),
                        {"org_id": entry.org_id},
                    ).scalar()
                    or 0
                )

                exists = conn.execute(
                    text(
                        "SELECT 1 FROM agents WHERE org_id = :org_id AND agent_id = :agent_id"
                    ),
                    {"org_id": entry.org_id, "agent_id": agent_id},
                ).fetchone()
        except Exception:
            logger.warning(
                "Failed to check agent limit, allowing request", exc_info=True
            )
            return

        if exists:
            with self._known_agents_lock:
                if entry.org_id not in self._known_agents:
                    self._known_agents[entry.org_id] = set()
                self._known_agents[entry.org_id].add(agent_id)
            return

        if agent_count >= plan_config.max_agents:
            raise AuthError(
                403,
                f"Agent limit reached ({agent_count}/{plan_config.max_agents}). "
                f"Upgrade your plan at https://app.tryvex.dev",
            )

    def _check_quota(self, entry: _CachedKey) -> None:
        """Enforce the monthly reliability quota for this validator's scope.

        Uses direct queries against the ``executions`` hypertable,
        filtered by scope:
        - ``verify`` scope → counts rows where ``confidence IS NOT NULL``
        - ``ingest``/``read`` scope → counts all rows
          (observations + verifications)

        ``memory`` is exempt: each service builds its own ``KeyValidator``
        with a fixed ``required_scope``, so a memory-scoped validator only
        ever serves memory traffic (the MCP server) and that traffic is
        metered by ``shared.usage``.  A key that also carries ``ingest`` /
        ``verify`` / ``read`` is still fully metered here whenever it is used
        against a reliability service, whose validator carries that scope.

        Fail-closed: if the query errors, the request is denied with 503.
        Results are cached per-org for 30 s to reduce DB pressure.
        """
        if self._required_scope not in RELIABILITY_SCOPES:
            return

        plan_config = get_plan_config(entry.plan, entry.plan_overrides)

        if self._required_scope == "verify":
            monthly_limit = plan_config.verifications_per_month
        else:
            monthly_limit = plan_config.observations_per_month

        # Unlimited plan — skip check entirely
        if monthly_limit < 0:
            return

        # Check quota cache
        cache_key = f"{entry.org_id}:{self._required_scope}"
        now = time.monotonic()
        with self._quota_cache_lock:
            cached = self._quota_cache.get(cache_key)
            if cached is not None:
                cached_usage, cached_at = cached
                if (now - cached_at) < self._quota_cache_ttl_s:
                    current_usage = cached_usage
                    self._enforce_quota_limit(
                        current_usage,
                        monthly_limit,
                        plan_config,
                        entry,
                    )
                    return

        # Cache miss or expired — query the DB
        if self._required_scope == "verify":
            query = """
                SELECT COUNT(*)
                FROM executions
                WHERE org_id = :org_id
                  AND timestamp >= date_trunc('month', NOW())
                  AND confidence IS NOT NULL
            """
        else:
            query = """
                SELECT COUNT(*)
                FROM executions
                WHERE org_id = :org_id
                  AND timestamp >= date_trunc('month', NOW())
            """

        try:
            with self._engine.connect() as conn:
                result = conn.execute(text(query), {"org_id": entry.org_id})
                current_usage = result.scalar() or 0
        except Exception:
            logger.error(
                "Quota check failed for org=%s scope=%s — denying request (fail-closed)",
                entry.org_id,
                self._required_scope,
                exc_info=True,
            )
            raise AuthError(
                503,
                "Quota check temporarily unavailable. Please retry shortly.",
                retry_after_seconds=30,
            ) from None

        # Update cache
        with self._quota_cache_lock:
            self._quota_cache[cache_key] = (current_usage, time.monotonic())

        self._enforce_quota_limit(current_usage, monthly_limit, plan_config, entry)

    def _enforce_quota_limit(
        self,
        current_usage: int,
        monthly_limit: int,
        plan_config: "PlanConfig",
        entry: _CachedKey,
    ) -> None:
        """Raise AuthError(429) if usage exceeds the monthly limit."""
        if current_usage >= monthly_limit:
            if not plan_config.overage_allowed:
                raise AuthError(
                    429,
                    f"Monthly quota exceeded ({current_usage}/{monthly_limit}). "
                    f"Upgrade your plan at https://app.tryvex.dev",
                    retry_after_seconds=3600,
                )
            logger.info(
                "Overage: org=%s plan=%s usage=%d limit=%d",
                entry.org_id,
                entry.plan,
                current_usage,
                monthly_limit,
            )

    def _check_rate_limit(self, entry: _CachedKey) -> None:
        """Enforce per-key sliding window rate limit."""
        now = time.monotonic()
        window = 60.0  # 1 minute

        with self._rate_lock:
            counter = self._rate_counters.get(entry.key_id)
            if counter is None:
                counter = deque()
                self._rate_counters[entry.key_id] = counter

            # Remove timestamps outside the window
            while counter and (now - counter[0]) > window:
                counter.popleft()

            if len(counter) >= entry.rate_limit_rpm:
                oldest = counter[0]
                retry_after = int(window - (now - oldest)) + 1
                raise AuthError(
                    429,
                    "Rate limit exceeded",
                    retry_after_seconds=max(retry_after, 1),
                )

            counter.append(now)

    def _track_usage(self, key_id: str) -> None:
        """Buffer a last_used_at timestamp and flush periodically."""
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        with self._usage_lock:
            self._usage_buffer[key_id] = now_iso

        # Check if it's time to flush
        if (time.monotonic() - self._last_flush) > self._flush_interval_s:
            self.flush_usage()


class AuthError(Exception):
    """Raised when API key validation fails.

    Carries the HTTP status code and detail message so that FastAPI
    route handlers (or a middleware) can convert it to the appropriate
    HTTP response.
    """

    def __init__(
        self,
        status_code: int,
        detail: str,
        retry_after_seconds: int | None = None,
    ) -> None:
        self.status_code = status_code
        self.detail = detail
        self.retry_after_seconds = retry_after_seconds
        super().__init__(detail)
