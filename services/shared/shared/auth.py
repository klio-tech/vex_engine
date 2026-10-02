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
- Owner-membership re-check for attributed keys (``KeyInfo.user_id``):
  every validation confirms the key's creator is still a member of the
  workspace, folded into the existing Supabase plan lookup so it costs no
  extra round-trip. A user removed from a workspace has their key stop
  working within one ``cache_ttl_s`` window, not "whenever someone
  remembers to revoke it." Unattributed keys (no ``created_by``, the
  documented back-compat case) and validators with no Supabase engine
  configured are unaffected -- see ``KeyValidator._query_plan_from_supabase``.
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
from shared.token_crypto import hash_token

logger = logging.getLogger("agentguard.auth")

VALID_SCOPES = frozenset({"ingest", "verify", "read", "memory"})

# Scopes whose monthly usage is metered by the reliability ``executions``
# hypertable.  The memory product meters itself through
# ``shared.usage`` (``memory_usage_counters``, enforced at the MCP/capture
# layer), so memory-scoped requests must neither pay for nor fail closed on a
# COUNT over a reliability table their tenant may never write to.
RELIABILITY_SCOPES = frozenset({"ingest", "verify", "read"})

#: Scopes granted to a workspace personal access token. A PAT is minted in
#: the workspace UI for the memory product; it deliberately never carries
#: the reliability scopes (``ingest`` / ``verify``), which stay engine-key
#: territory.
PAT_SCOPES = ("memory", "read")

#: Workspace DB: the PAT row itself. Since 2026-09-02 the workspace stores
#: only the SHA-256 of a token (``token_hash``; raw prefixes ``klio_pat_`` /
#: ``klio_bot_``, legacy ``plane_api_`` backfilled), so the lookup hashes the
#: presented value exactly like engine keys are matched. The hash is a
#: cross-repo contract pinned by ``tests/test_token_crypto.py`` — see
#: :mod:`shared.token_crypto`.
_PAT_TOKEN_SQL = text(
    """
    SELECT id, user_id, workspace_id, is_active, expired_at, deleted_at, is_service
    FROM api_tokens
    WHERE token_hash = :token_hash
    LIMIT 1
    """
)

#: Workspace DB: the living workspace a workspace-bound token belongs to.
_PAT_WORKSPACE_SQL = text(
    """
    SELECT id, slug, is_personal, owner_id
    FROM workspaces
    WHERE id = CAST(:ws AS uuid)
      AND deleted_at IS NULL
    LIMIT 1
    """
)

#: ``workspaces.kind`` of a team server (the workspace's
#: ``WORKSPACE_KIND_TEAM``). A community is a public room rather than the
#: caller's own team, so it is never a default destination for their memory.
_PAT_TEAM_SERVER_KIND = "team"

#: Two rows are enough to tell "exactly one" from "several"; which of the
#: several is never asked, because several means the caller has to choose.
_PAT_TEAM_SERVER_PROBE_LIMIT = 2

#: Workspace DB: the living team servers a workspace-less token's user is an
#: active member of. Same liveness predicates as the membership guard, so a
#: server chosen here always passes it. ``EXISTS`` rather than a join: one
#: server must be one row however many membership rows point at it.
#: Schema-qualified so the answer to "which server" can never come from a
#: same-named relation earlier on the search path.
_PAT_TEAM_SERVERS_SQL = text(
    """
    SELECT w.id, w.slug, w.is_personal, w.owner_id
    FROM public.workspaces w
    WHERE w.deleted_at IS NULL
      AND w.is_personal = false
      AND w.kind = :kind
      AND EXISTS (
          SELECT 1 FROM public.workspace_members m
          WHERE m.workspace_id = w.id
            AND m.member_id = CAST(:uid AS uuid)
            AND m.is_active = true
            AND m.deleted_at IS NULL
      )
    ORDER BY w.id
    LIMIT :probe_limit
    """
)

#: Workspace DB: the fallback for a workspace-less (user-level) token whose
#: user has no single unambiguous team server — their living personal
#: workspace.
_PAT_PERSONAL_WORKSPACE_SQL = text(
    """
    SELECT id, slug, is_personal, owner_id
    FROM workspaces
    WHERE owner_id = CAST(:uid AS uuid)
      AND is_personal = true
      AND deleted_at IS NULL
    LIMIT 1
    """
)

#: Workspace DB: best-effort usage stamp (see ``_touch_pat_last_used``).
_PAT_TOUCH_SQL = text("UPDATE api_tokens SET last_used = now() WHERE id = :id")

#: Engine DB: the slug the org-plan lookup keys on, read back after ensure.
_PAT_ORG_SLUG_SQL = text(
    "SELECT account_slug FROM organizations WHERE org_id = :org_id LIMIT 1"
)


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
    #: True only for workspace service/bot tokens (api_tokens.is_service).
    #: Gates capabilities reserved for non-human publishers — e.g. the chat
    #: bridge's explicit org-scope captures. Engine org keys are never
    #: service tokens.
    is_service: bool = False


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
    is_service: bool = False
    # Whether ``user_id`` is still a member of the workspace, as of the last
    # ``organizations``/Supabase query (see ``KeyValidator._query_db``).
    # ``None`` means the check was not evaluated at all — ``user_id`` is
    # ``None`` (unattributed key, the documented back-compat contract) or
    # Supabase is not configured. ``None`` must never be treated as "not a
    # member"; only an explicit ``False`` blocks the key. Rides the same
    # cache TTL as everything else on this dataclass, so a revoked
    # membership is re-checked, and takes effect, within ``cache_ttl_s``.
    is_member: bool | None = None
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
        workspace_database_url: str | None = None,
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
        self._workspace_engine: Engine | None = None
        if workspace_database_url:
            self._workspace_engine = create_engine(
                workspace_database_url,
                pool_size=2,
                max_overflow=2,
                pool_pre_ping=True,
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

        # Logged once (not per-request) the first time an attributed key is
        # validated with no Supabase engine configured, so the membership
        # check is silently skipped. See ``_query_plan_from_supabase``.
        self._warned_supabase_unconfigured = False
        self._warned_workspace_unconfigured = False
        self._warn_lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _live_entry(self, api_key: str) -> "_CachedKey":
        """Resolve a key to its cache entry, rejecting dead credentials.

        AUTHENTICATION ONLY — "is this a real, live key?". Every failure here
        is a 401, meaning the caller should present a different credential.
        Deliberately free of the metering in ``validate`` (rate limit, quota,
        agent limit, usage tracking) so it can be called from a layer in front
        of a ``validate`` that will run anyway; see ``check_credential``.

        Populates the cache on a miss, so a caller in front of ``validate``
        warms it rather than doubling database load.
        """
        key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()

        # 1. Cache lookup
        entry = self._cache_get(key_hash)

        # 2. Cache miss -> DB query. Engine keys FIRST, always; the
        # workspace-PAT arm runs only on an engine-keys miss, so an engine
        # key can never be shadowed by a workspace token that happens to
        # hold the same string.
        if entry is None:
            entry = self._query_db(key_hash)
            if entry is None:
                entry = self._query_workspace_pat(api_key)
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

        return entry

    def check_credential(self, api_key: str) -> None:
        """Raise ``AuthError(401)`` unless this key exists and is still live.

        Exists so the TRANSPORT layer can answer a stale credential with a
        real 401 instead of letting it through to fail later. Without that,
        an MCP client holding an expired or revoked key sees a healthy
        handshake, a full tool list, and an error only in the body of a 200 —
        so it never learns to re-authenticate, and degrades into a connector
        that looks connected and does nothing.

        Calls nothing that meters: ``validate`` still owns rate limiting,
        quota, agent limits and usage tracking, and running those twice per
        request would halve every caller's effective RPM.
        """
        self._live_entry(api_key)

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
        entry = self._live_entry(api_key)

        # 4.5. Check the owner is still a member of the workspace.
        #
        # Mirrors the OAuth grant resolver's "deliberately redundant"
        # re-check in oauth.py: a key's owner can be removed from the
        # workspace without the key itself ever being revoked, and without
        # this check the key would keep working indefinitely. ``is_member``
        # is only ever ``False`` when the check actually ran and the
        # membership row was absent -- ``None`` (unattributed key, or
        # Supabase not configured) always skips this branch, per the
        # back-compat contract on ``KeyInfo.user_id``.
        #
        # This rides the same cache as the rest of ``entry``, so a removed
        # membership stops the key within one ``cache_ttl_s`` window (a
        # cache hit re-checks ``entry.is_member`` on every call; a cache
        # miss re-queries it fresh via ``_query_db``).
        if entry.user_id is not None and entry.is_member is False:
            # Log with full context: this refusal reads identically for a
            # genuinely removed member and for a transient (stale cache
            # entry, a validation that raced a deploy window). Omar's first
            # connect hit the transient shape and the message below sent two
            # people debugging workspace membership that was never broken --
            # without this line there was nothing to pinpoint it afterwards.
            logger.warning(
                "membership guard refused key %s: owner %s is not an active "
                "member of org %s (cached answer, ttl %ss)",
                entry.key_id,
                entry.user_id,
                entry.org_id,
                self._cache_ttl_s,
            )
            raise AuthError(
                403,
                "This key's owner is not currently an active member of the "
                "workspace this key belongs to. If they were just added or "
                "this seems wrong, retry in a minute (membership answers are "
                "cached briefly) and restart your agent session; otherwise "
                "ask a workspace admin to re-add them or issue a new key.",
            )

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
            is_service=entry.is_service,
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
        if self._workspace_engine:
            self._workspace_engine.dispose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_membership_from_workspace(
        self, workspace_id: str, user_id: str | None
    ) -> bool | None:
        """Is the key's owner an active member of the org's linked workspace?

        ``None`` means "could not check, skip the guard": an unattributed key
        (no uuid ``created_by``) or an unconfigured workspace engine -- the
        Railway dark spare has no ``WORKSPACE_DATABASE_URL``, and a linked
        org's key must not brick there. Same fail-open contract the Supabase
        arm documents for its own unconfigured state.

        A query FAILURE is different from an absent configuration: the
        engine is configured and cannot answer, and treating that as
        "member" would turn every workspace-DB outage into an authz bypass.
        503, mirroring ``_query_plan_from_supabase``'s membership contract.
        """
        if user_id is None or not _is_uuid(user_id):
            return None
        if self._workspace_engine is None:
            self._warn_workspace_unconfigured_once(workspace_id)
            return None
        try:
            with self._workspace_engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT EXISTS ("
                        "SELECT 1 FROM workspace_members m "
                        "JOIN workspaces w ON w.id = m.workspace_id "
                        "WHERE w.id = CAST(:ws AS uuid) "
                        "AND m.member_id = CAST(:uid AS uuid) "
                        "AND m.is_active = true "
                        "AND m.deleted_at IS NULL "
                        "AND w.deleted_at IS NULL)"
                    ),
                    {"ws": workspace_id, "uid": user_id},
                ).fetchone()
            return bool(row[0]) if row is not None else False
        except Exception:
            logger.error(
                "workspace membership check failed for workspace %s",
                workspace_id,
                exc_info=True,
            )
            raise AuthError(
                503, "Authentication backend temporarily unavailable"
            ) from None

    def _warn_workspace_unconfigured_once(self, workspace_id: str) -> None:
        with self._warn_lock:
            if self._warned_workspace_unconfigured:
                return
            self._warned_workspace_unconfigured = True
        logger.warning(
            "org linked to workspace %s but WORKSPACE_DATABASE_URL is not "
            "configured; the key-owner membership guard is skipped for "
            "workspace-linked orgs on this process",
            workspace_id,
        )

    def _query_plan_from_supabase(
        self,
        account_slug: str,
        org_id: str | None = None,
        user_id: str | None = None,
    ) -> tuple:
        """Query plan data — and, when ``user_id`` is given, membership — from Supabase.

        The join prefers the **immutable** account UUID (``accounts.id``),
        which the dashboard writes into ``organizations.org_id`` when it
        provisions an org.  Account slugs are user-editable: renaming a
        workspace leaves the engine's cached ``organizations.account_slug``
        stale, and a slug-only join would then miss and silently downgrade a
        paying tenant to ``free`` until a dashboard page load re-heals the row.

        Legacy orgs predate that convention and carry a non-UUID ``org_id``
        (migration 003 backfilled ``account_slug = org_id`` for them), so the
        mutable-slug join is retained as a fallback.

        Membership
        ----------
        The same reasoning that motivates the UUID-preferring plan join
        motivates folding the membership check in here too: it is one more
        column on a query this method already issues once per cache miss,
        not a second round-trip to Supabase on the hot path. When
        ``user_id`` is a well-formed UUID, the SELECT also computes whether
        that user is a member of the resolved account (a ``LEFT``-style
        ``EXISTS`` against ``accounts_memberships``) and the third element
        of the returned tuple carries the answer:

        - ``None`` — membership was not evaluated: ``user_id`` is ``None``
          (unattributed key, the documented back-compat contract) or not a
          valid UUID, or Supabase is not configured. Callers must treat
          ``None`` as "skip the check", never as "not a member".
        - ``True`` / ``False`` — Supabase was queried and answered.

        Unlike the plan lookup (which fails open to ``"free"`` on any
        Supabase error, so a paying tenant is never hard-downed by a flaky
        query), a membership check that was actually requested fails
        CLOSED: a query error raises :class:`AuthError` (503) instead of
        returning a fallback. A transient Supabase outage must not look
        like "key is fine" (a silent allow -- the whole point of this
        method existing) nor like "owner was removed" (the wrong, and
        distinct, 403 -- see ``KeyValidator.validate``). This mirrors
        ``oauth.py``'s ``GrantStoreUnavailable`` -> 503 path, and this
        module's own ``_check_quota``, which already raises 503 rather than
        degrading on a reliability-quota query failure.

        Returns a ``(plan, plan_overrides, is_member)`` tuple. ``(plan,
        plan_overrides)`` keep their pre-existing fallback to ``("free",
        None)`` when Supabase is unconfigured or the account is not found;
        that behaviour is unchanged by this method learning to also answer
        membership.
        """
        if self._supabase_engine is None:
            if user_id is not None:
                self._warn_supabase_unconfigured_once(account_slug)
            return ("free", None, None)

        check_membership = user_id is not None and _is_uuid(user_id)

        try:
            with self._supabase_engine.connect() as conn:
                row = None
                if _is_uuid(org_id):
                    row = conn.execute(
                        text(
                            "SELECT a.vex_plan, a.vex_plan_overrides, "
                            "EXISTS (SELECT 1 FROM public.accounts_memberships m "
                            "WHERE m.account_id = a.id "
                            "AND m.user_id = CAST(:user_id AS uuid)) AS is_member "
                            "FROM accounts a "
                            "WHERE a.id = CAST(:account_id AS uuid) "
                            "AND a.is_personal_account = false LIMIT 1"
                        ),
                        {
                            "account_id": org_id,
                            "user_id": user_id if check_membership else None,
                        },
                    ).fetchone()

                if row is None:
                    row = conn.execute(
                        text(
                            "SELECT a.vex_plan, a.vex_plan_overrides, "
                            "EXISTS (SELECT 1 FROM public.accounts_memberships m "
                            "WHERE m.account_id = a.id "
                            "AND m.user_id = CAST(:user_id AS uuid)) AS is_member "
                            "FROM accounts a "
                            "WHERE a.slug = :slug AND a.is_personal_account = false LIMIT 1"
                        ),
                        {
                            "slug": account_slug,
                            "user_id": user_id if check_membership else None,
                        },
                    ).fetchone()

                if row is None:
                    return ("free", None, None)
                plan = row[0] or "free"
                overrides = (
                    row[1]
                    if isinstance(row[1], dict)
                    else (json.loads(row[1]) if row[1] else None)
                )
                is_member = bool(row[2]) if check_membership else None
                return (plan, overrides, is_member)
        except Exception:
            if check_membership:
                logger.error(
                    "Membership check failed for user=%s account=%s -- "
                    "denying request (fail-closed), not defaulting to "
                    "'free plan, allow' the way the plan lookup does",
                    user_id,
                    account_slug,
                    exc_info=True,
                )
                raise AuthError(
                    503,
                    "Membership check temporarily unavailable. Please retry shortly.",
                    retry_after_seconds=30,
                ) from None
            logger.warning(
                "Failed to query plan from Supabase, defaulting to free",
                exc_info=True,
            )
            return ("free", None, None)

    def _warn_supabase_unconfigured_once(self, account_slug: str) -> None:
        """Log once (not per-request) that the membership gate is disabled.

        A ``KeyValidator`` with no ``supabase_database_url`` cannot check
        membership at all, so every attributed key on it behaves as if its
        owner is always still a member -- this IS a bypass of the gate this
        module implements, and it is silent unless something logs it. Once
        is enough: this is a static configuration fact about the process,
        not a per-request event, and repeating it on every cache miss would
        just be noise an operator learns to ignore.
        """
        with self._warn_lock:
            if self._warned_supabase_unconfigured:
                return
            self._warned_supabase_unconfigured = True
        logger.warning(
            "KeyValidator has no Supabase engine configured (scope=%r); "
            "API keys will NOT be re-checked against workspace membership "
            "for account=%r or any other account. A user removed from a "
            "workspace keeps using their key indefinitely on this "
            "validator until it is explicitly revoked.",
            self._required_scope,
            account_slug,
        )

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
                        SELECT org_id, api_keys, account_slug, workspace_id
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

        # Find the matching key entry in the JSONB array first: the
        # membership check folded into the plan query below needs the
        # entry's owner (``created_by``) before it can run.
        key_entry = None
        for candidate in api_keys:
            if candidate.get("key_hash") == key_hash:
                key_entry = candidate
                break

        if key_entry is None:
            return None

        user_id = key_entry.get("created_by")
        plan_val, plan_overrides_val, is_member = self._query_plan_from_supabase(
            account_slug, org_id, user_id
        )
        # A workspace-linked org answers membership from the workspace DB,
        # not Supabase. The two disagree by construction: a key minted with a
        # WORKSPACE user uuid does not exist in Supabase at all, so the
        # legacy answer for every such key is "not a member" -- a 403 on a
        # valid credential. Workspace truth therefore OVERRIDES, not
        # supplements, on linked orgs; Supabase keeps answering only for
        # orgs the handshake has not touched.
        workspace_id = row[3] if len(row) > 3 else None
        if workspace_id is not None:
            is_member = self._query_membership_from_workspace(
                str(workspace_id), user_id
            )
        plan_config = get_plan_config(plan_val, plan_overrides_val)

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
            user_id=user_id,
            is_member=is_member,
        )

    # ------------------------------------------------------------------
    # Workspace personal access tokens (PATs)
    # ------------------------------------------------------------------

    def _query_workspace_pat(self, raw_key: str) -> _CachedKey | None:
        """Resolve a workspace personal access token to a cached entry.

        A PAT is a USER credential: ``user_id`` is the token's workspace
        user, and the org is the token's workspace org (for a workspace-less
        token, the one ``_pat_default_workspace`` picks), ensured via
        :func:`shared.membership.ensure_org_for_workspace` on first use.
        Everything downstream — the membership guard, node selection,
        project reach — already keys off those two fields, so NO authz is
        duplicated here.

        Runs only on an engine-keys miss (see ``_live_entry``); the caller
        caches the result exactly like a key entry, so revocation latency
        is the same ``cache_ttl_s`` window.

        Returns ``None`` — which the caller turns into the SAME 401 as an
        unknown key — for: no workspace engine configured, unknown token,
        inactive/expired/soft-deleted token, or a token whose workspace
        (or default workspace) cannot be resolved. No oracle distinguishes
        "revoked" from "never existed".

        Two databases, joined in Python: the workspace connection answers
        only about ``api_tokens`` / ``workspaces``; the engine connection
        answers only about ``organizations``. A workspace-DB failure is a
        503, not a 401 — an outage must not read as "invalid credential".
        """
        if self._workspace_engine is None:
            return None

        try:
            with self._workspace_engine.connect() as conn:
                token_row = conn.execute(
                    _PAT_TOKEN_SQL, {"token_hash": hash_token(raw_key)}
                ).fetchone()
                if token_row is None:
                    return None
                if not self._pat_token_alive(token_row):
                    return None
                if token_row.workspace_id is not None:
                    ws_row = conn.execute(
                        _PAT_WORKSPACE_SQL,
                        {"ws": str(token_row.workspace_id)},
                    ).fetchone()
                else:
                    ws_row = self._pat_default_workspace(
                        conn,
                        str(token_row.user_id),
                        is_service=bool(token_row.is_service),
                    )
                if ws_row is None:
                    return None
        except Exception:
            logger.error("workspace database error during PAT lookup", exc_info=True)
            raise AuthError(
                503, "Authentication backend temporarily unavailable"
            ) from None

        entry = self._pat_cached_key(token_row, ws_row)
        self._touch_pat_last_used(str(token_row.id))
        return entry

    @staticmethod
    def _pat_default_workspace(conn, user_id: str, *, is_service: bool):
        """The workspace a workspace-less (user-level) token acts in.

        The user's ONLY team server when they are an active member of exactly
        one, else their personal workspace. Someone with one server means that
        server when they say "my projects"; landing them on an empty personal
        node reads as lost data, because the call succeeds and returns
        nothing.

        Several team servers is a choice only the caller can make (the
        ``X-Klio-Node`` header): guessing would file a person's memory in the
        wrong company, so that case stays personal. Community servers never
        count, neither as the default nor as a reason to call it ambiguous.

        A service token is never defaulted into a server. It may publish at
        org scope, so its destination has to be stated — by binding the token
        to a workspace — and is not inferred from its user's memberships.

        The answer is cached with the rest of the entry, so a user who joins
        a second team server keeps the old default for up to ``cache_ttl_s``
        and then moves to the personal workspace until they select a node.

        Returns ``None`` when there is neither a single team server nor a
        living personal workspace.
        """
        if is_service:
            return conn.execute(
                _PAT_PERSONAL_WORKSPACE_SQL, {"uid": user_id}
            ).fetchone()
        team_servers = conn.execute(
            _PAT_TEAM_SERVERS_SQL,
            {
                "uid": user_id,
                "kind": _PAT_TEAM_SERVER_KIND,
                "probe_limit": _PAT_TEAM_SERVER_PROBE_LIMIT,
            },
        ).fetchall()
        if len(team_servers) == 1:
            return team_servers[0]
        return conn.execute(_PAT_PERSONAL_WORKSPACE_SQL, {"uid": user_id}).fetchone()

    @staticmethod
    def _pat_token_alive(token_row) -> bool:
        """Whether this ``api_tokens`` row is a live credential."""
        if not token_row.is_active or token_row.deleted_at is not None:
            return False
        if token_row.expired_at is not None:
            expired_at = token_row.expired_at
            if expired_at.tzinfo is None:
                expired_at = expired_at.replace(tzinfo=timezone.utc)
            if datetime.now(tz=timezone.utc) > expired_at:
                return False
        return True

    def _pat_cached_key(self, token_row, ws_row) -> _CachedKey:
        """Build the cache entry for a live PAT: ensure org, resolve plan."""
        # Lazy import: membership is a sibling with no dependency back on
        # this module, but auth.py is imported by every service at startup
        # and must not grow import weight for the non-PAT path.
        from shared.membership import ensure_org_for_workspace  # noqa: PLC0415

        workspace_id = str(ws_row.id)
        user_id = str(token_row.user_id)
        try:
            with self._engine.connect() as conn:
                org_id = ensure_org_for_workspace(
                    conn,
                    workspace_id=workspace_id,
                    slug=ws_row.slug,
                    is_personal=bool(ws_row.is_personal),
                    owner_user_id=(
                        str(ws_row.owner_id) if ws_row.owner_id is not None else None
                    ),
                )
                slug_row = conn.execute(
                    _PAT_ORG_SLUG_SQL, {"org_id": org_id}
                ).fetchone()
                conn.commit()
        except Exception:
            logger.error(
                "engine database error while provisioning the org for a PAT "
                "(workspace %s)",
                workspace_id,
                exc_info=True,
            )
            raise AuthError(500, "Internal authentication error") from None

        account_slug = (
            slug_row.account_slug if slug_row and slug_row.account_slug else org_id
        )
        # The existing org-plan path, exactly as the key arm resolves it.
        plan_val, plan_overrides_val, _ = self._query_plan_from_supabase(
            account_slug, org_id
        )
        plan_config = get_plan_config(plan_val, plan_overrides_val)

        # Same membership computation as the key path for a workspace-linked
        # org: a PAT user of the token's own workspace passes naturally.
        is_member = self._query_membership_from_workspace(workspace_id, user_id)

        # ``expires_at`` stays None on purpose: expiry was already answered
        # above as the oracle-free 401, and the cache TTL bounds how long a
        # newly-expired token can linger — the same latency as revocation.
        return _CachedKey(
            org_id=org_id,
            key_id=str(token_row.id),
            scopes=list(PAT_SCOPES),
            rate_limit_rpm=plan_config.max_rpm,
            expires_at=None,
            revoked=False,
            plan=plan_val,
            plan_overrides=plan_overrides_val,
            user_id=user_id,
            is_member=is_member,
            is_service=bool(getattr(token_row, "is_service", False)),
        )

    def _touch_pat_last_used(self, token_id: str) -> None:
        """Best-effort ``last_used`` stamp; NEVER fails the request.

        Throttled by construction: the PAT arm only runs on a cache miss,
        so this executes at most once per ``cache_ttl_s`` per token.
        """
        if self._workspace_engine is None:  # pragma: no cover - guarded upstream
            return
        try:
            with self._workspace_engine.connect() as conn:
                conn.execute(_PAT_TOUCH_SQL, {"id": token_id})
                conn.commit()
        except Exception:
            logger.warning(
                "failed to stamp last_used for PAT %s", token_id, exc_info=True
            )

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
