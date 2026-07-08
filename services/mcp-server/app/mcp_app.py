"""MCP protocol layer for the Klio memory product.

Builds a :class:`mcp.server.fastmcp.FastMCP` server over the
Streamable-HTTP transport and registers the eight Klio memory tools.

As of D2 the five *write* tools (``remember``, ``observe``, ``plan``,
``decide``, ``note``) are fully implemented: they resolve the tenant
principal, derive a stable brain ``session_id``, and persist via
:func:`shared.brain.write_scoped_memory`.  As of D3 the ``recall`` read
tool is implemented too: it resolves the principal, resolves optional
``space``/``project`` filters to ids (read-only, surfacing unknown names
as a warning), and reads via :func:`shared.memory.retrieve_memories_scoped`.
As of D4 the ``space`` management tool is implemented as well: it resolves
the principal and dispatches on ``action`` (``list`` / ``create`` /
``info`` / ``switch``) over :func:`shared.brain.list_spaces` /
``get_space`` / ``ensure_space``.  All eight tool bodies are now live.
The tool *schemas* have been the real ones throughout: each tool's
signature and argument descriptions define the JSON input schema that
``tools/list`` advertises, so filling in the bodies never changed the
public contract.

Every tool takes a FastMCP ``Context`` as its first parameter; FastMCP
injects it and EXCLUDES it from the public input schema, so the
``tools/list`` contract is unchanged.

Transport notes (mcp SDK 1.27.x):

* :meth:`FastMCP.streamable_http_app` returns a Starlette ASGI app that
  serves the Streamable-HTTP endpoint at ``streamable_http_path``
  (default ``/mcp``).  ``app.main`` mounts this under the FastAPI app.
* The Streamable-HTTP transport requires its session manager to be
  running for the lifetime of the app.  ``app.main`` enters
  ``mcp.session_manager.run()`` from the FastAPI lifespan — without that
  context the mounted route accepts connections but cannot service MCP
  requests.

Why set the ``X-Vex-Agent`` header: a client that omits it maps to a
single shared default agent (``mcp-default`` — see ``app.auth``), so its
memories pool under one agent_id.  The server instructions and the
``recall``/``remember`` tool descriptions tell agents to set the header
to scope memory to a distinct agent.
"""

import logging
from typing import Annotated, Any, Literal, Optional

from fastapi import HTTPException
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from shared import brain
from shared.memory import MemoryRecord, retract_memory, retrieve_memories_scoped
from shared.plan_limits import get_plan_config
from shared.usage import MemoryQuotaExceeded, check_memory_quota, record_usage
from sqlalchemy import text

from app.auth import (
    resolve_principal,
    resolve_principal_plan,
    resolve_project_from_request,
)
from app.config import (
    MCP_ALLOWED_HOSTS,
    MCP_ALLOWED_ORIGINS,
    MCP_DNS_REBINDING_PROTECTION,
)
from app.db import SessionLocal
from app.recall_log import log_recall_event
from app.session import brain_session_id

logger = logging.getLogger("agentguard.mcp.tools")

# Canonical, ordered list of the Klio tool names.  Tests assert the
# registered tool set matches this exactly.
EXPECTED_TOOL_NAMES: tuple[str, ...] = (
    "recall",
    "remember",
    "observe",
    "plan",
    "decide",
    "note",
    "space",
    "forget",
)

# Server-level guidance surfaced to MCP clients during ``initialize``.
# Encodes a proactive when-to-write protocol: agents are told to write
# durable knowledge without being asked, and steered away from persisting
# tool calls / transient steps (reinforces the signal-quality cleanup).
# Also reinforces the X-Vex-Agent contract at the protocol level.
SERVER_INSTRUCTIONS = (
    "Klio is the team's shared memory for AI agents. Memory is scoped to the "
    "org resolved from your X-Vex-Key and to an agent resolved from the "
    "X-Vex-Agent header; set X-Vex-Agent to a stable per-agent identifier "
    "(clients that omit it share one default agent). "
    "Use `recall` to retrieve relevant context before acting. "
    "PROACTIVELY persist durable knowledge as it arises — without being asked: "
    "when the user states a durable fact or preference, call `remember`; when a "
    "decision is made, call `decide`; when a plan is set, call `plan`; for other "
    "durable context, `note`. Do NOT persist tool calls, file reads, or "
    "transient steps. Use `forget` to retract a memory by id."
)

# Shared annotation for the optional ``space`` argument on the five write tools
# (remember/observe/plan/decide/note). Defined once so the schema description
# cannot drift between tools; each tool spells it ``space: SpaceParam = None``.
SpaceParam = Annotated[
    Optional[str],
    Field(
        description=(
            "Optional space slug to scope this memory to; created if it doesn't exist."
        )
    ),
]


async def _perform_write(
    ctx: Context,
    *,
    content: str,
    memory_type: str,
    metadata: Optional[dict] = None,
    scope: str = "org",
    space: Optional[str] = None,
) -> dict[str, Any]:
    """Shared body for the five Klio write tools.

    Resolves the tenant principal from the transport request, derives a stable
    brain ``session_id``, opens a short-lived DB session, and delegates to
    :func:`shared.brain.write_scoped_memory` (which dedups, inserts, and
    promotes the row to the org brain scope).

    Auth failures from :func:`resolve_principal` surface as Starlette
    ``HTTPException``; we translate them into a clean ``unauthorized`` result
    dict rather than letting the exception leak through the MCP layer (the MCP
    transport has no notion of HTTP status codes for tool errors).

    Space filing (D1): when ``space`` is provided, it is resolved to a
    ``space_id`` via :func:`shared.brain.ensure_space` (get-or-create, keyed on
    ``(org_id, slug)``) using the org_id from the resolved principal — NEVER
    caller input — so the space is inherently tenant-scoped and a write can
    never target another org's space. The provided string is used verbatim as
    the slug (mirroring how ``recall`` treats ``space`` as the slug, so a write
    and a later ``recall(space=...)`` converge on the same row) with
    ``name == slug``. Resolution is idempotent: a repeat write to the same slug
    reuses the existing space. Omitting ``space`` (or passing blank/whitespace)
    leaves ``space_id=None`` — today's org-brain behavior, unchanged.

    ``ensure_space`` is a synchronous DB helper called inline here, matching the
    established pattern in :func:`_perform_recall` (which resolves its ``space``
    filter via :func:`shared.brain.find_space_id` inline) and the inline project
    resolution below.

    Args:
        ctx: FastMCP-injected context; ``ctx.request_context.request`` is the
            Starlette request for the Streamable-HTTP transport.
        content: The memory text to store.
        memory_type: The brain memory type for this tool.
        metadata: Optional metadata persisted on the row.
        scope: ``"org"`` (default) for the shared team brain, or ``"agent"`` to
            write a row private to the caller's resolved ``agent_id`` (only
            ``recall(scope='agent')`` for that same agent sees it).
        space: Optional space slug to scope this memory to; the space is created
            (get-or-create) under the caller's org if it does not exist. Omitted
            or blank means the org-wide brain (``space_id=None``).

    Returns:
        The result dict from :func:`shared.brain.write_scoped_memory`, or a
        clean error dict on auth failure / unexpected error.
    """
    request = ctx.request_context.request
    if request is None:
        # The Streamable-HTTP transport always attaches the Starlette request
        # (mcp 1.27.x), so this only fires under a transport that does not —
        # there is no request to authenticate, so deny rather than guess a
        # principal.
        logger.error("brain write got no transport request; cannot authenticate")
        return {
            "stored": False,
            "error": "unauthorized",
            "detail": "no request context",
        }

    _attach_client_name(ctx, request)

    try:
        org_id, agent_id, plan, plan_overrides = resolve_principal_plan(request)
    except HTTPException as exc:
        return {"stored": False, "error": "unauthorized", "detail": str(exc.detail)}

    session_id = brain_session_id(request, org_id, agent_id)

    db = SessionLocal()
    try:
        # Enforce the org's monthly memory quota BEFORE doing any write work.
        # ``check_memory_quota`` raises ``MemoryQuotaExceeded`` for a hard block
        # (capped plan at/over limit, no overage) and otherwise returns an
        # ``over_cap`` flag (True when allowed via overage). It fails OPEN on any
        # metering error, so a counter hiccup never blocks a real write. Pass the
        # account's ``plan_overrides`` so a custom cap is honored here exactly as
        # the retention sweep honors it.
        try:
            over_cap = check_memory_quota(
                db,
                org_id,
                get_plan_config(plan, plan_overrides),
                "memory",
                plan_name=plan,
            )
        except MemoryQuotaExceeded as exc:
            return {
                "stored": False,
                "error": "quota_exceeded",
                "detail": exc.message,
                "suggested_plan": exc.suggested_plan,
            }

        # Best-effort project filing from the X-Vex-Project header (read-only
        # resolve; fail open to unfiled — filing is relevance, not security).
        header_project_id = resolve_project_from_request(request, db, org_id=org_id)

        # Resolve the optional space slug to a space_id (get-or-create, keyed on
        # (org_id, slug)). The org_id comes from the resolved principal — never
        # caller input — so the space is inherently tenant-scoped. The slug is
        # STRIPPED before both the blank-guard and storage so a padded value like
        # "  x  " files under slug "x" and a later recall(space="x") converges on
        # the same space (rather than creating an unreachable padded-slug space).
        # A blank / whitespace-only value is treated as "no space" (space_id stays
        # None, backward compatible) rather than creating a stray empty-slug space.
        stripped_space = space.strip() if space is not None else ""
        space_id: Optional[str] = (
            brain.ensure_space(
                db, org_id=org_id, name=stripped_space, slug=stripped_space
            )
            if stripped_space
            else None
        )

        result = await brain.write_scoped_memory(
            db,
            org_id=org_id,
            agent_id=agent_id,
            content=content,
            memory_type=memory_type,
            session_id=session_id,
            space_id=space_id,
            project_id=header_project_id,
            metadata=metadata,
            scope=scope,
        )
        # Only meter an actual insert: a deduplicated write stores no new row, so
        # it must not consume the org's monthly memory allowance.
        if result.get("stored"):
            record_usage(db, org_id, "memory", over_cap=over_cap)
        return result
    except Exception:
        logger.exception("brain write failed (memory_type=%s)", memory_type)
        return {"stored": False, "error": "write_failed"}
    finally:
        db.close()


# Sentinel project value meaning "search across all projects" (no project
# filter). Compared case-insensitively so "any"/"ANY"/"Any" all work.
_PROJECT_ANY = "any"

# Appended to every recall result as a top-level ``guidance`` field. On hosted
# clients we cannot force a write, so we remind the agent — at the moment it is
# already engaged with Klio (each recall), refreshed rather than only at
# initialize — to persist what matters. Deliberately steers toward DURABLE
# knowledge and away from tool-call/step noise so it reinforces (not undoes)
# the signal-quality work.
RECALL_WRITE_NUDGE = (
    "If this conversation produced durable facts, decisions, or preferences, "
    "persist them now with remember / decide / note. Save durable knowledge "
    "— not tool calls, file reads, or transient steps."
)


def _attach_client_name(ctx: Context, request: Any) -> None:
    """Expose the MCP client's self-reported name for OAuth agent identity.

    Reads the ``clientInfo.name`` from the ``initialize`` handshake
    (``ctx.session.client_params.clientInfo``) and writes it to
    ``request.state.mcp_client_name`` so that :func:`app.auth._oauth_agent_id`
    can derive a stable, human-readable agent id for OAuth principals.

    Best-effort: any exception (SDK shape change, missing session, None
    client_params) is swallowed silently.  When the attribute is absent the
    OAuth agent id simply falls back to the raw ``client_id`` — no memory
    operation is blocked.

    Confirmed SDK attribute path (mcp 1.27.x):
        ``ctx.session``                          → ``ServerSession``
        ``ctx.session.client_params``            → ``InitializeRequestParams | None``
        ``ctx.session.client_params.clientInfo`` → ``Implementation``
        ``ctx.session.client_params.clientInfo.name`` → ``str``
    """
    try:
        client_info = ctx.session.client_params.clientInfo
        if client_info and client_info.name:
            request.state.mcp_client_name = client_info.name
    except Exception:
        pass


def _serialize_record(
    record: MemoryRecord,
    project_label: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Project a :class:`MemoryRecord` to a JSON-safe dict for a recall result.

    Deliberately EXCLUDES ``embedding`` (a large float vector that is useless
    to the agent and bloats the response) and the verification-only / internal
    fields (``session_id``, ``execution_id``, ``status``, ``superseded_by``,
    ``sequence_number``). ``created_at`` is rendered as an ISO-8601 string (or
    ``None``) since ``datetime`` is not JSON-serializable.

    ``relevance`` carries the cosine-similarity score (0–1, rounded to 3 dp)
    populated by :func:`shared.memory.retrieve_memories_scoped`; ``None`` when
    not available (e.g. direct construction in tests).

    ``project`` carries the human-readable project label dict
    (``{"id": ..., "name": ...}``) resolved once per recall response, or
    ``None`` when no project scope is active.
    """
    return {
        "id": str(record.id),
        "memory_type": record.memory_type,
        "content": record.content,
        "scope": record.scope,
        "confidence": record.confidence,
        "relevance": record.relevance,
        # space_id/project_id are ``sa.UUID()`` columns, so a raw ``text()``
        # SELECT yields ``uuid.UUID`` objects (not strings). Stringify them —
        # like ``id`` above — so the result is plain-``json``-serializable and
        # the id fields are type-consistent across the payload.
        "space_id": str(record.space_id) if record.space_id is not None else None,
        "project_id": (
            str(record.project_id) if record.project_id is not None else None
        ),
        "project": project_label,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "metadata": record.metadata,
    }


def _serialize_space(s: dict[str, Any]) -> dict[str, Any]:
    """Project a brain-layer space dict to a JSON-safe dict for a tool result.

    The brain helpers (:func:`shared.brain.list_spaces` / ``get_space``) already
    stringify ``id`` (a UUID column) but leave ``created_at`` as a ``datetime``
    so this layer controls its serialization — exactly the split used on the
    recall path. We render ``created_at`` as an ISO-8601 string (or ``None``)
    here so NOTHING in the returned tool result is a ``uuid.UUID`` or
    ``datetime`` (both blow up ``json.dumps``; we just fixed this class of bug on
    the recall path and must not reintroduce it).
    """
    created_at = s.get("created_at")
    return {
        "id": s["id"],  # already a str (brain helper stringifies the UUID column)
        "name": s.get("name"),
        "slug": s.get("slug"),
        "embedding_model": s.get("embedding_model"),
        "created_at": created_at.isoformat() if created_at is not None else None,
    }


async def _perform_space(
    ctx: Context,
    *,
    action: str,
    name: Optional[str],
    slug: Optional[str],
) -> dict[str, Any]:
    """Shared body for the Klio ``space`` management tool.

    Resolves the tenant principal from the transport request, opens a
    short-lived DB session, and dispatches on ``action``:

    * ``list``   — enumerate the org's spaces (read-only).
    * ``create`` — get-or-create a space by ``(org_id, slug)`` via
      :func:`shared.brain.ensure_space` (idempotent: a repeat returns the
      existing id).
    * ``info``   — look up a single space by slug (read-only).
    * ``switch`` — validate a space exists and tell the caller how to scope
      reads to it (see the switch rationale in the dispatch below).

    The org_id is taken from the resolved principal (never caller-supplied
    input), so a tenant can only ever list/create/inspect its own spaces.

    Auth/session boilerplate mirrors :func:`_perform_recall` / ``_perform_write``
    inline rather than being factored out — the three tool bodies duplicate this
    today and a consolidation pass is tracked separately (do not refactor the
    already-reviewed bodies here).

    Args:
        ctx: FastMCP-injected context; ``ctx.request_context.request`` is the
            Starlette request for the Streamable-HTTP transport.
        action: One of ``list`` | ``create`` | ``info`` | ``switch`` (the schema
            already constrains it to this set).
        name: Human-readable space name (required, with ``slug``, for
            ``create``).
        slug: Space slug (required for ``create`` / ``info`` / ``switch``).

    Returns:
        An action-tagged result dict, or a clean error dict on auth failure /
        unexpected error.
    """
    request = ctx.request_context.request
    if request is None:
        # The Streamable-HTTP transport always attaches the Starlette request
        # (mcp 1.27.x); a transport that does not gives us nothing to
        # authenticate, so deny rather than guess a principal.
        logger.error("brain space got no transport request; cannot authenticate")
        return {"error": "unauthorized", "detail": "no request context"}

    _attach_client_name(ctx, request)

    try:
        org_id, _agent_id = resolve_principal(request)
    except HTTPException as exc:
        return {"error": "unauthorized", "detail": str(exc.detail)}

    db = SessionLocal()
    try:
        if action == "list":
            spaces = brain.list_spaces(db, org_id=org_id)
            return {
                "action": "list",
                "spaces": [_serialize_space(s) for s in spaces],
                "count": len(spaces),
            }

        if action == "create":
            # Both fields are mandatory and must be non-empty; bail before
            # touching the DB so a half-specified create never reaches
            # ensure_space.
            if not (name and name.strip()) or not (slug and slug.strip()):
                return {"action": "create", "error": "name and slug are required"}
            # ensure_space is idempotent (race-safe upsert on (org_id, slug)); a
            # repeat create returns the existing id, which is the intended
            # get-or-create semantics. No embedding_model lever in v1 — created
            # spaces inherit the default model (per-space embedding is a separate
            # future concern).
            space_id = brain.ensure_space(db, org_id=org_id, name=name, slug=slug)
            return {
                "action": "create",
                "created": True,
                "id": space_id,
                "slug": slug,
            }

        if action == "info":
            if not slug:
                return {"action": "info", "error": "slug is required"}
            s = brain.get_space(db, org_id=org_id, slug=slug)
            if s is None:
                return {"action": "info", "warning": f"unknown space: {slug}"}
            return {"action": "info", "space": _serialize_space(s)}

        if action == "switch":
            if not slug:
                return {"action": "switch", "error": "slug is required"}
            s = brain.get_space(db, org_id=org_id, slug=slug)
            if s is None:
                return {"action": "switch", "warning": f"unknown space: {slug}"}
            # RATIONALE: the brain server holds NO durable per-connection "active
            # space" state — the write tools write unfiled and recall takes an
            # explicit `space=` arg. So `switch` does not pretend to persist an
            # active selection (which would be fake stateful behavior across
            # stateless connections); it instead validates the space exists and
            # instructs the caller to pass `space=<slug>` on subsequent `recall`
            # calls. This is honest about the server's statelessness.
            return {
                "action": "switch",
                "space": _serialize_space(s),
                "note": (
                    "Pass space='" + slug + "' to recall to scope reads to this space."
                ),
            }

        # The schema (Literal) constrains `action` to the four handled cases, so
        # this is unreachable via the MCP contract; guard defensively anyway.
        return {"error": "unknown_action", "action": action}
    except Exception:
        logger.exception("brain space failed (action=%s)", action)
        return {"error": "space_failed"}
    finally:
        db.close()


async def _perform_recall(
    ctx: Context,
    *,
    query: str,
    scope: str,
    space: Optional[str],
    project: Optional[str],
    limit: int = 10,
) -> dict[str, Any]:
    """Shared body for the Klio ``recall`` read tool.

    Resolves the tenant principal from the transport request, resolves any
    user-facing ``space`` slug / ``project`` identifier to ids (read-only — an
    unknown name short-circuits to an empty result with a ``warning`` rather
    than silently widening the search), then delegates to
    :func:`shared.memory.retrieve_memories_scoped`.

    This mirrors :func:`_perform_write` but is a pure READ: no commit, and the
    org_id is taken from the resolved principal (never caller-supplied input)
    so a tenant can only ever recall its own memories.

    Args:
        ctx: FastMCP-injected context; ``ctx.request_context.request`` is the
            Starlette request for the Streamable-HTTP transport.
        query: Natural-language search query.
        scope: ``"org"`` (default) or ``"agent"`` (the schema restricts it to
            these two, so :func:`retrieve_memories_scoped` never raises
            ``ValueError`` for a valid call).
        space: Optional space slug; resolved to a ``space_id`` filter.
        project: Optional project identifier; resolved to a ``project_id``
            filter. The sentinel ``"any"`` (case-insensitive) means
            cross-project (no project filter). When ``None`` AND the caller
            sends an ``X-Vex-Project`` header, the header is resolved as a
            fail-open fallback (unresolvable header yields org-wide recall, not
            an error — explicit ``project`` args keep their strict warn-and-empty
            semantics).
        limit: Maximum number of memories to return (clamped to 1–50).

    Returns:
        ``{"memories": [...], "count": N}`` on success, ``{"memories": [],
        "count": 0, "warning": ...}`` for an unknown space/project filter, or a
        clean error dict on auth failure / unexpected error.
    """
    request = ctx.request_context.request
    if request is None:
        # The Streamable-HTTP transport always attaches the Starlette request
        # (mcp 1.27.x); a transport that does not gives us nothing to
        # authenticate, so deny rather than guess a principal.
        logger.error("brain recall got no transport request; cannot authenticate")
        return {
            "error": "unauthorized",
            "detail": "no request context",
            "guidance": RECALL_WRITE_NUDGE,
        }

    _attach_client_name(ctx, request)

    try:
        org_id, agent_id, plan, plan_overrides = resolve_principal_plan(request)
    except HTTPException as exc:
        return {
            "error": "unauthorized",
            "detail": str(exc.detail),
            "guidance": RECALL_WRITE_NUDGE,
        }

    db = SessionLocal()
    try:
        # Enforce the org's monthly recall quota BEFORE any retrieval work, so a
        # capped tenant is blocked cleanly rather than after spending an
        # embedding/search round trip. ``check_memory_quota`` raises
        # ``MemoryQuotaExceeded`` on a hard block and otherwise returns the
        # ``over_cap`` flag; it fails OPEN on any metering error. Pass the
        # account's ``plan_overrides`` so a custom cap is honored here exactly as
        # the retention sweep honors it.
        try:
            recall_over_cap = check_memory_quota(
                db,
                org_id,
                get_plan_config(plan, plan_overrides),
                "recall",
                plan_name=plan,
            )
        except MemoryQuotaExceeded as exc:
            return {
                "error": "quota_exceeded",
                "detail": exc.message,
                "suggested_plan": exc.suggested_plan,
                "guidance": RECALL_WRITE_NUDGE,
            }

        # Resolve the optional space-slug filter. The slug is STRIPPED so it
        # converges with the write path (which stores the stripped slug): a
        # padded recall(space="  x  ") looks up slug "x". A blank / whitespace-
        # only value is treated as "no space filter" (space_id stays None →
        # org-wide), mirroring the write side rather than erroring "unknown
        # space". A genuinely non-blank slug that is not found is still surfaced
        # (not silently dropped) so an over-narrow filter never masquerades as an
        # org-wide search returning unrelated memories.
        stripped_space = space.strip() if space is not None else ""
        space_id: Optional[str] = None
        if stripped_space:
            space_id = brain.find_space_id(db, org_id=org_id, slug=stripped_space)
            if space_id is None:
                return {
                    "memories": [],
                    "count": 0,
                    "warning": f"unknown space: {stripped_space}",
                    "guidance": RECALL_WRITE_NUDGE,
                }

        # Resolve the optional project filter. The "any" sentinel means
        # cross-project: no project filter at all.
        project_id: Optional[str] = None
        if project is not None and project.lower() != _PROJECT_ANY:
            project_id = brain.find_project_id(db, org_id=org_id, identifier=project)
            if project_id is None:
                return {
                    "memories": [],
                    "count": 0,
                    "warning": f"unknown project: {project}",
                    "guidance": RECALL_WRITE_NUDGE,
                }

        # No explicit project arg: fall back to the X-Vex-Project header
        # (fail-open — an unresolvable header means org-wide, matching the
        # read-path philosophy; explicit args keep strict warn-and-empty).
        if project is None:
            project_id = resolve_project_from_request(request, db, org_id=org_id)

        # agent_id is always forwarded; retrieve_memories_scoped ignores it
        # unless scope == "agent".
        records = await retrieve_memories_scoped(
            db,
            org_id,
            query,
            scope=scope,
            agent_id=agent_id,
            space_id=space_id,
            project_id=project_id,
            limit=min(max(limit, 1), 50),
        )
        # Best-effort recall activity log. This tool body already does its DB I/O
        # inline (no anyio), so call the helper directly; it swallows its own
        # errors so a logging failure can never break the recall.
        log_recall_event(
            db,
            org_id=org_id,
            agent_id=agent_id,
            project_id=project_id,
            query_text=(query or None),
            result_count=len(records),
            source="mcp",
        )
        # Meter the recall against the org's monthly allowance (best-effort;
        # ``record_usage`` never raises). A successful recall always counts,
        # regardless of how many records it returned.
        record_usage(db, org_id, "recall", over_cap=recall_over_cap)

        # Resolve a human-readable project label once for the entire response.
        project_label: Optional[dict[str, Any]] = None
        if project_id is not None:
            row = db.execute(
                text(
                    "SELECT display_name FROM projects "
                    "WHERE id = :id AND org_id = :org_id"
                ),
                {"id": project_id, "org_id": org_id},
            ).fetchone()
            if row is not None:
                project_label = {"id": project_id, "name": row.display_name}
        memories = [_serialize_record(r, project_label=project_label) for r in records]
        return {
            "memories": memories,
            "count": len(memories),
            "guidance": RECALL_WRITE_NUDGE,
        }
    except Exception:
        logger.exception("brain recall failed (scope=%s)", scope)
        return {"error": "recall_failed", "guidance": RECALL_WRITE_NUDGE}
    finally:
        db.close()


async def _perform_forget(ctx: Context, *, memory_id: str) -> dict[str, Any]:
    """Shared body for the Klio ``forget`` tool: retract one memory by id.

    Resolves the tenant principal from the transport request and delegates to
    :func:`shared.memory.retract_memory`, which enforces org_id and agent
    authorization (cross-org ids and cross-agent scope='agent' rows are each
    rejected with distinct clean error dicts).

    Auth failures from :func:`resolve_principal` surface as a clean
    ``unauthorized`` result dict rather than letting an HTTPException propagate
    through the MCP layer (no HTTP status codes in tool errors).

    Args:
        ctx: FastMCP-injected context; ``ctx.request_context.request`` is the
            Starlette request for the Streamable-HTTP transport.
        memory_id: UUID string of the memory to retract (from a prior recall).

    Returns:
        ``{"forgotten": True, "id": <id>}`` on success, or a clean error dict
        on auth failure, id validation failure, or unexpected error.
    """
    request = ctx.request_context.request
    if request is None:
        logger.error("brain forget got no transport request; cannot authenticate")
        return {
            "forgotten": False,
            "error": "unauthorized",
            "detail": "no request context",
        }

    _attach_client_name(ctx, request)

    try:
        org_id, agent_id = resolve_principal(request)
    except HTTPException as exc:
        return {"forgotten": False, "error": "unauthorized", "detail": str(exc.detail)}

    db = SessionLocal()
    try:
        return retract_memory(
            db, org_id=org_id, agent_id=agent_id, memory_id=memory_id, via="mcp"
        )
    except Exception:
        logger.exception("brain forget failed")
        return {"forgotten": False, "error": "forget_failed"}
    finally:
        db.close()


def build_mcp_server() -> FastMCP:
    """Construct a fresh :class:`FastMCP` with the eight Klio tools.

    Returns a new instance on every call so tests get isolated servers
    and so the app factory controls the lifecycle.  All eight tool bodies
    are implemented (D2-D4, T11); schemas are final.
    """
    # Keep the default ``streamable_http_path="/mcp"``.  ``app.main``
    # mounts this server's ASGI app at the application root, so the
    # transport endpoint lands exactly at ``/mcp`` (no trailing-slash
    # redirect).  Mounting at a ``/mcp`` prefix with a ``"/"`` transport
    # path instead would force Starlette to serve at ``/mcp/`` and 307
    # redirect bare ``/mcp`` POSTs, which the MCP client does not follow.
    #
    # Pass ``transport_security`` EXPLICITLY: when it is left ``None``,
    # FastMCP auto-enables DNS-rebinding protection with a hardcoded
    # localhost allow-list (``localhost:*`` etc.), which rejects
    # legitimate non-localhost traffic with HTTP 421.  We drive it from
    # config instead (see ``app.config`` for the rationale — auth is the
    # real boundary and the deployment Host is environment-specific).
    mcp = FastMCP(
        name="klio-memory",
        instructions=SERVER_INSTRUCTIONS,
        transport_security=_build_transport_security(),
    )
    _register_tools(mcp)
    return mcp


def _build_transport_security() -> TransportSecuritySettings:
    """Construct the SDK transport-security settings from config.

    Defaults to DNS-rebinding protection OFF (see ``app.config``).  When
    enabled, the configured host/origin allow-lists are applied verbatim,
    supporting the SDK's ``host:*`` wildcard-port syntax.
    """
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=MCP_DNS_REBINDING_PROTECTION,
        allowed_hosts=list(MCP_ALLOWED_HOSTS),
        allowed_origins=list(MCP_ALLOWED_ORIGINS),
    )


def _register_tools(mcp: FastMCP) -> None:
    """Register all eight Klio tools on the given server.

    Each tool's typed signature drives the JSON input schema advertised
    by ``tools/list``.  All bodies are implemented (D2-D4, T11).
    """

    @mcp.tool(
        name="recall",
        description=(
            "Retrieve relevant memories for the current org/team. "
            "Args: query (required); optional scope ('org' default | "
            "'agent'), space, project, limit. Set the X-Vex-Agent header to "
            "scope by agent."
        ),
    )
    async def recall(
        ctx: Context,
        query: Annotated[
            str, Field(description="What to search the team's memory for.")
        ],
        scope: Annotated[
            Literal["org", "agent"],
            Field(description="Memory scope: 'org' (default) or 'agent'."),
        ] = "org",
        space: Annotated[
            Optional[str],
            Field(description="Optional space slug to scope the search."),
        ] = None,
        project: Annotated[
            Optional[str],
            Field(description="Optional project identifier to scope the search."),
        ] = None,
        limit: Annotated[
            int,
            Field(
                description="Max memories to return (1-50, default 10).",
                ge=1,
                le=50,
            ),
        ] = 10,
    ) -> dict[str, Any]:
        return await _perform_recall(
            ctx, query=query, scope=scope, space=space, project=project, limit=limit
        )

    @mcp.tool(
        name="remember",
        description=(
            "Store a durable fact or preference for the team. Call this PROACTIVELY "
            "whenever the user reveals one — e.g. 'I'm interviewing at Hive.co for a "
            "senior PM role' — not only when explicitly asked to remember. Arg: content."
        ),
    )
    async def remember(
        ctx: Context,
        content: Annotated[
            str, Field(description="The durable fact or preference to store.")
        ],
        scope: Annotated[
            Literal["org", "agent"],
            Field(
                description=(
                    "'org' (default, shared team brain) or 'agent' "
                    "(private to this X-Vex-Agent)."
                )
            ),
        ] = "org",
        space: SpaceParam = None,
    ) -> dict[str, Any]:
        return await _perform_write(
            ctx,
            content=content,
            memory_type="memory",
            metadata={"source": "mcp"},
            scope=scope,
            space=space,
        )

    @mcp.tool(
        name="observe",
        description="Log something the agent saw/did this session. Arg: content.",
    )
    async def observe(
        ctx: Context,
        content: Annotated[
            str, Field(description="What the agent observed or did this session.")
        ],
        scope: Annotated[
            Literal["org", "agent"],
            Field(
                description=(
                    "'org' (default, shared team brain) or 'agent' "
                    "(private to this X-Vex-Agent)."
                )
            ),
        ] = "org",
        space: SpaceParam = None,
    ) -> dict[str, Any]:
        return await _perform_write(
            ctx,
            content=content,
            memory_type="observation",
            metadata={"source": "mcp"},
            scope=scope,
            space=space,
        )

    @mcp.tool(
        name="plan",
        description="Record a forward-looking plan when one is set. Arg: content.",
    )
    async def plan(
        ctx: Context,
        content: Annotated[
            str, Field(description="The forward-looking plan to record.")
        ],
        scope: Annotated[
            Literal["org", "agent"],
            Field(
                description=(
                    "'org' (default, shared team brain) or 'agent' "
                    "(private to this X-Vex-Agent)."
                )
            ),
        ] = "org",
        space: SpaceParam = None,
    ) -> dict[str, Any]:
        return await _perform_write(
            ctx,
            content=content,
            memory_type="plan",
            metadata={"source": "mcp"},
            scope=scope,
            space=space,
        )

    @mcp.tool(
        name="decide",
        description="Record a decision and its rationale as soon as one is made. Args: content, optional rationale.",
    )
    async def decide(
        ctx: Context,
        content: Annotated[str, Field(description="The decision that was made.")],
        rationale: Annotated[
            Optional[str],
            Field(description="Optional rationale behind the decision."),
        ] = None,
        scope: Annotated[
            Literal["org", "agent"],
            Field(
                description=(
                    "'org' (default, shared team brain) or 'agent' "
                    "(private to this X-Vex-Agent)."
                )
            ),
        ] = "org",
        space: SpaceParam = None,
    ) -> dict[str, Any]:
        metadata = {"source": "mcp"}
        if rationale is not None:
            metadata["rationale"] = rationale
        return await _perform_write(
            ctx,
            content=content,
            memory_type="decision",
            metadata=metadata,
            scope=scope,
            space=space,
        )

    @mcp.tool(
        name="note",
        description="Free-form note for durable context that isn't a fact, decision, or plan. Arg: content.",
    )
    async def note(
        ctx: Context,
        content: Annotated[str, Field(description="The free-form note to store.")],
        scope: Annotated[
            Literal["org", "agent"],
            Field(
                description=(
                    "'org' (default, shared team brain) or 'agent' "
                    "(private to this X-Vex-Agent)."
                )
            ),
        ] = "org",
        space: SpaceParam = None,
    ) -> dict[str, Any]:
        return await _perform_write(
            ctx,
            content=content,
            memory_type="note",
            metadata={"source": "mcp"},
            scope=scope,
            space=space,
        )

    @mcp.tool(
        name="space",
        description=(
            "Manage spaces. Args: action (list|create|info|switch), optional name/slug."
        ),
    )
    async def space(
        ctx: Context,
        action: Annotated[
            Literal["list", "create", "info", "switch"],
            Field(description="The space operation to perform."),
        ],
        name: Annotated[
            Optional[str],
            Field(description="Human-readable space name (for 'create')."),
        ] = None,
        slug: Annotated[
            Optional[str],
            Field(description="Space slug (for 'info'/'switch'/'create')."),
        ] = None,
    ) -> dict[str, Any]:
        return await _perform_space(ctx, action=action, name=name, slug=slug)

    @mcp.tool(
        name="forget",
        description=(
            "Retract a memory by id (from a prior recall result) so it never "
            "surfaces again. Arg: memory_id."
        ),
    )
    async def forget(
        ctx: Context,
        memory_id: Annotated[
            str, Field(description="The id of the memory to retract.")
        ],
    ) -> dict[str, Any]:
        return await _perform_forget(ctx, memory_id=memory_id)
