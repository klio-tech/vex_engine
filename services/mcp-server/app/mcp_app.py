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

import functools
import hashlib
import logging
from typing import Annotated, Any, Literal, Optional

import anyio
from fastapi import HTTPException
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from shared import artifact_storage, artifacts, brain, graph_store, projects
from shared.compression.router import compress as route_compress
from shared.compression.router import estimate_tokens
from shared.compression.structural import compress_payload
from shared.memory import (
    _NO_CREDENTIALS_RULE,
    MemoryRecord,
    call_llm,
    retract_memory,
    retrieve_memories_scoped,
)
from shared.plan_limits import get_plan_config
from shared.promotion import promote_memory
from shared.usage import MemoryQuotaExceeded, check_memory_quota, record_usage
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

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
from app.directory import resolve_org_member_by_email
from app.compression_cache import (
    OriginalTooLarge,
    handle_is_expired,
    load_original,
    store_original,
)
from app.identity import resolve_request_user_id
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
    "artifact_put",
    "artifact_get",
    "space",
    "forget",
    "share",
    "project_create",
    "project_list",
    "project_members",
    "project_grant",
    "klio_retrieve",
    "compress",
    "explore",
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


def _explicit_write_scope(
    db: Any,
    *,
    requested_scope: str,
    org_id: str,
    user_id: Optional[str],
    project_id: Optional[str],
) -> str:
    """Decide the scope for an EXPLICIT write (``remember``/``decide``/…).

    Calling one of these tools IS the act of sharing, so unlike a hook capture
    the result is never ``private``. The only question is *how* shared::

        project  — when a project resolved AND the author is a member of it
        org      — otherwise (and always, for the deprecated 'agent' request)

    Why membership is required before writing ``scope='project'``: that scope is
    a boundary, and a write into it by a non-member is wrong in both directions
    at once. The author could not read back their own memory (they are not a
    member), and a non-member would have injected content into a space they
    have no standing in — a way to write across a boundary you cannot read
    across. Falling back to ``org`` is the conservative answer and is exactly
    today's behaviour for the 106 auto-created projects that have no members:
    the row is still FILED under the project (``project_id`` is set, ranking it
    first on that project's recall) but it is not FENCED by it.

    ``requested_scope='agent'`` is honoured as-is for backward compatibility;
    it is deprecated in favour of ``private``, which explicit tools do not
    write.
    """
    if requested_scope == "agent":
        return "agent"
    if project_id and projects.is_member(
        db, org_id=org_id, project_id=project_id, user_id=user_id
    ):
        return "project"
    return "org"


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
        scope: The scope the caller ASKED for — ``"org"`` (default) or the
            deprecated ``"agent"``. The scope actually written is decided by
            :func:`_explicit_write_scope`, which may narrow ``org`` to
            ``project``. Explicit tools never write ``private``: calling one is
            the act of sharing.
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

    # Authorship, not visibility: an explicit write is shared either way, and
    # this records who taught the team it. ``None`` is legal and simply means
    # the row has no author.
    user_id = resolve_request_user_id(request)

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

        effective_scope = _explicit_write_scope(
            db,
            requested_scope=scope,
            org_id=org_id,
            user_id=user_id,
            project_id=header_project_id,
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
            scope=effective_scope,
            user_id=user_id,
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


def _dedup_confirm(
    db: Any,
    s3: Any,
    key: str,
    *,
    org_id: str,
    pending_id: str,
    existing: dict,
) -> dict[str, Any]:
    """Finalise a confirm whose bytes turned out to duplicate an active artifact.

    The org already holds these exact bytes, so the caller is pointed at the
    existing artifact — same contract as the inline path's dedup — and the
    redundant upload is cleaned up rather than left to rot:

    * the uploaded object is deleted (best-effort: a failure here must not turn
      a successful dedup into an error; the sweep is not a factor because the
      object belongs to a row that is about to stop existing, so a failed
      delete is logged and left for an operator);
    * the ``pending`` row is deleted, freeing one of the org's reservation
      slots instead of parking a GC candidate for 24 hours.

    No card is written: the existing artifact already has one, and a second
    card pointing at the same artifact would be a duplicate recall result.
    """
    try:
        artifact_storage.delete(s3, key)
    except Exception:
        logger.warning(
            "artifact confirm dedup: could not delete the redundant object for "
            "artifact_id=%s; the row is gone, the object may need manual cleanup",
            pending_id,
            exc_info=True,
        )
    try:
        artifacts.delete_pending(db, org_id=org_id, artifact_id=pending_id)
    except Exception:
        # The abandoned-upload sweep collects it within PENDING_TTL_HOURS, so
        # this is untidy rather than harmful. Never fail the dedup over it.
        logger.warning(
            "artifact confirm dedup: could not delete pending row %s; the GC "
            "sweep will collect it",
            pending_id,
            exc_info=True,
        )
    return {
        "stored": True,
        "artifact_id": existing["id"],
        "version": existing["version"],
        "status": "active",
        "deduplicated": True,
    }


def _resolve_artifact_project(
    request: Any, db: Any, *, org_id: str, project: Optional[str]
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """Resolve where an artifact should be filed. Returns ``(project_id, error)``.

    Precedence follows ``recall``'s existing rule rather than inventing a new
    one: an explicit ``project`` argument wins, and only when it is absent does
    the ``X-Vex-Project`` header decide. Before this existed, ``artifact_put``
    advertised a ``project`` argument in its schema and then overwrote it with
    the header unconditionally — an agent passing ``project="apollo"`` with no
    header got ``project_id = None``, and a later ``recall(project="apollo")``
    never surfaced the card, with no error anywhere to explain it.

    Also as in ``recall``: an explicit identifier that resolves to nothing is
    SURFACED, not silently widened. On the write side that means refusing the
    write — filing the artifact org-wide when the caller named a project is a
    silent misfile, and the artifact is immutable so it cannot be corrected in
    place. The header path keeps failing open (unfiled), matching every other
    write tool.

    ``brain.find_project_id`` is read-only and org-scoped; it never creates a
    project, so a caller cannot mint tenant rows through this argument.
    """
    stripped = project.strip() if project is not None else ""
    if not stripped:
        return resolve_project_from_request(request, db, org_id=org_id), None

    project_id = brain.find_project_id(db, org_id=org_id, identifier=stripped)
    if project_id is None:
        return None, {
            "stored": False,
            "error": "unknown_project",
            "detail": (
                f"unknown project: {stripped}. Omit `project` to file this "
                "artifact org-wide, or create the project first."
            ),
        }
    return project_id, None


async def _perform_artifact_put(
    ctx: Context,
    *,
    title: str,
    summary: str,
    kind: str,
    content: Optional[str] = None,
    mime_type: Optional[str] = None,
    size_bytes: Optional[int] = None,
    space: Optional[str] = None,
    project: Optional[str] = None,
    supersedes: Optional[str] = None,
    artifact_id: Optional[str] = None,
) -> dict[str, Any]:
    """Shared body for ``artifact_put``.

    Mirrors :func:`_perform_write`'s auth and session handling exactly. Three
    behaviours select on the arguments present: inline (``content`` given),
    presigned request (neither ``content`` nor ``artifact_id``), and confirm
    (``artifact_id`` given).

    Quota is enforced on each branch before that branch does anything durable,
    but *what* it can be enforced against differs, and the difference matters:

    * **inline** — the bytes are in hand, so the per-file limit and the storage
      pool are both checked against their real size before any row is created.
    * **request** — nothing exists yet, so the checks run against the CLIENT'S
      DECLARED size. That size is mandatory here and is SIGNED into the upload
      URL (``presign_put``'s ``ContentLength``), so it is not merely a hint the
      client may ignore: an upload of any other size fails signature
      verification at the storage layer. The declared size therefore both gates
      the strategy and caps the URL, and ``artifacts.MAX_PENDING_PER_ORG``
      bounds how many such capped reservations an org may hold at once.
      Server-measured values still win at confirm time; the declared size is
      never persisted.
    * **confirm** — the only branch with a server-measured number. The size is
      read via ``HEAD`` (no transfer) and checked against both limits BEFORE
      the object is streamed to compute its checksum, so rejecting an
      over-limit upload never costs a full download.

    Blocking boto3 calls in this coroutine run through
    ``anyio.to_thread.run_sync``: hashing a multi-GB object inline would stall
    the event loop for the entire transfer.
    """
    request = ctx.request_context.request
    if request is None:
        logger.error("artifact_put got no transport request; cannot authenticate")
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

    cfg = get_plan_config(plan, plan_overrides)
    session_id = brain_session_id(request, org_id, agent_id)

    db = SessionLocal()
    try:
        if content is not None:
            body = content.encode("utf-8")

            if len(body) > artifacts.INLINE_MAX_BYTES:
                return {
                    "stored": False,
                    "error": "too_large_for_inline",
                    "detail": (
                        f"{len(body)} bytes exceeds the "
                        f"{artifacts.INLINE_MAX_BYTES}-byte inline limit. Call "
                        "artifact_put without `content` to get an upload URL."
                    ),
                }

            if len(body) > cfg.artifact_max_file_bytes:
                return {
                    "stored": False,
                    "error": "file_too_large",
                    "detail": (
                        f"{len(body)} bytes exceeds the {cfg.artifact_max_file_bytes}-byte "
                        f"per-file limit on the {plan} plan."
                    ),
                    "suggested_plan": "max",
                }

            used = artifacts.org_storage_bytes(db, org_id=org_id)
            if used + len(body) > cfg.artifact_storage_bytes:
                return {
                    "stored": False,
                    "error": "storage_quota_exceeded",
                    "detail": (
                        f"{used} of {cfg.artifact_storage_bytes} bytes used on the "
                        f"{plan} plan. Reads still work; forget an artifact or upgrade."
                    ),
                    "suggested_plan": "max",
                }

            checksum = hashlib.sha256(body).hexdigest()
            existing = artifacts.find_active_by_checksum(
                db, org_id=org_id, checksum=checksum
            )
            if existing:
                return {
                    "stored": True,
                    "artifact_id": existing["id"],
                    "version": existing["version"],
                    "status": "active",
                    "deduplicated": True,
                }

            # Project resolution FIRST: it is the only one of the two that can
            # reject the write, and ``ensure_space`` CREATES a row. Running
            # ensure_space first left an orphan space behind on every
            # ``unknown_project`` rejection — a typo'd project name retried in a
            # loop silently fills the org's space list with junk it never asked
            # for. Resolve what can fail before creating anything durable.
            project_id, project_error = _resolve_artifact_project(
                request, db, org_id=org_id, project=project
            )
            if project_error is not None:
                return project_error

            stripped_space = space.strip() if space is not None else ""
            space_id = (
                brain.ensure_space(
                    db, org_id=org_id, name=stripped_space, slug=stripped_space
                )
                if stripped_space
                else None
            )

            new_id = artifacts.create_pending(
                db,
                org_id=org_id,
                agent_id=agent_id,
                title=title,
                summary=summary,
                kind=kind,
                mime_type=mime_type or "text/plain",
                space_id=space_id,
                project_id=project_id,
                supersedes=supersedes,
            )
            artifacts.activate(
                db,
                org_id=org_id,
                artifact_id=new_id,
                size_bytes=len(body),
                checksum=checksum,
                inline_text=content,
                storage_uri=None,
            )

            try:
                await _write_artifact_card(
                    db,
                    org_id=org_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    artifact_id=new_id,
                    title=title,
                    summary=summary,
                    kind=kind,
                    size_bytes=len(body),
                    space_id=space_id,
                    project_id=project_id,
                )
            except Exception:
                # The card is the ONLY discovery mechanism (see module docstring
                # above _write_artifact_card): without it the artifact is active,
                # counted against the org's storage pool, and permanently
                # invisible to recall. artifacts.create_pending/activate each
                # commit independently of this write (write_scoped_memory commits
                # internally and is shared with the verification path, so it
                # cannot be folded into one transaction), so a card-write failure
                # here would otherwise leave a paid-for, invisible row behind.
                # Compensate by retracting the artifact we just activated so it
                # stops counting against the pool and the caller can retry
                # cleanly. Log with the artifact id FIRST, before attempting the
                # compensating retract, so the original failure is never lost if
                # the retract itself also fails.
                logger.exception(
                    "artifact card write failed after activate; retracting "
                    "artifact_id=%s to avoid an invisible, paid-for row",
                    new_id,
                )
                try:
                    artifacts.retract(db, org_id=org_id, artifact_id=new_id)
                except Exception:
                    # Guarded: a failure here must never mask the original
                    # card-write exception. Any row that slips through past
                    # this point is findable in logs by artifact_id above.
                    logger.exception(
                        "compensating retract also failed for artifact_id=%s; "
                        "row may still be active with no card",
                        new_id,
                    )
                return {"stored": False, "error": "card_write_failed"}

            if supersedes:
                artifacts.mark_superseded(db, org_id=org_id, artifact_id=supersedes)
                _hide_card_for_artifact(db, org_id=org_id, artifact_id=supersedes)

            return {
                "stored": True,
                "artifact_id": new_id,
                "version": 1,
                "status": "active",
            }

        # ── Confirm: the client says the bytes are uploaded ──────────────
        if artifact_id is not None:
            row = artifacts.get_artifact(db, org_id=org_id, artifact_id=artifact_id)
            if row is None or row["status"] != "pending":
                # A cross-org id reads as not-found, never as a permission error.
                return {"stored": False, "error": "not_found"}

            s3 = artifact_storage.get_artifact_s3_client()
            key = artifact_storage.object_key(org_id, artifact_id, row["version"])

            # ── Step 1: size, from HEAD alone. No bytes move. ─────────────
            # The presigned PUT is uncapped server-side, so whatever the client
            # declared, the object here may be arbitrarily large. Both limits
            # are enforced against ContentLength BEFORE anything reads the
            # body: measuring by downloading would make the REJECTION path the
            # expensive one, which is an authenticated remote DoS available to
            # the cheapest plan and unstoppable by quota.
            measured_size = await anyio.to_thread.run_sync(
                artifact_storage.head_size, s3, key
            )
            if measured_size is None:
                return {
                    "stored": False,
                    "error": "no_bytes_uploaded",
                    "detail": "No object found at the upload URL. Upload, then confirm.",
                }

            # The declared size only chose a strategy; the quota is charged on
            # what was actually stored.
            if measured_size > cfg.artifact_max_file_bytes:
                return {
                    "stored": False,
                    "error": "file_too_large",
                    "detail": (
                        f"{measured_size} bytes exceeds the "
                        f"{cfg.artifact_max_file_bytes}-byte per-file limit on the "
                        f"{plan} plan."
                    ),
                    "suggested_plan": "max",
                }

            used = artifacts.org_storage_bytes(db, org_id=org_id)
            if used + measured_size > cfg.artifact_storage_bytes:
                return {
                    "stored": False,
                    "error": "storage_quota_exceeded",
                    "detail": (
                        f"{used} of {cfg.artifact_storage_bytes} bytes used on the "
                        f"{plan} plan."
                    ),
                    "suggested_plan": "max",
                }

            # ── Step 2: checksum, streamed. Only now are the bytes read. ──
            # Off the event loop for the same reason: this is the one call here
            # whose cost scales with object size.
            measured_checksum = await anyio.to_thread.run_sync(
                artifact_storage.checksum, s3, key
            )

            # Dedup on the MEASURED checksum, exactly as the inline path does.
            # Without this, identical bytes make ``activate``'s UPDATE violate
            # ``ux_artifacts_org_checksum_active``; the IntegrityError becomes a
            # generic ``internal_error`` and the pending row is stuck forever,
            # unconfirmable and indistinguishable from a server bug.
            existing = artifacts.find_active_by_checksum(
                db, org_id=org_id, checksum=measured_checksum
            )
            if existing is not None and existing["id"] != artifact_id:
                return _dedup_confirm(
                    db,
                    s3,
                    key,
                    org_id=org_id,
                    pending_id=artifact_id,
                    existing=existing,
                )

            try:
                artifacts.activate(
                    db,
                    org_id=org_id,
                    artifact_id=artifact_id,
                    size_bytes=measured_size,
                    checksum=measured_checksum,
                    inline_text=None,
                    storage_uri=artifact_storage.storage_uri(
                        org_id, artifact_id, row["version"]
                    ),
                )
            except IntegrityError:
                # The concurrent-confirm race: two callers confirm identical
                # bytes at once, both pass the dedup lookup above, and the
                # partial unique index picks a winner. The loser lands here and
                # must resolve to the winning artifact, not ``internal_error``.
                db.rollback()
                winner = artifacts.find_active_by_checksum(
                    db, org_id=org_id, checksum=measured_checksum
                )
                if winner is None:
                    # Not a dedup collision — some other constraint. Do not
                    # dress a real failure up as a success.
                    logger.exception(
                        "artifact confirm hit a non-dedup IntegrityError for "
                        "artifact_id=%s",
                        artifact_id,
                    )
                    return {"stored": False, "error": "internal_error"}
                return _dedup_confirm(
                    db, s3, key, org_id=org_id, pending_id=artifact_id, existing=winner
                )

            try:
                await _write_artifact_card(
                    db,
                    org_id=org_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    artifact_id=artifact_id,
                    title=row["title"],
                    summary=row["summary"],
                    kind=row["kind"],
                    size_bytes=measured_size,
                    space_id=row["space_id"],
                    project_id=row["project_id"],
                )
            except Exception:
                # Same compensating-retract pattern as the inline path above:
                # activate() already committed, so a card-write failure here
                # would otherwise leave a paid-for, invisible artifact. Log
                # with the artifact id FIRST, before attempting the
                # compensating retract, so the original failure is never lost
                # if the retract itself also fails.
                logger.exception(
                    "artifact card write failed after confirm/activate; "
                    "retracting artifact_id=%s to avoid an invisible, "
                    "paid-for row",
                    artifact_id,
                )
                try:
                    artifacts.retract(db, org_id=org_id, artifact_id=artifact_id)
                except Exception:
                    # Guarded: a failure here must never mask the original
                    # card-write exception. Any row that slips through past
                    # this point is findable in logs by artifact_id above.
                    logger.exception(
                        "compensating retract also failed for artifact_id=%s; "
                        "row may still be active with no card",
                        artifact_id,
                    )
                return {"stored": False, "error": "card_write_failed"}

            if supersedes:
                artifacts.mark_superseded(db, org_id=org_id, artifact_id=supersedes)
                _hide_card_for_artifact(db, org_id=org_id, artifact_id=supersedes)

            return {
                "stored": True,
                "artifact_id": artifact_id,
                "version": row["version"],
                "status": "active",
            }

        # ── Request: reserve an id and hand back an upload URL ───────────
        if not mime_type:
            return {
                "stored": False,
                "error": "mime_type_required",
                "detail": "Provide `mime_type` when omitting `content`.",
            }

        # A declared size is mandatory on this branch, because it is what gets
        # SIGNED into the upload URL (``presign_put``'s ``ContentLength``). No
        # declared size means no cap to sign, i.e. a URL through which an
        # object of any size can be uploaded — the exact hole that made
        # ``MAX_PENDING_PER_ORG``'s bound false. The previous
        # ``size_bytes or 0`` also sailed through the pool check below, since
        # zero always fits.
        declared = size_bytes if size_bytes is not None else 0
        if declared <= 0:
            return {
                "stored": False,
                "error": "size_bytes_required",
                "detail": (
                    "Provide a positive `size_bytes` when omitting `content`: "
                    "the upload URL is signed for exactly that many bytes, so "
                    "an upload of any other size is rejected."
                ),
            }

        if declared > cfg.artifact_max_file_bytes:
            return {
                "stored": False,
                "error": "file_too_large",
                "detail": (
                    f"{declared} bytes exceeds the {cfg.artifact_max_file_bytes}-byte "
                    f"per-file limit on the {plan} plan."
                ),
                "suggested_plan": "max",
            }

        # Multipart is not implemented, so refuse here rather than further down.
        #
        # This branch used to sit AFTER create_pending and told the caller to
        # "use the Klio CLI, which performs the multipart upload". No such
        # command exists, so the caller was handed an artifact_id it could
        # never fulfil — and the reservation it burned counted against
        # MAX_PENDING_PER_ORG until GC collected it 24 hours later. Refusing
        # before any row is created costs the caller nothing and says something
        # true.
        if declared > artifact_storage.MULTIPART_THRESHOLD_BYTES:
            return {
                "stored": False,
                "error": "too_large_for_single_upload",
                "detail": (
                    f"{declared} bytes exceeds the "
                    f"{artifact_storage.MULTIPART_THRESHOLD_BYTES}-byte limit for a "
                    "single upload. Multipart upload is not available yet, so an "
                    "artifact this large cannot be stored in this release."
                ),
            }

        # The pool, against the DECLARED size. Weaker than the confirm-time
        # check (nothing verifies this number), but its absence was a complete
        # bypass: an org could loop this branch, collect presigned PUTs, and
        # upload unbounded bytes through them with nothing counting or
        # rejecting anything until GC ran 24 hours later.
        used = artifacts.org_storage_bytes(db, org_id=org_id)
        if used + declared > cfg.artifact_storage_bytes:
            return {
                "stored": False,
                "error": "storage_quota_exceeded",
                "detail": (
                    f"{used} of {cfg.artifact_storage_bytes} bytes used on the "
                    f"{plan} plan. Reads still work; forget an artifact or upgrade."
                ),
                "suggested_plan": "max",
            }

        # The declared size is unverifiable, so bound the blast radius by
        # capping live reservations too: without this, `size_bytes=1` a
        # thousand times yields a thousand uncounted upload URLs. See
        # ``artifacts.MAX_PENDING_PER_ORG`` for the number and its reasoning.
        pending = artifacts.pending_count(db, org_id=org_id)
        if pending >= artifacts.MAX_PENDING_PER_ORG:
            return {
                "stored": False,
                "error": "too_many_pending_uploads",
                "detail": (
                    f"{pending} uploads are already awaiting confirmation "
                    f"(limit {artifacts.MAX_PENDING_PER_ORG}). Confirm or "
                    "abandon them before starting more."
                ),
            }

        # Project first, then space — same ordering and same reason as the
        # inline branch above: never create a space row for a write that is
        # about to be rejected.
        project_id, project_error = _resolve_artifact_project(
            request, db, org_id=org_id, project=project
        )
        if project_error is not None:
            return project_error

        stripped_space = space.strip() if space is not None else ""
        space_id = (
            brain.ensure_space(
                db, org_id=org_id, name=stripped_space, slug=stripped_space
            )
            if stripped_space
            else None
        )

        new_id = artifacts.create_pending(
            db,
            org_id=org_id,
            agent_id=agent_id,
            title=title,
            summary=summary,
            kind=kind,
            mime_type=mime_type,
            space_id=space_id,
            project_id=project_id,
            supersedes=supersedes,
        )

        s3 = artifact_storage.get_artifact_s3_client()
        key = artifact_storage.object_key(org_id, new_id, 1)

        return {
            "stored": False,
            "artifact_id": new_id,
            # Signed for exactly ``declared`` bytes: an upload of any other
            # size fails signature verification at the storage layer, so this
            # URL is genuinely capped and not merely advertised as capped.
            "upload_url": artifact_storage.presign_put(
                s3, key, content_length=declared
            ),
            "method": "PUT",
            "expires_in": artifact_storage.PRESIGN_TTL_SECONDS,
            "detail": "Upload the bytes, then call artifact_put again with this artifact_id.",
        }
    except Exception:
        logger.exception("artifact_put failed")
        return {"stored": False, "error": "internal_error"}
    finally:
        db.close()


_TEXT_MIME_PREFIXES = ("text/",)
_TEXT_MIME_EXACT = {"application/json", "application/xml", "application/yaml"}


def _is_text_mime(mime_type: str) -> bool:
    # MIME types commonly carry a parameter (e.g. "application/json;
    # charset=utf-8"); match against the normalized type/subtype only, not
    # the raw header value, or a genuinely-text record gets misclassified as
    # binary.
    normalized = mime_type.split(";", 1)[0].strip().lower()
    return normalized.startswith(_TEXT_MIME_PREFIXES) or normalized in _TEXT_MIME_EXACT


async def _perform_artifact_get(
    ctx: Context,
    *,
    artifact_id: str,
    mode: str = "text",
    offset: int = 0,
) -> dict[str, Any]:
    """Shared body for ``artifact_get``.

    Text is returned one bounded page at a time: a recall result can point at a
    multi-megabyte artifact, and a context window is the real constraint.
    Binary artifacts are never guessed at — v1 does no extraction, so the caller
    is told to use ``mode='url'``.

    **Every blocking S3 call here runs through ``anyio.to_thread.run_sync``**,
    for the same reason the confirm path in :func:`_perform_artifact_put` does:
    boto3 is synchronous, and a network round trip taken inline stalls every
    other MCP request sharing this replica's event loop.

    **The page is bounded at the storage layer, not sliced after the fact.**
    This used to call ``artifact_storage.get_text``, which did
    ``get_object(...)["Body"].read().decode("utf-8")`` — an unbounded
    whole-object read with no size gate at all. Plans allow up to 1GB per file,
    so returning 51,200 characters of a 900MB ``text/plain`` artifact meant
    buffering 900MB plus a UTF-8 decode into RAM, on the loop: an authenticated
    caller could stall the replica by reading page one. ``get_text_page`` takes
    the character window and issues a byte ``Range`` sized from it, so the
    transfer is proportional to the page rather than the object. See that
    function for why a character-addressed page must still be anchored at
    byte 0 and how it stays bounded regardless.
    """
    request = ctx.request_context.request
    if request is None:
        return {"error": "unauthorized", "detail": "no request context"}

    _attach_client_name(ctx, request)

    try:
        org_id, _agent_id, _plan, _overrides = resolve_principal_plan(request)
    except HTTPException as exc:
        return {"error": "unauthorized", "detail": str(exc.detail)}

    db = SessionLocal()
    try:
        row = artifacts.get_artifact(db, org_id=org_id, artifact_id=artifact_id)
        if row is None or row["status"] not in ("active", "superseded"):
            return {"error": "not_found"}

        head = {
            "artifact_id": row["id"],
            "title": row["title"],
            "kind": row["kind"],
        }

        if mode == "url":
            s3 = artifact_storage.get_artifact_s3_client()
            key = artifact_storage.object_key(org_id, row["id"], row["version"])
            if row["inline_text"] is not None:
                # Inline artifacts have no object yet; materialise one so the
                # URL contract holds for every kind. The row deliberately stays
                # inline (``storage_uri`` NULL) — Postgres remains the body of
                # record, and the quota is charged once, not twice.
                #
                # The materialised object is deterministic and idempotent: the
                # key is ``object_key(org_id, id, version)`` and artifacts are
                # immutable, so every call re-puts identical bytes at the
                # identical key. A stale copy is therefore harmless — it can
                # only ever be overwritten with itself, never orphaned under a
                # second identity. And because the key is derived rather than
                # recorded, the GC purge pass finds it without consulting
                # ``storage_uri`` (see ``scheduler._purge_dead_artifacts``), so
                # it is destroyed with the artifact rather than outliving it.
                await anyio.to_thread.run_sync(
                    artifact_storage.put_text,
                    s3,
                    key,
                    row["inline_text"],
                    row["mime_type"],
                )
            url = await anyio.to_thread.run_sync(artifact_storage.presign_get, s3, key)
            return {
                **head,
                # Never logged: a presigned URL is a bearer credential.
                "url": url,
                "expires_in": artifact_storage.PRESIGN_TTL_SECONDS,
            }

        if row["inline_text"] is not None:
            # Already in memory and capped at INLINE_MAX_BYTES (64KB), so
            # slicing it is bounded by construction.
            full = row["inline_text"]
            page = full[offset : offset + artifacts.TEXT_PAGE_CHARS]
            has_more = (offset + len(page)) < len(full)
        elif _is_text_mime(row["mime_type"]):
            # A character is at least one byte in UTF-8, so an offset at or
            # beyond the object's SERVER-MEASURED size is certainly past its
            # last character. Answering that from the row costs nothing and
            # keeps a caller from asking S3 to range over the whole object with
            # an out-of-range offset — the one input on this path that scales
            # the transfer, since the range is otherwise sized from it.
            size_bytes = row["size_bytes"] or 0
            if offset >= size_bytes:
                return {**head, "text": "", "offset": offset, "has_more": False}

            s3 = artifact_storage.get_artifact_s3_client()
            key = artifact_storage.object_key(org_id, row["id"], row["version"])
            page, has_more = await anyio.to_thread.run_sync(
                functools.partial(
                    artifact_storage.get_text_page,
                    s3,
                    key,
                    offset=offset,
                    limit=artifacts.TEXT_PAGE_CHARS,
                )
            )
        else:
            return {
                **head,
                "error": "not_text",
                "detail": (
                    f"{row['mime_type']} is not text. Call artifact_get with "
                    "mode='url' to download it."
                ),
            }

        return {
            **head,
            "text": page,
            "offset": offset,
            "has_more": has_more,
        }
    except Exception:
        logger.exception("artifact_get failed")
        return {"error": "internal_error"}
    finally:
        db.close()


async def _write_artifact_card(
    db: Any,
    *,
    org_id: str,
    agent_id: str,
    session_id: str,
    artifact_id: str,
    title: str,
    summary: str,
    kind: str,
    size_bytes: int,
    space_id: Optional[str],
    project_id: Optional[str],
) -> None:
    """Write the one memory row that makes an artifact discoverable.

    Dedup is disabled: two artifacts may legitimately carry near-identical
    summaries, and silently collapsing their cards would make one of them
    permanently unfindable.
    """
    await brain.write_scoped_memory(
        db,
        org_id=org_id,
        agent_id=agent_id,
        content=artifacts.card_content(title, summary, kind, size_bytes),
        memory_type="artifact",
        session_id=session_id,
        space_id=space_id,
        project_id=project_id,
        metadata={"artifact_id": artifact_id, "source": "mcp"},
        dedup=False,
        scope="org",
    )


def _hide_card_for_artifact(db: Any, *, org_id: str, artifact_id: str) -> None:
    """Hide an artifact's card from recall without deleting the row."""
    db.execute(
        text(
            "UPDATE session_memories SET recall_hidden = TRUE "
            "WHERE org_id = :org_id AND memory_type = 'artifact' "
            "AND metadata->>'artifact_id' = :artifact_id"
        ),
        {"org_id": org_id, "artifact_id": artifact_id},
    )
    db.commit()


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
        # EXTRACTED (stated or done) vs INFERRED (an LLM deduced it) vs
        # AMBIGUOUS. Returned so a caller can WEIGH a memory rather than
        # treating a model's deduction as something the user said. Distinct
        # from ``confidence`` above, which is a float and always has been.
        "provenance": record.provenance,
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
        scope: ``"org"`` (default — the visibility ladder: the team brain plus
            the caller's own private and project memories), ``"private"``
            (theirs alone), or the deprecated ``"agent"``. The schema restricts
            it to these three, so :func:`retrieve_memories_scoped` never raises
            ``ValueError`` for a valid call — except for ``private`` without a
            resolvable user, which is caught before the call.
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

    # The read half of the visibility ladder. Forwarded on every scope, not
    # just 'private': at the default scope it is what unions in the caller's
    # own private captures and their project memberships.
    user_id = resolve_request_user_id(request)
    if scope == "private" and not user_id:
        return {
            "memories": [],
            "count": 0,
            "error": "unattributed_caller",
            "detail": (
                "scope='private' needs a caller we can attribute to a person. "
                "This request resolved to no user, so there are no memories "
                "that could be yours."
            ),
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
            user_id=user_id,
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
        payload = {
            "memories": memories,
            "count": len(memories),
            "guidance": RECALL_WRITE_NUDGE,
        }
        # Structural compression is lossless, so it is unconditional: there is
        # no information to opt into keeping. Nulls go and fields identical on
        # every row are hoisted into `common`, which is additive — an existing
        # consumer reading `memories[i]["content"]` is unaffected. See the
        # compression design spec.
        #
        # Wrapped even though compress_payload fails open internally: this is a
        # request path, and a compressor is never allowed to be the reason a
        # recall fails. The uncompressed payload is always a correct answer.
        try:
            payload, _stats = compress_payload(payload)
        except Exception:
            logger.warning(
                "recall compression raised; returning the uncompressed payload",
                exc_info=True,
            )
        return payload
    except Exception:
        logger.exception("brain recall failed (scope=%s)", scope)
        return {"error": "recall_failed", "guidance": RECALL_WRITE_NUDGE}
    finally:
        db.close()


async def _perform_compress(
    ctx: Context, *, content: str, lossy: bool = False
) -> dict[str, Any]:
    """Shared body for the Klio ``compress`` tool.

    The response pairs two fields and the pairing is the whole contract:
    ``lossy=true`` is only ever returned alongside a ``handle`` that
    ``klio_retrieve`` resolves, and ``lossy=false`` is never returned with
    one. A handle with nothing behind it, or lossy output with no way back,
    are both broken promises.

    **The handle is a separate field, never a marker inside the text.** Models
    echo inline markers back, and an echoed marker inside a diff or a source
    file corrupts it.

    **If the original cannot be stored, the content comes back unchanged.**
    Returning compressed-but-unrecoverable output would be the one failure the
    originals cache exists to prevent, so an oversized original or a database
    error costs the compression, not the content.

    Fail-open throughout: every error path returns the input with
    ``lossy=false``.
    """
    request = ctx.request_context.request
    if request is None:
        return {"error": "unauthorized", "detail": "no request context"}

    _attach_client_name(ctx, request)

    try:
        org_id, _agent_id, _plan, _overrides = resolve_principal_plan(request)
    except HTTPException as exc:
        return {"error": "unauthorized", "detail": str(exc.detail)}

    user_id = resolve_request_user_id(request)

    def _unchanged() -> dict[str, Any]:
        tokens = estimate_tokens(content) if isinstance(content, str) else 0
        return {
            "compressed": content,
            "original_tokens": tokens,
            "compressed_tokens": tokens,
            "lossy": False,
        }

    if not isinstance(content, str) or not content:
        return _unchanged()

    try:
        result = route_compress(content, allow_lossy=lossy)
    except Exception:
        logger.exception("compression failed; returning the content unchanged")
        return _unchanged()

    payload = {
        "compressed": result.text,
        "original_tokens": result.original_tokens,
        "compressed_tokens": result.compressed_tokens,
        "lossy": result.lossy,
    }
    if not result.lossy:
        return payload

    db = SessionLocal()
    try:
        payload["handle"] = store_original(
            db, org_id=org_id, user_id=user_id, text=content
        )
    except OriginalTooLarge:
        logger.info("original exceeds the inline cache; returning it uncompressed")
        return _unchanged()
    except Exception:
        logger.exception("could not store the original; returning it uncompressed")
        return _unchanged()
    finally:
        db.close()

    return payload


async def _perform_klio_retrieve(
    ctx: Context, *, handle: str, offset: int = 0
) -> dict[str, Any]:
    """Shared body for ``klio_retrieve``: recover a compressed original.

    Paging is deliberately the ``artifact_get`` contract — ``TEXT_PAGE_CHARS``
    per page, ``offset`` to continue, ``has_more`` to stop — so an agent that
    knows one knows the other.

    **Refusals carry no signal.** A handle belonging to another user, another
    org, or nobody at all returns the same ``not_found``. ``expired`` is told
    only to the owner: for anyone else it would confirm the handle is real,
    which is exactly what ``not_found`` exists to avoid.
    """
    request = ctx.request_context.request
    if request is None:
        return {"error": "unauthorized", "detail": "no request context"}

    _attach_client_name(ctx, request)

    try:
        org_id, _agent_id, _plan, _overrides = resolve_principal_plan(request)
    except HTTPException as exc:
        return {"error": "unauthorized", "detail": str(exc.detail)}

    user_id = resolve_request_user_id(request)

    db = SessionLocal()
    try:
        original = load_original(db, org_id=org_id, user_id=user_id, handle=handle)
        if original is None:
            # Only the owner of a real-but-stale row learns it expired.
            if handle_is_expired(db, org_id=org_id, user_id=user_id, handle=handle):
                return {
                    "error": "expired",
                    "detail": (
                        "This original has passed its 24-hour retention window "
                        "and is no longer recoverable."
                    ),
                }
            return {"error": "not_found"}

        page = original[offset : offset + artifacts.TEXT_PAGE_CHARS]
        return {
            "text": page,
            "offset": offset,
            "has_more": (offset + len(page)) < len(original),
        }
    except Exception:
        logger.exception("klio_retrieve failed")
        return {"error": "internal_error"}
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
        # An id may name an artifact or a memory. Try the artifact first: a hit
        # must retract the bytes AND hide the card, or recall keeps surfacing a
        # card pointing at something that is gone.
        #
        # A non-UUID id (memory ids are not always UUIDs) makes the ::uuid cast
        # raise and leaves the transaction aborted, which would break the memory
        # path below. Roll back before falling through.
        try:
            if artifacts.retract(db, org_id=org_id, artifact_id=memory_id):
                _hide_card_for_artifact(db, org_id=org_id, artifact_id=memory_id)
                return {"forgotten": True, "kind": "artifact", "id": memory_id}
        except Exception:
            db.rollback()
            logger.debug("forget: %s is not an artifact id, trying memory", memory_id)

        # user_id gates private rows: without it every private row in the org
        # looks unowned and a teammate could retract memories they cannot read.
        return retract_memory(
            db,
            org_id=org_id,
            agent_id=agent_id,
            memory_id=memory_id,
            via="mcp",
            user_id=resolve_request_user_id(request),
        )
    except Exception:
        logger.exception("brain forget failed")
        return {"forgotten": False, "error": "forget_failed"}
    finally:
        db.close()


async def _perform_share(ctx: Context, *, memory_id: str, to: str) -> dict[str, Any]:
    """Shared body for the ``share`` tool: promote one of MY private memories.

    Authorization lives entirely in :func:`shared.promotion.promote_memory`,
    whose SELECT carries ``user_id = :user_id`` so a row belonging to someone
    else is never fetched, let alone modified. This layer's job is to resolve
    the caller and to refuse cleanly when it cannot.

    Sharing to ``'project'`` additionally requires membership of the project
    the memory is filed under. Without that check a user could push a private
    memory into a project boundary they have no standing in — writing across a
    line they cannot read across.
    """
    request = ctx.request_context.request
    if request is None:
        logger.error("brain share got no transport request; cannot authenticate")
        return {
            "shared": False,
            "error": "unauthorized",
            "detail": "no request context",
        }

    _attach_client_name(ctx, request)

    try:
        org_id, _agent_id = resolve_principal(request)
    except HTTPException as exc:
        return {"shared": False, "error": "unauthorized", "detail": str(exc.detail)}

    user_id = resolve_request_user_id(request)
    if not user_id:
        return {
            "shared": False,
            "error": "unattributed_caller",
            "detail": (
                "Sharing requires a caller we can attribute to a person. This "
                "request resolved to no user, so it owns no private memories."
            ),
        }

    db = SessionLocal()
    try:
        if to == "project":
            row = db.execute(
                text(
                    """
                    SELECT project_id FROM session_memories
                    WHERE id = :id AND org_id = :org_id AND user_id = :user_id
                      AND status = 'active'
                    LIMIT 1
                    """
                ),
                {"id": memory_id, "org_id": org_id, "user_id": user_id},
            ).fetchone()
            if row is None or row.project_id is None:
                # Same answer for "not yours", "does not exist" and "not filed
                # under a project" — see promote_memory on why not_found.
                return {"shared": False, "error": "not_found"}
            if not projects.is_member(
                db, org_id=org_id, project_id=str(row.project_id), user_id=user_id
            ):
                return {
                    "shared": False,
                    "error": "not_a_member",
                    "detail": (
                        "You are not a member of the project this memory is "
                        "filed under, so you cannot share it there."
                    ),
                }

        return promote_memory(
            db, org_id=org_id, memory_id=memory_id, user_id=user_id, to=to, via="mcp"
        )
    except Exception:
        logger.exception("brain share failed (to=%s)", to)
        return {"shared": False, "error": "share_failed"}
    finally:
        db.close()


async def _synthesize_entity_summary(brief: dict[str, Any]) -> Optional[str]:
    """LLM-synthesize the 2-3 sentence entity summary from the VISIBLE slice.

    Only the caller's own visible evidence (already ladder-filtered by
    ``graph_store.entity_brief``) reaches the prompt, so the summary can never
    say more than the caller could read row by row. Returns ``None`` on any
    failure — explore degrades to a summary-less brief rather than erroring.
    """
    contents = [m.content for m in brief.get("memories", [])[:20] if m.content]
    if not contents:
        return None
    prompt = (
        f"Synthesize what this team knows about '{brief['entity']}' "
        "in 2-3 sentences, from the memories below. State only what the "
        "memories support. "
        + _NO_CREDENTIALS_RULE
        + "\n\n- "
        + "\n- ".join(contents)
        + '\n\nRespond with JSON: {"summary": "<your summary>"}'
    )
    try:
        result = await call_llm(prompt)
    except Exception:
        logger.warning("explore: summary synthesis raised", exc_info=True)
        return None
    if not isinstance(result, dict):
        return None
    summary = result.get("summary")
    return summary if isinstance(summary, str) and summary else None


def _contributor_labels(brief: dict[str, Any], user_id: Optional[str]) -> list[str]:
    """Contributors as COUNTS, never identities.

    ``["you", "2 teammates"]`` when the caller authored part of the visible
    slice, ``["3 teammates"]`` when not, ``[]`` when nothing visible is
    attributed. No user id — not even the caller's own — appears in the
    payload: ids are principals, and a brief that echoed them would hand every
    MCP client a directory of who wrote what.
    """
    total = int(brief.get("contributors", 0))
    caller_contributed = bool(user_id) and any(
        m.user_id == user_id for m in brief.get("memories", [])
    )
    teammates = total - (1 if caller_contributed else 0)
    labels: list[str] = []
    if caller_contributed:
        labels.append("you")
    if teammates > 0:
        labels.append(f"{teammates} teammate" + ("s" if teammates != 1 else ""))
    return labels


async def _perform_explore(ctx: Context, *, entity: str) -> dict[str, Any]:
    """Shared body for the ``explore`` tool: an entity brief, or not_found.

    The visibility work happens in :func:`shared.graph_store.entity_brief`
    (derived from evidence, ladder in SQL, both edge endpoints checked); this
    layer resolves the caller and shapes the payload. Two hard rules live
    here:

    * ``None`` from the store — covering a nonexistent entity, another org's
      entity, and one whose every evidencing memory is invisible to this
      caller — becomes ONE opaque ``not_found`` payload. Never an empty brief
      (it would confirm the name exists), never ``forbidden``.
    * The raw evidence rows do NOT ship. The brief is the product; the rows
      are reachable via ``recall`` under the same ladder.
    """
    request = ctx.request_context.request
    if request is None:
        logger.error("explore got no transport request; cannot authenticate")
        return {"error": "unauthorized", "detail": "no request context"}

    _attach_client_name(ctx, request)

    try:
        org_id, _agent_id = resolve_principal(request)
    except HTTPException as exc:
        return {"error": "unauthorized", "detail": str(exc.detail)}

    user_id = resolve_request_user_id(request)

    db = SessionLocal()
    try:
        brief = graph_store.entity_brief(
            db, org_id=org_id, viewer_user_id=user_id, entity=entity
        )
        if brief is None:
            return {"error": "not_found", "entity": entity}

        summary = await _synthesize_entity_summary(brief)
        return {
            "entity": brief["entity"],
            "kind": brief["kind"],
            "summary": summary,
            # The summary is an on-the-fly LLM deduction over the caller's
            # visible slice; labelled so a client never mistakes it for
            # something a person stated.
            "summary_provenance": "INFERRED" if summary else None,
            "decisions": brief["decisions"],
            "superseded": brief["superseded"],
            "contributors": _contributor_labels(brief, user_id),
            "provenance": brief["provenance"],
            "evidence_count": brief["evidence_count"],
        }
    except Exception:
        logger.exception("explore failed")
        return {"error": "explore_failed"}
    finally:
        db.close()


def _serialize_project(p: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe projection of a :func:`shared.projects.list_projects` row."""
    return {
        "id": p["id"],
        "name": p["display_name"],
        "git_remote": p["git_remote"],
        "repo_root_path": p["repo_root_path"],
        "member_count": p["member_count"],
        "last_seen_at": (
            p["last_seen_at"].isoformat() if p.get("last_seen_at") else None
        ),
        "created_at": p["created_at"].isoformat() if p.get("created_at") else None,
    }


async def _perform_project_create(
    ctx: Context,
    *,
    name: str,
    git_remote: Optional[str],
    repo_root_path: Optional[str],
) -> dict[str, Any]:
    """Create (or adopt) a project and make the caller its administrator.

    ``ensure_project`` already auto-creates a project on any write that carries
    a git remote or repo root — that is how 106 of them exist. So this tool is
    for NAMING and ORGANISING ahead of use, and for one thing auto-creation
    cannot do: establishing who administers it.

    Adoption rule for a project that already exists:

    * the caller is already a member → their role is raised to ``admin``;
    * the project has NO members at all → the caller becomes its first admin.
      Auto-created projects are all in this state, and claiming one grants
      nothing retroactively: a project with no members has no readable
      ``scope='project'`` rows, because nothing could have been written into a
      boundary nobody was inside;
    * someone else already administers it → ``forbidden``. This is the case
      that must not fall through to a silent claim.
    """
    request = ctx.request_context.request
    if request is None:
        return {
            "created": False,
            "error": "unauthorized",
            "detail": "no request context",
        }

    _attach_client_name(ctx, request)

    try:
        org_id, _agent_id = resolve_principal(request)
    except HTTPException as exc:
        return {"created": False, "error": "unauthorized", "detail": str(exc.detail)}

    user_id = resolve_request_user_id(request)
    if not user_id:
        return {
            "created": False,
            "error": "unattributed_caller",
            "detail": (
                "Creating a project makes you its administrator, which needs a "
                "caller we can attribute to a person."
            ),
        }

    stripped = (name or "").strip()
    if not stripped:
        return {"created": False, "error": "name is required"}
    if not git_remote and not repo_root_path:
        # Mirrors ck_projects_has_identity: a project with neither is
        # unrepresentable and, worse, unresolvable by every later lookup.
        return {
            "created": False,
            "error": "identity_required",
            "detail": (
                "Provide git_remote or repo_root_path. A project is keyed by "
                "one of them, and without either nothing can resolve it later."
            ),
        }

    db = SessionLocal()
    try:
        project_id = brain.ensure_project(
            db,
            org_id=org_id,
            display_name=stripped,
            git_remote=git_remote,
            repo_root_path=repo_root_path,
        )
        existing = projects.member_count(db, org_id=org_id, project_id=project_id)
        already_member = projects.is_member(
            db, org_id=org_id, project_id=project_id, user_id=user_id
        )
        if existing and not already_member:
            return {
                "created": False,
                "id": project_id,
                "error": "forbidden",
                "detail": (
                    "This project already exists and is administered by someone "
                    "else. Ask one of its admins for access."
                ),
            }

        projects.add_member(
            db,
            org_id=org_id,
            project_id=project_id,
            user_id=user_id,
            role=projects.ROLE_ADMIN,
            granted_by=user_id,
        )
        projects.record_grant_audit(
            db,
            org_id=org_id,
            project_id=project_id,
            granted_to=user_id,
            granted_to_email=None,
            granted_by=user_id,
            role=projects.ROLE_ADMIN,
            surface=projects.SURFACE_MCP,
        )
        return {
            "created": True,
            "id": project_id,
            "name": stripped,
            "role": projects.ROLE_ADMIN,
            # 0 means this call established the project's first administrator —
            # either a brand-new project or an adopted auto-created one.
            "members_before": existing,
        }
    except ValueError as exc:
        return {"created": False, "error": "invalid_project", "detail": str(exc)}
    except Exception:
        logger.exception("project_create failed")
        return {"created": False, "error": "project_create_failed"}
    finally:
        db.close()


async def _perform_project_list(ctx: Context) -> dict[str, Any]:
    """List the org's projects. Names and member COUNTS only — never contents."""
    request = ctx.request_context.request
    if request is None:
        return {"error": "unauthorized", "detail": "no request context"}

    _attach_client_name(ctx, request)

    try:
        org_id, _agent_id = resolve_principal(request)
    except HTTPException as exc:
        return {"error": "unauthorized", "detail": str(exc.detail)}

    db = SessionLocal()
    try:
        rows = projects.list_projects(db, org_id=org_id)
        return {
            "projects": [_serialize_project(p) for p in rows],
            "count": len(rows),
        }
    except Exception:
        logger.exception("project_list failed")
        return {"error": "project_list_failed"}
    finally:
        db.close()


async def _perform_project_members(ctx: Context, *, project: str) -> dict[str, Any]:
    """List a project's members.

    Readable by any member of the project. A non-member gets ``not_found``
    rather than a member list: who has access to a project is itself part of
    what the boundary protects, and confirming a project exists to someone
    outside it is a needless disclosure.
    """
    request = ctx.request_context.request
    if request is None:
        return {"error": "unauthorized", "detail": "no request context"}

    _attach_client_name(ctx, request)

    try:
        org_id, _agent_id = resolve_principal(request)
    except HTTPException as exc:
        return {"error": "unauthorized", "detail": str(exc.detail)}

    user_id = resolve_request_user_id(request)

    db = SessionLocal()
    try:
        project_id = brain.find_project_id(db, org_id=org_id, identifier=project)
        if project_id is None:
            return {"error": "not_found", "detail": f"unknown project: {project}"}
        if not projects.is_member(
            db, org_id=org_id, project_id=project_id, user_id=user_id
        ):
            return {"error": "not_found", "detail": f"unknown project: {project}"}
        members = projects.list_members(db, org_id=org_id, project_id=project_id) or []
        return {
            "project": {"id": project_id},
            "members": [
                {
                    "user_id": m["user_id"],
                    "role": m["role"],
                    "granted_by": m["granted_by"],
                    "granted_at": (
                        m["granted_at"].isoformat() if m.get("granted_at") else None
                    ),
                }
                for m in members
            ],
            "count": len(members),
        }
    except Exception:
        logger.exception("project_members failed")
        return {"error": "project_members_failed"}
    finally:
        db.close()


async def _perform_project_grant(
    ctx: Context, *, project: str, user: str, role: str
) -> dict[str, Any]:
    """Grant a teammate access to a project they are not yet in.

    This is the tool that converts a prompt injection into real data access, so
    it is the most constrained one in the file. Four gates, in order, each of
    which fails closed:

    1. The caller must resolve to a person. An unattributable request grants
       nothing.
    2. The project must resolve within the caller's own org.
    3. The caller must ADMINISTER that project — not merely belong to it, and
       certainly not merely belong to the org. This is what holds the blast
       radius at "a project you already control".
    4. The grantee is named by EMAIL and must already be a member of the
       caller's org, resolved against the org's own membership records
       (:mod:`app.directory`). A raw user id is refused outright: accepting one
       would let injected text name a principal directly, and an agent has no
       legitimate way to know a UUID anyway.

    Then, and only then, the grant is written and audited.
    """
    request = ctx.request_context.request
    if request is None:
        return {
            "granted": False,
            "error": "unauthorized",
            "detail": "no request context",
        }

    _attach_client_name(ctx, request)

    try:
        org_id, _agent_id = resolve_principal(request)
    except HTTPException as exc:
        return {"granted": False, "error": "unauthorized", "detail": str(exc.detail)}

    caller = resolve_request_user_id(request)
    if not caller:
        return {
            "granted": False,
            "error": "unattributed_caller",
            "detail": (
                "Granting project access requires a caller we can attribute to "
                "a person."
            ),
        }

    email = (user or "").strip()
    if "@" not in email:
        return {
            "granted": False,
            "error": "email_required",
            "detail": (
                "`user` must be a teammate's email address. User ids are not "
                "accepted here."
            ),
        }
    if role not in (projects.ROLE_MEMBER, projects.ROLE_ADMIN):
        return {"granted": False, "error": "invalid_role"}

    db = SessionLocal()
    try:
        project_id = brain.find_project_id(db, org_id=org_id, identifier=project)
        if project_id is None:
            return {
                "granted": False,
                "error": "not_found",
                "detail": f"unknown project: {project}",
            }

        if not projects.is_admin(
            db, org_id=org_id, project_id=project_id, user_id=caller
        ):
            return {
                "granted": False,
                "error": "forbidden",
                "detail": (
                    "You may only grant access to projects you administer. "
                    "Create the project, or ask one of its admins."
                ),
            }

        grantee = resolve_org_member_by_email(org_id, email)
        if not grantee:
            return {
                "granted": False,
                "error": "not_in_org",
                "detail": (
                    f"{email} is not a member of this organisation, so they "
                    "cannot be granted access to its projects."
                ),
            }

        ok = projects.add_member(
            db,
            org_id=org_id,
            project_id=project_id,
            user_id=grantee,
            role=role,
            granted_by=caller,
        )
        if not ok:
            return {"granted": False, "error": "not_found"}

        projects.record_grant_audit(
            db,
            org_id=org_id,
            project_id=project_id,
            granted_to=grantee,
            granted_to_email=email,
            granted_by=caller,
            role=role,
            surface=projects.SURFACE_MCP,
        )
        logger.info(
            "project_grant: %s granted %s (%s) role=%s on project %s via mcp",
            caller,
            grantee,
            email,
            role,
            project_id,
        )
        return {
            "granted": True,
            "project_id": project_id,
            "user_id": grantee,
            "email": email,
            "role": role,
        }
    except Exception:
        logger.exception("project_grant failed")
        return {"granted": False, "error": "project_grant_failed"}
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
            "scope by agent. "
            "Results with memory_type 'artifact' are stored files or documents: "
            "call artifact_get with the metadata's artifact_id to read them. "
            "The result is structurally compressed: null fields are omitted "
            "from a row, and fields identical on every row may be hoisted "
            "into a top-level `common` object — read a memory as `common` "
            "merged with the row. `id`, `content` and `scope` always stay on "
            "the row itself."
        ),
    )
    async def recall(
        ctx: Context,
        query: Annotated[
            str, Field(description="What to search the team's memory for.")
        ],
        scope: Annotated[
            Literal["org", "agent", "private"],
            Field(
                description=(
                    "'org' (default) returns the team's shared brain PLUS your "
                    "own private memories and any project you belong to. "
                    "'private' narrows to your own memories alone. 'agent' is "
                    "deprecated."
                )
            ),
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
        name="artifact_put",
        description=(
            "Store a durable artifact — a document, a JSON record, or a file — "
            "and make it discoverable to every agent in the org. Pass `content` "
            "for text up to 64KB (one call). Omit `content` to get an upload URL "
            "for larger or binary data, then call again with the returned "
            "`artifact_id` to confirm. Args: title, summary, kind "
            "('document'|'record'|'file')."
        ),
    )
    async def artifact_put(
        ctx: Context,
        title: Annotated[
            str, Field(description="Short name for the artifact.", max_length=200)
        ],
        summary: Annotated[
            str,
            Field(
                description="One line describing what this is; other agents find it by this.",
                max_length=500,
            ),
        ],
        kind: Annotated[
            Literal["document", "record", "file"],
            Field(
                description="'document' (prose), 'record' (JSON), or 'file' (binary)."
            ),
        ],
        content: Annotated[
            Optional[str],
            Field(description="Inline text, up to 64KB. Omit for the upload-URL path."),
        ] = None,
        mime_type: Annotated[
            Optional[str],
            Field(description="MIME type. Required when content is omitted."),
        ] = None,
        size_bytes: Annotated[
            Optional[int],
            Field(
                description=(
                    "Byte size of the upload. Required when content is "
                    "omitted: the upload URL is signed for exactly this many "
                    "bytes, so an upload of any other size is rejected."
                )
            ),
        ] = None,
        space: SpaceParam = None,
        project: Annotated[
            Optional[str], Field(description="Optional project id.")
        ] = None,
        supersedes: Annotated[
            Optional[str], Field(description="Artifact id this replaces.")
        ] = None,
        artifact_id: Annotated[
            Optional[str], Field(description="Confirm an upload started earlier.")
        ] = None,
    ) -> dict[str, Any]:
        return await _perform_artifact_put(
            ctx,
            title=title,
            summary=summary,
            kind=kind,
            content=content,
            mime_type=mime_type,
            size_bytes=size_bytes,
            space=space,
            project=project,
            supersedes=supersedes,
            artifact_id=artifact_id,
        )

    @mcp.tool(
        name="artifact_get",
        description=(
            "Fetch an artifact surfaced by recall. mode='text' returns its "
            "contents one page at a time (use `offset` to continue); mode='url' "
            "returns a short-lived download link for binary content."
        ),
    )
    async def artifact_get(
        ctx: Context,
        artifact_id: Annotated[
            str, Field(description="From a recall result's metadata.")
        ],
        mode: Annotated[
            Literal["text", "url"],
            Field(description="'text' (default) or 'url' for a download link."),
        ] = "text",
        offset: Annotated[
            int, Field(description="Character offset for paging text.", ge=0)
        ] = 0,
    ) -> dict[str, Any]:
        return await _perform_artifact_get(
            ctx, artifact_id=artifact_id, mode=mode, offset=offset
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

    @mcp.tool(
        name="share",
        description=(
            "Promote one of YOUR OWN private memories to the team, or to the "
            "project it is filed under. Passive session capture is private by "
            "default; this is how something becomes shared. You can only share "
            "memories you wrote. Args: memory_id, to ('org' default | "
            "'project')."
        ),
    )
    async def share(
        ctx: Context,
        memory_id: Annotated[
            str, Field(description="Id of your private memory, from a recall result.")
        ],
        to: Annotated[
            Literal["org", "project"],
            Field(
                description=(
                    "'org' (default) shares with everyone in the org; "
                    "'project' shares with members of the project this memory "
                    "is filed under."
                )
            ),
        ] = "org",
    ) -> dict[str, Any]:
        return await _perform_share(ctx, memory_id=memory_id, to=to)

    @mcp.tool(
        name="project_create",
        description=(
            "Name a project and become its administrator. NOTE: projects are "
            "already created automatically the first time a memory is written "
            "from a repo, so you do NOT need to call this before writing — use "
            "it to give a project a proper name ahead of time, and to become "
            "the admin who can grant teammates access to it. Args: name, "
            "optional git_remote, optional repo_root_path (one of the two is "
            "required)."
        ),
    )
    async def project_create(
        ctx: Context,
        name: Annotated[str, Field(description="Human-readable project name.")],
        git_remote: Annotated[
            Optional[str],
            Field(description="Canonical git remote URL; the preferred identity."),
        ] = None,
        repo_root_path: Annotated[
            Optional[str],
            Field(description="Repository root path, for projects with no remote."),
        ] = None,
    ) -> dict[str, Any]:
        return await _perform_project_create(
            ctx, name=name, git_remote=git_remote, repo_root_path=repo_root_path
        )

    @mcp.tool(
        name="klio_retrieve",
        description=(
            "Recover the full original behind a compression handle (the "
            "`handle` field returned alongside compressed content). Returns "
            "the text one page at a time — use `offset` to continue, exactly "
            "as with artifact_get. Originals are kept for 24 hours."
        ),
    )
    async def klio_retrieve(
        ctx: Context,
        handle: Annotated[
            str, Field(description="The `kc_...` handle issued with compressed output.")
        ],
        offset: Annotated[
            int, Field(description="Character offset for paging text.", ge=0)
        ] = 0,
    ) -> dict[str, Any]:
        return await _perform_klio_retrieve(ctx, handle=handle, offset=offset)

    @mcp.tool(
        name="compress",
        description=(
            "Compress content before putting it in context — tool output, "
            "JSON, source files, logs, documentation. Returns the compressed "
            "text with a before/after token count. By default only lossless "
            "compression is applied. Pass lossy=true to allow the "
            "information-dropping compressors (source code keeps signatures "
            "and drops bodies; prose drops low-value tokens); when it does, "
            "you also get a `handle` that klio_retrieve turns back into the "
            "full original for the next 24 hours. Content under ~500 tokens, "
            "and anything we cannot confidently identify, is returned "
            "untouched. Reading compressed JSON: it is minified; a key is "
            "omitted from any row where its value was null (per row, not only "
            "when null everywhere); and keys whose value is identical on every "
            "row are hoisted into a `common` object that applies to all of "
            "them — read a row as `common` merged with the row. `id`, "
            "`content` and `scope` always stay on the row. A `_klio` object "
            "carries the metadata that reverses all of it; no information is "
            "lost."
        ),
    )
    async def compress(
        ctx: Context,
        content: Annotated[str, Field(description="The text to compress.")],
        lossy: Annotated[
            bool,
            Field(
                description=(
                    "Allow compressors that drop information. A `handle` is "
                    "returned with the result so the original can be "
                    "recovered."
                )
            ),
        ] = False,
    ) -> dict[str, Any]:
        return await _perform_compress(ctx, content=content, lossy=lossy)

    @mcp.tool(
        name="explore",
        description=(
            "Explore everything the team's memory knows about one entity — a "
            "system, person, project, tool, or concept — as a structured "
            "brief: a synthesized summary, decisions in order, what was "
            "superseded and by what, contributor counts, and provenance "
            "(stated vs inferred). Built from YOUR visible slice of memory "
            "(org + your projects + your private notes), so teammates may see "
            "different briefs. Arg: entity (its name)."
        ),
    )
    async def explore(
        ctx: Context,
        entity: Annotated[
            str,
            Field(description="The entity name to explore, e.g. 'auth service'."),
        ],
    ) -> dict[str, Any]:
        return await _perform_explore(ctx, entity=entity)

    @mcp.tool(
        name="project_list",
        description=(
            "List the projects in this org, with how many members each has. "
            "No arguments."
        ),
    )
    async def project_list(ctx: Context) -> dict[str, Any]:
        return await _perform_project_list(ctx)

    @mcp.tool(
        name="project_members",
        description=(
            "List who has access to a project. You must be a member of it. "
            "Arg: project (name, git remote, path, or id)."
        ),
    )
    async def project_members(
        ctx: Context,
        project: Annotated[
            str, Field(description="Project name, git remote, repo path, or id.")
        ],
    ) -> dict[str, Any]:
        return await _perform_project_members(ctx, project=project)

    @mcp.tool(
        name="project_grant",
        description=(
            "Give a teammate access to a project you administer. `user` is "
            "their EMAIL ADDRESS — they must already be in this org. You can "
            "only grant on projects you administer, and every grant is "
            "audit-logged. Args: project, user (email), optional role "
            "('member' default | 'admin')."
        ),
    )
    async def project_grant(
        ctx: Context,
        project: Annotated[
            str, Field(description="Project name, git remote, repo path, or id.")
        ],
        user: Annotated[
            str,
            Field(
                description=(
                    "The teammate's EMAIL address, e.g. 'priya@oppla.ai'. Not "
                    "a user id."
                )
            ),
        ],
        role: Annotated[
            Literal["member", "admin"],
            Field(description="'member' (default) can read; 'admin' can also grant."),
        ] = "member",
    ) -> dict[str, Any]:
        return await _perform_project_grant(ctx, project=project, user=user, role=role)
