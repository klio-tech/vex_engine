"""REST capture layer for Klio Cloud passive memory.

The MCP tools (:mod:`app.mcp_app`) cover the *active* path — an agent
explicitly calling ``recall`` / ``remember``.  This module adds the *passive*
path used by the thin ``klio hook`` client: it turns Claude Code (and
compatible) lifecycle hook events into brain writes, and serves a lightweight
recall for session-start context injection, all over plain REST.  A hook script
therefore needs only an HTTP ``POST`` — no MCP handshake — which keeps the
client a dumb forwarder.

Three endpoints, all mounted under ``/capture``:

* ``POST /capture/event`` — store one passive memory (default an
  ``observation``), e.g. from a ``UserPromptSubmit`` hook.
* ``POST /capture/transcript`` — distil a finished session: extract discrete
  ``fact`` rows and one ``summary`` from the transcript, e.g. from a ``Stop``
  hook.  The transcript is the richest source and only exists at session end,
  so we mine it then rather than storing the whole thing.  Secrets and PII in
  the transcript are redacted via :func:`shared.redaction.redact` BEFORE the
  text is sent to the external LLM for fact extraction/summarisation (defense
  in depth: :func:`shared.brain.write_scoped_memory` re-screens whatever
  comes back at the storage chokepoint).
* ``POST /capture/recall`` — read memories for context injection at
  ``SessionStart``.  With a query it does vector recall; with an empty query it
  falls back to recency (most-recent first), so a fresh session always gets
  *something* even before the curator has synthesised anything.

Auth reuses the same ``X-Vex-Key`` (+ optional ``X-Vex-Agent``) bridge as the
MCP tools, via :func:`app.auth.resolve_principal_dep`, so capture is scoped to
exactly the org/agent the key authorises.  Nothing here modifies ``shared/*``:
writes reuse :func:`shared.brain.write_scoped_memory`, reads reuse
:func:`shared.memory.retrieve_memories_scoped`, and project tagging reuses
:func:`shared.brain.ensure_project` / ``find_project_id`` — the exact same
primitives the MCP tools use.

Project tagging is *best-effort*: the client may send ``git_remote`` and/or
``repo_root`` (Claude Code provides ``cwd``).  When present, writes are filed
under the resolved project; when absent or unresolved, writes go unfiled
(``project_id = NULL``) and recall falls back to org-wide.  Capture never fails
because project resolution failed.

Async note: handlers are ``async def`` and ``await`` the async brain/memory
helpers (which embed/LLM-call) but those helpers also perform inline synchronous
DB I/O — the established house pattern across the memory substrate (see
:func:`shared.brain.write_scoped_memory`).  We do not wrap the DB calls in
threads here.
"""

import logging
from typing import Any, Literal, Optional

import anyio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from shared import brain
from shared.auth import KeyInfo
from shared.memory import extract_facts, generate_summary, retrieve_memories_scoped
from shared.redaction import redact
from shared.plan_limits import get_plan_config
from shared.usage import MemoryQuotaExceeded, check_memory_quota, record_usage
from sqlalchemy import text

from app.auth import resolve_principal_dep, verify_api_key
from app.db import SessionLocal
from app.recall_log import log_recall_event

logger = logging.getLogger("agentguard.mcp.capture")

router = APIRouter(prefix="/capture", tags=["capture"])


def _check_memory_quota_402(
    db, org_id: str, plan: str, kind: str, overrides: dict | None = None
) -> bool:
    """Run the memory quota check, translating a hard block to HTTP 402.

    Returns ``over_cap``. Synchronous; call via anyio.to_thread from handlers.
    ``overrides`` is the account's ``plan_overrides`` (from the validated key);
    passing it keeps real-time enforcement consistent with the retention sweep,
    which also merges overrides via ``get_plan_config(plan, overrides)``.
    """
    try:
        return check_memory_quota(
            db, org_id, get_plan_config(plan, overrides), kind, plan_name=plan
        )
    except MemoryQuotaExceeded as exc:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "quota_exceeded",
                "metric": exc.metric,
                "limit": exc.limit,
                "message": exc.message,
                "suggested_plan": exc.suggested_plan,
            },
        ) from exc

# Memory types a passive capture *event* is allowed to write.  Deliberately
# EXCLUDES the curator's output types ('summary'/'fact') so a hook can never
# inject a synthesised type, and excludes arbitrary free strings.  Mirrors the
# RAW input set the curator consumes.
_CAPTURE_EVENT_TYPES: frozenset[str] = frozenset(
    {"observation", "note", "memory", "plan", "decision"}
)
_DEFAULT_EVENT_TYPE = "observation"

# Upper bound (in characters) on the transcript text sent to the LLM for fact
# extraction / summarisation.  Keeps cost and latency bounded on long sessions;
# we keep the most-recent TAIL since that is the freshest context.
_MAX_TRANSCRIPT_CHARS = 16_000

# Hard cap on how many memories a single recall returns, regardless of the
# requested ``limit`` (a defensive bound on response size / DB work).
_MAX_RECALL_LIMIT = 50

# Stable session-id PREFIX for passive captures.  Like the MCP write path
# (:mod:`app.session`), the suffix groups a client's writes so dedup works
# across calls; it is NOT an authorization input (tenancy is enforced by org_id
# / agent_id on every query).
_CAPTURE_SESSION_PREFIX = "klio-hook"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CaptureEventRequest(BaseModel):
    """One passive memory to store (e.g. a user prompt)."""

    content: str = Field(min_length=1, description="The memory text to store.")
    memory_type: str = Field(
        default=_DEFAULT_EVENT_TYPE,
        description=(
            "One of observation|note|memory|plan|decision; anything else is "
            "coerced to 'observation'."
        ),
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Optional metadata persisted on the row."
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Client session id; groups dedup within a session.",
    )
    scope: Literal["org", "agent"] = Field(
        default="org",
        description=(
            "Memory scope. 'org' (default) writes the shared team brain — "
            "visible to every agent on recall (backward compatible). 'agent' "
            "writes a row private to the caller's X-Vex-Agent, returned only by "
            "recall(scope='agent') for that same agent."
        ),
    )
    git_remote: Optional[str] = Field(
        default=None, description="Canonical git remote URL of the working repo."
    )
    repo_root: Optional[str] = Field(
        default=None, description="Filesystem repo root (e.g. the hook's cwd)."
    )


class TranscriptMessage(BaseModel):
    """A single message in a session transcript."""

    role: Optional[str] = Field(default=None, description="Speaker role, if known.")
    content: str = Field(default="", description="Message text.")


class CaptureTranscriptRequest(BaseModel):
    """A finished session's transcript to distil into facts + a summary."""

    messages: list[TranscriptMessage] = Field(
        default_factory=list, description="Ordered session messages."
    )
    session_id: Optional[str] = Field(
        default=None, description="Client session id; groups dedup within a session."
    )
    git_remote: Optional[str] = Field(default=None)
    repo_root: Optional[str] = Field(default=None)


class RecallRequest(BaseModel):
    """A context-injection read for session start (or ad-hoc recall)."""

    query: str = Field(
        default="",
        description="Search text; empty falls back to most-recent memories.",
    )
    limit: int = Field(default=12, ge=1, description="Max memories to return.")
    scope: Literal["org", "agent"] = Field(default="org")
    space: Optional[str] = Field(default=None, description="Optional space slug.")
    git_remote: Optional[str] = Field(default=None)
    repo_root: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_session_id(org_id: str, agent_id: str, session_id: Optional[str]) -> str:
    """Derive a stable dedup-grouping session id for a passive capture.

    Uses the client-supplied ``session_id`` when present (groups a single
    Claude Code session's writes), else falls back to ``<org_id>:<agent_id>`` so
    dedup stays stable per principal rather than scattering across calls.
    """
    suffix = session_id if session_id else f"{org_id}:{agent_id}"
    return f"{_CAPTURE_SESSION_PREFIX}:{suffix}"


def _project_display_name(git_remote: Optional[str], repo_root: Optional[str]) -> str:
    """Derive a human-readable project name from a remote URL or repo path.

    Prefers the last path segment of the git remote (sans a trailing ``.git``),
    falling back to the repo-root basename, then a generic ``"project"``.
    """
    if git_remote:
        name = git_remote.rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".git"):
            name = name[:-4]
        if name:
            return name
    if repo_root:
        base = repo_root.rstrip("/").rsplit("/", 1)[-1]
        if base:
            return base
    return "project"


def _ensure_project_id(
    db: Any, *, org_id: str, git_remote: Optional[str], repo_root: Optional[str]
) -> Optional[str]:
    """Best-effort get-or-create of the project for a write; never raises.

    Returns the project id when ``git_remote`` or ``repo_root`` is supplied and
    resolution succeeds, else ``None`` (the write proceeds unfiled).  Any error
    is logged and swallowed so capture never fails because of project tagging.
    """
    if not git_remote and not repo_root:
        return None
    try:
        return brain.ensure_project(
            db,
            org_id=org_id,
            display_name=_project_display_name(git_remote, repo_root),
            git_remote=git_remote,
            repo_root_path=repo_root,
        )
    except Exception:
        logger.warning(
            "capture: ensure_project failed for org %s; writing unfiled",
            org_id,
            exc_info=True,
        )
        return None


def _find_project_id(
    db: Any, *, org_id: str, git_remote: Optional[str]
) -> Optional[str]:
    """Best-effort project lookup for a read; never raises.

    Only resolves by ``git_remote`` (the reliable cross-machine identity).  A
    miss returns ``None``, which the recall path treats as "no project filter"
    (org-wide) rather than an empty result — the right default for passive
    session-start injection, which must not be silenced by an unresolved repo.
    """
    if not git_remote:
        return None
    try:
        return brain.find_project_id(db, org_id=org_id, identifier=git_remote)
    except Exception:
        logger.warning(
            "capture: find_project_id failed for org %s; recalling org-wide",
            org_id,
            exc_info=True,
        )
        return None


def _combine_messages(messages: list[TranscriptMessage]) -> str:
    """Flatten transcript messages into one text blob, tail-capped and redacted.

    Empty messages are dropped; a role prefix is added when known.  The result
    is truncated to the most-recent :data:`_MAX_TRANSCRIPT_CHARS` characters so
    a very long session does not blow up the LLM call.

    Secrets and PII are redacted via :func:`shared.redaction.redact` BEFORE
    the text is sent to an external LLM for fact extraction/summarisation —
    secrets never leave the infrastructure via the mining path.
    :func:`shared.brain.write_scoped_memory` re-screens whatever comes back
    (defense in depth), so a redaction failure here can fail open to the
    chokepoint guarantee.
    """
    parts: list[str] = []
    for m in messages:
        content = (m.content or "").strip()
        if not content:
            continue
        role = (m.role or "").strip()
        parts.append(f"{role}: {content}" if role else content)
    combined = "\n".join(parts)
    if len(combined) > _MAX_TRANSCRIPT_CHARS:
        combined = combined[-_MAX_TRANSCRIPT_CHARS:]
    # Redact BEFORE this text is sent to the external LLM for fact
    # extraction/summarisation — secrets never leave the infrastructure via
    # the mining path. write_scoped_memory re-screens whatever comes back
    # (defense in depth), so a redaction failure here can fail open to the
    # chokepoint guarantee.
    try:
        combined = redact(combined).clean_text
    except Exception:
        logger.warning("capture: transcript redaction failed; relying on chokepoint")
    return combined


def _serialize_memory(record: Any) -> dict[str, Any]:
    """Project a memory (``MemoryRecord`` OR a raw recency ``Row``) to JSON.

    Both expose ``id`` / ``memory_type`` / ``content`` / ``created_at`` via
    attribute access, so one serializer covers the vector-recall and recency
    paths.  Only the fields a context block needs are emitted (no embedding,
    no internal verification columns); ``id`` is stringified and ``created_at``
    is ISO-8601 (or ``None``).
    """
    created_at = getattr(record, "created_at", None)
    return {
        "id": str(record.id),
        "memory_type": record.memory_type,
        "content": record.content,
        "created_at": created_at.isoformat() if created_at is not None else None,
    }


def _recent_memories(
    db: Any, *, org_id: str, project_id: Optional[str], limit: int
) -> list[dict[str, Any]]:
    """Return the most-recent active org memories (recency fallback).

    Used when recall is called with an empty query (session start).  Tenant- and
    brain-scoped (``org_id`` + ``scope='org'`` + ``status='active'``) so neither
    verification's ``scope='session'`` rows nor other orgs' rows are returned.
    When ``project_id`` is given, rows for that project OR unfiled (``NULL``)
    rows surface — the same NULL-surfacing semantics as
    :func:`shared.memory.retrieve_memories_scoped`.
    """
    rows = db.execute(
        text(
            """
            SELECT id, memory_type, content, created_at
            FROM session_memories
            WHERE org_id = :org_id
              AND scope = 'org'
              AND status = 'active'
              AND (
                :project_id IS NULL
                OR project_id = :project_id
                OR project_id IS NULL
              )
            ORDER BY created_at DESC
            LIMIT :limit
            """
        ),
        {"org_id": org_id, "project_id": project_id, "limit": limit},
    ).fetchall()
    return [_serialize_memory(r) for r in rows]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/event")
async def capture_event(
    req: CaptureEventRequest,
    principal: tuple[str, str] = Depends(resolve_principal_dep),
    key_info: KeyInfo = Depends(verify_api_key),
) -> dict[str, Any]:
    """Store one passive memory (default ``observation``).

    Coerces an unknown ``memory_type`` to ``observation`` (the schema accepts a
    free string for forward-compat, but only the RAW set is allowed through),
    tags ``source`` in metadata, best-effort files it under a project, and
    persists via :func:`shared.brain.write_scoped_memory` (dedup on).
    """
    org_id, agent_id = principal
    memory_type = (
        req.memory_type
        if req.memory_type in _CAPTURE_EVENT_TYPES
        else _DEFAULT_EVENT_TYPE
    )
    metadata = dict(req.metadata or {})
    metadata.setdefault("source", "hook")

    db = SessionLocal()
    try:
        # ensure_project is synchronous (blocking psycopg2); run it off the
        # event loop per its documented threading contract (shared.brain).
        project_id = await anyio.to_thread.run_sync(
            lambda: _ensure_project_id(
                db, org_id=org_id, git_remote=req.git_remote, repo_root=req.repo_root
            )
        )
        over_cap = await anyio.to_thread.run_sync(
            lambda: _check_memory_quota_402(
                db, org_id, key_info.plan, "memory", key_info.plan_overrides
            )
        )
        result = await brain.write_scoped_memory(
            db,
            org_id=org_id,
            agent_id=agent_id,
            content=req.content,
            memory_type=memory_type,
            session_id=_capture_session_id(org_id, agent_id, req.session_id),
            project_id=project_id,
            metadata=metadata,
            scope=req.scope,
        )
        if result.get("stored"):
            await anyio.to_thread.run_sync(
                lambda: record_usage(db, org_id, "memory", over_cap=over_cap)
            )
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception("capture event failed (memory_type=%s)", memory_type)
        return {"stored": False, "error": "capture_failed"}
    finally:
        db.close()


@router.post("/transcript")
async def capture_transcript(
    req: CaptureTranscriptRequest,
    principal: tuple[str, str] = Depends(resolve_principal_dep),
    key_info: KeyInfo = Depends(verify_api_key),
) -> dict[str, Any]:
    """Distil a finished session into discrete ``fact`` rows and one ``summary``.

    Extraction failures are tolerated independently: a failed summary still
    allows facts to be stored and vice-versa.  Each synthesised memory is written
    via :func:`shared.brain.write_scoped_memory` (dedup on), so re-sending the
    same transcript is idempotent.  The curator never reprocesses these (it reads
    only RAW types, not ``fact`` / ``summary``).
    """
    org_id, agent_id = principal
    combined = _combine_messages(req.messages)
    if not combined:
        return {"stored": 0, "facts": 0, "summary": False, "reason": "empty_transcript"}

    db = SessionLocal()
    try:
        project_id = await anyio.to_thread.run_sync(
            lambda: _ensure_project_id(
                db, org_id=org_id, git_remote=req.git_remote, repo_root=req.repo_root
            )
        )
        session_id = _capture_session_id(org_id, agent_id, req.session_id)
        base_meta = {"source": "hook-transcript"}
        stored = 0
        over_cap = await anyio.to_thread.run_sync(
            lambda: _check_memory_quota_402(
                db, org_id, key_info.plan, "memory", key_info.plan_overrides
            )
        )

        try:
            summary = await generate_summary(combined)
        except Exception:
            logger.warning("capture: generate_summary raised for org %s", org_id)
            summary = None
        if summary:
            result = await brain.write_scoped_memory(
                db,
                org_id=org_id,
                agent_id=agent_id,
                content=summary,
                memory_type="summary",
                session_id=session_id,
                project_id=project_id,
                metadata=base_meta,
            )
            if result.get("stored"):
                stored += 1
                await anyio.to_thread.run_sync(
                    lambda: record_usage(db, org_id, "memory", over_cap=over_cap)
                )

        try:
            facts = await extract_facts(combined)
        except Exception:
            logger.warning("capture: extract_facts raised for org %s", org_id)
            facts = []
        for fact in facts:
            if not fact:
                continue
            result = await brain.write_scoped_memory(
                db,
                org_id=org_id,
                agent_id=agent_id,
                content=fact,
                memory_type="fact",
                session_id=session_id,
                project_id=project_id,
                metadata=base_meta,
            )
            if result.get("stored"):
                stored += 1
                await anyio.to_thread.run_sync(
                    lambda: record_usage(db, org_id, "memory", over_cap=over_cap)
                )

        return {"stored": stored, "facts": len(facts), "summary": bool(summary)}
    except HTTPException:
        raise
    except Exception:
        logger.exception("capture transcript failed")
        return {"stored": 0, "error": "capture_failed"}
    finally:
        db.close()


@router.post("/recall")
async def capture_recall(
    req: RecallRequest,
    principal: tuple[str, str] = Depends(resolve_principal_dep),
    key_info: KeyInfo = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return memories for context injection.

    With a non-empty ``query`` this is vector recall via
    :func:`shared.memory.retrieve_memories_scoped`; with an empty ``query`` it
    falls back to recency (most-recent first) so a fresh ``SessionStart`` always
    has something to inject.  Project (via ``git_remote``) and space filters are
    BEST-EFFORT: an unresolved filter widens to org-wide rather than returning
    empty — appropriate for passive injection (unlike the strict MCP ``recall``
    tool, which warns + empties on an unknown explicit filter).
    """
    org_id, agent_id = principal
    limit = min(req.limit, _MAX_RECALL_LIMIT)

    db = SessionLocal()
    try:
        # The synchronous brain/recency reads run off the event loop per the
        # shared.brain threading contract (blocking psycopg2 I/O).
        project_id = await anyio.to_thread.run_sync(
            lambda: _find_project_id(db, org_id=org_id, git_remote=req.git_remote)
        )
        query = req.query.strip()
        recall_over_cap = await anyio.to_thread.run_sync(
            lambda: _check_memory_quota_402(
                db, org_id, key_info.plan, "recall", key_info.plan_overrides
            )
        )

        if not query:
            memories = await anyio.to_thread.run_sync(
                lambda: _recent_memories(
                    db, org_id=org_id, project_id=project_id, limit=limit
                )
            )
        else:
            # Best-effort space filter: unknown slug → org-wide (None), never
            # empty.
            space_id: Optional[str] = None
            if req.space:
                space_id = await anyio.to_thread.run_sync(
                    lambda: brain.find_space_id(db, org_id=org_id, slug=req.space)
                )

            records = await retrieve_memories_scoped(
                db,
                org_id,
                query,
                scope=req.scope,
                agent_id=agent_id,
                space_id=space_id,
                project_id=project_id,
                limit=limit,
            )
            memories = [_serialize_memory(r) for r in records]

        # Best-effort recall activity log for BOTH the recency and vector paths.
        # log_recall_event swallows its own errors, but it does synchronous
        # psycopg2 I/O, so run it off the event loop per the shared.brain
        # threading contract. A logging failure must NEVER break the recall, so
        # this stays inside the try (the finally still closes the session).
        await anyio.to_thread.run_sync(
            lambda: log_recall_event(
                db,
                org_id=org_id,
                agent_id=agent_id,
                project_id=project_id,
                query_text=(query or None),
                result_count=len(memories),
                source="hook",
            )
        )
        await anyio.to_thread.run_sync(
            lambda: record_usage(db, org_id, "recall", over_cap=recall_over_cap)
        )
        return {"memories": memories, "count": len(memories)}
    except HTTPException:
        raise
    except Exception:
        logger.exception("capture recall failed (scope=%s)", req.scope)
        return {"memories": [], "count": 0, "error": "recall_failed"}
    finally:
        db.close()
