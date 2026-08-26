"""REST capture layer for Klio Cloud passive memory.

The MCP tools (:mod:`app.mcp_app`) cover the *active* path — an agent
explicitly calling ``recall`` / ``remember``.  This module adds the *passive*
path used by the thin ``klio hook`` client: it turns Claude Code (and
compatible) lifecycle hook events into brain writes, and serves a lightweight
recall for session-start context injection, all over plain REST.  A hook script
therefore needs only an HTTP ``POST`` — no MCP handshake — which keeps the
client a dumb forwarder.

Four endpoints, all mounted under ``/capture``:

* ``POST /capture/event`` — store one passive memory (default an
  ``observation``), e.g. from a ``UserPromptSubmit`` hook.
* ``POST /capture/transcript`` — distil a finished session: extract discrete
  memory rows, each classified into the product taxonomy
  (``decision`` / ``plan`` / ``fact`` / ``note`` / ``observation``, see
  :mod:`shared.taxonomy`), plus one ``summary``, e.g. from a ``Stop``
  hook.  The transcript is the richest source and only exists at session end,
  so we mine it then rather than storing the whole thing.  Secrets and PII in
  the transcript are redacted via :func:`shared.redaction.redact` BEFORE the
  text is sent to the external LLM for fact extraction/summarisation (defense
  in depth: :func:`shared.brain.write_scoped_memory` re-screens whatever
  comes back at the storage chokepoint).
* ``POST /capture/recall`` — read memories for context injection at
  ``SessionStart``.  With a query it does vector recall; with an empty query it
  falls back to recency (most-recent first) within the resolved project.  It
  may return NOTHING, and that is a correct answer rather than a degraded one:
  a project with no memories yet, or a query nothing in the brain is relevant
  to, gets an empty list instead of the org's newest unrelated rows.
* ``POST /capture/forget`` — retract one memory by id (hook-client mirror of
  the MCP ``forget`` tool).  Returns 404 for unknown or cross-org ids, 403
  for cross-agent scope='agent' rows.

Auth reuses the same ``X-Vex-Key`` (+ optional ``X-Vex-Agent``) bridge as the
MCP tools: the key is validated by ``Depends(verify_api_key)`` (threadpool +
the ``dependency_overrides`` test seam), and each route body then resolves its
principal through :func:`app.identity.resolve_active_principal` — the ONE
resolver — after seeding the validated key into the per-request memo
(:func:`app.identity.seed_request_key_info`), so an ``X-Klio-Node`` header
scopes every capture to the SELECTED node and no second validation token is
spent.  Nothing here modifies ``shared/*``:
writes reuse :func:`shared.brain.write_scoped_memory`, reads reuse
:func:`shared.memory.retrieve_memories_scoped`, and project tagging reuses
:func:`shared.brain.ensure_project` / ``find_project_id`` — the exact same
primitives the MCP tools use.

Project tagging is *best-effort*: the client may send ``git_remote`` and/or
``repo_root`` (Claude Code provides ``cwd``), and BOTH the write and the read
now try the remote first and the path second (the read used to accept only a
remote, which is why 95 % of memories were written with a project and 95 % of
recalls ignored it).  Capture never fails because project resolution failed.

**What an unresolved project MEANS differs between the two recall paths, and
the difference is deliberate.**

* Empty query → :func:`_recent_memories`.  Fenced symmetrically with the write
  path: resolved → ``project_id = :project_id``; unresolved → ``project_id IS
  NULL``.  No widening in either direction.  This path orders by nothing but
  ``created_at``, so every row it admits beyond the fence is chosen by its
  clock alone — there is no ranking to justify letting more in.
* Non-empty query → :func:`shared.memory.retrieve_memories_scoped`.  Resolved
  → two-tier (same-project rows first, unfiled rows backfilling).
  **Unresolved → still org-wide, unchanged.**  That path can rank what it lets
  in, and narrowing an unresolved read to the unfiled bucket would silently
  empty per-query recall for every caller that sends no project.

The practical consequence, until the Node proxy starts sending ``repo_root``
(``npm/src/proxy/recall.ts`` sends neither identifier today): **per-query proxy
recalls remain org-wide and exactly as cross-contaminated as before.**  What
this module fixes on its own is the ``klio hook`` path, which already sends
both identifiers — that is SessionStart injection, ~87 % of recalls.

Async note: handlers are ``async def`` and ``await`` the async brain/memory
helpers (which embed/LLM-call) but those helpers also perform inline synchronous
DB I/O — the established house pattern across the memory substrate (see
:func:`shared.brain.write_scoped_memory`).  We do not wrap the DB calls in
threads here.
"""

import hashlib
import logging
from typing import Any, Literal, Optional

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from shared import brain, branches, claims
from shared.auth import KeyInfo
from shared.memory import (
    extract_typed_memories,
    generate_summary,
    retract_memory,
    retrieve_memories_scoped,
)
from shared import metrics
from shared.evidence_stream import build_captured_payload, publish_captured
from shared.redaction import redact
from shared.plan_limits import get_plan_config
from shared.usage import MemoryQuotaExceeded, check_memory_quota, record_usage
from shared import project_scope
from sqlalchemy import text

from app.auth import verify_api_key
from app.db import SessionLocal
from app.identity import (
    resolve_active_principal,
    resolve_user_id,
    seed_request_key_info,
)
from app.recall_log import log_recall_event

logger = logging.getLogger("agentguard.mcp.capture")

router = APIRouter(prefix="/capture", tags=["capture"])


# ── instrumentation helpers ───────────────────────────────────────────
#
# Thin wrappers so each call site is one line, and so a metrics fault can never
# propagate into a capture. Both go through ``metrics.record``, which swallows.


def record_capture(outcome: str) -> None:
    """Record one capture attempt by outcome."""
    metrics.record(metrics.CAPTURE_EVENTS.inc, outcome=outcome)


def record_redaction(hit: bool) -> None:
    """Record that the redactor found and removed something.

    Hits, not calls. A count of calls is a count of captures, which is already
    measured; what nobody can otherwise see is how much sensitive material is
    actually flowing through — and a hit rate that drops to zero is how a
    silently broken redactor presents.
    """
    if hit:
        metrics.record(metrics.CAPTURE_REDACTIONS.inc)


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
#
# It is, however, a PRIVACY input, because a dedup hit hands the matching row's
# id back to the caller. See :func:`_capture_session_id` for why the principal
# is part of the key and not just the client's string.
_CAPTURE_SESSION_PREFIX = "klio-hook"

#: ``session_memories.session_id`` is ``VARCHAR(256)`` (migration 017). One
#: segment of the derived id is client-controlled, so the result is bounded
#: here rather than left to fail at the column.
_MAX_SESSION_ID_CHARS = 256


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
    git_branch: Optional[str] = Field(
        default=None,
        description=(
            "The repo's current git branch, best-effort. Binds the capture to "
            "the project's context branch of the same name (auto-created). "
            "'main'/'master' (any case), blank, or omitted mean main. Ignored "
            "unless a project resolves and the caller is attributable."
        ),
    )


class TranscriptMessage(BaseModel):
    """A single message in a session transcript."""

    role: Optional[str] = Field(default=None, description="Speaker role, if known.")
    content: str = Field(default="", description="Message text.")


class CaptureTranscriptRequest(BaseModel):
    """A finished session's transcript to distil into typed memories + a summary."""

    messages: list[TranscriptMessage] = Field(
        default_factory=list, description="Ordered session messages."
    )
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Ordered tool invocations for the session (evidence loop). "
            "Optional and backward compatible: omitted or empty is a no-op."
        ),
    )
    session_id: Optional[str] = Field(
        default=None, description="Client session id; groups dedup within a session."
    )
    git_remote: Optional[str] = Field(default=None)
    repo_root: Optional[str] = Field(default=None)
    git_branch: Optional[str] = Field(
        default=None,
        description=(
            "The repo's current git branch, best-effort; same contract as "
            "CaptureEventRequest.git_branch."
        ),
    )


class RecallRequest(BaseModel):
    """A context-injection read for session start (or ad-hoc recall)."""

    query: str = Field(
        default="",
        description="Search text; empty falls back to most-recent memories.",
    )
    limit: int = Field(default=12, ge=1, description="Max memories to return.")
    scope: Literal["org", "agent", "private"] = Field(
        default="org",
        description=(
            "'org' (default) returns the visibility ladder — the shared brain "
            "plus the caller's own private and project memories. 'private' "
            "narrows to the caller's own memories alone. 'agent' is deprecated."
        ),
    )
    space: Optional[str] = Field(default=None, description="Optional space slug.")
    git_remote: Optional[str] = Field(default=None)
    repo_root: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Client session id, same value the transcript capture sends. "
            "Optional and backward compatible: omitted is a no-op. Used to "
            "derive the SAME session identity _capture_session_id computes "
            "for a transcript capture, so the recall audit row can be joined "
            "to the session it was served into."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hook_write_scope(
    request: Request,
    key_info: Optional[KeyInfo],
    declared_scope: str,
) -> tuple[str, Optional[str]]:
    """Decide the scope and owner for a passive hook capture.

    This is the routing rule the user silo turns on. Hooks are not a decision —
    nobody chose to publish when a ``Stop`` hook fired — so a capture we can
    attribute to a person becomes theirs::

        attributable   -> ('private', <user_id>)
        unattributable -> (<declared_scope>, None)   + a warning

    On the unattributable branch we do NOT write private, and we do NOT refuse.
    Refusing would break every legacy key overnight for no gain in safety.
    Writing private would be worse than either: migration 031's
    ``ck_session_memories_private_has_owner`` rejects an owner-less private row
    outright, and even without the constraint such a row is unreachable by
    every caller INCLUDING the person who wrote it — data quietly destroyed
    while the API reports success. So the write falls back to what the client
    declared, which is exactly today's behaviour, and logs.

    The fallback honours ``declared_scope`` rather than forcing ``'org'`` so
    that a client already asking for the deprecated ``'agent'`` scope keeps
    getting it. An attributable caller's declared scope is ignored on purpose
    -- with one exception, ``'org'``, which is an explicit publication (the
    chat bridge declares it for plot messages; see the inline comment below).
    Everything else collapses to ``private``: strictly narrower, so the
    override can only ever reduce who sees the row.

    Returns:
        A ``(scope, user_id)`` pair.
    """
    user_id = resolve_user_id(request, key_info)
    if user_id:
        # One exception to the discard rule, added for chat capture: an
        # attributable caller declaring EXACTLY 'org' is making an explicit
        # publication -- the plot bridge says it on purpose, because a plot
        # message was said in the open room and forcing it private made the
        # team's own conversation visible to nobody but the workspace bot.
        # Only SERVICE tokens may publish this way -- a human's hook client
        # declaring org still collapses to private, because nobody chose to
        # publish when a Stop hook fired. The protected direction is unchanged:
        # nothing is widened silently
        # (hooks never declare a scope), 'private' stays private, and any
        # other declared value still falls back to the private default.
        if declared_scope == "org" and getattr(key_info, "is_service", False):
            return "org", user_id
        return "private", user_id
    logger.warning(
        "capture: request could not be attributed to a user; writing scope=%s "
        "instead of private (org=%s, key=%s)",
        declared_scope,
        getattr(key_info, "org_id", None),
        getattr(key_info, "key_id", None),
    )
    return declared_scope, None


def _capture_session_id(
    org_id: str,
    agent_id: str,
    session_id: Optional[str],
    *,
    user_id: Optional[str] = None,
) -> str:
    """Derive a stable dedup-grouping session id for a passive capture.

    The client-supplied ``session_id`` groups a single Claude Code session's
    writes; the principal segment in front of it decides WHOSE bucket that is.

    **Why the principal has to be in the key.** A dedup hit returns the matching
    row's id and memory_type to the caller. ``find_similar_memory`` narrows the
    window to ``user_id = :user_id`` for ``scope='private'`` for exactly that
    reason, but every other scope falls back to the historical session window —
    ``session_id AND org_id AND status``, with no ``scope`` and no ``user_id``
    predicate. This value used to be ``klio-hook:<client string>``, so an
    unattributable caller (whose writes land at ``org`` scope, and who
    therefore gets the wide window) could name the bucket an attributed user's
    captures land in, submit a guess, and learn from the response whether that
    person had already stored near-identical text — and receive the id of their
    private row. The key is the only thing that ever separated them.

    ``user_id`` first, then ``agent_id``, mirroring the identity ladder: an
    attributed caller's bucket follows the PERSON across agents, and an
    unattributable one is namespaced by the agent principal it does have. The
    two can never collide because the marker (``u`` / ``a``) distinguishes them
    even if a user id and an agent id were somehow equal.

    The result is bounded: ``session_memories.session_id`` is ``VARCHAR(256)``
    and the client controls one segment, so an over-long id is folded into a
    digest of itself rather than truncated. Truncation would merge two of one
    principal's sessions — harmless — but hashing keeps them distinct for no
    extra cost, and the principal segment is never at risk either way.

    Changing this key regroups dedup buckets once: a capture that would have
    deduped against a pre-existing row under the old key is stored instead.
    That is a one-off duplicate, not a loss.
    """
    principal = f"u:{user_id}" if user_id else f"a:{agent_id}"
    suffix = session_id if session_id else "default"
    candidate = f"{_CAPTURE_SESSION_PREFIX}:{org_id}:{principal}:{suffix}"
    if len(candidate) <= _MAX_SESSION_ID_CHARS:
        return candidate

    digest = hashlib.sha256(suffix.encode("utf-8")).hexdigest()[:32]
    return f"{_CAPTURE_SESSION_PREFIX}:{org_id}:{principal}:h:{digest}"[
        :_MAX_SESSION_ID_CHARS
    ]


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


def _rollback(db: Any, what: str, org_id: str) -> None:
    """Clear an aborted transaction so the request's session stays usable.

    Project resolution is best-effort by design — a miss files the write
    unfiled rather than failing it. But Postgres aborts the whole transaction
    on the first failed statement, so swallowing the error without a ROLLBACK
    leaves the very session the capture or recall is about to run on refusing
    every statement. The degradation the ``except`` promises is "unfiled", not
    "the request dies two statements later for an unrelated-looking reason".

    Guarded in turn: a session too broken to roll back must not convert a
    best-effort path into a raising one.
    """
    try:
        db.rollback()
    except Exception:
        logger.warning(
            "capture: rollback after a failed %s also failed for org %s",
            what,
            org_id,
            exc_info=True,
        )


def _ensure_project_id(
    db: Any,
    *,
    org_id: str,
    git_remote: Optional[str],
    repo_root: Optional[str],
    user_id: Optional[str] = None,
) -> Optional[str]:
    """Best-effort get-or-create of the project for a write; never raises.

    Returns the project id when ``git_remote`` or ``repo_root`` is supplied and
    resolution succeeds, else ``None`` (the write proceeds unfiled).  Any error
    is logged and swallowed so capture never fails because of project tagging.

    ``user_id`` is forwarded as ``created_by``: this is the auto-create path
    (a hook capture, not the explicit ``project_create`` tool), so it is the
    ONLY place a freshly-created project gets an admin seated at all. An
    unattributed hook (``user_id=None``) still files the write under the
    project — best-effort project tagging is unaffected — it just seats
    nobody, matching ``ensure_project``'s contract.
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
            created_by=user_id,
        )
    except Exception:
        _rollback(db, "ensure_project", org_id)
        logger.warning(
            "capture: ensure_project failed for org %s; writing unfiled",
            org_id,
            exc_info=True,
        )
        return None


#: Git default-branch names that ARE main: they map to ``branch_id = NULL``
#: (main is the absence of a branch), never to a registry row. Compared
#: case-insensitively.
_MAIN_GIT_BRANCHES = frozenset({"main", "master"})


def _capture_branch(
    db: Any,
    *,
    org_id: str,
    project_id: Optional[str],
    user_id: Optional[str],
    git_branch: Optional[str],
) -> Optional[dict]:
    """The context branch a hook capture should land on, or ``None``.

    Returns the whole registry row rather than just its id, because a branch
    bound to a work item (migration 051) files its captures under that work
    item too — that is the whole point of a feature branch: you create it
    once against the work item, and everything the hooks capture while you
    are standing on it is filed there without anyone naming it again.

    The hook auto-bind (Track C): inside a git repo the hook sends the branch
    it is standing on, and the capture lands on the project's context branch
    of the same name — auto-created via :func:`shared.branches.ensure_branch`.

    ``None`` — the write stays on main — in every guarded case, each a plan
    rule rather than a convenience:

    * no ``git_branch``, or it is ``main``/``master`` (case-insensitive): the
      repo's default branch IS main;
    * no project resolved: a branch always belongs to a project (global
      constraint), so the write falls through to its normal scope;
    * unattributable caller (``user_id`` is ``None``): auto-creating registry
      rows on behalf of nobody would mint branches no one owns — the same
      caller whose captures cannot be private cannot bind branches either.

    Best-effort, never raises: capture must not fail because branch tagging
    did (the same contract as :func:`_ensure_project_id`). A refused name
    (over-long, say) or a database error logs and lands the write on main.
    """
    if not git_branch or not project_id or not user_id:
        return None
    stripped = git_branch.strip()
    if not stripped or stripped.lower() in _MAIN_GIT_BRANCHES:
        return None
    try:
        row = branches.ensure_branch(
            db,
            org_id=org_id,
            project_id=project_id,
            name=stripped,
            created_by=user_id,
        )
        return row
    except Exception:
        _rollback(db, "ensure_branch", org_id)
        logger.warning(
            "capture: ensure_branch(%r) failed for org %s; writing on main",
            stripped,
            org_id,
            exc_info=True,
        )
        return None


def _find_project_id(
    db: Any,
    *,
    org_id: str,
    git_remote: Optional[str],
    repo_root: Optional[str] = None,
) -> Optional[str]:
    """Best-effort project lookup for a read; never raises.

    Resolves by ``git_remote`` first and falls back to ``repo_root``, which is
    the fix for the single largest source of cross-project memory leakage in
    production. This function used to return ``None`` the moment ``git_remote``
    was falsy — but the WRITE path (:func:`_ensure_project_id`) has always
    passed BOTH identifiers, and :func:`shared.brain.find_project_id` has
    always resolved ``repo_root_path``. So writes filed by path and reads
    refused to. Measured over production at the time of the fix:

    * 79 of 118 projects in the org have no ``git_remote`` at all — including
      every non-git working directory;
    * 23,895 of 25,101 visible memories (95.2 %) carry a ``project_id``;
    * 604 of 637 hook recalls (94.8 %) resolved ``project_id = NULL``.

    95 % of memories were written with a project and 95 % of recalls ignored
    it, which is why a session about one repo was injected with facts from two
    unrelated ones. Side-by-side replay of the same query with and without the
    project resolved returned entirely different, and entirely better, rows.

    Order matters and is not arbitrary. ``git_remote`` is stable across
    machines and checkouts; ``repo_root`` is a local path that two developers
    will not share. Trying the remote first means a repo cloned to a different
    directory still resolves to the one project, and the path is only consulted
    when there is no remote to be more specific than it.

    A miss returns ``None``, and what that MEANS depends on which recall path
    consumes it — do not read this function as installing one fence:

    * :func:`_recent_memories` (empty query) treats ``None`` as **unfiled**:
      ``project_id IS NULL``, symmetric with what the write path does from the
      same directory. This is the path that changed.
    * :func:`shared.memory.retrieve_memories_scoped` (non-empty query) treats
      ``None`` as **org-wide**, exactly as before. Its two-tier filter applies
      only when a project resolves; with ``project_id=None`` it still takes the
      ``_fetch("TRUE", {})`` branch. Narrowing it here would empty per-query
      recall for every caller that sends no project — which today includes the
      Node proxy.
    """
    for identifier in (git_remote, repo_root):
        if not identifier:
            continue
        try:
            resolved = brain.find_project_id(db, org_id=org_id, identifier=identifier)
        except Exception:
            _rollback(db, "find_project_id", org_id)
            logger.warning(
                "capture: find_project_id failed for org %s; recalling unfiled",
                org_id,
                exc_info=True,
            )
            return None
        if resolved:
            return resolved
    return None


def _combine_messages(messages: list[TranscriptMessage]) -> str:
    """Flatten transcript messages into one text blob, tail-capped and redacted.

    Empty messages are dropped; a role prefix is added when known.  The result
    is truncated to the most-recent :data:`_MAX_TRANSCRIPT_CHARS` characters so
    a very long session does not blow up the LLM call.

    Secrets and PII are redacted via :func:`shared.redaction.redact` BEFORE
    the text is sent to an external LLM for fact extraction/summarisation.

    This call is the ONLY thing standing between a transcript and the LLM
    vendor. :func:`shared.brain.write_scoped_memory` re-screens whatever comes
    back, but that protects *storage*, not *egress* — it cannot un-send a
    request. So if redaction fails here the raw transcript is sent to the
    vendor and the chokepoint does not cover it.
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
    # extraction/summarisation. This is the only egress guard on the mining
    # path: write_scoped_memory re-screens what comes back, but that is a
    # storage guarantee, not an egress one, and it cannot un-send a request.
    # The except below therefore does NOT fail safe — it proceeds with the
    # unredacted transcript. Left as-is deliberately (availability over
    # egress) and called out here so the trade-off is visible rather than
    # implied by a comment claiming secrets never leave.
    try:
        screened = redact(combined).clean_text
        record_redaction(hit=screened != combined)
        combined = screened
    except Exception:
        record_capture("error")
        logger.warning("capture: transcript redaction failed; relying on chokepoint")
    return combined


def _serialize_memory(record: Any) -> dict[str, Any]:
    """Project a memory (``MemoryRecord`` OR a raw recency ``Row``) to JSON.

    Both expose ``id`` / ``memory_type`` / ``content`` / ``created_at`` via
    attribute access, so one serializer covers the vector-recall and recency
    paths.  Only the fields a context block needs are emitted (no embedding,
    no internal verification columns); ``id`` is stringified and ``created_at``
    is ISO-8601 (or ``None``).

    ``revision`` (migration 043) is the row's current CAS token — present when
    the caller's row/query selected it, ``None`` otherwise (``getattr`` rather
    than attribute access so this stays additive over any pre-existing caller
    that has not started selecting the column).
    """
    created_at = getattr(record, "created_at", None)
    revision = getattr(record, "revision", None)
    return {
        "id": str(record.id),
        "memory_type": record.memory_type,
        "content": record.content,
        "created_at": created_at.isoformat() if created_at is not None else None,
        "revision": str(revision) if revision is not None else None,
    }


def _recent_memories(
    db: Any,
    *,
    org_id: str,
    project_id: Optional[str],
    limit: int,
    user_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return the most-recent active memories the caller may read.

    Used when recall is called with an empty query (session start).  Tenant-
    scoped (``org_id`` + ``status='active'``) so neither verification's
    ``scope='session'`` rows nor other orgs' rows are returned.  Rows with
    ``recall_hidden = TRUE`` are excluded: these are raw passive-hook
    observations stored as curator fuel and must not surface to agents at
    session start (mirrors the hidden-row contract of
    :func:`shared.memory.retrieve_memories_scoped`).
    Context-branch rows (``branch_id IS NOT NULL``, Track C) are excluded the
    same way — session start reads MAIN only, with no opt-in here on purpose:
    this path ranks by nothing but the clock, so a branch's freshest
    half-formed draft would otherwise open every session.
    When ``project_id`` is given, rows for that project OR unfiled (``NULL``)
    rows surface — the same NULL-surfacing semantics as
    :func:`shared.memory.retrieve_memories_scoped`.

    The scope predicate is the SAME visibility ladder the vector path uses, and
    it had to change in this commit or it would have become the leak: hook
    captures now land at ``scope='private'``, so a query that still said
    ``scope = 'org'`` would have returned nothing at session start for exactly
    the people generating the most memories. It is written as SQL — including
    the ``project_members`` subquery — for the same reason as the vector path:
    a Python filter over a wider read is one bug away from returning the wider
    read.

    **The project predicate is a fence, and this is the fix for the reported
    symptom.** It used to be
    ``(:project_id IS NULL OR project_id = :project_id OR project_id IS NULL)``,
    which is two escape hatches in one line: an unresolved project matched
    every row in the org, and a resolved one still admitted every unfiled row.
    Combined with ``ORDER BY created_at DESC`` and no relevance term at all,
    that is literally "the last N things anyone wrote anywhere" — and it is how
    a session about the Klio proxy opened with two facts about a Vedic
    astrology app and two about a hiring app, all of them real rows, all filed
    under other projects, all recent.

    It is now **symmetric with the write path**:

    * project resolved → ``project_id = :project_id``. Rows from other
      projects cannot appear, and neither can unfiled ones.
    * project unresolved → ``project_id IS NULL``. A capture from a directory
      that resolves to no project is written unfiled; a recall from that same
      directory now reads exactly that bucket back.

    Why not keep the ``OR project_id IS NULL`` widening for a resolved project,
    given that most deliberate ``note``/``decision``/``plan`` memories are
    written unfiled through MCP? Because this path has **no relevance signal
    whatsoever** — it is an ``ORDER BY created_at DESC``. Every row it admits
    beyond the fence is chosen by nothing but its clock, so widening the fence
    does not add context, it adds coin flips. The query path
    (:func:`shared.memory.retrieve_memories_scoped`) keeps its two-tier
    softness precisely because it *can* rank what it lets in.

    An empty result is a legitimate answer here, not a failure — see the recall
    handler.
    """
    scope_arms = ["scope = 'org'"]
    params: dict[str, Any] = {
        "org_id": org_id,
        "project_id": project_id,
        "limit": limit,
    }
    if user_id:
        scope_arms.append("(scope = 'private' AND user_id = :user_id)")
        # Joined through projects.org_id for the same reason as the vector
        # path: project_members has no org_id, so a membership row in another
        # tenant would otherwise satisfy this arm here.
        scope_arms.append(
            "(scope = 'project' AND project_id IN ("
            "SELECT m.project_id FROM project_members m "
            "JOIN projects p ON p.id = m.project_id AND p.org_id = :org_id "
            "WHERE m.user_id = :user_id))"
        )
        params["user_id"] = user_id
    # Assembled from module-local literals only — every value that came from
    # outside this process (org_id, user_id, project_id, limit) is a bound
    # parameter. Same construction as the WHERE-clause assembly in
    # shared.memory.retrieve_memories_scoped.
    scope_sql = "(" + " OR ".join(scope_arms) + ")"
    # Two module-local literals, chosen by whether a project resolved. Not a
    # parameterised expression, because the two cases are different predicates
    # rather than one predicate with different values — writing them as one
    # SQL string is what produced the escape hatches this replaced.
    project_sql = "project_id = :project_id" if project_id else "project_id IS NULL"

    rows = db.execute(
        text(
            f"""
            SELECT id, memory_type, content, created_at, revision
            FROM session_memories
            WHERE org_id = :org_id
              AND {scope_sql}
              AND status = 'active'
              AND recall_hidden = FALSE
              AND branch_id IS NULL
              AND {project_sql}
            ORDER BY created_at DESC
            LIMIT :limit
            """
        ),
        params,
    ).fetchall()
    return [_serialize_memory(r) for r in rows]


async def enqueue_trace_outbox(*, payload: dict[str, Any]) -> None:
    """Publish one redacted transcript payload to ``sessions.captured``.

    Thin wrapper around :func:`shared.evidence_stream.publish_captured` so the
    call site in :func:`capture_transcript` has a single, mockable seam (tests
    patch ``app.capture.enqueue_trace_outbox`` directly rather than reaching
    into the Redis client). Raises on failure — the caller wraps this in its
    own try/except, because evidence publishing is best-effort: it must never
    fail the capture request itself.
    """
    await publish_captured(payload=payload)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/event")
async def capture_event(
    request: Request,
    req: CaptureEventRequest,
    key_info: KeyInfo = Depends(verify_api_key),
) -> dict[str, Any]:
    """Store one passive memory (default ``observation``).

    Coerces an unknown ``memory_type`` to ``observation`` (the schema accepts a
    free string for forward-compat, but only the RAW set is allowed through),
    tags ``source`` in metadata, best-effort files it under a project, and
    persists via :func:`shared.brain.write_scoped_memory` (dedup on).

    Scope is decided by :func:`_hook_write_scope`, not by the caller: an
    attributable hook capture is the author's private memory. Provenance stays
    ``EXTRACTED`` — a hook records what was said or done; it deduces nothing.
    """
    memory_type = (
        req.memory_type
        if req.memory_type in _CAPTURE_EVENT_TYPES
        else _DEFAULT_EVENT_TYPE
    )
    metadata = dict(req.metadata or {})
    metadata.setdefault("source", "hook")

    db = SessionLocal()
    try:
        # The ONE resolver (see app.identity): the dependency-validated key is
        # seeded into the per-request memo so this costs no second validation,
        # and the org that comes back is the ACTIVE node — an X-Klio-Node
        # header redirects the whole capture, and an unreachable node is a
        # 404 before anything is written.
        seed_request_key_info(request, key_info)
        org_id, agent_id, _plan, _overrides = resolve_active_principal(request, db)
        scope, user_id = _hook_write_scope(request, key_info, req.scope)
        # ensure_project is synchronous (blocking psycopg2); run it off the
        # event loop per its documented threading contract (shared.brain).
        project_id = await anyio.to_thread.run_sync(
            lambda: _ensure_project_id(
                db,
                org_id=org_id,
                git_remote=req.git_remote,
                repo_root=req.repo_root,
                user_id=user_id,
            )
        )
        # The project's default capture scope, applied only now that a project
        # has actually resolved. `_hook_write_scope` already returned the safe
        # answer; this can only widen it, and only when the project says so and
        # the member has not opted out. Unfiled captures never reach here with
        # a project, so they stay private.
        # ... unless the scope is already an explicit org publication (a
        # service token declared it — see _hook_write_scope). The comment
        # above says this resolution "can only widen"; that stopped being
        # true the day org became reachable upstream, and re-resolving here
        # silently re-narrowed every chat capture back to private. An
        # explicit publication is final.
        if user_id and scope == "private":
            scope = await anyio.to_thread.run_sync(
                lambda: project_scope.resolve_capture_scope(
                    db, org_id=org_id, project_id=project_id, user_id=user_id
                )
            )
        # See the transcript route: branch binding only where the project
        # opted into project-visible capture. The branch must never widen a
        # private capture.
        branch_id = None
        branch_work_item_id = None
        if scope == "project":
            branch_row = await anyio.to_thread.run_sync(
                lambda: _capture_branch(
                    db,
                    org_id=org_id,
                    project_id=project_id,
                    user_id=user_id,
                    git_branch=req.git_branch,
                )
            )
            if branch_row is not None:
                branch_id = branch_row["id"]
                branch_work_item_id = branch_row.get("work_item_id")
        over_cap = await anyio.to_thread.run_sync(
            lambda: _check_memory_quota_402(
                db, org_id, key_info.plan, "memory", key_info.plan_overrides
            )
        )
        # Raw tool/prompt observations are curator fuel, not recall
        # results; deliberate types (note/plan/decision/memory) stay
        # visible.
        result = await brain.write_scoped_memory(
            db,
            org_id=org_id,
            agent_id=agent_id,
            content=req.content,
            memory_type=memory_type,
            session_id=_capture_session_id(
                org_id, agent_id, req.session_id, user_id=user_id
            ),
            project_id=project_id,
            branch_id=branch_id,
            work_item_id=branch_work_item_id,
            metadata=metadata,
            scope=scope,
            user_id=user_id,
            recall_hidden=(memory_type == "observation"),
        )
        if result.get("stored"):
            await anyio.to_thread.run_sync(
                lambda: record_usage(db, org_id, "memory", over_cap=over_cap)
            )
        # ``rejected`` is not a failure — a dedup or a policy refusal is the
        # system working. It is counted apart from ``error`` so a rise in one
        # is never mistaken for the other.
        record_capture("accepted" if result.get("stored") else "rejected")
        return result
    except HTTPException:
        raise
    except Exception:
        record_capture("error")
        logger.exception("capture event failed (memory_type=%s)", memory_type)
        return {"stored": False, "error": "capture_failed"}
    finally:
        db.close()


@router.post("/transcript")
async def capture_transcript(
    request: Request,
    req: CaptureTranscriptRequest,
    key_info: KeyInfo = Depends(verify_api_key),
) -> dict[str, Any]:
    """Distil a finished session into typed memory rows and one ``summary``.

    Extraction failures are tolerated independently: a failed summary still
    allows the distilled memories to be stored and vice-versa.  Each synthesised memory is written
    via :func:`shared.brain.write_scoped_memory` (dedup on), so re-sending the
    same transcript is idempotent.  The curator never reprocesses these (it reads
    only RAW types, not ``fact`` / ``summary``).

    Scope follows :func:`_hook_write_scope`, same as ``/capture/event``: a
    transcript is the richest passive capture there is, and publishing one to
    the team on a ``Stop`` hook is the exact thing decision 3 forbids.
    """
    db = SessionLocal()
    try:
        # Resolution comes FIRST, before even the empty-transcript check: a
        # hook firing with an unreachable X-Klio-Node header must get the 404
        # refusal regardless of payload, never a success-shaped answer.
        # SessionLocal connects lazily, so an empty transcript still does no
        # session work.
        seed_request_key_info(request, key_info)
        org_id, agent_id, _plan, _overrides = resolve_active_principal(request, db)
        scope, user_id = _hook_write_scope(request, key_info, "org")
        combined = _combine_messages(req.messages)
        if not combined:
            record_capture("rejected")
            return {
                "stored": 0,
                "facts": 0,
                "summary": False,
                "reason": "empty_transcript",
            }

        project_id = await anyio.to_thread.run_sync(
            lambda: _ensure_project_id(
                db,
                org_id=org_id,
                git_remote=req.git_remote,
                repo_root=req.repo_root,
                user_id=user_id,
            )
        )
        # Same rule as /capture/event: the project's default is applied only
        # after a project has resolved, and can only widen the safe answer
        # `_hook_write_scope` already produced.
        # ... unless the scope is already an explicit org publication (a
        # service token declared it — see _hook_write_scope). The comment
        # above says this resolution "can only widen"; that stopped being
        # true the day org became reachable upstream, and re-resolving here
        # silently re-narrowed every chat capture back to private. An
        # explicit publication is final.
        if user_id and scope == "private":
            scope = await anyio.to_thread.run_sync(
                lambda: project_scope.resolve_capture_scope(
                    db, org_id=org_id, project_id=project_id, user_id=user_id
                )
            )
        # Branch binding is gated on the project having OPTED INTO
        # project-visible capture (default_scope='project', migration 047).
        # Nearly every developer is on a git feature branch nearly all the
        # time, so letting the branch's presence force scope='project' would
        # silently widen every private hook capture to the whole project --
        # the declared-scope-discarded defect class, again. A private capture
        # stays private and unbranched; a branch is a draft OF THE PROJECT,
        # and only exists where captures are project-visible to begin with.
        branch_id = None
        branch_work_item_id = None
        if scope == "project":
            branch_row = await anyio.to_thread.run_sync(
                lambda: _capture_branch(
                    db,
                    org_id=org_id,
                    project_id=project_id,
                    user_id=user_id,
                    git_branch=req.git_branch,
                )
            )
            if branch_row is not None:
                branch_id = branch_row["id"]
                branch_work_item_id = branch_row.get("work_item_id")
        session_id = _capture_session_id(
            org_id, agent_id, req.session_id, user_id=user_id
        )
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
                branch_id=branch_id,
                work_item_id=branch_work_item_id,
                metadata=base_meta,
                scope=scope,
                user_id=user_id,
            )
            if result.get("stored"):
                stored += 1
                await anyio.to_thread.run_sync(
                    lambda: record_usage(db, org_id, "memory", over_cap=over_cap)
                )

        # Typed distillation. Each item carries its own ``memory_type`` from
        # the taxonomy the Context workspace groups by, instead of every item
        # landing as ``fact`` because that was the only thing the old
        # facts-only prompt could return. Items that fail the durability rule
        # never reach this loop — see :mod:`shared.taxonomy`.
        try:
            memories = await extract_typed_memories(combined)
        except Exception:
            logger.warning("capture: extract_typed_memories raised for org %s", org_id)
            memories = []
        for item in memories:
            if not item.content:
                continue
            result = await brain.write_scoped_memory(
                db,
                org_id=org_id,
                agent_id=agent_id,
                content=item.content,
                memory_type=item.memory_type,
                session_id=session_id,
                project_id=project_id,
                branch_id=branch_id,
                work_item_id=branch_work_item_id,
                metadata=base_meta,
                scope=scope,
                user_id=user_id,
            )
            if result.get("stored"):
                stored += 1
                await anyio.to_thread.run_sync(
                    lambda: record_usage(db, org_id, "memory", over_cap=over_cap)
                )

        record_capture("accepted" if stored else "rejected")

        # Evidence loop: publish the redacted session for retention. Best
        # effort — a publish failure (bad Redis, redaction failure inside
        # build_captured_payload) must never fail the capture request, which
        # has already succeeded by this point.
        #
        # Uses the SAME derived `session_id` the distillation writes above
        # used (via `_capture_session_id`), not the raw client-supplied
        # `req.session_id`. The raw value is not unique per principal — two
        # different agents/users with no session_id (or the same client
        # string) would otherwise collapse onto one evidence identity, and
        # storage-worker's richer-wins upsert (migration 041) would let one
        # silently overwrite another's session_traces row.
        try:
            payload = build_captured_payload(
                org_id=org_id,
                session_id=session_id,
                messages=[m.model_dump() for m in req.messages],
                tool_calls=req.tool_calls,
                project_id=project_id,
                user_id=user_id,
                agent_id=agent_id,
            )
            if payload is not None:
                await enqueue_trace_outbox(payload=payload)
        except Exception:
            logger.warning("evidence publish failed; capture unaffected", exc_info=True)

        # ``facts`` stays in the response for wire compatibility with hook
        # clients already in the field: it is the count of distilled items,
        # which is exactly what it always was. ``by_kind`` is the additive
        # breakdown across the taxonomy.
        by_kind: dict[str, int] = {}
        for item in memories:
            by_kind[item.memory_type] = by_kind.get(item.memory_type, 0) + 1
        return {
            "stored": stored,
            "facts": len(memories),
            "by_kind": by_kind,
            "summary": bool(summary),
        }
    except HTTPException:
        raise
    except Exception:
        record_capture("error")
        logger.exception("capture transcript failed")
        return {"stored": 0, "error": "capture_failed"}
    finally:
        db.close()


@router.post("/recall")
async def capture_recall(
    request: Request,
    req: RecallRequest,
    key_info: KeyInfo = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return memories for context injection.

    With a non-empty ``query`` this is vector recall via
    :func:`shared.memory.retrieve_memories_scoped`; with an empty ``query`` it
    falls back to recency (most-recent first).

    **Recall may legitimately return zero memories, and callers must handle
    that as a normal answer.** Two changes here make it more likely than it
    used to be, both deliberately:

    * the project filter is now a fence rather than a widening (see
      :func:`_find_project_id` and :func:`_recent_memories`) — a session in a
      directory with no memories gets none, instead of getting somebody
      else's;
    * :func:`shared.ranking.apply_relevance_floor` drops rows that are not
      relevant enough to be worth injecting, and can drop all of them.

    Injecting nothing beats injecting four wrong things: an agent handed no
    context asks, and an agent handed confidently-wrong context builds on it.
    The Node proxy already reports ``reason: "empty"`` for this case.

    Project resolution takes ``git_remote`` then ``repo_root``; the space
    filter remains best-effort (an unknown slug widens rather than empties,
    unlike the strict MCP ``recall`` tool, which warns + empties).

    The caller's ``user_id`` is forwarded on BOTH branches. Without it a
    session-start injection would return the org brain and none of the caller's
    own private captures — which, post-routing, is most of what they wrote.
    ``scope='private'`` from an unattributable caller is refused rather than
    quietly widened to the org: see the guard below.
    """
    limit = min(req.limit, _MAX_RECALL_LIMIT)

    db = SessionLocal()
    try:
        # Same ordering as the capture writes: the active node is resolved
        # first, so a selected node scopes the read and an unreachable one is
        # a 404 rather than a quiet fallback to the key's org.
        seed_request_key_info(request, key_info)
        org_id, agent_id, _plan, _overrides = resolve_active_principal(request, db)
        user_id = resolve_user_id(request, key_info)
        if req.scope == "private" and not user_id:
            # retrieve_memories_scoped would raise; answer honestly instead of
            # letting a 500 look like "you have no private memories".
            return {
                "memories": [],
                "count": 0,
                "error": "unattributed_caller",
                "detail": (
                    "scope='private' needs a caller we can attribute to a "
                    "person. This request resolved to no user."
                ),
            }
        # The synchronous brain/recency reads run off the event loop per the
        # shared.brain threading contract (blocking psycopg2 I/O).
        project_id = await anyio.to_thread.run_sync(
            lambda: _find_project_id(
                db,
                org_id=org_id,
                git_remote=req.git_remote,
                repo_root=req.repo_root,
            )
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
                    db,
                    org_id=org_id,
                    project_id=project_id,
                    limit=limit,
                    user_id=user_id,
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
                user_id=user_id,
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
        #
        # session_id is derived through the SAME _capture_session_id call the
        # transcript handler uses (same org/agent/user/raw-session-id inputs),
        # not the raw req.session_id: the grader joins brain_recall_events to
        # session_traces on this derived identity, so a mismatched derivation
        # here would silently break every join.
        recall_session_id = _capture_session_id(
            org_id, agent_id, req.session_id, user_id=user_id
        )
        await anyio.to_thread.run_sync(
            lambda: log_recall_event(
                db,
                org_id=org_id,
                agent_id=agent_id,
                project_id=project_id,
                query_text=(query or None),
                result_count=len(memories),
                source="hook",
                memory_ids=[m["id"] for m in memories],
                session_id=recall_session_id,
            )
        )
        await anyio.to_thread.run_sync(
            lambda: record_usage(db, org_id, "recall", over_cap=recall_over_cap)
        )
        # Claims ride every recall response (Build 4), same as the MCP
        # `recall` tool: passive session-start injection is still a recall,
        # so an agent hitting this REST path gets the same active_claims
        # awareness. Best-effort: a claims-listing failure must never be the
        # reason a capture recall fails.
        try:
            active_claims = await anyio.to_thread.run_sync(
                lambda: claims.live_claims(db, org_id=org_id, project_id=project_id)
            )
        except Exception:
            logger.warning(
                "capture recall: live_claims lookup raised; omitting active_claims",
                exc_info=True,
            )
            active_claims = []
        return {
            "memories": memories,
            "count": len(memories),
            "active_claims": active_claims,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("capture recall failed (scope=%s)", req.scope)
        return {"memories": [], "count": 0, "error": "recall_failed"}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# /capture/forget
# ---------------------------------------------------------------------------


class ForgetRequest(BaseModel):
    """Retract one memory by id (hook-client mirror of the MCP forget tool)."""

    memory_id: str = Field(min_length=1, description="Memory id to retract.")


@router.post("/forget")
async def capture_forget(
    request: Request,
    req: ForgetRequest,
    key_info: KeyInfo = Depends(verify_api_key),
) -> dict[str, Any]:
    """Retract a memory. 404 unknown/cross-org/other-user-private, 403 cross-agent.

    Delegates to :func:`shared.memory.retract_memory` (synchronous DB I/O)
    which enforces org_id on every query, refuses another user's private row,
    and rejects cross-agent scope='agent' rows.  The route maps the
    storage-layer error codes to HTTP status codes: ``not_found`` /
    ``invalid_id`` → 404, ``forbidden`` → 403.

    ``user_id`` must be forwarded: without it every private row in the org
    looks unowned to ``retract_memory``, and a teammate could delete memories
    they are not permitted to even read.
    """
    db = SessionLocal()
    try:
        # Active-node resolution, same as every other capture route: a forget
        # aimed at a selected node retracts THERE, and an unreachable node is
        # a 404 before the retraction is attempted.
        seed_request_key_info(request, key_info)
        org_id, agent_id, _plan, _overrides = resolve_active_principal(request, db)
        user_id = resolve_user_id(request, key_info)
        result = await anyio.to_thread.run_sync(
            lambda: retract_memory(
                db,
                org_id=org_id,
                agent_id=agent_id,
                memory_id=req.memory_id,
                via="hook",
                user_id=user_id,
            )
        )
        if result.get("error") in ("not_found", "invalid_id"):
            raise HTTPException(status_code=404, detail=result["error"])
        if result.get("error") == "forbidden":
            raise HTTPException(status_code=403, detail="forbidden")
        return result
    finally:
        db.close()
