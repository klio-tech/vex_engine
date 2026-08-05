"""The `init` MCP tool — onboarding that fills a new user's workplace.

Spec: docs/superpowers/specs/2026-08-03-klio-init-onboarding-design.md.

`init(phase, ...)` is a structured interview the server conducts with the
agent: the agent does the reading (its conversation, the repo, two questions
to the user) and the server does the judging — validation, redaction, dedup,
provenance, filing. Three phases:

* ``start``  — return the versioned playbook and the caps.
* ``submit`` — accept candidate memories; every one goes through the SAME
  door as capture (:func:`shared.brain.write_scoped_memory`: redaction fail
  closed, dedup, embedding), at ``scope='private'`` ALWAYS.
* ``finish`` — summarise the run, write the marker note that makes re-runs
  detectable and countable.

Statefulness: none beyond the rows themselves. A run is identified by
``(org_id, user_id, UTC day)``; the per-run accepted count and the runs-per-day
count are DERIVED from the rows init already wrote (metadata tags), so a client
that ignores the playbook still cannot exceed the caps and no new table or
migration exists.

Security invariants (each mutation-checked in tests/test_init_tool.py):

1. No redaction bypass — candidates are stored only via
   ``write_scoped_memory``, the same chokepoint capture uses.
2. Private only — ``scope="private"`` is hard-coded at the single write call;
   no parameter, header, or payload field reaches it.
3. Identity is server-resolved — ``org_id`` from the key,
   ``user_id`` via :func:`app.identity.resolve_request_user_id`; the payload
   is never consulted.
4. Caps are server-side — enforced here, not in the playbook text.
5. Idempotence — private-scope dedup (user-scoped window, session-independent)
   makes a re-run collapse to ~zero new rows.
"""

from datetime import datetime, timezone
from typing import Any, Optional

import logging

from fastapi import HTTPException
from shared import brain
from shared.plan_limits import get_plan_config
from shared.usage import MemoryQuotaExceeded, check_memory_quota, record_usage
from sqlalchemy import text

from app.curator import RAW_MEMORY_TYPES
from app.db import SessionLocal

# Both halves of the caller's identity come from ``app.identity``, never one
# from here and one from ``app.auth``. ``app.identity.request_key_info``
# memoizes the validated KeyInfo on ``request.state``, and only the identity
# module's ``resolve_principal_plan`` writes that memo. Pairing ``app.auth``'s
# copy with ``resolve_request_user_id`` left the memo empty, so ``init``
# validated the key twice per request — two rate-limit tokens for one call, and
# on the token that trips it is the SECOND call that raises, the one deciding
# whether this request has a person behind it.
from app.identity import resolve_principal_plan, resolve_request_user_id

logger = logging.getLogger("agentguard.mcp.init")

# ---------------------------------------------------------------------------
# Caps (spec table "Caps and quality"). Server-enforced; the playbook merely
# repeats them so a well-behaved agent can prioritise before submitting.
# ---------------------------------------------------------------------------

MAX_CANDIDATES_PER_SUBMIT = 25
MAX_ACCEPTED_PER_RUN = 60
MAX_CANDIDATE_CHARS = 2000
MAX_RUNS_PER_DAY = 3

CAPS: dict[str, int] = {
    "candidates_per_submit": MAX_CANDIDATES_PER_SUBMIT,
    "accepted_per_run": MAX_ACCEPTED_PER_RUN,
    "content_chars_per_candidate": MAX_CANDIDATE_CHARS,
    "runs_per_day": MAX_RUNS_PER_DAY,
}

# Candidate sources (spec: phase `submit`). "import" is reserved for v1.1.
ALLOWED_SOURCES: frozenset[str] = frozenset({"conversation", "repo", "user", "import"})

# Candidate kinds: the established RAW input set the curator consumes —
# imported, not redeclared, so the two can never drift.
ALLOWED_KINDS: frozenset[str] = frozenset(RAW_MEMORY_TYPES)

# The scope every init row lands at. A module constant so tests can assert
# the value flows to the write call verbatim; deliberately NOT a parameter of
# any public function in this module.
_INIT_SCOPE = "private"

# Metadata tags. `source: "init"` marks a row as seeded by onboarding (the
# per-run accepted count is derived from it); `init_marker` marks the one
# note `finish` writes per run (the runs-per-day count is derived from it).
_META_SOURCE = "init"
_META_MARKER_KEY = "init_marker"

PLAYBOOK_VERSION = "v1.1"

# Versioned server-side so it can improve without a client update.
PLAYBOOK: dict[str, Any] = {
    "version": PLAYBOOK_VERSION,
    "steps": [
        {
            "id": "conversation",
            "title": "Distil this conversation",
            "instructions": (
                "Distil what you already know from this session: who the user "
                "is, what they are building, decisions already made, "
                "preferences already stated. Durable facts only — no "
                "transcripts, no play-by-play. Submit with "
                "source='conversation'."
            ),
        },
        {
            "id": "repo",
            "title": "Read the repo, if there is one",
            "instructions": (
                "Read CLAUDE.md, AGENTS.md, .cursor/rules, README, and the "
                "last ~50 commit subjects. Extract conventions, architecture "
                "decisions, active plans, and standing rules. Skip anything "
                "generated, boilerplate, or stale. Submit with source='repo', "
                "and pass the repo's git remote URL as `git_remote` so the "
                "memories are filed under the right project."
            ),
        },
        {
            "id": "user",
            "title": "Ask the user two questions, at most",
            "instructions": (
                "Ask: 'Anything your agents keep getting wrong that I should "
                "record?' and 'Anything here that should NOT be remembered?' "
                "The second is a consent gate, not decoration — anything the "
                "user marks off-limits must not be submitted. Submit their "
                "answers with source='user'."
            ),
        },
        {
            "id": "import",
            "title": "Offer to import memory from another assistant",
            "instructions": (
                "Only if the user offers it or asks for it: they can paste "
                "an export of their memory from another assistant (Claude's "
                "memory summary, a ChatGPT memory export, or similar). "
                "Distil the pasted text into durable facts and submit them "
                "with source='import'. Do not solicit exports yourself, do "
                "not read other tools' storage, and never submit the raw "
                "paste — distil first. Skip this step entirely if the user "
                "does not bring it up."
            ),
        },
    ],
    "do_not_submit": [
        "secrets, API keys, tokens, or credentials of any kind",
        "personal data (emails, phone numbers, addresses) beyond what the "
        "user explicitly asked to record",
        "tool-call logs, file contents, or transcripts",
        "anything the user marked off-limits",
    ],
    "submit_contract": {
        "shape": "{content, kind, source}",
        "kinds": sorted(ALLOWED_KINDS),
        "sources": sorted(ALLOWED_SOURCES),
    },
}


def _utc_day_start(now: Optional[datetime] = None) -> datetime:
    moment = now or datetime.now(timezone.utc)
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)


def _count_runs_today(db: Any, *, org_id: str, user_id: str) -> int:
    """How many onboarding runs this user FINISHED today (UTC).

    Derived from the marker notes `finish` writes — no run table exists. The
    marker is written with dedup OFF, so every finished run leaves exactly one
    countable row even when two runs summarise identically.
    """
    row = db.execute(
        text(
            """
            SELECT COUNT(*)
            FROM session_memories
            WHERE org_id = :org_id
              AND user_id = :user_id
              AND scope = 'private'
              AND memory_type = 'note'
              AND metadata->>:marker_key = 'true'
              AND created_at >= :day_start
            """
        ),
        {
            "org_id": org_id,
            "user_id": user_id,
            "marker_key": _META_MARKER_KEY,
            "day_start": _utc_day_start(),
        },
    ).fetchone()
    return int(row[0]) if row else 0


def _count_accepted_today(db: Any, *, org_id: str, user_id: str) -> int:
    """How many candidates init has ACCEPTED for this user today (UTC).

    The per-run cap is enforced against this count (a run == a UTC day, per
    the plan), so a client that skips `start` or reconnects mid-run still
    cannot exceed :data:`MAX_ACCEPTED_PER_RUN`. Marker notes are excluded —
    they are bookkeeping, not seeded knowledge.
    """
    row = db.execute(
        text(
            """
            SELECT COUNT(*)
            FROM session_memories
            WHERE org_id = :org_id
              AND user_id = :user_id
              AND scope = 'private'
              AND metadata->>'source' = :source
              AND COALESCE(metadata->>:marker_key, 'false') <> 'true'
              AND created_at >= :day_start
            """
        ),
        {
            "org_id": org_id,
            "user_id": user_id,
            "source": _META_SOURCE,
            "marker_key": _META_MARKER_KEY,
            "day_start": _utc_day_start(),
        },
    ).fetchone()
    return int(row[0]) if row else 0


def _unattributed() -> dict[str, Any]:
    return {
        "error": "unattributed_caller",
        "detail": (
            "init seeds YOUR private memory, so the request must resolve to a "
            "person — sign in via OAuth or use an API key created by your "
            "user account. This request resolved to no user."
        ),
    }


async def perform_init(
    ctx: Any,
    *,
    phase: str,
    candidates: Optional[list[dict[str, Any]]] = None,
    git_remote: Optional[str] = None,
) -> dict[str, Any]:
    """Shared body for the `init` tool. Auth, DB session, phase dispatch.

    Mirrors the auth/session boilerplate of ``app.mcp_app._perform_write``:
    principal from the transport request (never the payload), a short-lived
    DB session closed in ``finally``, HTTPException translated to a clean
    ``unauthorized`` result.
    """
    request = getattr(ctx.request_context, "request", None)
    if request is None:
        logger.error("init got no transport request; cannot authenticate")
        return {"error": "unauthorized", "detail": "no request context"}

    try:
        org_id, agent_id, plan, plan_overrides = resolve_principal_plan(request)
    except HTTPException as exc:
        return {"error": "unauthorized", "detail": str(exc.detail)}

    # The ONLY identity source. Never a header, never the payload.
    user_id = resolve_request_user_id(request)
    if not user_id:
        return _unattributed()

    db = SessionLocal()
    try:
        if phase == "start":
            return _phase_start(db, org_id=org_id, user_id=user_id)
        if phase == "submit":
            return await _phase_submit(
                db,
                org_id=org_id,
                agent_id=agent_id,
                user_id=user_id,
                plan=plan,
                plan_overrides=plan_overrides,
                candidates=candidates,
                git_remote=git_remote,
            )
        if phase == "finish":
            return await _phase_finish(
                db, org_id=org_id, agent_id=agent_id, user_id=user_id
            )
        # The schema constrains `phase`; guard defensively anyway.
        return {"error": "unknown_phase", "phase": phase}
    except Exception:
        logger.exception("init failed (phase=%s)", phase)
        return {"error": "init_failed"}
    finally:
        db.close()


def _phase_start(db: Any, *, org_id: str, user_id: str) -> dict[str, Any]:
    runs_today = _count_runs_today(db, org_id=org_id, user_id=user_id)
    if runs_today >= MAX_RUNS_PER_DAY:
        return {
            "error": "run_limit_reached",
            "detail": (
                f"init has already run {runs_today} times today (limit "
                f"{MAX_RUNS_PER_DAY} per day). Your workplace is seeded — "
                "use recall to read it, remember/note to add to it."
            ),
        }
    return {
        "phase": "start",
        "playbook": PLAYBOOK,
        "caps": CAPS,
        "runs_today": runs_today,
        "next": (
            "Perform the playbook steps, then call "
            "init(phase='submit', candidates=[...]) one or more times, then "
            "init(phase='finish')."
        ),
    }


def _validate_candidate(candidate: Any) -> Optional[dict[str, Any]]:
    """Return a refusal dict (sans index) for a bad candidate, else ``None``.

    Candidate-level, not batch-level: one malformed entry never sinks its
    neighbours (spec: "reject the candidate, not the batch").
    """
    if not isinstance(candidate, dict):
        return {
            "reason": "invalid_candidate",
            "detail": "expected {content, kind, source}",
        }
    kind = candidate.get("kind")
    if kind not in ALLOWED_KINDS:
        return {
            "reason": "invalid_kind",
            "detail": f"kind must be one of {sorted(ALLOWED_KINDS)}",
        }
    source = candidate.get("source")
    if source not in ALLOWED_SOURCES:
        return {
            "reason": "invalid_source",
            "detail": f"source must be one of {sorted(ALLOWED_SOURCES)}",
        }
    content = candidate.get("content")
    if not isinstance(content, str) or not content.strip():
        return {"reason": "empty_content", "detail": "content must be non-empty text"}
    if len(content) > MAX_CANDIDATE_CHARS:
        return {
            "reason": "content_too_long",
            "limit": MAX_CANDIDATE_CHARS,
        }
    return None


def _resolve_project_id(
    db: Any, *, org_id: str, git_remote: Optional[str]
) -> Optional[str]:
    """Best-effort, read-only project lookup for filing; never raises.

    The existing ``find_project_id`` path (same as capture's read side): it
    never CREATES a project, so onboarding cannot mint tenant rows, and a
    miss files the memories unfiled rather than failing the run.
    """
    if not git_remote or not git_remote.strip():
        return None
    try:
        return brain.find_project_id(db, org_id=org_id, identifier=git_remote.strip())
    except Exception:
        logger.warning(
            "init: find_project_id failed for org %s; filing unfiled",
            org_id,
            exc_info=True,
        )
        return None


async def _phase_submit(
    db: Any,
    *,
    org_id: str,
    agent_id: str,
    user_id: str,
    plan: str,
    plan_overrides: Optional[dict],
    candidates: Optional[list[dict[str, Any]]],
    git_remote: Optional[str],
) -> dict[str, Any]:
    if not candidates:
        return {
            "error": "no_candidates",
            "detail": (
                "phase='submit' needs candidates=[{content, kind, source}, ...]. "
                "Run init(phase='start') for the playbook."
            ),
        }
    if len(candidates) > MAX_CANDIDATES_PER_SUBMIT:
        return {
            "error": "too_many_candidates",
            "detail": (
                f"{len(candidates)} candidates in one call (limit "
                f"{MAX_CANDIDATES_PER_SUBMIT}). Prioritise the most durable, "
                "highest-signal facts and resubmit in smaller batches."
            ),
        }

    accepted_today = _count_accepted_today(db, org_id=org_id, user_id=user_id)
    remaining = MAX_ACCEPTED_PER_RUN - accepted_today
    if remaining <= 0:
        return {
            "error": "run_cap_reached",
            "detail": (
                f"this run already accepted {accepted_today} memories (limit "
                f"{MAX_ACCEPTED_PER_RUN} per run). Call init(phase='finish') "
                "to summarise what was seeded."
            ),
        }

    # The org's monthly memory quota applies to onboarding exactly as it
    # applies to every other write; a hard block refuses the call cleanly.
    try:
        over_cap = check_memory_quota(
            db, org_id, get_plan_config(plan, plan_overrides), "memory", plan_name=plan
        )
    except MemoryQuotaExceeded as exc:
        return {
            "error": "quota_exceeded",
            "detail": exc.message,
            "suggested_plan": exc.suggested_plan,
        }

    project_id = _resolve_project_id(db, org_id=org_id, git_remote=git_remote)
    session_id = _init_session_id(org_id, user_id)

    accepted: list[str] = []
    refused: list[dict[str, Any]] = []
    deduplicated = 0
    redacted_kinds: set[str] = set()

    for index, candidate in enumerate(candidates):
        problem = _validate_candidate(candidate)
        if problem is not None:
            refused.append({"index": index, **problem})
            continue
        if len(accepted) >= remaining:
            refused.append(
                {
                    "index": index,
                    "reason": "run_cap_reached",
                    "detail": f"limit {MAX_ACCEPTED_PER_RUN} accepted per run",
                }
            )
            continue

        source = candidate["source"]
        provenance = _provenance_for(source)
        kind = candidate["kind"]

        # THE DOOR. The same chokepoint capture and remember use:
        # redaction fail-closed, then user-scoped dedup, then embedding.
        # scope is the module constant 'private' — never an input.
        result = await brain.write_scoped_memory(
            db,
            org_id=org_id,
            agent_id=agent_id,
            content=candidate["content"],
            memory_type=kind,
            session_id=session_id,
            project_id=project_id,
            metadata={"source": _META_SOURCE, "init_source": source},
            scope=_INIT_SCOPE,
            user_id=user_id,
            provenance=provenance,
            # Raw observations are curator fuel, not recall results — the
            # same visibility rule capture applies.
            recall_hidden=(kind == "observation"),
        )

        if result.get("stored"):
            accepted.append(result["id"])
            record_usage(db, org_id, "memory", over_cap=over_cap)
        elif result.get("deduplicated"):
            deduplicated += 1
        else:
            refused.append(
                {
                    "index": index,
                    "reason": result.get("error", "write_failed"),
                }
            )
        if result.get("redacted"):
            redacted_kinds.update(result.get("redaction_kinds") or [])

    return {
        "phase": "submit",
        "accepted": accepted,
        "deduplicated": deduplicated,
        # Kinds only, never contents — the agent may relay this list.
        "redacted_kinds": sorted(redacted_kinds),
        "refused": refused,
        "remaining_run_capacity": max(remaining - len(accepted), 0),
    }


def _init_session_id(org_id: str, user_id: str) -> str:
    """Stable session id for init writes.

    Private-scope dedup ignores session_id (the window is the user's own
    rows), so this is grouping/traceability only — like capture's
    ``klio-hook:`` prefix.
    """
    return f"klio-init:{org_id}:{user_id}"


def _summarize_run_today(db: Any, *, org_id: str, user_id: str) -> dict[str, Any]:
    """Aggregate today's init rows: counts by kind and source, redaction KINDS.

    One pass over the same window the caps count against. Redactions surface
    as kinds only (e.g. ``stripe_secret_key``) — the contents were masked at
    the chokepoint and are not recoverable here, which is the point.
    """
    rows = db.execute(
        text(
            """
            SELECT memory_type,
                   metadata->>'init_source' AS init_source,
                   metadata->'redactions' AS redactions
            FROM session_memories
            WHERE org_id = :org_id
              AND user_id = :user_id
              AND scope = 'private'
              AND metadata->>'source' = :source
              AND COALESCE(metadata->>:marker_key, 'false') <> 'true'
              AND created_at >= :day_start
            """
        ),
        {
            "org_id": org_id,
            "user_id": user_id,
            "source": _META_SOURCE,
            "marker_key": _META_MARKER_KEY,
            "day_start": _utc_day_start(),
        },
    ).fetchall()

    by_kind: dict[str, int] = {}
    by_source: dict[str, int] = {}
    redacted_kinds: set[str] = set()
    for memory_type, init_source, redactions in rows:
        by_kind[memory_type] = by_kind.get(memory_type, 0) + 1
        label = init_source or "unknown"
        by_source[label] = by_source.get(label, 0) + 1
        if redactions:
            redacted_kinds.update(str(k) for k in redactions)
    return {
        "total": len(rows),
        "by_kind": by_kind,
        "by_source": by_source,
        "redacted_kinds": sorted(redacted_kinds),
    }


def _provenance_for(source: str) -> str:
    """EXTRACTED only when the user stated it directly. Everything the agent
    derived (conversation, repo) is its deduction — and a pasted export from
    another assistant is machine-distilled second-hand, so ``import`` is
    INFERRED too, never EXTRACTED."""
    return "EXTRACTED" if source == "user" else "INFERRED"


def _relay_paragraph(summary: dict[str, Any]) -> str:
    """The one paragraph the agent relays to the user verbatim."""
    total = summary.get("total", sum(summary["by_kind"].values()))
    if total == 0:
        return (
            "Klio now knows nothing new from this onboarding run — no "
            "candidates were accepted. Everything submitted was either a "
            "duplicate of what Klio already knew or was refused."
        )
    kinds = ", ".join(f"{n} {k}" for k, n in sorted(summary["by_kind"].items()))
    sources = ", ".join(
        f"{n} from {s}" for s, n in sorted(summary["by_source"].items())
    )
    text_out = (
        f"Klio now knows {total} thing{'s' if total != 1 else ''} about you "
        f"and your work: {kinds} "
        f"({sources}). They are stored PRIVATE to you — nothing was shared "
        "with your team; use the share tool to promote anything deliberately. "
        "Klio's curator will distil them into your knowledge graph on its "
        "next pass, so recall and explore get richer over the next hour."
    )
    if summary["redacted_kinds"]:
        text_out += (
            " Some submissions contained secrets or personal data "
            f"({', '.join(summary['redacted_kinds'])}); those values were "
            "masked before anything was stored."
        )
    return text_out


async def _phase_finish(
    db: Any, *, org_id: str, agent_id: str, user_id: str
) -> dict[str, Any]:
    summary = _summarize_run_today(db, org_id=org_id, user_id=user_id)

    # The marker note: makes re-runs detectable (spec, phase `finish`) and is
    # what the 3-runs-per-day cap counts. dedup OFF — two identical-looking
    # runs must still leave two countable markers, exactly like artifact
    # cards disable dedup to stay individually discoverable.
    marker_content = (
        f"Onboarded via init on {datetime.now(timezone.utc).date().isoformat()}, "
        f"{summary['total']} memories seeded"
    )
    marker = await brain.write_scoped_memory(
        db,
        org_id=org_id,
        agent_id=agent_id,
        content=marker_content,
        memory_type="note",
        session_id=_init_session_id(org_id, user_id),
        metadata={"source": _META_SOURCE, _META_MARKER_KEY: True},
        dedup=False,
        scope=_INIT_SCOPE,
        user_id=user_id,
    )

    return {
        "phase": "finish",
        "seeded": summary["total"],
        "by_kind": summary["by_kind"],
        "by_source": summary["by_source"],
        "redacted_kinds": summary["redacted_kinds"],
        "relay": _relay_paragraph(summary),
        "marker_id": marker.get("id"),
    }
