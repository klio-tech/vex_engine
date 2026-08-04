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
from sqlalchemy import text

from app.auth import resolve_principal_plan
from app.curator import RAW_MEMORY_TYPES
from app.db import SessionLocal
from app.identity import resolve_request_user_id

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
ALLOWED_SOURCES: frozenset[str] = frozenset({"conversation", "repo", "user"})

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

PLAYBOOK_VERSION = "v1"

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
    raise NotImplementedError


async def _phase_finish(
    db: Any, *, org_id: str, agent_id: str, user_id: str
) -> dict[str, Any]:
    raise NotImplementedError
