"""Org-scoped Klio brain curator: periodic synthesis of RAW memories.

The curator is a server-side background job that "tunes" an org's brain over
time. Periodically it reads the org's RAW brain memories — the discrete writes
made through the MCP write tools (``observation`` / ``note`` / ``memory`` /
``plan`` / ``decision``) — and synthesises higher-level memories back into the
same org brain: a single ``summary`` plus discrete ``fact`` rows. It reuses the
existing LLM helpers (:func:`shared.memory.generate_summary` /
:func:`shared.memory.extract_facts`) and the brain write path
(:func:`shared.brain.write_scoped_memory`) verbatim — nothing in ``shared`` is
modified.

Separation from verification:
    This module is deliberately independent of the verification worker
    (``services/async-worker``). The curator only ever reads ``scope='org'``
    rows — never verification's ``scope='session'`` rows, which live in the
    same ``session_memories`` table — so the two paths never interfere.

No self-reprocessing:
    The curator reads only the RAW input types
    (``observation``/``note``/``memory``/``plan``/``decision``) and never its
    own outputs (``summary``/``fact``). Combined with the per-org
    ``last_processed_at`` cursor (it only reads rows *newer* than the cursor),
    this makes a tick unable to loop on its own synthesised memories.

Cursor / idempotency:
    Progress is tracked per org in ``brain_curator_state`` (migration 021). A
    tick advances ``last_processed_at`` to the max ``created_at`` of the batch
    it processed. A tick that finds nothing new synthesises nothing and leaves
    ``last_processed_at`` untouched (only ``last_run_at`` is bumped), so re-runs
    are idempotent.

Async / DB I/O note:
    :func:`curator_tick` is ``async def`` yet performs inline *synchronous*
    psycopg2 DB I/O via ``db_session.execute(...)``. This matches the house
    pattern of :func:`shared.memory.write_memory` /
    :func:`shared.brain.write_scoped_memory`; do not wrap the DB calls in
    threads here.

Scheduling:
    This module exposes :func:`run_curator_once` as the callable a scheduler /
    cron would invoke. The periodic trigger itself (cron / scheduled job) is
    wired during deployment (F3); no live background task is registered in the
    app lifespan here.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from sqlalchemy import text

from shared.brain import write_scoped_memory
from shared.memory import extract_facts, generate_summary

logger = logging.getLogger("agentguard.mcp.curator")

# RAW input memory types the curator reads. Deliberately EXCLUDES the curator's
# own outputs ('summary'/'fact') so a tick cannot reprocess what it produced,
# and is read only under scope='org' so verification's scope='session' rows are
# never touched.
RAW_MEMORY_TYPES: tuple[str, ...] = (
    "observation",
    "note",
    "memory",
    "plan",
    "decision",
)

# Rendered as a SQL IN-list literal, e.g. "'observation','note',...". The values
# are a fixed module constant (never user input), so literal interpolation here
# is safe and keeps the IN-list inspectable in tests.
_RAW_TYPES_SQL: str = ", ".join(f"'{t}'" for t in RAW_MEMORY_TYPES)

DEFAULT_CURATOR_AGENT_ID: str = "klio-curator"
"""Agent principal under which synthesised memories are authored."""


def _now() -> datetime:
    """UTC now, isolated for clarity/testability."""
    return datetime.now(timezone.utc)


def _read_cursor(db_session: Any, org_id: str) -> Optional[datetime]:
    """Return this org's ``last_processed_at`` cursor, or ``None`` if unset.

    A missing state row (first ever tick for the org) yields ``None``, which the
    new-rows query treats as "no lower bound" (read from the beginning).
    """
    return db_session.execute(
        text(
            """
            SELECT last_processed_at
            FROM brain_curator_state
            WHERE org_id = :org_id
            """
        ),
        {"org_id": org_id},
    ).scalar()


def _read_new_raw_rows(
    db_session: Any,
    org_id: str,
    cursor: Optional[datetime],
    batch_size: int,
) -> list[Any]:
    """Read up to ``batch_size`` RAW org memories newer than ``cursor``.

    Tenant-scoped by ``org_id`` and locked to ``scope='org'`` + RAW types +
    ``status='active'`` so neither verification's session rows nor the curator's
    own ``summary``/``fact`` output are ever returned. Ordered ASC by
    ``created_at`` so the batch max is the last row (used to advance the cursor).
    """
    rows = db_session.execute(
        text(
            f"""
            SELECT id, content, created_at
            FROM session_memories
            WHERE org_id = :org_id
              AND scope = 'org'
              AND status = 'active'
              AND memory_type IN ({_RAW_TYPES_SQL})
              AND (:cursor IS NULL OR created_at > :cursor)
            ORDER BY created_at ASC
            LIMIT :batch_size
            """
        ),
        {"org_id": org_id, "cursor": cursor, "batch_size": batch_size},
    ).fetchall()
    return list(rows)


def _touch_last_run(db_session: Any, org_id: str) -> None:
    """Bump only ``last_run_at`` for an org, leaving the cursor untouched.

    Used on the no-op path (below threshold / no new rows). Race-safe upsert: a
    fresh org inserts a state row with the cursor still ``NULL``; an existing org
    only bumps ``last_run_at``.
    """
    db_session.execute(
        text(
            """
            INSERT INTO brain_curator_state (org_id, last_run_at)
            VALUES (:org_id, now())
            ON CONFLICT (org_id)
              DO UPDATE SET last_run_at = now()
            """
        ),
        {"org_id": org_id},
    )
    db_session.commit()


def _advance_cursor(
    db_session: Any,
    org_id: str,
    last_processed_at: datetime,
    processed_delta: int,
) -> None:
    """Advance the cursor to ``last_processed_at`` and add to ``processed_count``.

    Race-safe upsert against the ``brain_curator_state`` primary key
    (``org_id``): a fresh org inserts; an existing org folds the cursor advance,
    the ``processed_count`` increment, and the ``last_run_at`` bump into one
    atomic statement (matching the upsert idiom in :mod:`shared.brain`).
    """
    db_session.execute(
        text(
            """
            INSERT INTO brain_curator_state (
                org_id, last_processed_at, last_run_at, processed_count
            )
            VALUES (:org_id, :last_processed_at, now(), :processed_delta)
            ON CONFLICT (org_id)
              DO UPDATE SET
                last_processed_at = EXCLUDED.last_processed_at,
                last_run_at = now(),
                processed_count =
                    brain_curator_state.processed_count + :processed_delta
            """
        ),
        {
            "org_id": org_id,
            "last_processed_at": last_processed_at,
            "processed_delta": processed_delta,
        },
    )
    db_session.commit()


async def _synthesize_and_store(
    db_session: Any,
    *,
    org_id: str,
    agent_id: str,
    combined: str,
    input_count: int,
) -> int:
    """Run the LLM synthesis over ``combined`` and store the results.

    Resilient to LLM failure: if :func:`generate_summary` yields ``None`` the
    summary write is skipped but fact extraction is still attempted, and any
    individual write failure is logged without aborting the rest. Returns the
    number of memories that were *newly stored* (dedup hits do NOT count).
    """
    session_id = f"{agent_id}:{org_id}"
    base_meta = {"source": "curator", "input_count": input_count}
    stored = 0

    # --- summary --------------------------------------------------------
    try:
        summary = await generate_summary(combined)
    except Exception:
        logger.warning(
            "curator: generate_summary raised for org %s; skipping summary",
            org_id,
            exc_info=True,
        )
        summary = None

    if summary:
        if await _store_one(
            db_session,
            org_id=org_id,
            agent_id=agent_id,
            content=summary,
            memory_type="summary",
            session_id=session_id,
            metadata=base_meta,
        ):
            stored += 1
    else:
        logger.info("curator: no summary produced for org %s", org_id)

    # --- facts ----------------------------------------------------------
    try:
        facts = await extract_facts(combined)
    except Exception:
        logger.warning(
            "curator: extract_facts raised for org %s; no facts stored",
            org_id,
            exc_info=True,
        )
        facts = []

    for fact in facts:
        if not fact:
            continue
        if await _store_one(
            db_session,
            org_id=org_id,
            agent_id=agent_id,
            content=fact,
            memory_type="fact",
            session_id=session_id,
            metadata=base_meta,
        ):
            stored += 1

    return stored


async def _store_one(
    db_session: Any,
    *,
    org_id: str,
    agent_id: str,
    content: str,
    memory_type: str,
    session_id: str,
    metadata: dict,
) -> bool:
    """Write a single synthesised memory; return True only if newly stored.

    A dedup hit (``stored=False``) or an embedding failure counts as
    processed-but-not-new and returns ``False``. A raised exception is logged
    and swallowed (returns ``False``) so one bad write never aborts the tick.
    """
    try:
        result = await write_scoped_memory(
            db_session,
            org_id=org_id,
            agent_id=agent_id,
            content=content,
            memory_type=memory_type,
            session_id=session_id,
            metadata=metadata,
        )
    except Exception:
        logger.warning(
            "curator: write_scoped_memory raised for org %s (%s); skipping",
            org_id,
            memory_type,
            exc_info=True,
        )
        return False
    return bool(result.get("stored"))


async def curator_tick(
    db_session: Any,
    org_id: str,
    *,
    batch_size: int = 50,
    min_inputs: int = 3,
    agent_id: str = DEFAULT_CURATOR_AGENT_ID,
) -> dict:
    """Run one curation pass for a single org.

    Reads up to ``batch_size`` RAW ``scope='org'`` memories newer than the org's
    cursor; if at least ``min_inputs`` are found, synthesises a ``summary`` and
    discrete ``fact`` memories back into the org brain and advances the cursor to
    the batch's max ``created_at``. Below the threshold (including zero new rows)
    nothing is synthesised and only ``last_run_at`` is bumped, leaving the cursor
    in place — so repeated ticks with no new input are idempotent.

    Args:
        db_session: A synchronous SQLAlchemy session/connection exposing
            ``.execute()`` and ``.commit()``.
        org_id: The org (tenant) to curate. Always enforced on every query.
        batch_size: Max RAW memories to read in this tick.
        min_inputs: Minimum new RAW memories required before synthesis runs.
        agent_id: Principal authoring the synthesised memories.

    Returns:
        On the no-op path:
            ``{"synthesized": 0, "processed": <n>, "reason": "below_threshold"}``
        On the synthesis path:
            ``{"synthesized": <newly stored>, "processed": <n>,
            "cursor": <new last_processed_at ISO string>}``
    """
    cursor = _read_cursor(db_session, org_id)
    rows = _read_new_raw_rows(db_session, org_id, cursor, batch_size)

    if len(rows) < min_inputs:
        # Nothing (or too little) new: record the run without moving the cursor
        # so the next tick reconsiders the same rows once enough accumulate.
        _touch_last_run(db_session, org_id)
        return {
            "synthesized": 0,
            "processed": len(rows),
            "reason": "below_threshold",
        }

    combined = "\n".join(r.content for r in rows if r.content)
    synthesized = await _synthesize_and_store(
        db_session,
        org_id=org_id,
        agent_id=agent_id,
        combined=combined,
        input_count=len(rows),
    )

    # Rows are ordered ASC by created_at, so the last row carries the batch max.
    batch_max = rows[-1].created_at
    _advance_cursor(db_session, org_id, batch_max, len(rows))

    return {
        "synthesized": synthesized,
        "processed": len(rows),
        "cursor": batch_max.isoformat(),
    }


async def run_curator_once(
    session_factory: Callable[[], Any],
    *,
    batch_size: int = 50,
    min_inputs: int = 3,
) -> dict:
    """Run one curation pass across every org that has RAW org memories.

    This is the entry point a scheduler / cron invokes. It discovers candidate
    orgs (those with at least one ``scope='org'`` RAW memory) using a short-lived
    discovery session, then opens a fresh session per org for its
    :func:`curator_tick`. Every opened session is closed in a ``finally`` block.
    A failure in one org's tick is logged and does not abort the run for the
    remaining orgs.

    Args:
        session_factory: A zero-arg callable returning a new synchronous DB
            session each time (e.g. ``app.db.SessionLocal``).
        batch_size: Forwarded to :func:`curator_tick`.
        min_inputs: Forwarded to :func:`curator_tick`.

    Returns:
        ``{"orgs": <number of orgs ticked>, "synthesized_total": <sum>}``.
    """
    discovery = session_factory()
    try:
        org_rows = discovery.execute(
            text(
                f"""
                SELECT DISTINCT org_id
                FROM session_memories
                WHERE scope = 'org'
                  AND status = 'active'
                  AND memory_type IN ({_RAW_TYPES_SQL})
                """
            )
        ).fetchall()
    finally:
        discovery.close()

    org_ids = [row.org_id for row in org_rows]

    synthesized_total = 0
    for org_id in org_ids:
        session = session_factory()
        try:
            result = await curator_tick(
                session,
                org_id,
                batch_size=batch_size,
                min_inputs=min_inputs,
            )
            synthesized_total += int(result.get("synthesized", 0))
        except Exception:
            logger.warning(
                "curator: tick failed for org %s; continuing",
                org_id,
                exc_info=True,
            )
        finally:
            session.close()

    return {"orgs": len(org_ids), "synthesized_total": synthesized_total}
