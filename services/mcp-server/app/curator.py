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
    own outputs (``summary``/``fact``). Combined with the per-org keyset cursor
    (it only reads rows *past* the cursor), this makes a tick unable to loop on
    its own synthesised memories.

Reconciliation:
    After storing each new ``fact`` the curator runs a bounded reconciliation
    pass (:func:`reconcile_fact`): it vector-searches similar older active
    facts and asks an LLM judge whether the new fact directly contradicts each
    candidate. Contradicted older facts are marked superseded by the new one
    and excluded from subsequent recall. The pass is deliberately conservative
    — a stale fact surviving is acceptable; a live fact hidden by a false
    positive is not. The number of LLM judgments per tick is budget-capped via
    ``CURATOR_RECONCILE_MAX`` (default 50).

Cursor / idempotency:
    Progress is tracked per org in ``brain_curator_state`` (migration 021) as a
    **keyset** cursor over the composite ``(last_processed_at,
    last_processed_id)``. A tick advances both halves to the ``(created_at, id)``
    of the last row of the batch it processed (the batch is ordered
    ``created_at ASC, id ASC``). The composite tiebreaker is what prevents
    silently skipping a row that shares an identical ``created_at`` with the
    previous batch's last row at a ``LIMIT`` boundary — a plain
    ``created_at > cursor`` predicate would drop such a twin. A tick that finds
    nothing new synthesises nothing and leaves *both* cursor halves untouched
    (only ``last_run_at`` is bumped), so re-runs are idempotent.

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
import os
from datetime import datetime, timezone
from typing import Any, Callable, NamedTuple, Optional

from sqlalchemy import text

from shared.brain import write_scoped_memory
from shared.memory import call_llm, embed_text, extract_facts, generate_summary, supersede_memory

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

# Reconciliation: a new fact supersedes an older one only on DIRECT
# contradiction. Conservative by design — a stale fact surviving is
# acceptable; a live fact hidden is not.
RECONCILE_SIMILARITY_THRESHOLD = 0.85
RECONCILE_CANDIDATES_PER_FACT = 5


def _reconcile_budget() -> int:
    """Max LLM contradiction judgments per curator tick (cost bound)."""
    return int(os.environ.get("CURATOR_RECONCILE_MAX", "50"))


async def reconcile_fact(
    db_session: Any,
    *,
    org_id: str,
    new_fact_id: Any,
    new_fact_content: str,
    embedding: list[float],
) -> int:
    """Supersede older active facts that the new fact directly contradicts.

    Vector-searches the org's older active visible facts within
    RECONCILE_SIMILARITY_THRESHOLD of the new fact (top
    RECONCILE_CANDIDATES_PER_FACT), asks the LLM judge per candidate, and
    marks contradicted rows superseded_by the new fact. A judge failure
    skips that pair (never raises). Returns the number superseded.
    """
    rows = db_session.execute(
        text(
            """
            SELECT id, content
            FROM session_memories
            WHERE org_id = :org_id
              AND scope = 'org'
              AND status = 'active'
              AND memory_type = 'fact'
              AND id != :new_id
              AND 1 - (embedding <=> :embedding) >= :threshold
            ORDER BY embedding <=> :embedding
            LIMIT :k
            """
        ),
        {
            "org_id": org_id,
            "new_id": str(new_fact_id),
            "embedding": str(embedding),
            "threshold": RECONCILE_SIMILARITY_THRESHOLD,
            "k": RECONCILE_CANDIDATES_PER_FACT,
        },
    ).fetchall()

    superseded = 0
    for row in rows:
        prompt = (
            "Two facts from a team knowledge base:\n"
            f'OLDER: "{row.content}"\n'
            f'NEWER: "{new_fact_content}"\n\n'
            "Does the NEWER fact DIRECTLY contradict the OLDER one — i.e. "
            "they cannot both be true now? Merely related, overlapping, or "
            "complementary facts do NOT contradict. Respond with JSON: "
            '{"contradicts": true|false}'
        )
        try:
            verdict = await call_llm(prompt)
        except Exception:
            logger.warning("reconcile: judge raised; skipping pair", exc_info=True)
            continue
        if verdict and verdict.get("contradicts") is True:
            supersede_memory(db_session, row.id, new_fact_id)
            superseded += 1
    return superseded


class CursorState(NamedTuple):
    """The org's composite keyset cursor over ``(created_at, id)``.

    Both halves are ``None`` together on the first-ever tick for an org (no
    state row yet), which the new-rows query treats as "no lower bound" (read
    from the beginning). When set, ``last_processed_id`` is the UUID of the last
    ``session_memories`` row already processed at ``last_processed_at`` — the
    tiebreaker that makes pagination total and stable so same-``created_at`` rows
    are never skipped at a batch boundary.
    """

    last_processed_at: Optional[datetime]
    last_processed_id: Optional[Any]


def _now() -> datetime:
    """UTC now, isolated for clarity/testability."""
    return datetime.now(timezone.utc)


def _read_cursor(db_session: Any, org_id: str) -> CursorState:
    """Return this org's composite ``(last_processed_at, last_processed_id)``.

    A missing state row (first ever tick for the org) yields a
    :class:`CursorState` with both halves ``None``, which the new-rows query
    treats as "no lower bound" (read from the beginning).
    """
    row = db_session.execute(
        text(
            """
            SELECT last_processed_at, last_processed_id
            FROM brain_curator_state
            WHERE org_id = :org_id
            """
        ),
        {"org_id": org_id},
    ).fetchone()
    if row is None:
        return CursorState(None, None)
    return CursorState(row.last_processed_at, row.last_processed_id)


def _read_new_raw_rows(
    db_session: Any,
    org_id: str,
    cursor: CursorState,
    batch_size: int,
) -> list[Any]:
    """Read up to ``batch_size`` RAW org memories past the keyset ``cursor``.

    Tenant-scoped by ``org_id`` and locked to ``scope='org'`` + RAW types +
    ``status='active'`` so neither verification's session rows nor the curator's
    own ``summary``/``fact`` output are ever returned.

    Pagination is **keyset** over the composite ``(created_at, id)``: a row is
    new if its ``created_at`` is strictly past the cursor's, OR it shares the
    cursor's ``created_at`` but has a strictly greater ``id``. Combined with the
    matching ``ORDER BY created_at ASC, id ASC``, this gives a total, stable
    order so a row sharing an identical ``created_at`` with the previous batch's
    last row is never skipped at the ``LIMIT`` boundary. When there is no cursor
    yet (both halves ``None``) the ``:cursor_ts IS NULL`` arm matches everything.
    """
    rows = db_session.execute(
        text(
            f"""
            SELECT id, content, created_at, space_id, project_id
            FROM session_memories
            WHERE org_id = :org_id
              AND scope = 'org'
              AND status = 'active'
              AND memory_type IN ({_RAW_TYPES_SQL})
              AND (
                :cursor_ts IS NULL
                OR created_at > :cursor_ts
                OR (created_at = :cursor_ts AND id > :cursor_id)
              )
            ORDER BY created_at ASC, id ASC
            LIMIT :batch_size
            """
        ),
        {
            "org_id": org_id,
            "cursor_ts": cursor.last_processed_at,
            "cursor_id": cursor.last_processed_id,
            "batch_size": batch_size,
        },
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
    last_processed_id: Any,
    processed_delta: int,
) -> None:
    """Advance the composite cursor and add to ``processed_count``.

    The cursor advances to the batch's max ``(created_at, id)`` — supplied by the
    caller as ``last_processed_at`` / ``last_processed_id`` (the last row of a
    batch ordered ``created_at ASC, id ASC``). Race-safe upsert against the
    ``brain_curator_state`` primary key (``org_id``): a fresh org inserts; an
    existing org folds both cursor halves, the ``processed_count`` increment, and
    the ``last_run_at`` bump into one atomic statement (matching the upsert idiom
    in :mod:`shared.brain`).
    """
    db_session.execute(
        text(
            """
            INSERT INTO brain_curator_state (
                org_id, last_processed_at, last_processed_id,
                last_run_at, processed_count
            )
            VALUES (
                :org_id, :last_processed_at, :last_processed_id,
                now(), :processed_delta
            )
            ON CONFLICT (org_id)
              DO UPDATE SET
                last_processed_at = EXCLUDED.last_processed_at,
                last_processed_id = EXCLUDED.last_processed_id,
                last_run_at = now(),
                processed_count =
                    brain_curator_state.processed_count + :processed_delta
            """
        ),
        {
            "org_id": org_id,
            "last_processed_at": last_processed_at,
            "last_processed_id": last_processed_id,
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
    space_id: Optional[str] = None,
    project_id: Optional[str] = None,
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
        new_id = await _store_one(
            db_session,
            org_id=org_id,
            agent_id=agent_id,
            content=summary,
            memory_type="summary",
            session_id=session_id,
            metadata=base_meta,
            space_id=space_id,
            project_id=project_id,
        )
        if new_id:
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

    reconcile_budget = _reconcile_budget()
    for fact in facts:
        if not fact:
            continue
        new_id = await _store_one(
            db_session,
            org_id=org_id,
            agent_id=agent_id,
            content=fact,
            memory_type="fact",
            session_id=session_id,
            metadata=base_meta,
            space_id=space_id,
            project_id=project_id,
        )
        if new_id:
            stored += 1
            if reconcile_budget > 0:
                fact_embedding = await embed_text(fact)
                if fact_embedding is not None:
                    done = await reconcile_fact(
                        db_session,
                        org_id=org_id,
                        new_fact_id=new_id,
                        new_fact_content=fact,
                        embedding=fact_embedding,
                    )
                    reconcile_budget -= max(done, 1)

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
    space_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Optional[str]:
    """Write a single synthesised memory; return the new row id when newly stored.

    Returns the id string of the newly written row when ``stored=True``.
    Returns ``None`` on a dedup hit (``stored=False``), an embedding failure,
    or any raised exception. Exceptions are logged and swallowed so one bad
    write never aborts the tick. Callers treat a non-None return as a new
    stored row (truthiness preserved for counting).
    """
    try:
        result = await write_scoped_memory(
            db_session,
            org_id=org_id,
            agent_id=agent_id,
            content=content,
            memory_type=memory_type,
            session_id=session_id,
            space_id=space_id,
            project_id=project_id,
            metadata=metadata,
        )
    except Exception:
        logger.warning(
            "curator: write_scoped_memory raised for org %s (%s); skipping",
            org_id,
            memory_type,
            exc_info=True,
        )
        return None
    return result.get("id") if result.get("stored") else None


async def curator_tick(
    db_session: Any,
    org_id: str,
    *,
    batch_size: int = 50,
    min_inputs: int = 3,
    agent_id: str = DEFAULT_CURATOR_AGENT_ID,
) -> dict:
    """Run one curation pass for a single org.

    Reads up to ``batch_size`` RAW ``scope='org'`` memories past the org's keyset
    cursor; if at least ``min_inputs`` are found, synthesises a ``summary`` and
    discrete ``fact`` memories back into the org brain and advances the cursor to
    the batch's composite max ``(created_at, id)``. Below the threshold (including
    zero new rows) nothing is synthesised and only ``last_run_at`` is bumped,
    leaving both cursor halves in place — so repeated ticks with no new input are
    idempotent.

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

    # Partition the batch by (space_id, project_id), preserving order so each
    # partition's combined text and the cursor advance stay deterministic.
    # NULL space/project => a genuinely org-wide partition (surfaces everywhere).
    partitions: dict[tuple, list] = {}
    for r in rows:
        partitions.setdefault((r.space_id, r.project_id), []).append(r)

    synthesized = 0
    for (space_id, project_id), part_rows in partitions.items():
        # Per-partition threshold: a project needs real signal before it gets a
        # distilled fact. Sub-threshold rows stay recallable as raw observations.
        if len(part_rows) < min_inputs:
            continue
        combined = "\n".join(r.content for r in part_rows if r.content)
        synthesized += await _synthesize_and_store(
            db_session,
            org_id=org_id,
            agent_id=agent_id,
            combined=combined,
            input_count=len(part_rows),
            space_id=space_id,
            project_id=project_id,
        )

    # Rows are ordered (created_at ASC, id ASC), so the LAST row carries the
    # batch's composite max — the correct new keyset cursor.
    batch_max_at = rows[-1].created_at
    batch_max_id = rows[-1].id
    _advance_cursor(db_session, org_id, batch_max_at, batch_max_id, len(rows))

    return {
        "synthesized": synthesized,
        "processed": len(rows),
        "cursor": batch_max_at.isoformat(),
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
