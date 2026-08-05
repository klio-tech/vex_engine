"""Per-user Klio brain curator: periodic synthesis of RAW private memories.

The curator is a server-side background job that "tunes" a person's memory over
time. Periodically it reads one user's RAW private memories — the passive hook
captures and discrete writes (``observation`` / ``note`` / ``memory`` /
``plan`` / ``decision``), **including the ``recall_hidden`` ones**, which in
practice are nearly all of them — and synthesises higher-level memories back into the
**same user's private scope**: a single ``summary`` plus discrete ``fact`` rows.
It reuses the existing LLM helpers (:func:`shared.memory.generate_summary` and
the extended :func:`extract_graph` pass, which subsumed the old facts-only
extraction) and the brain write path
(:func:`shared.brain.write_scoped_memory`) verbatim. Since phase 2 of the
knowledge graph the same extraction pass also yields entities and edges,
stored through the :mod:`shared.graph_store` seam, every one ``INFERRED``.

Why per user, and why it never touches ``org``:
    Until phase 2 this job read ``scope='org'`` and wrote ``scope='org'``. Hook
    captures now land at ``scope='private'``. Left alone, the curator would
    have read one person's private captures and published distilled facts from
    them to the whole team — a privacy breach shipped as a feature, and
    precisely the failure decision 4 of the spec exists to prevent. So the read
    predicate, the write scope, and the cursor all became per user in the same
    change. There is no ``org`` arm left in this module: not on the read, not
    on the write, not on the org-discovery query.

Provenance:
    Everything this module produces is an LLM's deduction from other rows, so
    every write is stamped ``INFERRED``. That label is what lets a reader — and
    the dashboard — tell "Priya said this" from "a model concluded this from
    ten observations", and it survives promotion to the team unchanged.

Separation from verification:
    This module is deliberately independent of the verification worker
    (``services/async-worker``). The curator only ever reads
    ``scope='private'`` rows — never verification's ``scope='session'`` rows,
    which live in the same ``session_memories`` table — so the two paths never
    interfere.

No self-reprocessing:
    The curator reads only the RAW input types
    (``observation``/``note``/``memory``/``plan``/``decision``) and never its
    own outputs (``summary``/``fact``). Combined with the per-org keyset cursor
    (it only reads rows *past* the cursor), this makes a tick unable to loop on
    its own synthesised memories.

Hidden rows are fuel, not noise:
    The input predicate deliberately does NOT filter on ``recall_hidden``.
    That flag keeps raw hook captures out of *recall*; it says nothing about
    synthesis, and the two goals are compatible — the curator eats the noise,
    recall never sees it. Filtering on it was the reason the knowledge graph
    stayed empty: migration 026 backfilled ``recall_hidden=TRUE`` onto every
    passive capture, so ``recall_hidden = FALSE`` AND a RAW-only type list
    intersected to the empty set on real data. The separation of input from
    output now rests entirely on :data:`RAW_MEMORY_TYPES` and the cursor.

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
    Progress is tracked per ``(org_id, user_id)`` in ``brain_curator_state``
    (migration 021, PK widened by 031) as a
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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, NamedTuple, Optional

from sqlalchemy import text

from shared import graph_store
from shared.brain import write_scoped_memory
from shared.memory import (
    _NO_CREDENTIALS_RULE,
    call_llm,
    embed_text,
    generate_summary,
    supersede_memory,
)

logger = logging.getLogger("agentguard.mcp.curator")

# RAW input memory types the curator reads. Deliberately EXCLUDES the curator's
# own outputs ('summary'/'fact') so a tick cannot reprocess what it produced,
# and is read only under scope='private' so verification's scope='session' rows
# and the team's scope='org' brain are never touched. This type list is the
# ONLY thing separating input from output now that `recall_hidden` no longer
# narrows the batch: adding 'fact' or 'summary' here would close a feedback
# loop in which the curator re-digests its own distillate.
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

# Relations the extraction pass may draw. Deliberately NOT the full closed set
# from graph_store: `supersedes` is reconciliation's judgment (a compared pair,
# not a single-batch impression) and `derived_from` is bookkeeping the code
# records itself. An extractor that "notices" either is hallucinating history.
_EXTRACTABLE_RELATIONS: tuple[str, ...] = ("about", "relates_to")


def _entity_cap() -> int:
    """Max NEW entities one curator tick may create (god-entity guard part 3).

    Overflow is logged and dropped, never stored: a single runaway extraction
    must not be able to hairball the org's graph in one tick.
    """
    return int(os.environ.get("CURATOR_ENTITY_CAP", "12"))


@dataclass
class EntityBudget:
    """The per-TICK entity allowance, threaded through per-partition synthesis.

    Deliberately mutable (like the reconcile budget integer this module already
    threads): the whole point is that partitions within one tick draw down a
    single shared allowance rather than each getting a fresh one.
    """

    remaining: int


def _reconcile_budget() -> int:
    """Max LLM judge calls (``call_llm`` attempts) across all facts in one curator tick.

    Each unit of budget consumed by :func:`reconcile_fact` corresponds to one
    ``call_llm`` attempt (including attempts that fail and are skipped).  Because
    :func:`_synthesize_and_store` passes ``max_judgments=reconcile_budget`` into
    every :func:`reconcile_fact` call and decrements by ``judged`` (the number of
    attempts actually made), the total ``call_llm`` invocations for the tick is
    bounded by this value — not merely accounted after the fact.
    """
    return int(os.environ.get("CURATOR_RECONCILE_MAX", "50"))


async def extract_graph(combined: str) -> dict[str, Any]:
    """One extended LLM pass: facts PLUS the entities and edges they evidence.

    This replaces the old facts-only extraction call in the tick — same single
    ``call_llm`` round trip, extended prompt (spec decision 2: the curator is
    the graph builder precisely because it already reads every memory; no new
    LLM pass, no new cost centre).

    Output contract (always all three keys, empty lists on any failure)::

        {"facts": ["..."],
         "entities": [{"name": "auth service", "kind": "system"}],
         "edges": [{"src": 0, "relation": "about", "entity": "auth service"}]}

    Edge parsing accepts the spec's wire shape — ``{"src": <fact idx>,
    "<relation>": "<entity name>"}`` — and NORMALIZES it to the explicit form
    above. Only :data:`_EXTRACTABLE_RELATIONS` survive: an extractor claiming
    ``supersedes`` or ``derived_from`` is inventing history, and a
    non-integer ``src`` cannot be attached to anything. Malformed items are
    dropped one by one; they never take the batch down.
    """
    kinds = "|".join(sorted(graph_store.ENTITY_KINDS))
    prompt = (
        "Extract discrete, verifiable facts from the following notes, AND the "
        "named entities those facts concern. "
        + _NO_CREDENTIALS_RULE
        + "Entities must be specific named things (a service, a person, a "
        "project, a tool, a decision subject) — NEVER generic terms like "
        "'the code', 'the user', 'the system', weekdays, or bare programming "
        "language names. Prefer few precise entities over many vague ones."
        "\n\n" + combined + "\n\nRespond with JSON exactly of this shape:\n"
        '{"facts": ["<fact>", ...],\n'
        f' "entities": [{{"name": "<name>", "kind": "<{kinds}>"}}, ...],\n'
        ' "edges": [{"src": <fact index>, "about": "<entity name>"},\n'
        '           {"src": <fact index>, "relates_to": "<entity name>"}, ...]}'
    )

    empty: dict[str, Any] = {"facts": [], "entities": [], "edges": []}
    try:
        result = await call_llm(prompt)
    except Exception:
        logger.warning("curator: extract_graph LLM call raised", exc_info=True)
        return empty
    if not isinstance(result, dict):
        return empty

    facts_raw = result.get("facts", [])
    facts = (
        [f for f in facts_raw if isinstance(f, str)]
        if isinstance(facts_raw, list)
        else []
    )

    entities_raw = result.get("entities", [])
    entities = (
        [
            e
            for e in entities_raw
            if isinstance(e, dict) and isinstance(e.get("name"), str)
        ]
        if isinstance(entities_raw, list)
        else []
    )

    edges: list[dict[str, Any]] = []
    edges_raw = result.get("edges", [])
    if isinstance(edges_raw, list):
        for edge in edges_raw:
            if not isinstance(edge, dict):
                continue
            relation = next(
                (r for r in _EXTRACTABLE_RELATIONS if isinstance(edge.get(r), str)),
                None,
            )
            src = edge.get("src")
            if relation is None or not isinstance(src, int) or isinstance(src, bool):
                continue
            edges.append({"src": src, "relation": relation, "entity": edge[relation]})

    return {"facts": facts, "entities": entities, "edges": edges}


def _store_graph(
    db_session: Any,
    *,
    org_id: str,
    entities: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    fact_row_ids: dict[int, str],
    entity_budget: EntityBudget,
) -> int:
    """Persist one partition's extracted entities and edges via the GraphStore.

    Everything here is best-effort telemetry on top of the facts, and every
    write is ``INFERRED``: a refused entity (god-entity guard), an exhausted
    per-tick cap, or an edge whose fact deduped away (no new row → nothing to
    attach; drawing the edge to some OTHER row would be misattribution) is
    logged and skipped — a graph problem must never abort the tick.

    Returns the number of edges drawn.
    """
    entity_ids: dict[str, str] = {}
    for spec in entities:
        name = spec.get("name") or ""
        kind = spec.get("kind") or ""
        norm = graph_store.normalize_entity_name(name)
        if not norm or norm in entity_ids:
            continue
        if entity_budget.remaining <= 0:
            logger.warning(
                "curator: per-tick entity cap reached for org %s; dropping %r "
                "(and any further entities this tick)",
                org_id,
                norm,
            )
            break
        try:
            entity_ids[norm] = graph_store.add_entity(
                db_session, org_id=org_id, name=name, kind=kind
            )
            entity_budget.remaining -= 1
        except graph_store.EntityRefused as exc:
            logger.info("curator: entity refused for org %s: %s", org_id, exc)
        except Exception:
            logger.warning(
                "curator: add_entity failed for org %s (%r); skipping",
                org_id,
                norm,
                exc_info=True,
            )

    drawn = 0
    for edge in edges:
        src_row_id = fact_row_ids.get(edge.get("src"))
        entity_id = entity_ids.get(
            graph_store.normalize_entity_name(edge.get("entity") or "")
        )
        if src_row_id is None or entity_id is None:
            continue  # deduped fact, refused/capped entity, or bad index
        try:
            graph_store.add_edge(
                db_session,
                org_id=org_id,
                src_id=src_row_id,
                dst_kind="entity",
                dst_id=entity_id,
                relation=edge["relation"],
                provenance="INFERRED",
            )
            drawn += 1
        except Exception:
            logger.warning(
                "curator: add_edge failed for org %s; skipping",
                org_id,
                exc_info=True,
            )
    return drawn


async def reconcile_fact(
    db_session: Any,
    *,
    org_id: str,
    user_id: str,
    new_fact_id: Any,
    new_fact_content: str,
    embedding: list[float],
    max_judgments: int,
) -> tuple[int, int]:
    """Supersede older active facts that the new fact directly contradicts.

    Scoped to ONE user's private facts. Reconciliation supersedes rows, so a
    cross-user candidate query would let one person's session silently retire
    another person's memory — a write across the boundary rather than a read
    across it, and no less of a breach for it.

    Vector-searches that user's older active visible facts within
    RECONCILE_SIMILARITY_THRESHOLD of the new fact (top
    RECONCILE_CANDIDATES_PER_FACT), asks the LLM judge per candidate, and
    marks contradicted rows superseded_by the new fact. A judge failure
    skips that pair (never raises).

    The ``max_judgments`` parameter is a hard cap on the number of ``call_llm``
    attempts issued in this call: the candidate list is sliced to
    ``rows[:max_judgments]`` before the loop begins, so the caller can enforce a
    strict budget without relying solely on post-hoc accounting.

    Returns (superseded, judged): rows superseded and judge calls attempted.
    """
    rows = db_session.execute(
        text(
            """
            SELECT id, content
            FROM session_memories
            WHERE org_id = :org_id
              AND scope = 'private'
              AND user_id = :user_id
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
            "user_id": user_id,
            "new_id": str(new_fact_id),
            "embedding": str(embedding),
            "threshold": RECONCILE_SIMILARITY_THRESHOLD,
            "k": RECONCILE_CANDIDATES_PER_FACT,
        },
    ).fetchall()

    superseded = 0
    judged = 0
    for row in rows[:max_judgments]:
        judged += 1
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
            supersede_memory(db_session, row.id, new_fact_id, org_id=org_id)
            superseded += 1
            # Record the judgment it just made as a supersedes edge (spec:
            # "it already makes the judgment; today it only flips status").
            # Best-effort: the status flip is the contract, the edge is the
            # explanation, and a graph problem must never undo or abort a
            # reconciliation that already happened.
            try:
                graph_store.add_edge(
                    db_session,
                    org_id=org_id,
                    src_id=str(new_fact_id),
                    dst_kind="memory",
                    dst_id=str(row.id),
                    relation="supersedes",
                    provenance="INFERRED",
                )
            except Exception:
                logger.warning(
                    "reconcile: supersedes edge write failed for org %s; "
                    "the status flip stands",
                    org_id,
                    exc_info=True,
                )
    return superseded, judged


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


def _read_cursor(db_session: Any, org_id: str, user_id: str) -> CursorState:
    """Return this ``(org, user)``'s ``(last_processed_at, last_processed_id)``.

    A missing state row (first ever tick for the pair) yields a
    :class:`CursorState` with both halves ``None``, which the new-rows query
    treats as "no lower bound" (read from the beginning).

    Keyed by user as well as org because each user's private stream is
    paginated independently: a shared cursor would let one person's activity
    skip past another's unprocessed rows, permanently.
    """
    row = db_session.execute(
        text(
            """
            SELECT last_processed_at, last_processed_id
            FROM brain_curator_state
            WHERE org_id = :org_id AND user_id = :user_id
            """
        ),
        {"org_id": org_id, "user_id": user_id},
    ).fetchone()
    if row is None:
        return CursorState(None, None)
    return CursorState(row.last_processed_at, row.last_processed_id)


def _read_new_raw_rows(
    db_session: Any,
    org_id: str,
    user_id: str,
    cursor: CursorState,
    batch_size: int,
) -> list[Any]:
    """Read up to ``batch_size`` of ONE user's RAW private memories past ``cursor``.

    Scoped by ``org_id`` AND ``user_id`` and locked to ``scope='private'`` +
    ``status='active'``, and restricted to :data:`RAW_MEMORY_TYPES` so neither
    verification's ``scope='session'`` rows nor the curator's own
    ``summary``/``fact`` output are ever returned.

    Deliberately **not** filtered on ``recall_hidden``. That flag exists to keep
    raw hook noise out of *recall*; it was never a statement about synthesis.
    Filtering on it here starved the curator completely: migration 026 backfills
    ``recall_hidden=TRUE`` onto every passive hook capture, and those
    ``observation`` rows are the overwhelming majority of a user's RAW private
    stream — so a ``recall_hidden=FALSE`` predicate combined with a RAW-only
    type list matched nothing at all, and the knowledge graph could never
    populate. Hidden rows are fuel; hidden still means invisible to recall,
    because nothing here changes how they are read back.

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
              AND scope = 'private'
              AND user_id = :user_id
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
            "user_id": user_id,
            "cursor_ts": cursor.last_processed_at,
            "cursor_id": cursor.last_processed_id,
            "batch_size": batch_size,
        },
    ).fetchall()
    return list(rows)


def _touch_last_run(db_session: Any, org_id: str, user_id: str) -> None:
    """Bump only ``last_run_at`` for an ``(org, user)``, cursor untouched.

    Used on the no-op path (below threshold / no new rows). Race-safe upsert: a
    fresh pair inserts a state row with the cursor still ``NULL``; an existing
    pair only bumps ``last_run_at``. The conflict target is the widened PK
    ``(org_id, user_id)`` from migration 031 — conflicting on ``org_id`` alone
    no longer names a unique index and would raise.
    """
    db_session.execute(
        text(
            """
            INSERT INTO brain_curator_state (org_id, user_id, last_run_at)
            VALUES (:org_id, :user_id, now())
            ON CONFLICT (org_id, user_id)
              DO UPDATE SET last_run_at = now()
            """
        ),
        {"org_id": org_id, "user_id": user_id},
    )
    db_session.commit()


def _advance_cursor(
    db_session: Any,
    org_id: str,
    user_id: str,
    last_processed_at: datetime,
    last_processed_id: Any,
    processed_delta: int,
) -> None:
    """Advance the composite cursor and add to ``processed_count``.

    The cursor advances to the batch's max ``(created_at, id)`` — supplied by the
    caller as ``last_processed_at`` / ``last_processed_id`` (the last row of a
    batch ordered ``created_at ASC, id ASC``). Race-safe upsert against the
    ``brain_curator_state`` primary key (``(org_id, user_id)`` since migration
    031): a fresh pair inserts; an existing pair folds both cursor halves, the
    ``processed_count`` increment, and the ``last_run_at`` bump into one atomic
    statement (matching the upsert idiom in :mod:`shared.brain`).
    """
    db_session.execute(
        text(
            """
            INSERT INTO brain_curator_state (
                org_id, user_id, last_processed_at, last_processed_id,
                last_run_at, processed_count
            )
            VALUES (
                :org_id, :user_id, :last_processed_at, :last_processed_id,
                now(), :processed_delta
            )
            ON CONFLICT (org_id, user_id)
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
            "user_id": user_id,
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
    user_id: str,
    agent_id: str,
    combined: str,
    input_count: int,
    space_id: Optional[str] = None,
    project_id: Optional[str] = None,
    entity_budget: Optional[EntityBudget] = None,
) -> int:
    """Run the LLM synthesis over ``combined`` and store the results.

    Resilient to LLM failure: if :func:`generate_summary` yields ``None`` the
    summary write is skipped but fact extraction is still attempted, and any
    individual write failure is logged without aborting the rest. Returns the
    number of memories that were *newly stored* (dedup hits do NOT count).

    Since phase 2 of the knowledge graph the fact pass is the extended
    :func:`extract_graph` (same single LLM call): entities and edges ride
    along and are stored through the GraphStore seam AFTER the facts land, so
    every edge src is a real stored row. ``entity_budget`` is the shared
    per-tick allowance; a caller that omits it (direct test invocation) gets
    a fresh one at the configured cap.
    """
    budget = entity_budget if entity_budget is not None else EntityBudget(_entity_cap())
    # The session id carries user_id so the dedup window (which is keyed on
    # session_id for shared writes) cannot straddle two people.
    session_id = f"{agent_id}:{org_id}:{user_id}"
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
            user_id=user_id,
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

    # --- facts + graph (ONE extended LLM pass) ---------------------------
    try:
        graph = await extract_graph(combined)
    except Exception:
        logger.warning(
            "curator: extract_graph raised for org %s; no facts stored",
            org_id,
            exc_info=True,
        )
        graph = {"facts": [], "entities": [], "edges": []}
    facts = graph.get("facts", [])

    # Fact index -> newly stored row id. Dedup hits are deliberately ABSENT:
    # their edges are skipped rather than attached to somebody else's row.
    fact_row_ids: dict[int, str] = {}

    reconcile_budget = _reconcile_budget()
    for fact_idx, fact in enumerate(facts):
        if not fact:
            continue
        new_id = await _store_one(
            db_session,
            org_id=org_id,
            user_id=user_id,
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
            fact_row_ids[fact_idx] = new_id
            if reconcile_budget > 0:
                fact_embedding = await embed_text(fact)
                if fact_embedding is not None:
                    superseded, judged = await reconcile_fact(
                        db_session,
                        org_id=org_id,
                        user_id=user_id,
                        new_fact_id=new_id,
                        new_fact_content=fact,
                        embedding=fact_embedding,
                        max_judgments=reconcile_budget,
                    )
                    reconcile_budget -= max(judged, 1)

    # --- the graph itself -------------------------------------------------
    try:
        _store_graph(
            db_session,
            org_id=org_id,
            entities=graph.get("entities", []),
            edges=graph.get("edges", []),
            fact_row_ids=fact_row_ids,
            entity_budget=budget,
        )
    except Exception:
        logger.warning(
            "curator: graph storage failed for org %s; facts are unaffected",
            org_id,
            exc_info=True,
        )

    return stored


async def _store_one(
    db_session: Any,
    *,
    org_id: str,
    user_id: str,
    agent_id: str,
    content: str,
    memory_type: str,
    session_id: str,
    metadata: dict,
    space_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Optional[str]:
    """Write a single synthesised memory; return the new row id when newly stored.

    Always ``scope='private'`` to ``user_id`` and always
    ``provenance='INFERRED'``. Both are hardcoded rather than parameterised:
    there is no caller that should be able to make the curator publish, and no
    curator output that is anything other than a deduction. A parameter here
    would be a way for a future edit to make either happen by accident.

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
            scope="private",
            user_id=user_id,
            provenance="INFERRED",
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
    user_id: str,
    *,
    batch_size: int = 50,
    min_inputs: int = 3,
    agent_id: str = DEFAULT_CURATOR_AGENT_ID,
) -> dict:
    """Run one curation pass for a single ``(org, user)`` pair.

    Reads up to ``batch_size`` of that user's RAW ``scope='private'`` memories
    past their keyset cursor; if at least ``min_inputs`` are found, synthesises
    a ``summary`` and discrete ``fact`` memories back into **the same user's
    private scope** and advances the cursor to the batch's composite max
    ``(created_at, id)``. Below the threshold (including zero new rows) nothing
    is synthesised and only ``last_run_at`` is bumped, leaving both cursor
    halves in place — so repeated ticks with no new input are idempotent.

    Args:
        db_session: A synchronous SQLAlchemy session/connection exposing
            ``.execute()`` and ``.commit()``.
        org_id: The org (tenant) to curate. Always enforced on every query.
        user_id: The person to curate. Also enforced on every query — this is
            the predicate that keeps one person's distillation from being fed
            by another person's memories.
        batch_size: Max RAW memories to read in this tick.
        min_inputs: Minimum new RAW memories required before synthesis runs.
        agent_id: Principal authoring the synthesised memories.

    Returns:
        On the no-op path:
            ``{"synthesized": 0, "processed": <n>, "reason": "below_threshold"}``
        On the synthesis path:
            ``{"synthesized": <newly stored>, "processed": <n>,
            "cursor": <new last_processed_at ISO string>}``

    Raises:
        ValueError: If ``user_id`` is empty. There is no "curate the org"
            mode any more, and defaulting to one would publish private
            material.
    """
    if not user_id:
        raise ValueError("curator_tick requires a user_id; there is no org mode")

    cursor = _read_cursor(db_session, org_id, user_id)
    rows = _read_new_raw_rows(db_session, org_id, user_id, cursor, batch_size)

    if len(rows) < min_inputs:
        # Nothing (or too little) new: record the run without moving the cursor
        # so the next tick reconsiders the same rows once enough accumulate.
        _touch_last_run(db_session, org_id, user_id)
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
    # ONE entity allowance for the whole tick, drawn down across partitions —
    # the cap is per tick, not per partition.
    entity_budget = EntityBudget(_entity_cap())
    for (space_id, project_id), part_rows in partitions.items():
        # Per-partition threshold: a project needs real signal before it gets a
        # distilled fact. Sub-threshold rows stay recallable as raw observations.
        if len(part_rows) < min_inputs:
            continue
        combined = "\n".join(r.content for r in part_rows if r.content)
        synthesized += await _synthesize_and_store(
            db_session,
            org_id=org_id,
            user_id=user_id,
            agent_id=agent_id,
            combined=combined,
            input_count=len(part_rows),
            space_id=space_id,
            project_id=project_id,
            entity_budget=entity_budget,
        )

    # Rows are ordered (created_at ASC, id ASC), so the LAST row carries the
    # batch's composite max — the correct new keyset cursor.
    batch_max_at = rows[-1].created_at
    batch_max_id = rows[-1].id
    _advance_cursor(db_session, org_id, user_id, batch_max_at, batch_max_id, len(rows))

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
    """Run one curation pass for every ``(org, user)`` with RAW private memories.

    This is the entry point a scheduler / cron invokes. It discovers candidate
    ``(org_id, user_id)`` pairs (those with at least one ``scope='private'`` RAW
    memory) using a short-lived discovery session, then opens a fresh session
    per pair for its :func:`curator_tick`. Every opened session is closed in a
    ``finally`` block. A failure in one user's tick is logged and does not abort
    the run for the remaining users.

    The discovery query filters ``user_id IS NOT NULL``. It cannot fire for the
    37,605 pre-existing org rows or for any unattributable write, which is the
    intended outcome: the curator has nothing to say about memories with no
    owner, and it must never fall back to curating them as a group.

    Args:
        session_factory: A zero-arg callable returning a new synchronous DB
            session each time (e.g. ``app.db.SessionLocal``).
        batch_size: Forwarded to :func:`curator_tick`.
        min_inputs: Forwarded to :func:`curator_tick`.

    Returns:
        ``{"orgs": <distinct orgs touched>, "users": <pairs ticked>,
        "synthesized_total": <sum>}``.
    """
    discovery = session_factory()
    try:
        pair_rows = discovery.execute(
            text(
                f"""
                SELECT DISTINCT org_id, user_id
                FROM session_memories
                WHERE scope = 'private'
                  AND user_id IS NOT NULL
                  AND status = 'active'
                  AND memory_type IN ({_RAW_TYPES_SQL})
                """
            )
        ).fetchall()
    finally:
        discovery.close()

    pairs = [(row.org_id, row.user_id) for row in pair_rows]

    synthesized_total = 0
    for org_id, user_id in pairs:
        session = session_factory()
        try:
            result = await curator_tick(
                session,
                org_id,
                user_id,
                batch_size=batch_size,
                min_inputs=min_inputs,
            )
            synthesized_total += int(result.get("synthesized", 0))
        except Exception:
            logger.warning(
                "curator: tick failed for org %s user %s; continuing",
                org_id,
                user_id,
                exc_info=True,
            )
        finally:
            session.close()

    return {
        "orgs": len({org_id for org_id, _ in pairs}),
        "users": len(pairs),
        "synthesized_total": synthesized_total,
    }
