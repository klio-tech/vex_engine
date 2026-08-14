"""Grade a stored session and attribute served memories — honestly.

Design corrections (binding):
- ONE verify() per session; verdict lives ONCE in session_outcomes.
- recall_outcomes rows carry attribution (used/usage_score), never a verdict.
- Final output = last assistant message from the STRUCTURED trace.
- verify() receives session_memories so memory_consistency actually runs.
Missing evidence grades as verdict='unknown' — coverage gaps must be visible.

Every query here is org-predicated: recalls are fetched WHERE org_id=:org_id
AND session_id=:session_id, and the served-memory lookup is WHERE
org_id=:org_id AND id = ANY(:ids), so grading can never cross an org
boundary even if a session_id were guessed or leaked.

The served-memory ``ids`` bind is cast explicitly — ``CAST(:ids AS uuid[])``
— because a plain bind of a Python list of uuid strings adapts to an
untyped Postgres array literal (effectively ``text[]``), and Postgres
refuses to compare that against ``session_memories.id`` (``uuid``) via
``= ANY(...)`` without the cast (DatatypeMismatch). This mirrors the fix
already applied in ``app.recall_log.log_recall_event`` (mcp-server) for the
same array-typing reason; scalar uuid-string binds elsewhere in this module
(``id``, ``recall_event_id``, ``memory_id``, ...) do not need it — Postgres
accepts a text literal directly as ``uuid`` input outside of array
comparison.

``session_memories`` rows are hydrated into real :class:`shared.memory.MemoryRecord`
objects via :func:`shared.memory._row_to_record` — the SAME hydrator every
other reader of this table uses (``shared.memory.recall_memories`` et al.) —
rather than passed to ``verify()`` as bare ``{id, content}`` dicts.
``engine.memory_consistency._format_memories`` does ATTRIBUTE access
(``mem.memory_type``, ``mem.sequence_number``, ``mem.content``) on whatever
``session_memories`` gives it; a dict there raises ``AttributeError`` inside
``verify()``'s ``asyncio.gather`` for every real (non-mocked) call, which
this module's own ``except Exception`` around ``verify()`` would silently
swallow — degrading every graded session to ``verdict='unknown'`` and
defeating the point of passing ``session_memories`` at all. The SELECT below
matches the canonical 16-column order ``_row_to_record`` expects (see its
docstring and ``shared.memory.recall_memories``'s semantic/summary queries).

Re-grading a session (duplicate ``sessions.stored`` events are DESIGNED —
storage-worker emits one even when its own upsert lost the richer-wins race,
see ``app.trace_consumer.handle_captured``) must be a no-op, not a poison
message. ``session_outcomes`` has a real ``UNIQUE(org_id, session_id,
grader_version)`` constraint (migration 042); the INSERT below adds
``RETURNING id`` to ``ON CONFLICT ... DO NOTHING`` so a second grade for the
same session+grader_version can DETECT that no row was written and skip
attribution entirely — inserting ``recall_outcomes`` rows against a
freshly-minted ``session_outcome_id`` that was never persisted would raise
``ForeignKeyViolation``, and because that happens after the consumer's
``xack`` window, the message would never ACK and get redelivered forever.
"""

import logging
import math
import uuid
from typing import Any, Optional

from sqlalchemy import text

from engine.pipeline import verify
from shared.memory import _row_to_record, embed_text
from shared.traces import fetch_trace

logger = logging.getLogger("agentguard.outcome-grader")

GRADER_VERSION = "v1"
USED_THRESHOLD = 0.5

#: Column order MUST match ``shared.memory._row_to_record``'s expected
#: positional layout (its docstring's "first 16 columns, required").
_MEMORY_COLUMNS = (
    "id, session_id, org_id, agent_id, execution_id, memory_type, content, "
    "embedding, confidence, sequence_number, scope, status, superseded_by, "
    "metadata, created_at, updated_at"
)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _final_output(messages: list[dict[str, Any]]) -> Optional[str]:
    """Last ``role == "assistant"`` message with content, from the
    STRUCTURED trace messages — never string-prefix hunting over raw text.
    """
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            return str(m["content"])
    return None


async def grade_stored_session(
    db: Any,
    s3: Any,
    *,
    org_id: str,
    session_id: str,
    grader_version: str = GRADER_VERSION,
) -> dict[str, Any]:
    """Grade one stored session: ONE verify() call, ONE session_outcomes
    row, and one recall_outcomes attribution row per (recall, served
    memory) pair.

    Returns ``{"session_outcome_id": str | None, "attributions": int}``.
    ``session_outcome_id`` is ``None`` in two distinct cases the caller
    cannot tell apart from the return value alone (neither needs to raise —
    both are legitimate no-ops for the calling consumer, which should ACK
    either way):

    1. Nothing to grade: no recall events with served memory_ids for this
       session.
    2. Already graded: a session_outcomes row for
       ``(org_id, session_id, grader_version)`` already exists (duplicate
       ``sessions.stored`` delivery) — attribution is skipped entirely
       rather than risk a foreign key pointing at an id that was never
       persisted.

    Best-effort: a ``verify()`` or ``embed_text()`` failure degrades the
    verdict/scores to unknown/null rather than raising into the consumer
    loop that calls this function.
    """
    recalls = (
        db.execute(
            text(
                """
            SELECT id, memory_ids, session_id FROM brain_recall_events
            WHERE org_id = :org_id AND session_id = :session_id
              AND memory_ids IS NOT NULL
            """
            ),
            {"org_id": org_id, "session_id": session_id},
        )
        .mappings()
        .all()
    )
    if not recalls:
        return {"session_outcome_id": None, "attributions": 0}

    served_ids = sorted({str(mid) for r in recalls for mid in (r["memory_ids"] or [])})
    memory_rows = db.execute(
        text(
            f"""
            SELECT {_MEMORY_COLUMNS} FROM session_memories
            WHERE org_id = :org_id AND id = ANY(CAST(:ids AS uuid[]))
            """
        ),
        {"org_id": org_id, "ids": served_ids},
    ).fetchall()
    memories = [_row_to_record(row) for row in memory_rows]

    trace = fetch_trace(db, s3, org_id=org_id, session_id=session_id)
    output = _final_output(trace["messages"]) if trace else None

    verdict_str, confidence, mem_consistency = "unknown", None, None
    scores: dict[str, Optional[float]] = {str(m.id): None for m in memories}
    if output:
        try:
            result = await verify(
                output,
                session_memories=memories,
            )
            verdict_str = result.action
            confidence = result.confidence
            mc = getattr(result, "checks", {}) or {}
            if "memory_consistency" in mc:
                mem_consistency = getattr(mc["memory_consistency"], "score", None)
        except Exception:
            logger.warning(
                "grader: verify failed org=%s session=%s",
                org_id,
                session_id,
                exc_info=True,
            )
        try:
            out_emb = await embed_text(output)
            if out_emb:
                for m in memories:
                    m_emb = await embed_text(str(m.content))
                    if m_emb:
                        scores[str(m.id)] = _cosine(out_emb, m_emb)
        except Exception:
            logger.warning(
                "grader: attribution embedding failed org=%s session=%s",
                org_id,
                session_id,
                exc_info=True,
            )

    session_outcome_id = str(uuid.uuid4())
    inserted = db.execute(
        text(
            """
            INSERT INTO session_outcomes
              (id, org_id, session_id, verdict, confidence, memory_consistency,
               grader_version)
            VALUES (:id, :org_id, :session_id, :verdict, :confidence,
                    :memory_consistency, :grader_version)
            ON CONFLICT (org_id, session_id, grader_version) DO NOTHING
            RETURNING id
            """
        ),
        {
            "id": session_outcome_id,
            "org_id": org_id,
            "session_id": session_id,
            "verdict": verdict_str,
            "confidence": confidence,
            "memory_consistency": mem_consistency,
            "grader_version": grader_version,
        },
    ).first()

    if inserted is None:
        # ON CONFLICT DO NOTHING matched zero rows: a session_outcomes row
        # for this (org_id, session_id, grader_version) already exists —
        # this is a re-delivery of sessions.stored (see module docstring),
        # not a new session to grade. session_outcome_id above was never
        # written, so it must NOT be used as a foreign key for
        # recall_outcomes inserts (that would raise ForeignKeyViolation and
        # leave the message permanently unacked/redelivered). Skip
        # attribution entirely and let the caller ACK the message.
        logger.info(
            "grader: org=%s session=%s already graded for grader_version=%s; "
            "skipping duplicate attribution",
            org_id,
            session_id,
            grader_version,
        )
        db.commit()
        return {"session_outcome_id": None, "attributions": 0}

    attributions = 0
    for recall in recalls:
        for mid in recall["memory_ids"] or []:
            score = scores.get(str(mid))
            db.execute(
                text(
                    """
                    INSERT INTO recall_outcomes
                      (id, org_id, recall_event_id, memory_id,
                       session_outcome_id, used, usage_score)
                    VALUES (:id, :org_id, :recall_event_id, :memory_id,
                            :session_outcome_id, :used, :usage_score)
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "org_id": org_id,
                    "recall_event_id": str(recall["id"]),
                    "memory_id": str(mid),
                    "session_outcome_id": session_outcome_id,
                    "used": bool(score is not None and score >= USED_THRESHOLD),
                    "usage_score": score,
                },
            )
            attributions += 1
    db.commit()
    return {"session_outcome_id": session_outcome_id, "attributions": attributions}
