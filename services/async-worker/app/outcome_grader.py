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
"""

import logging
import math
import uuid
from typing import Any, Optional

from sqlalchemy import text

from engine.pipeline import verify
from shared.memory import embed_text
from shared.traces import fetch_trace

logger = logging.getLogger("agentguard.outcome-grader")

GRADER_VERSION = "v1"
USED_THRESHOLD = 0.5


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
    db: Any, s3: Any, *, org_id: str, session_id: str,
    grader_version: str = GRADER_VERSION,
) -> dict[str, Any]:
    """Grade one stored session: ONE verify() call, ONE session_outcomes
    row, and one recall_outcomes attribution row per (recall, served
    memory) pair.

    Returns ``{"session_outcome_id": str | None, "attributions": int}``.
    ``session_outcome_id`` is ``None`` only when there is nothing to grade
    (no recall events with served memory_ids for this session) — a session
    with recall evidence always gets a session_outcomes row, even when the
    trace or output is missing (verdict='unknown').

    Best-effort: a ``verify()`` or ``embed_text()`` failure degrades the
    verdict/scores to unknown/null rather than raising into the consumer
    loop that calls this function.
    """
    recalls = db.execute(
        text(
            """
            SELECT id, memory_ids, session_id FROM brain_recall_events
            WHERE org_id = :org_id AND session_id = :session_id
              AND memory_ids IS NOT NULL
            """
        ),
        {"org_id": org_id, "session_id": session_id},
    ).mappings().all()
    if not recalls:
        return {"session_outcome_id": None, "attributions": 0}

    served_ids = sorted({str(mid) for r in recalls for mid in (r["memory_ids"] or [])})
    memories = db.execute(
        text(
            """
            SELECT id, content FROM session_memories
            WHERE org_id = :org_id AND id = ANY(CAST(:ids AS uuid[]))
            """
        ),
        {"org_id": org_id, "ids": served_ids},
    ).mappings().all()

    trace = fetch_trace(db, s3, org_id=org_id, session_id=session_id)
    output = _final_output(trace["messages"]) if trace else None

    verdict_str, confidence, mem_consistency = "unknown", None, None
    scores: dict[str, Optional[float]] = {str(m["id"]): None for m in memories}
    if output:
        try:
            result = await verify(
                output,
                session_memories=[dict(m) for m in memories],
            )
            verdict_str = result.action
            confidence = result.confidence
            mc = getattr(result, "checks", {}) or {}
            if "memory_consistency" in mc:
                mem_consistency = getattr(mc["memory_consistency"], "score", None)
        except Exception:
            logger.warning(
                "grader: verify failed org=%s session=%s", org_id, session_id,
                exc_info=True,
            )
        try:
            out_emb = await embed_text(output)
            if out_emb:
                for m in memories:
                    m_emb = await embed_text(str(m["content"]))
                    if m_emb:
                        scores[str(m["id"])] = _cosine(out_emb, m_emb)
        except Exception:
            logger.warning(
                "grader: attribution embedding failed org=%s session=%s",
                org_id, session_id, exc_info=True,
            )

    session_outcome_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO session_outcomes
              (id, org_id, session_id, verdict, confidence, memory_consistency,
               grader_version)
            VALUES (:id, :org_id, :session_id, :verdict, :confidence,
                    :memory_consistency, :grader_version)
            ON CONFLICT (org_id, session_id, grader_version) DO NOTHING
            """
        ),
        {"id": session_outcome_id, "org_id": org_id, "session_id": session_id,
         "verdict": verdict_str, "confidence": confidence,
         "memory_consistency": mem_consistency, "grader_version": grader_version},
    )

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
                {"id": str(uuid.uuid4()), "org_id": org_id,
                 "recall_event_id": str(recall["id"]), "memory_id": str(mid),
                 "session_outcome_id": session_outcome_id,
                 "used": bool(score is not None and score >= USED_THRESHOLD),
                 "usage_score": score},
            )
            attributions += 1
    db.commit()
    return {"session_outcome_id": session_outcome_id, "attributions": attributions}
