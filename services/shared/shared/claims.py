"""Work claims: advisory, semantic, TTL'd awareness (Build 4).

NOT locks. A claim records intent; collisions are detected semantically
("auth refactor" ~ "fix login token expiry") because exact-key locking is
useless against natural language. TTL + released_at handle crashed agents
with no cleanup job. Enforcement is the client's business: cloud agents get
awareness (recall responses carry ``active_claims``), local ``PreToolUse``
hooks may block — this module has no opinion on that, it only records and
surfaces.

No network call is ever made while holding an open transaction:
:func:`create_claim` embeds the description BEFORE the INSERT, exactly the
ordering :func:`shared.memory.supersede_with_revision` documents (phase 3 of
its docstring) as the fix for a slow embed pinning a checked-out connection
against a transaction-mode pooler (Neon PgBouncer in prod). There is no
read-then-write here to roll back first — the content is known upfront, so
there is nothing to release before the embed.
"""

import logging
import uuid
from typing import Any, Optional

from sqlalchemy import text

from shared.memory import embed_text

logger = logging.getLogger("klio.claims")

#: Cosine similarity at/above which a live claim is surfaced as a collision.
COLLISION_THRESHOLD = 0.85

#: TTL clamp, minutes. A caller-supplied ttl_minutes outside this range is
#: silently clamped rather than rejected -- an advisory claim with a
#: pathological TTL (0, negative, or effectively-forever) is a nuisance, not
#: a security concern, so failing the whole claim over it would be the wrong
#: trade.
_MIN_TTL_MINUTES = 1
_MAX_TTL_MINUTES = 240


def _clamp_ttl(ttl_minutes: int) -> int:
    return min(max(int(ttl_minutes), _MIN_TTL_MINUTES), _MAX_TTL_MINUTES)


async def create_claim(
    db: Any,
    *,
    org_id: str,
    description: str,
    agent_id: str,
    user_id: Optional[str] = None,
    project_id: Optional[str] = None,
    ttl_minutes: int = 15,
) -> dict[str, Any]:
    """Register a work claim and return any semantic collisions.

    Embeds ``description`` first (best-effort: an embedding failure never
    blocks registration, it only disables collision detection for this
    claim), then inserts. If the embed succeeded, a second query searches
    live claims in the same org for cosine similarity >=
    :data:`COLLISION_THRESHOLD`, ordered by similarity, capped at 5.

    "Live" is the same predicate everywhere in this module:
    ``released_at IS NULL AND expires_at > now()``.

    Returns:
        ``{"claim_id": <uuid str>, "collisions": [{"id", "description",
        "agent_id", "similarity"}, ...]}``. ``collisions`` is always a list,
        never absent -- an embed failure yields an empty list rather than
        omitting the key.
    """
    claim_id = str(uuid.uuid4())
    emb: Optional[list[float]] = None
    try:
        emb = await embed_text(description)
    except Exception:
        # embed_text already catches its own backend errors and returns
        # None; this guards the (currently theoretical) case of a future
        # change making it raise. Either way, the claim still registers.
        logger.warning(
            "claims: embed failed; registering without collision detection",
            exc_info=True,
        )

    db.execute(
        text(
            """
            INSERT INTO work_claims
              (id, org_id, project_id, agent_id, user_id, description,
               embedding, expires_at)
            VALUES (:id, :org_id, :project_id, :agent_id, :user_id,
                    :description, :embedding,
                    now() + make_interval(mins => :ttl))
            """
        ),
        {
            "id": claim_id,
            "org_id": org_id,
            "project_id": project_id,
            "agent_id": agent_id,
            "user_id": user_id,
            "description": description,
            "embedding": str(emb) if emb else None,
            "ttl": _clamp_ttl(ttl_minutes),
        },
    )

    collisions: list[dict[str, Any]] = []
    if emb:
        emb_str = str(emb)
        rows = db.execute(
            text(
                """
                SELECT id, description, agent_id,
                       1 - (embedding <=> :emb) AS similarity
                FROM work_claims
                WHERE org_id = :org_id AND id != :id
                  AND released_at IS NULL AND expires_at > now()
                  AND embedding IS NOT NULL
                  AND 1 - (embedding <=> :emb) >= :threshold
                ORDER BY similarity DESC LIMIT 5
                """
            ),
            {
                "emb": emb_str,
                "org_id": org_id,
                "id": claim_id,
                "threshold": COLLISION_THRESHOLD,
            },
        ).mappings().all()
        collisions = [dict(r) for r in rows]

    db.commit()
    return {"claim_id": claim_id, "collisions": collisions}


def release_claim(db: Any, *, org_id: str, claim_id: str, agent_id: str) -> bool:
    """Release a claim -- owner-scoped: only the agent that made the claim
    can release it.

    ``claim_id`` is UUID-validated before any query (the same discipline
    :func:`shared.memory.retract_memory` and :func:`shared.memory.
    supersede_with_revision` apply to their own id-shaped inputs): an
    unparseable id is refused as a clean ``False`` rather than reaching the
    ``id = :claim_id`` comparison against a UUID column, which would
    otherwise raise an uncaught driver error.

    Deliberately NOT gated on ``expires_at > now()``: an owner releasing a
    claim that already expired is still a meaningful, honest action (it
    stamps ``released_at`` rather than leaving the row to look like it just
    timed out), mirroring why :func:`shared.memory.retract_memory` retracts
    a memory regardless of its ``status`` -- the only thing that must not
    happen twice is releasing an ALREADY-released claim, which
    ``released_at IS NULL`` alone prevents.

    Returns:
        ``True`` iff a not-yet-released claim owned by ``agent_id`` in
        ``org_id`` was released; ``False`` for a malformed id, an unknown
        id, an id owned by a different agent, an id in a different org, or
        an already-released claim.
    """
    try:
        parsed_id = uuid.UUID(str(claim_id))
    except (ValueError, AttributeError, TypeError):
        return False

    result = db.execute(
        text(
            """
            UPDATE work_claims SET released_at = now()
            WHERE id = :claim_id AND org_id = :org_id
              AND agent_id = :agent_id AND released_at IS NULL
            """
        ),
        {"claim_id": str(parsed_id), "org_id": org_id, "agent_id": agent_id},
    )
    db.commit()
    return bool(result.rowcount)


def live_claims(
    db: Any, *, org_id: str, project_id: Optional[str] = None, limit: int = 10
) -> list[dict[str, Any]]:
    """List live claims for ``org_id`` (optionally narrowed to a project).

    Injected into every recall response under ``"active_claims"`` -- the
    design is that agents already recall before working, so they cannot NOT
    see live claims; there is no separate opt-in surface. Never returns the
    ``embedding`` column: descriptions are user text that already flows into
    recall responses via memories, but the raw vector has no business
    leaving the database.

    "Live" matches :func:`create_claim`'s collision predicate exactly:
    ``released_at IS NULL AND expires_at > now()``.
    """
    rows = db.execute(
        text(
            """
            SELECT id, description, agent_id, user_id, expires_at
            FROM work_claims
            WHERE org_id = :org_id AND released_at IS NULL
              AND expires_at > now()
              AND (:project_id::uuid IS NULL OR project_id = :project_id)
            ORDER BY created_at DESC LIMIT :limit
            """
        ),
        {"org_id": org_id, "project_id": project_id, "limit": limit},
    ).mappings().all()
    return [dict(r) for r in rows]
