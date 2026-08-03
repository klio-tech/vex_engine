"""The GraphStore seam: the ONLY module that knows the graph lives in Postgres.

Contract (from the 2026-08-02 knowledge-graph spec):

* ``add_entity`` / ``add_edge`` — the write half. Closed sets validated here,
  the god-entity guard lives here (so no writer can bypass it), and because
  the schema deliberately carries no foreign keys, *this layer* is the
  referential-integrity check for edge endpoints.
* ``neighbors`` / ``entity_brief`` — the read half. Every read takes the
  viewer and applies the visibility ladder IN SQL, on the destination row of
  every edge. Edges are a tenancy leak class: a private memory linked to an
  org entity must never surface through a walk to someone who could not read
  it directly, so the destination predicate here has the same rigor as
  ``org_id`` everywhere else. Never fetch wide and filter in Python.
* ``sweep_orphan_edges`` — edges are swept, not FK-cascaded, so a ``forget``
  can never fail because of an edge.

Entity visibility is DERIVED, never stored: an entity is visible to a caller
iff the ladder grants them at least one memory evidencing it. There is no
entity-scope column to drift out of sync, and ``entity_brief`` answers
``None`` (→ not_found) rather than an empty brief when the caller can see no
evidence — an empty brief would confirm the name exists.

This module is the future graph-engine boundary (HelixDB-class): its contract
is graph+vector, its implementation is Postgres, and nothing outside it may
write SQL against ``entities`` or ``memory_edges`` — an invariant test greps
the tree for violations. A later engine swap reimplements this one file.
"""

import logging
import uuid
from typing import Any, Optional

from sqlalchemy import text

from shared.memory import MemoryRecord, _row_to_record

logger = logging.getLogger("agentguard.shared.graph_store")

# ---------------------------------------------------------------------------
# Closed sets — mirror the CHECK constraints from migration 035 exactly.
# ---------------------------------------------------------------------------

ENTITY_KINDS: frozenset[str] = frozenset(
    {"system", "person", "concept", "decision_subject", "tool", "other"}
)
EDGE_RELATIONS: frozenset[str] = frozenset(
    {"about", "supersedes", "derived_from", "relates_to"}
)
PROVENANCE_LABELS: frozenset[str] = frozenset({"EXTRACTED", "INFERRED", "AMBIGUOUS"})

ENTITY_NAME_MAX_LEN = 200

# God-entity guard (graphify issue #726): names so generic that an entity for
# them accretes edges from everywhere and turns the graph into a hairball.
# Compared against the NORMALIZED name. Extend deliberately, never trim
# casually — every removal re-opens the hairball.
ENTITY_NAME_BLOCKLIST: frozenset[str] = frozenset(
    {
        # pronouns / deixis
        "it",
        "this",
        "that",
        "they",
        "them",
        "he",
        "she",
        "we",
        "you",
        "i",
        # the trivial trio and their bare forms
        "the code",
        "the user",
        "the system",
        "the team",
        "the project",
        "the app",
        "the application",
        "code",
        "user",
        "users",
        "system",
        "team",
        "project",
        "app",
        "application",
        "data",
        "file",
        "files",
        "database",
        "server",
        "test",
        "tests",
        "bug",
        "issue",
        "error",
        "work",
        "task",
        "thing",
        "stuff",
        # days
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "today",
        "tomorrow",
        "yesterday",
        # languages — a language name says nothing about THIS team's knowledge
        "python",
        "javascript",
        "typescript",
        "java",
        "go",
        "golang",
        "rust",
        "ruby",
        "c",
        "c++",
        "c#",
        "sql",
        "html",
        "css",
        "bash",
        "shell",
    }
)

# An entity accumulating edges from everywhere is an extraction bug, not
# knowledge. Alarmed (logged at WARNING), not blocked: blocking would make an
# unrelated write fail because of an unrelated extraction problem.
ENTITY_DEGREE_ALARM = 200


class EntityRefused(ValueError):
    """The god-entity guard refused this entity name.

    A distinct type so writers (the curator) can log-and-skip the refusal
    without also swallowing genuine programming errors (unknown kind, blank
    org), which stay plain ``ValueError``.
    """


def normalize_entity_name(name: str) -> str:
    """Lower-case, trim, and collapse internal whitespace — the dedup key."""
    return " ".join((name or "").split()).lower()


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def add_entity(
    db: Any,
    *,
    org_id: str,
    name: str,
    kind: str,
    embedding: Optional[list[float]] = None,
) -> str:
    """Upsert an entity on ``(org_id, norm_name, kind)``; return its id.

    Race-safe ``INSERT ... ON CONFLICT ... RETURNING`` against the
    ``uq_entities_org_norm_name_kind`` arbiter, same idiom as
    :func:`shared.brain.ensure_project`: concurrent callers converge on one
    row instead of raising. On a hit the display ``name`` is NOT thrashed
    (first spelling wins) and an ``embedding`` only fills a NULL — it never
    overwrites one already stored.

    The god-entity guard lives HERE, not in the curator, so no writer can
    bypass it: a blocklisted or blank/oversized name raises
    :class:`EntityRefused`; an unknown ``kind`` raises plain ``ValueError``
    (that is a programming error, not an extraction-quality problem).
    """
    if not org_id:
        raise ValueError("add_entity requires an org_id")
    if kind not in ENTITY_KINDS:
        raise ValueError(
            f"unknown entity kind: {kind!r} (allowed: {sorted(ENTITY_KINDS)})"
        )

    norm = normalize_entity_name(name)
    if not norm:
        raise EntityRefused("entity name is blank")
    if len(norm) > ENTITY_NAME_MAX_LEN or len(name) > ENTITY_NAME_MAX_LEN:
        raise EntityRefused(f"entity name exceeds {ENTITY_NAME_MAX_LEN} chars")
    if norm in ENTITY_NAME_BLOCKLIST:
        raise EntityRefused(f"entity name too generic to be knowledge: {norm!r}")

    row = db.execute(
        text(
            """
            INSERT INTO entities (id, org_id, name, norm_name, kind, embedding)
            VALUES (:id, :org_id, :name, :norm_name, :kind, :embedding)
            ON CONFLICT (org_id, norm_name, kind)
            DO UPDATE SET
                embedding = COALESCE(entities.embedding, EXCLUDED.embedding)
            RETURNING id
            """
        ),
        {
            "id": str(uuid.uuid4()),
            "org_id": org_id,
            "name": name.strip(),
            "norm_name": norm,
            "kind": kind,
            "embedding": str(embedding) if embedding is not None else None,
        },
    ).fetchone()
    db.commit()
    return str(row.id)


def add_edge(
    db: Any,
    *,
    org_id: str,
    src_id: str,
    dst_kind: str,
    dst_id: str,
    relation: str,
    provenance: str = "INFERRED",
    created_by: Optional[str] = None,
) -> str:
    """Insert (or idempotently re-find) a typed edge; return its id.

    Store-layer referential integrity — the schema has no FKs by design, so
    THIS is where an edge to nowhere is refused: ``src_id`` must name a
    ``session_memories`` row in this org, and ``dst_id`` must name a memory
    or entity in this org. An endpoint in another tenant is treated exactly
    like one that does not exist (:class:`LookupError`), because as far as
    this org is concerned it doesn't.

    Idempotent on the ``(org_id, src_id, dst_kind, dst_id, relation)`` unique
    key via the no-op ``DO UPDATE`` upsert idiom (``DO NOTHING`` would
    suppress ``RETURNING``).
    """
    if dst_kind not in ("memory", "entity"):
        raise ValueError(f"unknown dst_kind: {dst_kind!r}")
    if relation not in EDGE_RELATIONS:
        raise ValueError(
            f"unknown relation: {relation!r} (allowed: {sorted(EDGE_RELATIONS)})"
        )
    if provenance not in PROVENANCE_LABELS:
        raise ValueError(f"unknown provenance: {provenance!r}")

    src_exists = db.execute(
        text("SELECT 1 FROM session_memories WHERE id = :id AND org_id = :org_id"),
        {"id": str(src_id), "org_id": org_id},
    ).fetchone()
    if src_exists is None:
        raise LookupError(f"edge src {src_id} is not a memory in org {org_id}")

    if dst_kind == "memory":
        dst_sql = "SELECT 1 FROM session_memories WHERE id = :id AND org_id = :org_id"
    else:
        dst_sql = "SELECT 1 FROM entities WHERE id = :id AND org_id = :org_id"
    dst_exists = db.execute(
        text(dst_sql), {"id": str(dst_id), "org_id": org_id}
    ).fetchone()
    if dst_exists is None:
        raise LookupError(f"edge dst {dst_id} ({dst_kind}) not found in org {org_id}")

    row = db.execute(
        text(
            """
            INSERT INTO memory_edges
                (id, org_id, src_id, dst_kind, dst_id, relation, provenance,
                 created_by)
            VALUES
                (:id, :org_id, :src_id, :dst_kind, :dst_id, :relation,
                 :provenance, :created_by)
            ON CONFLICT ON CONSTRAINT uq_memory_edges_org_src_dst_relation
            DO UPDATE SET relation = EXCLUDED.relation
            RETURNING id
            """
        ),
        {
            "id": str(uuid.uuid4()),
            "org_id": org_id,
            "src_id": str(src_id),
            "dst_kind": dst_kind,
            "dst_id": str(dst_id),
            "relation": relation,
            "provenance": provenance,
            "created_by": created_by,
        },
    ).fetchone()
    db.commit()

    if dst_kind == "entity":
        _alarm_on_degree(db, org_id=org_id, entity_id=str(dst_id))
    return str(row.id)


def _alarm_on_degree(db: Any, *, org_id: str, entity_id: str) -> None:
    """God-entity guard, part 2: alarm (never block) on runaway degree."""
    try:
        degree = db.execute(
            text(
                "SELECT count(*) FROM memory_edges "
                "WHERE org_id = :org_id AND dst_kind = 'entity' AND dst_id = :dst"
            ),
            {"org_id": org_id, "dst": entity_id},
        ).scalar()
        if degree is not None and degree > ENTITY_DEGREE_ALARM:
            logger.warning(
                "god-entity alarm: entity %s in org %s has degree %d (> %d) — "
                "likely an extraction bug, not knowledge",
                entity_id,
                org_id,
                degree,
                ENTITY_DEGREE_ALARM,
            )
    except Exception:
        # The alarm is telemetry; it must never fail the write that fired it.
        logger.warning("god-entity degree check failed", exc_info=True)


# ---------------------------------------------------------------------------
# The visibility ladder — built once, shared by every read in this module
# ---------------------------------------------------------------------------


def _ladder_clause(
    *,
    scope: str,
    viewer_user_id: Optional[str],
    agent_id: Optional[str],
    alias: str,
) -> tuple[str, dict[str, Any]]:
    """The scope arms of :func:`shared.memory.retrieve_memories_scoped`, for a
    destination row aliased ``alias``.

    Kept semantically identical to the recall ladder ON PURPOSE: stage-B
    expansion composes with stage A, and a row expansion admits that recall
    would refuse is a leak. Same arms, same membership subquery (with the
    same ``JOIN projects p ... AND p.org_id`` — ``project_members`` has no
    org_id of its own, so without the join a membership row in ANOTHER tenant
    would satisfy this arm), same validation, same refusal to widen.

    Returns ``(sql_fragment, params)``; the fragment is one parenthesized
    OR-group ANDed into the caller's WHERE.
    """
    if scope == "private":
        if not viewer_user_id:
            raise ValueError("viewer_user_id is required when scope='private'")
        return (
            f"({alias}.scope = 'private' AND {alias}.user_id = :viewer_user_id)",
            {"viewer_user_id": viewer_user_id},
        )
    if scope == "agent" and agent_id is None:
        raise ValueError("agent_id is required when scope='agent'")
    if scope not in ("org", "agent"):
        raise ValueError(f"unknown scope: {scope!r}")

    params: dict[str, Any] = {}
    arms = [f"{alias}.scope = 'org'"]
    if scope == "agent":
        arms.append(
            f"({alias}.scope = 'agent' AND {alias}.agent_id = :ladder_agent_id)"
        )
        params["ladder_agent_id"] = agent_id
    if viewer_user_id:
        arms.append(
            f"({alias}.scope = 'private' AND {alias}.user_id = :viewer_user_id)"
        )
        arms.append(
            f"({alias}.scope = 'project' AND {alias}.project_id IN ("
            "SELECT m.project_id FROM project_members m "
            "JOIN projects p ON p.id = m.project_id "
            "AND p.org_id = :org_id "
            "WHERE m.user_id = :viewer_user_id))"
        )
        params["viewer_user_id"] = viewer_user_id
    return "(" + " OR ".join(arms) + ")", params


# The one-hop candidate set: every memory connected to the entry set by an
# edge in THIS org — directly (either direction) or as a sibling through a
# shared entity. Counting rows per candidate gives the connectedness signal
# stage C re-ranks on.
_HOP_SQL = """
        SELECT e.dst_id AS other_id
        FROM memory_edges e
        WHERE e.org_id = :org_id AND e.dst_kind = 'memory'
          AND e.src_id = ANY(CAST(:entry_ids AS uuid[]))
        UNION ALL
        SELECT e.src_id
        FROM memory_edges e
        WHERE e.org_id = :org_id AND e.dst_kind = 'memory'
          AND e.dst_id = ANY(CAST(:entry_ids AS uuid[]))
        UNION ALL
        SELECT e2.src_id
        FROM memory_edges e1
        JOIN memory_edges e2
          ON e2.org_id = :org_id
         AND e2.dst_kind = 'entity'
         AND e2.dst_id = e1.dst_id
        WHERE e1.org_id = :org_id AND e1.dst_kind = 'entity'
          AND e1.src_id = ANY(CAST(:entry_ids AS uuid[]))
"""


def neighbors(
    db: Any,
    *,
    org_id: str,
    viewer_user_id: Optional[str],
    memory_ids: list[str],
    scope: str = "org",
    agent_id: Optional[str] = None,
    space_id: Optional[str] = None,
    project_id: Optional[str] = None,
    depth: int = 1,
    limit: int = 5,
) -> dict[str, Any]:
    """One-hop, LADDER-SCOPED expansion around a set of entry memories.

    Returns ``{"records": [MemoryRecord...], "connections": {id: n}}`` where
    ``connections`` counts the edges tying each returned memory into the
    entry set (the stage-C connectedness signal). Entry rows themselves are
    never returned.

    Security contract (invariants 1, 3, 4 of the spec):

    * The DESTINATION memory of every hop passes the caller's visibility
      ladder, in SQL — the same arms, in the same shape, as
      ``retrieve_memories_scoped``. Expansion can therefore never introduce
      a row the ladder would not return directly.
    * The destination row is additionally pinned to ``m.org_id = :org_id``.
      The edge's own org_id is NOT trusted for this: a tampered or buggy
      cross-org edge must grant nothing, so the org check rides on the
      destination row itself.
    * ``space_id`` / ``project_id`` mirror recall's relevance filters so a
      filtered recall cannot be widened by its own expansion: with a
      ``project_id``, rows filed under OTHER projects never surface.

    ``depth`` is hard-capped at 1 in v1 — a value other than 1 raises rather
    than silently truncating, so a future depth-2 caller cannot believe it
    got what it asked for.
    """
    if depth != 1:
        raise ValueError("neighbors() depth is hard-capped at 1 in v1")
    if not memory_ids:
        return {"records": [], "connections": {}}

    ladder_sql, ladder_params = _ladder_clause(
        scope=scope, viewer_user_id=viewer_user_id, agent_id=agent_id, alias="m"
    )

    clauses = [
        "m.org_id = :org_id",
        "m.status = 'active'",
        "m.superseded_by IS NULL",
        "m.recall_hidden = FALSE",
        "m.id != ALL(CAST(:entry_ids AS uuid[]))",
        ladder_sql,
    ]
    params: dict[str, Any] = {
        "org_id": org_id,
        "entry_ids": [str(i) for i in memory_ids],
        "limit": limit,
        **ladder_params,
    }
    if space_id is not None:
        clauses.append("(m.space_id = :space_id OR m.space_id IS NULL)")
        params["space_id"] = space_id
    if project_id is not None:
        clauses.append("(m.project_id = :project_id OR m.project_id IS NULL)")
        params["project_id"] = project_id

    where_sql = "\n              AND ".join(clauses)
    rows = db.execute(
        text(
            f"""
            WITH hop AS ({_HOP_SQL}),
            scored AS (
                SELECT other_id, count(*) AS connections
                FROM hop
                GROUP BY other_id
            )
            SELECT m.id, m.session_id, m.org_id, m.agent_id, m.execution_id,
                   m.memory_type, m.content, m.embedding, m.confidence,
                   m.sequence_number, m.scope, m.status, m.superseded_by,
                   m.metadata, m.created_at, m.updated_at, m.space_id,
                   m.project_id, m.user_id, m.provenance,
                   scored.connections
            FROM scored
            JOIN session_memories m
              ON m.id = scored.other_id
             AND m.org_id = :org_id
            WHERE {where_sql}
            ORDER BY scored.connections DESC, m.created_at DESC
            LIMIT :limit
            """
        ),
        params,
    ).fetchall()

    records: list[MemoryRecord] = []
    connections: dict[str, int] = {}
    for row in rows:
        record = _row_to_record(row[:-1])
        records.append(record)
        connections[str(record.id)] = int(row[-1])
    return {"records": records, "connections": connections}


# ---------------------------------------------------------------------------
# entity_brief — derived visibility, assembled from the caller's slice
# ---------------------------------------------------------------------------

_BRIEF_EVIDENCE_LIMIT = 50


def entity_brief(
    db: Any,
    *,
    org_id: str,
    viewer_user_id: Optional[str],
    entity: str,
    scope: str = "org",
    agent_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Assemble the caller's view of an entity, or ``None`` for not_found.

    ``None`` covers, indistinguishably: no such entity, an entity in another
    org, and an entity whose every evidencing memory is invisible to this
    caller. All three MUST answer identically — an empty brief (or a distinct
    "forbidden") would confirm the name exists, and entity visibility is
    derived from evidence, never stored.

    The brief is assembled from the caller's full cascade (their ladder),
    evaluated together: two teammates exploring the same entity may see
    different briefs, which is the feature. Contributors are a COUNT — the
    number of distinct attributed authors across the visible slice — never a
    list of ids or names.
    """
    norm = normalize_entity_name(entity)
    if not norm:
        return None

    entity_rows = db.execute(
        text(
            "SELECT id, name, kind FROM entities "
            "WHERE org_id = :org_id AND norm_name = :norm"
        ),
        {"org_id": org_id, "norm": norm},
    ).fetchall()
    if not entity_rows:
        return None
    entity_ids = [str(r.id) for r in entity_rows]

    ladder_sql, ladder_params = _ladder_clause(
        scope=scope, viewer_user_id=viewer_user_id, agent_id=agent_id, alias="m"
    )

    # The visible evidence: memories tied to this entity by an edge, passing
    # the caller's ladder, in THIS org (the org check rides on the memory row
    # — a tampered edge grants nothing).
    evidence_rows = db.execute(
        text(
            f"""
            SELECT DISTINCT
                   m.id, m.session_id, m.org_id, m.agent_id, m.execution_id,
                   m.memory_type, m.content, m.embedding, m.confidence,
                   m.sequence_number, m.scope, m.status, m.superseded_by,
                   m.metadata, m.created_at, m.updated_at, m.space_id,
                   m.project_id, m.user_id, m.provenance
            FROM memory_edges e
            JOIN session_memories m
              ON m.id = e.src_id
             AND m.org_id = :org_id
            WHERE e.org_id = :org_id
              AND e.dst_kind = 'entity'
              AND e.dst_id = ANY(CAST(:entity_ids AS uuid[]))
              AND m.status = 'active'
              AND m.recall_hidden = FALSE
              AND {ladder_sql}
            ORDER BY m.created_at ASC
            LIMIT :limit
            """
        ),
        {
            "org_id": org_id,
            "entity_ids": entity_ids,
            "limit": _BRIEF_EVIDENCE_LIMIT,
            **ladder_params,
        },
    ).fetchall()

    if not evidence_rows:
        return None  # not_found — same answer as "no such entity"

    memories = [_row_to_record(row) for row in evidence_rows]
    evidence_ids = [str(m.id) for m in memories]

    # Which entity row does the visible evidence actually support? (A name
    # can exist under several kinds; report the kind the caller can see.)
    kind_row = db.execute(
        text(
            """
            SELECT ent.name, ent.kind, count(*) AS n
            FROM memory_edges e
            JOIN entities ent ON ent.id = e.dst_id AND ent.org_id = :org_id
            WHERE e.org_id = :org_id
              AND e.dst_kind = 'entity'
              AND e.dst_id = ANY(CAST(:entity_ids AS uuid[]))
              AND e.src_id = ANY(CAST(:evidence_ids AS uuid[]))
            GROUP BY ent.name, ent.kind
            ORDER BY n DESC
            LIMIT 1
            """
        ),
        {
            "org_id": org_id,
            "entity_ids": entity_ids,
            "evidence_ids": evidence_ids,
        },
    ).fetchone()
    display_name = kind_row.name if kind_row else entity_rows[0].name
    kind = kind_row.kind if kind_row else entity_rows[0].kind

    decisions = [
        {
            "id": str(m.id),
            "content": m.content,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "provenance": m.provenance,
        }
        for m in memories
        if m.memory_type == "decision"
    ]

    # Supersessions within the visible slice. This query returns ``m.content``
    # — the RETIRED memory's own text — so it needs every read-path guard the
    # evidence query above has, minus the one that would defeat its purpose.
    #
    # * ``status != 'retracted'`` and NOT ``status = 'active'``: the retired
    #   side is status='superseded' by construction (that IS the point), so an
    #   active-only filter would empty the feature. But a retracted row is
    #   forgotten, and forgotten means forgotten, the graph included — its
    #   content must never come back as ``old_content``. Without this a
    #   ``forget`` on a superseded memory would succeed and change nothing
    #   the caller can observe.
    # * ``recall_hidden = FALSE``: hidden raw captures are curator fuel that
    #   NO read path returns. Omitting it here was a live hole in the store's
    #   contract even while the only supersedes-edge writer never made one.
    superseded_rows = db.execute(
        text(
            f"""
            SELECT e.src_id AS new_id, m.id AS old_id, m.content AS old_content,
                   e.created_at AS when_at
            FROM memory_edges e
            JOIN session_memories m
              ON m.id = e.dst_id
             AND m.org_id = :org_id
            WHERE e.org_id = :org_id
              AND e.relation = 'supersedes'
              AND e.dst_kind = 'memory'
              AND e.src_id = ANY(CAST(:evidence_ids AS uuid[]))
              AND m.status != 'retracted'
              AND m.recall_hidden = FALSE
              AND {ladder_sql}
            ORDER BY e.created_at ASC
            """
        ),
        {"org_id": org_id, "evidence_ids": evidence_ids, **ladder_params},
    ).fetchall()
    superseded = [
        {
            "old": str(r.old_id),
            "old_content": r.old_content,
            "by": str(r.new_id),
            "when": r.when_at.isoformat() if r.when_at else None,
        }
        for r in superseded_rows
    ]

    contributors = len({m.user_id for m in memories if m.user_id})
    provenance_tally: dict[str, int] = {}
    for m in memories:
        provenance_tally[m.provenance] = provenance_tally.get(m.provenance, 0) + 1

    return {
        "entity": display_name,
        "kind": kind,
        "memories": memories,
        "evidence_count": len(memories),
        "decisions": decisions,
        "superseded": superseded,
        "contributors": contributors,
        "provenance": provenance_tally,
    }


# ---------------------------------------------------------------------------
# The sweep — orphan edges are removed, never FK-cascaded
# ---------------------------------------------------------------------------


def sweep_orphan_edges(db: Any, *, org_id: str) -> int:
    """Delete this org's edges whose endpoints are gone or retracted.

    An endpoint is orphaned when the memory row no longer exists, when it was
    ``forget``-ed (``status='retracted'`` — forgotten means forgotten, the
    graph included), or when a dst entity row vanished. ``superseded`` rows
    are NOT orphans: edges into them are the supersession history this whole
    feature exists to keep.

    Returns the number of edges removed.
    """
    result = db.execute(
        text(
            """
            DELETE FROM memory_edges e
            WHERE e.org_id = :org_id
              AND (
                NOT EXISTS (
                    SELECT 1 FROM session_memories s
                    WHERE s.id = e.src_id AND s.org_id = e.org_id
                      AND s.status != 'retracted'
                )
                OR (
                    e.dst_kind = 'memory'
                    AND NOT EXISTS (
                        SELECT 1 FROM session_memories s
                        WHERE s.id = e.dst_id AND s.org_id = e.org_id
                          AND s.status != 'retracted'
                    )
                )
                OR (
                    e.dst_kind = 'entity'
                    AND NOT EXISTS (
                        SELECT 1 FROM entities ent
                        WHERE ent.id = e.dst_id AND ent.org_id = e.org_id
                    )
                )
              )
            """
        ),
        {"org_id": org_id},
    )
    db.commit()
    return int(result.rowcount or 0)


#: How long an entity is immune from collection after it is created.
#: ``add_entity`` and ``add_edge`` are two statements: a retention run landing
#: between them would delete a row the caller is in the middle of evidencing.
#: Generous on purpose — the cost of waiting a day is one dead row; the cost of
#: losing the race is data the user just wrote.
ENTITY_SWEEP_GRACE_HOURS: int = 24


def sweep_orphan_entities(db: Any, *, org_id: str) -> int:
    """Delete this org's entities that nothing evidences any more.

    Retention deletes from ``session_memories``; :func:`sweep_orphan_edges`
    then collects the edges whose endpoints vanished. Nothing collected the
    ENTITIES those edges evidenced, so an entity outlived every memory that
    ever mentioned it — forever, and invisibly, since ``entity_brief`` answers
    ``None`` for an entity with no visible evidence. Growth is the small half
    of that; a named thing surviving the deletion of everything that named it
    is the retention-policy half.

    **Evidence is counted without the visibility ladder, deliberately.** An
    entity evidenced only by one user's private memory is still evidenced.
    The ladder decides what a CALLER is shown; a retention sweep is not a
    caller, and scoping the anti-join to what some viewer can see would delete
    one user's data because another user could not see it. So the ``NOT
    EXISTS`` below matches ANY edge into the entity, whatever the source
    memory's scope, owner or project.

    Ordering: run this AFTER :func:`sweep_orphan_edges`, which is what turns
    "every evidencing memory was deleted or retracted" into "no edges remain".
    Running it first is safe but collects nothing — edges into dead memories
    still count as evidence until they are themselves swept.

    Same-org edges only. A cross-org edge into this entity is tampered data
    (``add_edge`` refuses to create one) and grants nothing anywhere else in
    this module; letting it pin an entity row would hand another tenant a
    veto over this org's retention.

    Returns the number of entities removed.
    """
    result = db.execute(
        text(
            """
            DELETE FROM entities e
            WHERE e.org_id = :org_id
              AND e.created_at < NOW() - make_interval(hours => :grace_hours)
              AND NOT EXISTS (
                    SELECT 1 FROM memory_edges m
                    WHERE m.org_id = e.org_id
                      AND m.dst_kind = 'entity'
                      AND m.dst_id = e.id
              )
            """
        ),
        {"org_id": org_id, "grace_hours": ENTITY_SWEEP_GRACE_HOURS},
    )
    db.commit()
    return int(result.rowcount or 0)
