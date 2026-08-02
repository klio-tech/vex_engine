"""Knowledge graph phase 1: entities and typed memory edges.

Two tables behind the ``GraphStore`` seam (``shared/graph_store.py``), exactly
as the 2026-08-02 knowledge-graph spec draws them:

* ``entities`` — org-scoped named things (system/person/concept/…), deduped by
  ``(org_id, norm_name, kind)``. Visibility is DERIVED from evidence at read
  time and deliberately not stored here: there is no scope column to drift.
* ``memory_edges`` — typed, provenance-stamped edges from a memory to a memory
  or an entity. ``relation`` and ``dst_kind`` are CHECK-constrained closed
  sets from day one — scope had no CHECK once and typos created invisible
  rows.

No foreign keys on the edge endpoints: ``dst_id`` is polymorphic
(memory-or-entity), and an edge must never be the reason a ``forget`` fails.
Integrity is enforced by the store layer and asserted by tests; orphan edges
are swept, not cascaded.

Backfill (the only one): ``supersedes`` edges from ``status='superseded'``
rows WHERE the successor id is genuinely recoverable — ``superseded_by`` is
set AND still names a row in the same org. Where it is not, no edge: we do
not invent history. The edges land ``INFERRED`` because the underlying
judgment was the reconciliation LLM's. ``derived_from`` is NOT backfilled —
curator input ids were never recorded, so there is nothing honest to write.

Revision ID: 035
Revises: 034
Create Date: 2026-08-02
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "035"
down_revision = "034"
branch_labels = None
depends_on = None

ENTITY_KINDS = ("system", "person", "concept", "decision_subject", "tool", "other")
EDGE_RELATIONS = ("about", "supersedes", "derived_from", "relates_to")
PROVENANCE_LABELS = ("EXTRACTED", "INFERRED", "AMBIGUOUS")


def _in_list(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{v}'" for v in values)


def upgrade() -> None:
    op.create_table(
        "entities",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("norm_name", sa.String(200), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.CheckConstraint(
            f"kind IN ({_in_list(ENTITY_KINDS)})",
            name="ck_entities_kind",
        ),
        sa.UniqueConstraint(
            "org_id", "norm_name", "kind", name="uq_entities_org_norm_name_kind"
        ),
    )
    # The entity-level semantic entry point. Raw SQL like every other pgvector
    # column in this repo (see 016/017); 1024 matches session_memories since
    # migration 017 so one embedder serves both tables.
    op.execute("ALTER TABLE entities ADD COLUMN embedding vector(1024)")

    op.create_table(
        "memory_edges",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", sa.String(64), nullable=False),
        # Always a session_memories.id — enforced by the store layer, not an
        # FK, so a forget can never be blocked by an edge.
        sa.Column("src_id", UUID(as_uuid=True), nullable=False),
        sa.Column("dst_kind", sa.String(8), nullable=False),
        sa.Column("dst_id", UUID(as_uuid=True), nullable=False),
        sa.Column("relation", sa.String(16), nullable=False),
        sa.Column(
            "provenance",
            sa.String(16),
            nullable=False,
            server_default="INFERRED",
        ),
        sa.Column("created_by", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.CheckConstraint(
            "dst_kind IN ('memory', 'entity')",
            name="ck_memory_edges_dst_kind",
        ),
        sa.CheckConstraint(
            f"relation IN ({_in_list(EDGE_RELATIONS)})",
            name="ck_memory_edges_relation",
        ),
        sa.CheckConstraint(
            f"provenance IN ({_in_list(PROVENANCE_LABELS)})",
            name="ck_memory_edges_provenance",
        ),
        sa.UniqueConstraint(
            "org_id",
            "src_id",
            "dst_kind",
            "dst_id",
            "relation",
            name="uq_memory_edges_org_src_dst_relation",
        ),
    )
    # Traversal both ways: forward from a memory, reverse into a destination.
    op.create_index("ix_memory_edges_org_src", "memory_edges", ["org_id", "src_id"])
    op.create_index(
        "ix_memory_edges_org_dst", "memory_edges", ["org_id", "dst_kind", "dst_id"]
    )

    # ---- backfill: recoverable supersessions only ----------------------
    # A supersedes edge points FROM the successor TO the row it retired,
    # matching the direction the reconciliation writes going forward. The
    # join to the successor row is what makes "recoverable" mean something:
    # superseded_by must be non-NULL AND still name a live row IN THE SAME
    # ORG. NULL or dangling successors produce nothing — no invented history.
    op.execute(
        """
        INSERT INTO memory_edges
            (id, org_id, src_id, dst_kind, dst_id, relation, provenance)
        SELECT gen_random_uuid(), old.org_id, new.id, 'memory', old.id,
               'supersedes', 'INFERRED'
        FROM session_memories old
        JOIN session_memories new
          ON new.id = old.superseded_by
         AND new.org_id = old.org_id
        WHERE old.status = 'superseded'
          AND old.superseded_by IS NOT NULL
        ON CONFLICT ON CONSTRAINT uq_memory_edges_org_src_dst_relation
        DO NOTHING
        """
    )


def downgrade() -> None:
    op.drop_index("ix_memory_edges_org_dst", table_name="memory_edges")
    op.drop_index("ix_memory_edges_org_src", table_name="memory_edges")
    op.drop_table("memory_edges")
    op.drop_table("entities")
