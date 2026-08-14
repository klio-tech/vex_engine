"""Work claims: advisory, semantic, TTL'd awareness (Build 4).

``work_claims`` is a NEW table, not an extension of anything existing — a
claim is not a memory (no ``status``/``superseded_by`` machinery, no
recall-visibility ladder) and not a lock (no row-level mutual exclusion; two
agents can claim overlapping work and both succeed — the point is that the
second one FINDS OUT, semantically, via ``create_claim``'s collision search).

Columns mirror the shape ``shared.claims`` expects: ``org_id`` is always the
tenancy fence; ``project_id``/``user_id`` are nullable identity refinements
with no FK (same rationale as ``memory_edges`` in migration 035 — a claim
must never be the reason a project or user row becomes undeletable);
``agent_id`` is free text, matching every other agent identifier column in
this schema (``session_memories.agent_id`` et al.), never a foreign key to a
users/agents table that does not exist. ``expires_at`` is NOT NULL — every
claim is TTL'd by construction, computed server-side by
``shared.claims.create_claim`` via ``now() + make_interval(...)`` so no
caller-supplied clock skew can mint a claim that never expires.
``released_at`` is the other half of "live": a claim is live iff
``released_at IS NULL AND expires_at > now()``, checked identically by
``create_claim``'s collision search and ``live_claims``' listing — there is
no ``status`` enum to keep in sync between the two.

``embedding vector(1024)`` via raw SQL, not a SQLAlchemy column type — same
reason as ``entities.embedding`` in migration 035 and ``session_memories``
since migration 017: this repo has no SQLAlchemy-level pgvector column type,
only the ``pgvector`` Python client library. 1024 matches the e5 embedder
already serving ``session_memories`` and ``entities`` so ``shared.memory.
embed_text`` needs no per-table branching. Nullable: ``create_claim`` embeds
BEFORE the INSERT (never inside an open transaction — see the module
docstring on ``shared/claims.py`` for why), but embed failure is
non-fatal — the claim still registers with no collision detection, per the
Build 4 spec.

The ``(org_id, expires_at)`` index serves both hot paths: ``live_claims``'s
``WHERE org_id = :org_id AND released_at IS NULL AND expires_at > now()``
listing (recall-response injection, so this runs on every recall) and the
same predicate inside ``create_claim``'s collision search.

Revision ID: 044
Revises: 043
Create Date: 2026-08-13
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "044"
down_revision: Union[str, None] = "043"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "work_claims",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("agent_id", sa.Text(), nullable=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Raw SQL like every other pgvector column in this repo (see 017/035):
    # no SQLAlchemy-level pgvector column type exists here, only the client
    # library. 1024 matches the e5 embedder already serving session_memories
    # and entities.
    op.execute("ALTER TABLE work_claims ADD COLUMN embedding vector(1024)")
    op.create_index(
        "ix_work_claims_org_expires",
        "work_claims",
        ["org_id", "expires_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_work_claims_org_expires", table_name="work_claims")
    op.drop_table("work_claims")
