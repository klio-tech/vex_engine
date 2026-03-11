"""Add session_memories table with pgvector embedding support.

Stores per-session memory entries (summaries, facts, procedural knowledge)
with vector embeddings for semantic similarity search. Uses pgvector's HNSW
index for approximate nearest-neighbour lookups on the embedding column.

Revision ID: 016
Revises: 015
Create Date: 2026-03-08
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "016"
down_revision: Union[str, None] = "015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension (idempotent)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create the session_memories table
    op.create_table(
        "session_memories",
        sa.Column(
            "id",
            sa.UUID(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("session_id", sa.String(256), nullable=False),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("agent_id", sa.String(128), nullable=False),
        sa.Column("execution_id", sa.String(64), nullable=True),
        sa.Column("memory_type", sa.String(32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("sequence_number", sa.Integer(), nullable=True),
        sa.Column(
            "scope",
            sa.String(32),
            nullable=False,
            server_default="session",
        ),
        sa.Column(
            "status",
            sa.String(16),
            nullable=False,
            server_default="active",
        ),
        sa.Column("superseded_by", sa.UUID(), nullable=True),
        sa.Column("metadata", JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
        ),
    )

    # Add the embedding column via raw SQL (pgvector type not in SA core)
    op.execute("ALTER TABLE session_memories ADD COLUMN embedding vector(1536)")

    # --- Indexes ---

    # Primary lookup: fetch memories for a specific session
    op.create_index(
        "ix_session_memories_session_lookup",
        "session_memories",
        ["session_id", "org_id", "scope", "status"],
    )

    # Future: agent-level memory retrieval
    op.create_index(
        "ix_session_memories_agent_lookup",
        "session_memories",
        ["agent_id", "org_id", "scope"],
    )

    # Future: org-wide memory retrieval
    op.create_index(
        "ix_session_memories_org_lookup",
        "session_memories",
        ["org_id", "scope"],
    )

    # HNSW index for approximate nearest-neighbour vector search
    op.execute(
        "CREATE INDEX ix_session_memories_embedding_hnsw "
        "ON session_memories USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.drop_table("session_memories")
    # NOTE: we intentionally do NOT drop the vector extension here;
    # other tables or migrations may depend on it.
