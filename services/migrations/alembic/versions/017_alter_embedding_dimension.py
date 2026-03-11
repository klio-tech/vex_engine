"""Alter session_memories embedding column from vector(1536) to vector(1024).

Switches from OpenAI text-embedding-3-small (1536 dims) to Together AI
intfloat/multilingual-e5-large-instruct (1024 dims). The HNSW index is
dropped and recreated to match the new dimension.

Revision ID: 017
Revises: 016
Create Date: 2026-03-08
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "017"
down_revision: Union[str, None] = "016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop HNSW index before altering column type
    op.execute("DROP INDEX IF EXISTS ix_session_memories_embedding_hnsw")

    # Alter embedding column dimension (table is empty, no re-embed needed)
    op.execute("ALTER TABLE session_memories ALTER COLUMN embedding TYPE vector(1024)")

    # Recreate HNSW index with new dimension
    op.execute(
        "CREATE INDEX ix_session_memories_embedding_hnsw "
        "ON session_memories USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_session_memories_embedding_hnsw")
    op.execute("ALTER TABLE session_memories ALTER COLUMN embedding TYPE vector(1536)")
    op.execute(
        "CREATE INDEX ix_session_memories_embedding_hnsw "
        "ON session_memories USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
