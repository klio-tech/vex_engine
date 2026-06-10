"""Klio brain: recall_hidden flag for raw passive captures.

ADDITIVE. Adds ``session_memories.recall_hidden`` (default FALSE). Rows with
``recall_hidden = TRUE`` are curator fuel only: both recall paths exclude
them. The data backfill hides existing passive hook observations
(``memory_type='observation'`` AND ``metadata->>'source'`` IN
('hook','hook-tool')) — explicit MCP ``observe`` rows (source='mcp') stay
visible.

Revision ID: 026
Revises: 025
Create Date: 2026-06-10
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "026"
down_revision: Union[str, None] = "025"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "session_memories",
        sa.Column(
            "recall_hidden",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
    )
    # Hide existing passive hook observations (curator fuel, not recall noise).
    op.execute(
        """
        UPDATE session_memories
        SET recall_hidden = TRUE
        WHERE memory_type = 'observation'
          AND metadata->>'source' IN ('hook', 'hook-tool')
        """
    )
    # Partial index: recall always filters recall_hidden = FALSE.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_session_memories_recall_visible
        ON session_memories (org_id, scope, status)
        WHERE recall_hidden = FALSE
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_session_memories_recall_visible;")
    op.drop_column("session_memories", "recall_hidden")
