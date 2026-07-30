"""Klio artifacts: retention must not orphan artifact cards.

Replaces ``enforce_memory_retention`` (migration 025), which deleted every
``session_memories`` row past the plan window with no ``memory_type`` filter.
An artifact card swept while its artifact is still active leaves bytes that
are billed, listed, and unreachable by recall.

A card now outlives the window exactly as long as its artifact does, and
becomes collectable the moment the artifact is retracted or superseded.

Revision ID: 028
Revises: 027
Create Date: 2026-07-30
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

revision: str = "028"
down_revision: Union[str, None] = "027"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION enforce_memory_retention(
            p_org_id TEXT, p_retention_days INT
        ) RETURNS void AS $$
        BEGIN
            DELETE FROM session_memories sm
            WHERE sm.org_id = p_org_id
              AND sm.created_at < NOW() - (p_retention_days || ' days')::interval
              AND NOT (
                sm.memory_type = 'artifact'
                AND EXISTS (
                    SELECT 1 FROM artifacts a
                    WHERE a.id = (sm.metadata->>'artifact_id')::uuid
                      AND a.status = 'active'
                )
              );
        END;
        $$ LANGUAGE plpgsql;
        """
    )


def downgrade() -> None:
    # Restore the unfiltered 025 body verbatim.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION enforce_memory_retention(
            p_org_id TEXT, p_retention_days INT
        ) RETURNS void AS $$
        BEGIN
            DELETE FROM session_memories
            WHERE org_id = p_org_id
              AND created_at < NOW() - (p_retention_days || ' days')::interval;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
