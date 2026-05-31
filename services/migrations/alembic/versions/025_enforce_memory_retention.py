"""Klio memory retention: per-org session_memories pruning function.

ADDITIVE ONLY. Adds an ``enforce_memory_retention(org_id, days)`` SQL function
that deletes ``session_memories`` rows older than the plan's window. Mirrors
``enforce_plan_retention`` (migration 008) but targets the memory table. The
retention cron (run_retention.py) calls it per org.

Revision ID: 025
Revises: 024
Create Date: 2026-06-01
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

revision: str = "025"
down_revision: Union[str, None] = "024"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
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


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS enforce_memory_retention(TEXT, INT);")
