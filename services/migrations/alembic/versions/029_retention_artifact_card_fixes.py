"""Klio artifacts: fix cross-org leak and malformed-uuid abort in retention.

Migration 028 protected artifact cards from the retention sweep while their
artifact is active, but review found two Important issues:

1. The EXISTS subquery checked ``a.status = 'active'`` without constraining
   ``a.org_id = p_org_id``. A session_memories row whose ``artifact_id``
   happened to reference an *active* artifact in a *different* org would be
   incorrectly preserved past its own org's retention window.

2. ``(sm.metadata->>'artifact_id')::uuid`` raises on any non-null value that
   is not a valid UUID. That exception aborts the entire DELETE for the
   calling org, so a single malformed artifact card fails retention for the
   whole org. No writer produces ``memory_type = 'artifact'`` rows yet, so
   this was dormant, but it sits on a nightly cron path with org-wide blast
   radius and must be closed before the writer lands.

This migration replaces ``enforce_memory_retention`` again (CREATE OR
REPLACE, additive) to:
  - add ``a.org_id = p_org_id`` to the EXISTS predicate, and
  - guard the uuid cast with a regex format check so a malformed
    ``artifact_id`` simply fails to match (row is swept) instead of raising.

Revision ID: 029
Revises: 028
Create Date: 2026-07-30
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

revision: str = "029"
down_revision: Union[str, None] = "028"
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
                AND (sm.metadata->>'artifact_id') ~*
                    '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
                AND EXISTS (
                    SELECT 1 FROM artifacts a
                    WHERE a.id = (sm.metadata->>'artifact_id')::uuid
                      AND a.org_id = p_org_id
                      AND a.status = 'active'
                )
              );
        END;
        $$ LANGUAGE plpgsql;
        """
    )


def downgrade() -> None:
    # Restore migration 028's body verbatim (org_id-unchecked, unguarded cast).
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
