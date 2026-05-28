"""Add datasets, experiments tables and experiment columns on executions.

Revision ID: 019
Revises: 018
Create Date: 2026-03-27
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

revision: str = "019"
down_revision: Union[str, None] = "018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- datasets ---
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
            org_id          TEXT            NOT NULL,
            name            TEXT            NOT NULL,
            items           JSONB           NOT NULL DEFAULT '[]'::jsonb,
            created_by      TEXT            NOT NULL,
            created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_datasets_org_id ON datasets (org_id)")

    # --- experiments ---
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
            org_id          TEXT            NOT NULL,
            name            TEXT            NOT NULL,
            description     TEXT,
            mode            TEXT            NOT NULL DEFAULT 'offline',
            status          TEXT            NOT NULL DEFAULT 'draft',
            variants        JSONB           NOT NULL DEFAULT '[]'::jsonb,
            dataset_id      UUID            REFERENCES datasets (id) ON DELETE SET NULL,
            created_by      TEXT            NOT NULL,
            created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
            CONSTRAINT chk_experiments_mode CHECK (mode IN ('offline', 'live')),
            CONSTRAINT chk_experiments_status CHECK (status IN ('draft', 'running', 'completed'))
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_experiments_org_id ON experiments (org_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_experiments_org_status ON experiments (org_id, status)"
    )

    # --- add experiment columns to executions ---
    op.execute("ALTER TABLE executions ADD COLUMN IF NOT EXISTS experiment_id UUID")
    op.execute("ALTER TABLE executions ADD COLUMN IF NOT EXISTS variant TEXT")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_executions_experiment "
        "ON executions (experiment_id, variant) "
        "WHERE experiment_id IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_executions_experiment")
    op.execute("ALTER TABLE executions DROP COLUMN IF EXISTS variant")
    op.execute("ALTER TABLE executions DROP COLUMN IF EXISTS experiment_id")
    op.execute("DROP INDEX IF EXISTS ix_experiments_org_status")
    op.execute("DROP INDEX IF EXISTS ix_experiments_org_id")
    op.execute("DROP TABLE IF EXISTS experiments")
    op.execute("DROP INDEX IF EXISTS ix_datasets_org_id")
    op.execute("DROP TABLE IF EXISTS datasets")
