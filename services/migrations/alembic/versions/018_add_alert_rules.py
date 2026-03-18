"""Add integration_connections, alert_rules, and alert_rule_channels tables.

Revision ID: 018
Revises: 017
Create Date: 2026-03-18
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

revision: str = "018"
down_revision: Union[str, None] = "017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- integration_connections ---
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS integration_connections (
            id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
            org_id          TEXT            NOT NULL,
            provider        TEXT            NOT NULL,
            nango_connection_id TEXT        NOT NULL,
            workspace_name  TEXT,
            status          TEXT            NOT NULL DEFAULT 'connected',
            created_by      UUID            NOT NULL,
            created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
            CONSTRAINT chk_integration_connections_status
                CHECK (status IN ('connected', 'disconnected'))
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_integration_connections_org "
        "ON integration_connections (org_id)"
    )
    op.execute(
        "CREATE UNIQUE INDEX idx_integration_connections_org_provider "
        "ON integration_connections (org_id, provider, nango_connection_id)"
    )

    # --- alert_rules ---
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS alert_rules (
            id                      UUID                PRIMARY KEY DEFAULT gen_random_uuid(),
            account_id              TEXT                NOT NULL,
            name                    TEXT                NOT NULL,
            confidence_threshold    DOUBLE PRECISION    NOT NULL,
            enabled                 BOOLEAN             NOT NULL DEFAULT true,
            created_by              UUID                NOT NULL,
            created_at              TIMESTAMPTZ         NOT NULL DEFAULT now(),
            updated_at              TIMESTAMPTZ         NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_alert_rules_account ON alert_rules (account_id)"
    )

    # --- alert_rule_channels ---
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS alert_rule_channels (
            id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
            alert_rule_id       UUID            NOT NULL REFERENCES alert_rules (id) ON DELETE CASCADE,
            channel_type        TEXT            NOT NULL,
            connection_id       UUID            REFERENCES integration_connections (id) ON DELETE SET NULL,
            slack_channel_id    TEXT,
            slack_channel_name  TEXT,
            created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
            CONSTRAINT chk_alert_rule_channels_type CHECK (channel_type IN ('slack'))
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_alert_rule_channels_rule "
        "ON alert_rule_channels (alert_rule_id)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS alert_rule_channels")
    op.execute("DROP TABLE IF EXISTS alert_rules")
    op.execute("DROP TABLE IF EXISTS integration_connections")
