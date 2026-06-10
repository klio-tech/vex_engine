"""Klio memory metering: monthly usage counters.

ADDITIVE ONLY. Introduces ``memory_usage_counters``, a per-(org, month) rollup
of billable memory usage (memories captured + recalls), incremented on the write
and recall paths. Enforcement reads one row per request; the counter survives
retention deletes (unlike COUNT over session_memories). Nothing existing is
touched.

Revision ID: 024
Revises: 023
Create Date: 2026-06-01
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "024"
down_revision: Union[str, None] = "023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_usage_counters",
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("period_month", sa.Date(), nullable=False),
        sa.Column(
            "memories_captured", sa.BigInteger(), server_default="0", nullable=False
        ),
        sa.Column("recalls", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column(
            "memories_overage", sa.BigInteger(), server_default="0", nullable=False
        ),
        sa.Column(
            "recalls_overage", sa.BigInteger(), server_default="0", nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("org_id", "period_month"),
    )


def downgrade() -> None:
    op.drop_table("memory_usage_counters")
