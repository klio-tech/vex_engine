"""Recall evidence with an HONEST join (Build 2).

ADDITIVE ONLY. The session grade lives ONCE in session_outcomes. Per-memory
rows in recall_outcomes carry ATTRIBUTION (used / usage_score) and staleness —
never a copied session verdict. Querying "how did this memory do" joins
attribution to the session outcome explicitly, so nobody mistakes session
weather for memory quality.
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "042"
down_revision: Union[str, None] = "041"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "brain_recall_events",
        sa.Column(
            "memory_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=True,
        ),
    )
    op.add_column(
        "brain_recall_events", sa.Column("session_id", sa.Text(), nullable=True)
    )
    op.create_index(
        "ix_brain_recall_events_org_session",
        "brain_recall_events",
        ["org_id", "session_id"],
    )
    op.create_table(
        "session_outcomes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("verdict", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Numeric(), nullable=True),
        sa.Column("memory_consistency", sa.Numeric(), nullable=True),
        sa.Column("grader_version", sa.Text(), nullable=False),
        sa.Column(
            "graded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "org_id",
            "session_id",
            "grader_version",
            name="uq_session_outcomes_org_session_grader",
        ),
    )
    op.create_table(
        "recall_outcomes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column(
            "recall_event_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("brain_recall_events.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("memory_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "session_outcome_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("session_outcomes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "used", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column("usage_score", sa.Numeric(), nullable=True),
        sa.Column(
            "served_stale",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_recall_outcomes_org_memory", "recall_outcomes", ["org_id", "memory_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_recall_outcomes_org_memory", table_name="recall_outcomes")
    op.drop_table("recall_outcomes")
    op.drop_table("session_outcomes")
    op.drop_index(
        "ix_brain_recall_events_org_session", table_name="brain_recall_events"
    )
    op.drop_column("brain_recall_events", "session_id")
    op.drop_column("brain_recall_events", "memory_ids")
