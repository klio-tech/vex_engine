"""session_traces: retained, redacted session evidence (Build 1).

ADDITIVE ONLY. Rows point at one S3 object holding the REDACTED structured
payload {messages, tool_calls}. message_count enables richer-transcript-wins
upserts (a second Stop hook with more messages replaces the first).
"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "041"
down_revision: Union[str, None] = "040"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "session_traces",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("agent_id", sa.Text(), nullable=True),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("transcript_ref", sa.Text(), nullable=False),
        sa.Column("message_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("tool_calls_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("redaction_version", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("org_id", "session_id", name="uq_session_traces_org_session"),
    )
    op.create_index("ix_session_traces_org_created", "session_traces",
                    ["org_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_session_traces_org_created", table_name="session_traces")
    op.drop_table("session_traces")
