"""Klio brain: recall activity audit log.

ADDITIVE ONLY. Introduces ``brain_recall_events``, an append-only audit log of
memory recall/read requests so a dashboard can later surface per-agent recall
activity ("Klio Cloud activity surfacing"). Nothing existing is touched: no
columns are added to or removed from any prior table, no existing index is
altered, and the verification memory path is unaffected.

Rows are written by the recall surfaces (the REST ``/capture/recall`` endpoint
and the MCP ``recall`` tool) after a read completes; the table is never read on
the hot path. ``id`` is an app-supplied UUID (the caller mints it) rather than a
server default, keeping inserts self-describing and avoiding a dependency on a
particular UUID extension. The two indexes serve the dashboard's only two query
shapes: most-recent activity per org (``org_id, created_at``) and per-agent
drill-down (``org_id, agent_id``). A plain ``(org_id, created_at)`` btree index
backs a ``created_at DESC`` scan fine, so no descending expression is needed.

``project_id`` is intentionally a bare UUID with no foreign key (unlike
``session_memories.project_id``, which references ``projects.id``) because this
is an append-only audit log that must outlive project deletion, so we avoid the
FK-check overhead and cascade coupling on the write path.

Revision ID: 022
Revises: 021
Create Date: 2026-05-29
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "022"
down_revision: Union[str, None] = "021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "brain_recall_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("agent_id", sa.Text(), nullable=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=True),
        sa.Column(
            "result_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_brain_recall_events_org_created",
        "brain_recall_events",
        ["org_id", "created_at"],
    )
    op.create_index(
        "ix_brain_recall_events_org_agent",
        "brain_recall_events",
        ["org_id", "agent_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_brain_recall_events_org_agent", table_name="brain_recall_events")
    op.drop_index(
        "ix_brain_recall_events_org_created", table_name="brain_recall_events"
    )
    op.drop_table("brain_recall_events")
