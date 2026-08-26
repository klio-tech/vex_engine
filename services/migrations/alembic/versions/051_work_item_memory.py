"""Work-item memory: filing memories and branches under a work item.

Revision ID: 051
Revises: 050

A work item is the finest grain a memory can be filed against. Both columns
are a FILING dimension, never a boundary: a work item's visibility in the
workspace is exactly its project's visibility, so inventing a
``scope='work_item'`` would create a wall the workspace itself does not have
— and every wall is a place a leak can hide. The boundary stays ``scope``
(private / project / org), unchanged.

``session_memories.work_item_id`` — NULL means "not filed under any work
item", which is every existing row by construction, so every existing reader
keeps working the day the column appears.

``context_branches.work_item_id`` — a branch bound to a work item is a
feature branch for it. The branch still BELONGS to its project (merge
promotes to project main, unchanged), so the open-name unique index is left
exactly as migration 050 wrote it: one open branch name per project, which is
what git does with one repo.

No FK on either column: the ids name rows in the WORKSPACE database, a
different server, so a foreign key is not expressible. Validation happens at
the resolution layer (:mod:`shared.work_items`), which refuses a work item
whose project is not linked into the caller's org.
"""

import sqlalchemy as sa
from alembic import op

revision = "051"
down_revision = "050"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "session_memories",
        sa.Column(
            "work_item_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=True
        ),
    )
    # Partial, like ix_session_memories_branch: the overwhelming majority of
    # rows are not filed under a work item, and indexing their NULLs buys
    # nothing. The org_id lead column matches how every read is filtered.
    op.create_index(
        "ix_session_memories_work_item",
        "session_memories",
        ["org_id", "work_item_id"],
        postgresql_where=sa.text("work_item_id IS NOT NULL"),
    )
    op.add_column(
        "context_branches",
        sa.Column(
            "work_item_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=True
        ),
    )
    op.create_index(
        "ix_context_branches_work_item",
        "context_branches",
        ["org_id", "work_item_id"],
        postgresql_where=sa.text("work_item_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_context_branches_work_item", table_name="context_branches")
    op.drop_column("context_branches", "work_item_id")
    op.drop_index("ix_session_memories_work_item", table_name="session_memories")
    op.drop_column("session_memories", "work_item_id")
