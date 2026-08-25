"""Context branches: the registry and the memory column.

Revision ID: 050
Revises: 049

``context_branches`` gives a project git-shaped context branches whose
lifecycle lives in a row, not a naming convention. ``session_memories``
gains a nullable ``branch_id``: NULL **is** main — every existing row is on
main by construction, and every existing reader keeps working the day the
column appears.

The partial unique index enforces one OPEN branch per (org, project, name);
a merged/discarded name is reusable, exactly like git. No FK on
``session_memories.branch_id`` — matches the repo's edges/``superseded_by``
convention of unconstrained cross-refs plus sweeps.
"""

import sqlalchemy as sa
from alembic import op

revision = "050"
down_revision = "049"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "context_branches",
        sa.Column(
            "id",
            sa.dialects.postgresql.UUID(as_uuid=False),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column(
            "project_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=False
        ),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="open"),
        sa.Column("created_by", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('open','merged','discarded')",
            name="ck_context_branches_status",
        ),
    )
    # One OPEN branch per (org, project, name); a merged/discarded name is
    # reusable, exactly like git.
    op.create_index(
        "uq_context_branches_open_name",
        "context_branches",
        ["org_id", "project_id", "name"],
        unique=True,
        postgresql_where=sa.text("status = 'open'"),
    )
    op.create_index(
        "ix_context_branches_project", "context_branches", ["org_id", "project_id"]
    )
    op.add_column(
        "session_memories",
        sa.Column(
            "branch_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=True
        ),
    )
    op.create_index(
        "ix_session_memories_branch",
        "session_memories",
        ["branch_id"],
        postgresql_where=sa.text("branch_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_session_memories_branch", table_name="session_memories")
    op.drop_column("session_memories", "branch_id")
    op.drop_index("ix_context_branches_project", table_name="context_branches")
    op.drop_index("uq_context_branches_open_name", table_name="context_branches")
    op.drop_table("context_branches")
