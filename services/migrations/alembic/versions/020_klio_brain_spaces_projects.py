"""Klio brain: spaces + projects tables, nullable space/project on session_memories.

ADDITIVE ONLY. Existing session_memories rows are unaffected (new columns
default NULL). Verification memory paths never set these columns.

Revision ID: 020
Revises: 019
Create Date: 2026-05-28
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "020"
down_revision: Union[str, None] = "019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "spaces",
        sa.Column(
            "id",
            sa.UUID(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("slug", sa.String(200), nullable=False),
        sa.Column("embedding_model", sa.String(200), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
        ),
        sa.UniqueConstraint("org_id", "slug", name="uq_spaces_org_slug"),
    )
    op.create_table(
        "projects",
        sa.Column(
            "id",
            sa.UUID(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column(
            "space_id",
            sa.UUID(),
            sa.ForeignKey("spaces.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("git_remote", sa.Text(), nullable=True),
        sa.Column("repo_root_path", sa.Text(), nullable=True),
        sa.Column("display_name", sa.String(200), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "ix_projects_org_remote",
        "projects",
        ["org_id", "git_remote"],
        unique=True,
        postgresql_where=sa.text("git_remote IS NOT NULL"),
    )
    op.create_index(
        "ix_projects_org_path",
        "projects",
        ["org_id", "repo_root_path"],
        unique=True,
        postgresql_where=sa.text("git_remote IS NULL AND repo_root_path IS NOT NULL"),
    )
    op.add_column(
        "session_memories",
        sa.Column(
            "space_id",
            sa.UUID(),
            sa.ForeignKey("spaces.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "session_memories",
        sa.Column(
            "project_id",
            sa.UUID(),
            sa.ForeignKey("projects.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_session_memories_space", "session_memories", ["space_id"]
    )
    op.create_index(
        "ix_session_memories_project", "session_memories", ["project_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_session_memories_project", table_name="session_memories")
    op.drop_index("ix_session_memories_space", table_name="session_memories")
    op.drop_column("session_memories", "project_id")
    op.drop_column("session_memories", "space_id")
    op.drop_index("ix_projects_org_path", table_name="projects")
    op.drop_index("ix_projects_org_remote", table_name="projects")
    op.drop_table("projects")
    op.drop_table("spaces")
