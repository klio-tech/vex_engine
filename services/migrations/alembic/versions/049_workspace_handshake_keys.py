"""Workspace handshake keys: the immutable Plane <-> engine join.

Revision ID: 049
Revises: 048
"""

import sqlalchemy as sa
from alembic import op

revision = "049"
down_revision = "048"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # UUID, not String: Plane ids are uuid4 and the column should refuse
    # anything that is not one. Nullable because every existing org predates
    # the handshake; unique where present because the match is one-to-one.
    op.add_column(
        "organizations",
        sa.Column(
            "workspace_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=True
        ),
    )
    op.create_index(
        "uq_organizations_workspace_id",
        "organizations",
        ["workspace_id"],
        unique=True,
        postgresql_where=sa.text("workspace_id IS NOT NULL"),
    )
    op.add_column(
        "projects",
        sa.Column(
            "plane_project_id",
            sa.dialects.postgresql.UUID(as_uuid=False),
            nullable=True,
        ),
    )
    op.create_index(
        "uq_projects_plane_project_id",
        "projects",
        ["plane_project_id"],
        unique=True,
        postgresql_where=sa.text("plane_project_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_projects_plane_project_id", table_name="projects")
    op.drop_column("projects", "plane_project_id")
    op.drop_index("uq_organizations_workspace_id", table_name="organizations")
    op.drop_column("organizations", "workspace_id")
