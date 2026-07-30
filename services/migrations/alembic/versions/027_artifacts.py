"""Klio artifacts: durable files, documents and records.

ADDITIVE ONLY. Introduces ``artifacts``. Bytes live either inline
(``inline_text``, text <= 64KB) or in the ``klio-artifacts`` bucket
(``storage_uri``) — never both, never neither, once the row is active.

``size_bytes`` and ``checksum`` are NULL while ``status='pending'``: the
bytes do not exist yet, and a client's declared size is never stored. Both
are written from server-side measurement at confirm time.

Nothing existing is touched. Migration 028 amends the retention function so
cards for active artifacts survive the sweep.

Revision ID: 027
Revises: 026
Create Date: 2026-07-30
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "027"
down_revision: Union[str, None] = "026"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "artifacts",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("space_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("agent_id", sa.String(256), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("mime_type", sa.String(128), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("checksum", sa.String(64), nullable=True),
        sa.Column("inline_text", sa.Text(), nullable=True),
        sa.Column("storage_uri", sa.Text(), nullable=True),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("supersedes", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("status", sa.String(16), server_default="pending", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    # An active artifact has exactly one home for its bytes. A pending row has
    # neither yet. Violating this means a row that reads as present but has no
    # content, so it is a constraint rather than application discipline.
    op.execute(
        """
        ALTER TABLE artifacts ADD CONSTRAINT ck_artifacts_one_body CHECK (
            status <> 'active'
            OR (inline_text IS NULL) <> (storage_uri IS NULL)
        )
        """
    )
    op.execute(
        """
        ALTER TABLE artifacts ADD CONSTRAINT ck_artifacts_active_measured CHECK (
            status <> 'active'
            OR (size_bytes IS NOT NULL AND checksum IS NOT NULL)
        )
        """
    )
    op.execute(
        "ALTER TABLE artifacts ADD CONSTRAINT ck_artifacts_kind "
        "CHECK (kind IN ('document', 'record', 'file'))"
    )
    op.execute(
        "ALTER TABLE artifacts ADD CONSTRAINT ck_artifacts_status "
        "CHECK (status IN ('pending', 'active', 'superseded', 'retracted'))"
    )

    op.create_index("ix_artifacts_org_status", "artifacts", ["org_id", "status"])
    op.create_index(
        "ix_artifacts_org_space_status", "artifacts", ["org_id", "space_id", "status"]
    )
    # Deduplication: identical bytes in one org resolve to one artifact.
    op.execute(
        """
        CREATE UNIQUE INDEX ux_artifacts_org_checksum_active
        ON artifacts (org_id, checksum)
        WHERE status = 'active'
        """
    )
    # The GC pass sweeps pending rows by age.
    op.create_index(
        "ix_artifacts_status_created", "artifacts", ["status", "created_at"]
    )


def downgrade() -> None:
    # Cards would otherwise point at nothing.
    op.execute("DELETE FROM session_memories WHERE memory_type = 'artifact'")
    op.drop_table("artifacts")
