"""Klio compression: cache of pre-compression originals.

Lossy compression is only acceptable if the original can be recovered, so this
table is the recovery contract. It is deliberately NOT part of session_memories:
compressing a log must never publish it into a team's shared brain, never be
curated, and never appear in recall.

Siloed by (org_id, user_id) exactly like memory — a handle belonging to another
user must be unreachable. user_id is nullable because an unattributable caller
still deserves a working compressor; such a row is simply unreachable by any
user-scoped read, which is the correct outcome rather than a special case.

Retention is 24 hours by default: retrieval happens within a session or not at
all. Storage mirrors artifacts — small originals inline, large ones in object
storage.

Revision ID: 034
Revises: 033
Create Date: 2026-08-01
"""

import sqlalchemy as sa
from alembic import op

revision = "034"
down_revision = "033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "compression_cache",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("user_id", sa.String(64), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("inline_text", sa.Text(), nullable=True),
        sa.Column("storage_uri", sa.Text(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    # Dedup: agents re-read the same file constantly. Scoped by org so one
    # tenant's content can never satisfy another's lookup.
    op.create_index(
        "ix_compression_cache_dedup",
        "compression_cache",
        ["org_id", "content_hash"],
    )
    # The expiry sweep scans by time and must not seq-scan the table.
    op.create_index(
        "ix_compression_cache_expires",
        "compression_cache",
        ["expires_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_compression_cache_expires", table_name="compression_cache")
    op.drop_index("ix_compression_cache_dedup", table_name="compression_cache")
    op.drop_table("compression_cache")
