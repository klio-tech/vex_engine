"""Klio brain: curator cursor/state table.

ADDITIVE ONLY. Introduces ``brain_curator_state``, a per-org bookmark the
server-side curator uses to remember how far it has synthesised an org's RAW
brain memories. Nothing existing is touched: no columns are added to or removed
from any prior table, and the verification memory path is unaffected.

The curator (see ``services/mcp-server/app/curator.py``) reads only
``scope='org'`` RAW memories past the composite ``(last_processed_at,
last_processed_id)`` keyset cursor and advances the cursor after each batch, so
the bookmark is the sole bit of durable state it needs. The ``last_processed_id``
half is a UUID matching ``session_memories.id``; together with
``last_processed_at`` it gives a total, stable order that keyset pagination needs
to avoid skipping rows that share an identical ``created_at`` at a batch
boundary.

Revision ID: 021
Revises: 020
Create Date: 2026-05-28
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "021"
down_revision: Union[str, None] = "020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "brain_curator_state",
        sa.Column("org_id", sa.Text(), primary_key=True),
        sa.Column("last_processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_processed_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "last_run_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "processed_count",
            sa.BigInteger(),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("brain_curator_state")
