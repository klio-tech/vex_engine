"""Optimistic-concurrency revision on session_memories (Build 3).

ADDITIVE ONLY. ``revision`` is a UUID that changes every time a row's belief
content is superseded through the CAS path (:func:`shared.memory.
supersede_with_revision`). A caller that read a memory's ``revision`` on
recall and passes it back as ``base_revision`` on a write gets an atomic
guarantee: the write only lands if nobody else moved the belief in between.
Existing rows get a value from ``gen_random_uuid()`` via the column's
``server_default`` — no backfill script needed, and every future write bumps
it going forward.

``gen_random_uuid()`` is already relied on elsewhere in this schema (migration
020's ``spaces``/``projects`` primary keys) with no ``CREATE EXTENSION
pgcrypto`` anywhere in this history. Postgres 13+ ships ``gen_random_uuid()``
in core, so no extension is created here either — consistent with what the
repo already does.

Revision ID: 043
Revises: 042
Create Date: 2026-08-13
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "043"
down_revision: Union[str, None] = "042"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "session_memories",
        sa.Column(
            "revision",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("session_memories", "revision")
