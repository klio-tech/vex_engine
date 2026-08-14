"""Optimistic-concurrency revision on session_memories (Build 3).

ADDITIVE ONLY. ``revision`` is a UUID that changes every time a row's belief
content is superseded through the CAS path (:func:`shared.memory.
supersede_with_revision`). A caller that read a memory's ``revision`` on
recall and passes it back as ``base_revision`` on a write gets an atomic
guarantee: the write only lands if nobody else moved the belief in between.

Four steps, not one ``ADD COLUMN ... NOT NULL DEFAULT gen_random_uuid()``:
``gen_random_uuid()`` is VOLATILE, so Postgres cannot use the fast-default
path (constant defaults only) and instead rewrites every existing row under
an ``ACCESS EXCLUSIVE`` lock — on this table, the busiest one in the schema,
that is an outage. Instead:

1. ``ADD COLUMN revision UUID`` — nullable, no default. Metadata-only, no
   table rewrite, no lock held beyond the DDL itself.
2. Batched backfill — ``UPDATE ... WHERE revision IS NULL LIMIT :batch_size``
   in a loop until no rows remain. Each batch is its own short transaction,
   so no single lock is held for the duration of the whole table.
3. ``ALTER COLUMN ... SET DEFAULT gen_random_uuid()`` — metadata-only; every
   row inserted from here on gets one automatically.
4. ``ALTER COLUMN ... SET NOT NULL`` — by now every row already has a value,
   so this is a validation scan, not a rewrite.

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

#: Rows touched per backfill transaction. Small enough that one batch's lock
#: is never the long pole against a live write workload; large enough that a
#: multi-million-row table backfills in a bounded number of round trips.
_BACKFILL_BATCH_SIZE = 5000


def upgrade() -> None:
    op.add_column(
        "session_memories",
        sa.Column("revision", postgresql.UUID(as_uuid=True), nullable=True),
    )

    bind = op.get_bind()
    while True:
        result = bind.execute(
            sa.text(
                """
                UPDATE session_memories
                SET revision = gen_random_uuid()
                WHERE id IN (
                    SELECT id FROM session_memories
                    WHERE revision IS NULL
                    LIMIT :batch_size
                )
                """
            ),
            {"batch_size": _BACKFILL_BATCH_SIZE},
        )
        if result.rowcount == 0:
            break

    op.alter_column(
        "session_memories",
        "revision",
        server_default=sa.text("gen_random_uuid()"),
    )
    op.alter_column(
        "session_memories",
        "revision",
        nullable=False,
    )


def downgrade() -> None:
    op.drop_column("session_memories", "revision")
