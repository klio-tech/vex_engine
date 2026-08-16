"""Second vector column for the E5-prefixed embedding space.

ADDITIVE ONLY. Nothing reads ``embedding_v2`` until an operator sets
``KLIO_EMBEDDING_SPACE=v2``, and nothing stops reading ``embedding`` at all.
This migration is safe to apply and leave applied indefinitely.

**Why a second column instead of re-embedding in place.** The production model
is ``intfloat/multilingual-e5-large-instruct``, which requires ``passage: ``
on documents and ``query: `` on searches. The engine sent bare text for both,
which compressed 18,367 memories into a 0.08-wide cosine band — the best match
in the whole corpus beat a random one by 0.085. Fixing the prefixes changes
every vector, and a prefixed query against an unprefixed corpus ranks *worse*
than the consistent-but-wrong status quo, so there is a window in which recall
is broken and the only question is how long.

It was measured, not estimated: 63,196 embedded rows, and the production
LiteLLM gateway sustains ~15.7 embeddings/second at concurrency 16 (0.68 s p50
per call). That is **≈67 minutes**. An hour of knowingly-degraded recall over
somebody's live memory is not a deploy window, so the corpus is re-embedded
into a second column while the first keeps serving, and the read path switches
by env var the moment the backfill reports complete. A half-finished backfill
leaves recall exactly as good as it was yesterday, which is the property that
made this the only acceptable option.

**Two steps, and the second one is the slow one.**

1. ``ADD COLUMN embedding_v2 vector(1024)`` — nullable, no default.
   Metadata-only: Postgres's fast-default path applies to a NULL default, so
   there is no table rewrite and no lock held beyond the catalogue update.
2. ``CREATE INDEX CONCURRENTLY`` for the HNSW index. On this table the v1
   index is 485 MB, and a plain ``CREATE INDEX`` holds ``ACCESS EXCLUSIVE``
   for its whole build — on the busiest table in the schema, at deploy time,
   that is an outage. ``CONCURRENTLY`` cannot run inside a transaction, and
   Alembic wraps every migration in one, so it runs inside
   ``op.get_context().autocommit_block()`` — the same escape hatch migration
   043 uses for its batched backfill.

   The index is built here, while the column is entirely NULL, rather than
   after the backfill. That makes the build itself instant (pgvector indexes
   no NULL rows) and lets the backfill maintain the index incrementally as it
   goes, which costs the backfill some throughput and buys the operator a
   sequence with no second migration step to forget.

RE-RUNNABLE. ``autocommit_block()`` commits step 1 the moment it is entered, so
a failure in step 2 leaves this migration partially applied while Alembic still
believes 046 never ran. Both steps are therefore guarded:

* the ``ADD COLUMN`` is skipped when the column already exists;
* ``CREATE INDEX CONCURRENTLY`` is the one DDL that can leave a *broken*
  artefact behind — an interrupted build leaves an ``indisvalid = false`` index
  that is never used by the planner and never repaired by re-running the same
  ``CREATE INDEX IF NOT EXISTS``. It is dropped and rebuilt when invalid, and
  skipped when valid.

**Storage.** ~259 MB of column (63,196 rows × 1024 dims × 4 bytes) plus an
HNSW index that will settle near the v1 index's 485 MB. Roughly 750 MB, on a
table that is already 920 MB. That is the price of the atomic switch, and it
is refunded by a follow-up migration that drops ``embedding`` once ``v2`` has
been the read path long enough to trust.

Revision ID: 046
Revises: 045
Create Date: 2026-08-16
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "046"
down_revision: Union[str, None] = "045"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "session_memories"
COLUMN = "embedding_v2"
INDEX = "ix_session_memories_embedding_v2_hnsw"

#: Matches migration 017's dimension for the v1 column. Same model, same
#: dimensionality — only the prompt prefix differs between the two spaces.
DIMENSIONS = 1024


def _column_exists(bind: sa.engine.Connection, table: str, column: str) -> bool:
    return bool(
        bind.execute(
            sa.text(
                """
                SELECT 1 FROM information_schema.columns
                WHERE table_name = :table AND column_name = :column
                """
            ),
            {"table": table, "column": column},
        ).first()
    )


def _index_state(bind: sa.engine.Connection, index: str) -> "str | None":
    """``"valid"``, ``"invalid"``, or ``None`` when the index does not exist.

    ``indisvalid`` is the field that separates "already built" from "a previous
    ``CREATE INDEX CONCURRENTLY`` died half way". The second state is the
    dangerous one: the index exists, so ``IF NOT EXISTS`` skips it, and the
    planner will never use it. It has to be dropped and rebuilt.
    """
    row = bind.execute(
        sa.text(
            """
            SELECT i.indisvalid
            FROM pg_class c
            JOIN pg_index i ON i.indexrelid = c.oid
            WHERE c.relname = :index
            """
        ),
        {"index": index},
    ).first()
    if row is None:
        return None
    return "valid" if row[0] else "invalid"


def upgrade() -> None:
    bind = op.get_bind()
    if not _column_exists(bind, TABLE, COLUMN):
        op.execute(f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} vector({DIMENSIONS})")

    # Entering the block commits the ADD COLUMN above and switches the
    # connection to autocommit, which is the only mode CREATE INDEX
    # CONCURRENTLY can run in.
    with op.get_context().autocommit_block():
        bind = op.get_bind()
        state = _index_state(bind, INDEX)
        if state == "invalid":
            bind.execute(sa.text(f"DROP INDEX IF EXISTS {INDEX}"))
            state = None
        if state is None:
            bind.execute(
                sa.text(
                    # m / ef_construction spelled out rather than left to
                    # pgvector's defaults, matching migration 017's v1 index
                    # exactly. The two indexes must be built the same way or
                    # a ranking difference after the switch is unattributable
                    # — it could be the prefixes or it could be the graph.
                    f"CREATE INDEX CONCURRENTLY {INDEX} "
                    f"ON {TABLE} USING hnsw ({COLUMN} vector_cosine_ops) "
                    f"WITH (m = 16, ef_construction = 64)"
                )
            )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.get_bind().execute(sa.text(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX}"))
    op.execute(f"ALTER TABLE {TABLE} DROP COLUMN IF EXISTS {COLUMN}")
