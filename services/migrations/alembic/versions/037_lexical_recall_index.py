"""Lexical recall: a GIN full-text index over ``session_memories.content``.

Recall had one retrieval strategy, a dense vector fetch. Dense vectors have no
private direction for a rare literal — ``KLIO_METRICS_TOKEN`` is compressed
toward the neighbourhood of ``metrics`` and ``token`` — so the one row that
contains the identifier can rank below rows that are merely about the subject.
This index is the second arm.

**Why the expression is bounded at 100,000 characters, and why that is not a
detail.** ``content`` is unbounded ``TEXT``; the engine's existing truncation
applies only to embedding *input* and never to stored content. Postgres
refuses a ``tsvector`` larger than 1 MB. An index on the unbounded expression
therefore converts a write that succeeds today into a hard failure — measured
on pg16, not assumed::

    CREATE INDEX ... GIN (to_tsvector('english', content));
    INSERT ... 120k distinct tokens;
    ERROR:  string is too long for tsvector (1477986 bytes, max 1048575 bytes)

The same INSERT with ``left(content, 100000)`` in the expression returns
``INSERT 0 1`` and still stores all 1,088,894 characters. Adding search must
not break writing. The cost is that a literal appearing only past 100,000
characters of a single memory is not lexically searchable; the vector arm
still covers that row, and a tail that is merely semantically searchable is a
far better outcome than a rejected write.

The bound and the configuration are NOT written out here. They come from
``shared.lexical``, which the query side imports too. Postgres matches an
expression index by expression *equality*: if the two copies ever drifted by a
single character the index would silently stop being used, and the only
symptom would be a sequential scan nobody is watching for.

**No ``CONCURRENTLY``.** Alembic runs each migration inside a transaction and
``CREATE INDEX CONCURRENTLY`` cannot run in one. A plain build takes an
``ACCESS EXCLUSIVE`` lock for its duration, so the duration is what matters,
and it was measured locally against ``timescale/timescaledb-ha:pg16`` at the
production row count of ~46,000:

* representative corpus (51 MB of content, avg 1,153 chars/row):
  **5.4 s**, index 2.7 MB;
* deliberate worst case (111 MB, every token a distinct hash, avg 2,529
  chars/row): **20.4 s**, index 399 MB.

Production content is prose and repeats its vocabulary, so the first figure is
the expectation and the second is the ceiling. Both are short enough for a
deploy-time migration; neither rewrites the table.

**Downgrade** drops the index. Nothing else is touched — no column is added,
no row is modified — so the rollback is total and the re-upgrade is a rebuild.

Revision ID: 037
Revises: 036
Create Date: 2026-08-06
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

from shared.lexical import FTS_INDEX_NAME, fts_document_expression

revision: str = "037"
down_revision: Union[str, None] = "036"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "session_memories"


def upgrade() -> None:
    op.execute(
        f"CREATE INDEX {FTS_INDEX_NAME} ON {TABLE} "
        f"USING GIN ({fts_document_expression()})"
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {FTS_INDEX_NAME}")
