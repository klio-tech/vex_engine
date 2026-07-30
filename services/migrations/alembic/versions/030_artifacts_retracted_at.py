"""Klio artifacts: make the purge grace window mean what it says.

ADDITIVE ONLY. Adds ``artifacts.retracted_at`` and an index to support the
purge sweep.

``PURGE_GRACE_HOURS`` exists so an accidental ``forget`` can still be undone:
the bytes survive a week before the sweep destroys them. The window was keyed
off ``created_at``, which only delivers that for content younger than the
window. For everything older — which is most real content — the grace period
was ZERO: an artifact created eight days ago and retracted one minute ago was
purgeable on the very next sweep tick, so the single tool call the constant
exists to protect against was in fact unrecoverable the moment it landed. That
is the opposite of "errs conservative", and it fails silently and exactly for
the mature, valuable documents an undelete matters most for.

``retracted_at`` records when a row actually became dead, so the window can
key off death rather than birth.

**Backfill.** Rows that are ALREADY dead when this migration runs have no
recorded death time, and inventing one (``now()``) would hand every
long-retracted row a fresh week of storage the customer did not ask to keep
paying for. They are left NULL and ``collect_purgeable`` reads
``COALESCE(retracted_at, created_at)``, preserving today's behaviour for
exactly that finite, pre-existing set and applying the real window to every
row retracted from here on.

Revision ID: 030
Revises: 029
Create Date: 2026-07-30
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "030"
down_revision: Union[str, None] = "029"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "artifacts",
        sa.Column("retracted_at", sa.DateTime(timezone=True), nullable=True),
    )

    # ``collect_purgeable`` scans dead rows by effective death time. Partial on
    # the dead statuses because live rows are the overwhelming majority and
    # never match this predicate.
    op.execute(
        """
        CREATE INDEX ix_artifacts_dead_retracted_at
        ON artifacts (COALESCE(retracted_at, created_at))
        WHERE status IN ('retracted', 'superseded')
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_artifacts_dead_retracted_at")
    op.drop_column("artifacts", "retracted_at")
