"""Artifact cards filed under a project must stop at that project.

**This migration NARROWS visibility, deliberately.** Every artifact card with
``memory_type='artifact' AND project_id IS NOT NULL AND scope='org'`` becomes
``scope='project'``: it stops being readable by the whole org and becomes
readable only by that project's members.

Why narrowing is the safe direction here, and not a data loss:

``artifact_put`` accepts a ``project`` argument and wrote ``project_id`` onto
the card, but hard-coded the card's ``scope`` to ``'org'``. On an org-scoped
row ``project_id`` is a relevance filter and explicitly NOT a boundary (see
the visibility ladder in ``shared.memory``), so a caller who named a project —
and could only reasonably read that as "share this with these people" — got an
artifact readable by every agent in the org instead. These rows are exactly
the set where the caller's stated intent and the stored scope disagree, and
this migration makes the storage match the intent. Nothing is deleted; the
rows keep their project, their content and their ids.

The scope of the narrowing is bounded by construction: only ``artifact`` cards
are touched, only those that already carry a ``project_id``, and only those
still at ``scope='org'``. Fact/decision/plan rows, org-wide artifacts (no
project), and anything already at another scope are left exactly as they are —
in particular the large population of ordinary ``scope='org'`` rows that carry
a ``project_id`` purely as a relevance hint, which must stay org-readable.

**Downgrade.** The ids this upgrade actually changed are recorded in
``session_memories_artifact_scope_backup`` first, and the downgrade restores
``scope='org'`` on exactly those ids before dropping the table. A blanket
"set every project-scoped artifact card back to org" would be wrong in the
one direction that matters: cards written at project scope AFTER this
migration were narrow *by intent*, and widening them on a rollback would leak
them org-wide — the very surprise this change exists to remove. Those rows are
therefore left narrow. A rolled-back deployment writes new cards at org scope
again and reads these older ones as project-scoped, which is restrictive but
never leaky.

Revision ID: 036
Revises: 035
Create Date: 2026-08-03
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

revision: str = "036"
down_revision: Union[str, None] = "035"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

BACKUP_TABLE = "session_memories_artifact_scope_backup"


def upgrade() -> None:
    # Record what we are about to change BEFORE changing it, so the downgrade
    # reverses this migration rather than guessing at it. Dropped first so a
    # re-run after a downgrade/upgrade cycle starts from a clean ledger.
    op.execute(f"DROP TABLE IF EXISTS {BACKUP_TABLE}")
    op.execute(
        f"""
        CREATE TABLE {BACKUP_TABLE} AS
        SELECT id
        FROM session_memories
        WHERE memory_type = 'artifact'
          AND project_id IS NOT NULL
          AND scope = 'org'
        """
    )
    op.execute(f"ALTER TABLE {BACKUP_TABLE} ADD PRIMARY KEY (id)")

    # The narrowing itself. Predicate is identical to the one above, so the
    # ledger and the update cover the same set within this transaction.
    op.execute(
        """
        UPDATE session_memories
        SET scope = 'project', updated_at = NOW()
        WHERE memory_type = 'artifact'
          AND project_id IS NOT NULL
          AND scope = 'org'
        """
    )


def downgrade() -> None:
    # Only the rows this migration narrowed go back. Rows written at project
    # scope afterwards were narrow by intent and are left alone: re-widening
    # them would publish org-wide exactly what a caller asked to keep inside a
    # project.
    op.execute(
        f"""
        UPDATE session_memories m
        SET scope = 'org', updated_at = NOW()
        FROM {BACKUP_TABLE} b
        WHERE m.id = b.id
          AND m.memory_type = 'artifact'
          AND m.scope = 'project'
        """
    )
    op.execute(f"DROP TABLE IF EXISTS {BACKUP_TABLE}")
