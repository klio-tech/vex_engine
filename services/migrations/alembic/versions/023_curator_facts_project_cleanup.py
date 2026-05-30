"""Delete legacy unscoped curator output (cross-project bleed cleanup).

ADDITIVE / DATA-ONLY. The curator historically wrote its distilled
``summary``/``fact`` rows with ``project_id = NULL`` (no project scope), and
recall surfaces NULL-project rows in every project — so those facts leaked into
unrelated projects' sessions. The curator is now project-aware (it tags output
with the source partition's project). This one-time cleanup removes the legacy
unscoped curator output that is already bleeding.

Strictly scoped to curator output: only ``metadata->>'source' = 'curator'`` rows
of type ``summary``/``fact`` with ``project_id IS NULL`` are deleted. Raw
observations (any project) and genuine user ``remember`` notes
(``source != 'curator'``) are untouched. Idempotent: a re-run or a fresh DB
matches nothing. Downgrade is a no-op — the deleted rows are regenerable derived
data, so there is nothing to restore.

Revision ID: 023
Revises: 022
Create Date: 2026-05-31
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "023"
down_revision: Union[str, None] = "022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        DELETE FROM session_memories
        WHERE metadata->>'source' = 'curator'
          AND project_id IS NULL
          AND memory_type IN ('summary', 'fact')
        """
    )


def downgrade() -> None:
    # No-op: deleted rows are regenerable derived data; nothing to restore.
    pass
