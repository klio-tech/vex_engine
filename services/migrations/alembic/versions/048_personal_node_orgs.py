"""Personal node orgs: a user's own account becomes an org row.

WHY THIS EXISTS. Four engine call sites filter ``is_personal_account = false``,
so a user's personal Supabase account resolves to no ``organizations`` row and
cannot be acted in. The personal node is the one part of the model that must
always exist: it is where solo agent work accumulates before any team is
involved.

``personal_owner_user_id`` records whose node a row is. The partial unique
index makes "one personal node per user" the database's job rather than a
convention -- a second one would silently split a person's own memory in two,
which is invisible until they notice something they wrote is gone.

The column is a bare ``String(64)`` holding a Supabase user UUID and is
deliberately NOT foreign-keyed: the engine holds no local users table, because
a local copy of Supabase identity would be a cache with no invalidation.

NO BACKFILL. Existing rows keep ``personal_owner_user_id IS NULL`` and are
unaffected; the partial index ignores them. Applying this migration changes
nothing observable until code calls ``ensure_personal_org``.

Revision ID: 048
Revises: 047
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "048"
down_revision: Union[str, None] = "047"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "organizations",
        sa.Column("personal_owner_user_id", sa.String(64), nullable=True),
    )
    # One personal node per user, enforced here rather than by convention: a
    # second one would silently split a person's own memory in two.
    op.create_index(
        "uq_organizations_personal_owner",
        "organizations",
        ["personal_owner_user_id"],
        unique=True,
        postgresql_where=sa.text("personal_owner_user_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_organizations_personal_owner", table_name="organizations")
    op.drop_column("organizations", "personal_owner_user_id")
