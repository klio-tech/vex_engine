"""Per-project default capture scope, and a per-member opt-out.

WHY THIS EXISTS. The 2026-08-01 user silo routed passive hook capture to
``scope='private'`` whenever the request could be attributed to a person. That
was deliberate and it fails closed. It also made the shared brain unreachable
in practice: promotion is per-memory and manual, and almost all volume arrives
through hooks. Measured 2026-08-18, the reference org held 6,543 active private
rows against 1 org-scoped row, with a single promotion in seventeen days.

The user-silo spec named this outcome as a risk and wrote its own reversal
condition: "promotions per user per week ... if it stays near zero, the model
is wrong and should change rather than be defended." This migration is that
change.

Columns:

* ``projects.default_scope`` — where captures FILED UNDER THIS PROJECT land.
  ``'private'`` is the default for every existing and new row, so applying this
  migration changes nothing observable. That property is the point: schema can
  land before code, and code can land before anyone opts in.

* ``project_members.scope_override`` — a member's own opt-out. NULL means
  "follow the project". ``'private'`` means this member's captures stay private
  in this project no matter what the project says. The CHECK deliberately
  permits ONLY ``'private'``: an override exists to narrow, never to widen. A
  member who could set ``'org'`` here would be publishing into a project whose
  administrators chose privacy, which is the same disclosure this column is
  meant to prevent, pointed the other way.

NO BACKFILL, EVER. Both columns default to today's behaviour. Nothing here
rewrites the scope of a single existing row, and nothing downstream may either:
those 6,543 rows were captured under a promise of privacy, and a setting that
retroactively publishes them is an incident rather than a feature.
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "047"
down_revision: Union[str, None] = "046"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column(
            "default_scope",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'private'"),
        ),
    )
    op.create_check_constraint(
        "ck_projects_default_scope",
        "projects",
        "default_scope IN ('private','project','org')",
    )

    op.add_column(
        "project_members",
        sa.Column("scope_override", sa.Text(), nullable=True),
    )
    op.create_check_constraint(
        "ck_project_members_scope_override",
        "project_members",
        "scope_override IS NULL OR scope_override = 'private'",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_project_members_scope_override", "project_members", type_="check"
    )
    op.drop_column("project_members", "scope_override")
    op.drop_constraint("ck_projects_default_scope", "projects", type_="check")
    op.drop_column("projects", "default_scope")
