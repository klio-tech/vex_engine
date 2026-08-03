"""Make ``project_grant_audit`` able to record a REVOKE.

Migration 032 built the table around a single verb. Its docstring says so —
*every grant is audit-logged* — and the shape agrees: ``role`` is NOT NULL with
``CHECK (role IN ('member','admin'))`` and there is no column that says what
happened. A revoke therefore cannot be written down. The only rows the table
accepts are rows that read, to anyone querying it later, as a grant.

That is worse than a missing row. Writing a revoke as ``role = 'member'``
would put a *false* grant into the append-only record: reconstructing an
incident from this table would show access being handed out at the exact
moment it was taken away. And leaving revokes out entirely means the sequence
of ACL changes is not reconstructible at all — a member removed from a project
leaves no trace, which is precisely how an access change becomes invisible.

So: one column, ``action``, CHECK-constrained to the two verbs.

**Why the default is ``'grant'``.** The engine's ``record_grant_audit`` inserts
an explicit column list that does not mention ``action`` (see
``shared/projects.py``). A NOT NULL column with no default would break every
MCP grant the moment this migration applied and before the engine was
redeployed — an ACL change refused because its audit row would not insert. The
default makes the existing statement keep writing correct rows, so engine and
dashboard can be deployed in any order relative to this migration, and the
column can still be NOT NULL.

**``role`` stays NOT NULL and stays CHECK-constrained.** A revoke records the
role the member actually held when it was taken away, which is real
information — "Priya lost admin on this project" and "Priya lost member" are
different events — and it satisfies the existing constraint without weakening
it. Nothing about 032's constraint is touched here.

The table remains append-only: a revoke is a NEW row, never an update or a
delete of the grant it undoes.

**The downgrade refuses while a revoke is on record.** Dropping ``action``
does not merely remove a column: every revoke left in the table becomes
indistinguishable from a grant, which is the exact corruption described four
paragraphs up — an incident reconstruction that shows access being handed out
at the moment it was taken away. Set ``KLIO_ALLOW_AUDIT_LOSS=1`` to accept
that deliberately.

Rows that are all ``action = 'grant'`` downgrade freely, and so does an empty
table: the column's default puts ``'grant'`` back on the way up, so nothing is
lost and nothing is misrepresented. Refusing there would make the guard noise,
and a guard people route around by reflex protects nothing.

Revision ID: 033
Revises: 032
Create Date: 2026-08-01
"""

import os

import sqlalchemy as sa
from alembic import op

revision = "033"
down_revision = "032"
branch_labels = None
depends_on = None

AUDIT_LOSS_OVERRIDE = "KLIO_ALLOW_AUDIT_LOSS"


def upgrade() -> None:
    op.add_column(
        "project_grant_audit",
        sa.Column(
            "action",
            sa.String(16),
            nullable=False,
            server_default=sa.text("'grant'"),
        ),
    )
    op.create_check_constraint(
        "ck_project_grant_audit_action",
        "project_grant_audit",
        "action IN ('grant', 'revoke')",
    )
    # "when did this project last lose a member" is the query that motivated
    # the column, and it is answered from the same (project, time) prefix the
    # existing index already provides — so no new index. The action column is
    # a filter over an already-narrow slice, not an access path.


def downgrade() -> None:
    non_grants = (
        op.get_bind()
        .execute(
            sa.text("SELECT count(*) FROM project_grant_audit WHERE action <> 'grant'")
        )
        .scalar()
    )
    if non_grants and os.environ.get(AUDIT_LOSS_OVERRIDE) != "1":
        raise RuntimeError(
            f"Refusing to downgrade 033: dropping column project_grant_audit.action "
            f"would leave {non_grants} recorded revoke(s) indistinguishable from "
            f"grants. The rows survive, but they start asserting the opposite of "
            f"what happened — an audit trail that reads as access being granted at "
            f"the moment it was taken away is worse than a missing row. Export the "
            f"table first, e.g. `pg_dump --data-only --table=project_grant_audit`, "
            f"then re-run this downgrade with {AUDIT_LOSS_OVERRIDE}=1 to accept the "
            f"loss deliberately."
        )

    op.drop_constraint("ck_project_grant_audit_action", "project_grant_audit")
    op.drop_column("project_grant_audit", "action")
