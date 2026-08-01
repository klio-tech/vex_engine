"""User silo phase 2/3: an audit trail for project ACL changes.

``project_grant`` is reachable over MCP, which means an LLM can change who
reads a project's memories. The spec accepts that on purpose and holds the
blast radius with two constraints, one of which is this table: *every grant is
audit-logged — who, what, when, and via which surface*.

Append-only by intent. There is no update path and no delete path in the code;
a demotion or a re-grant writes a NEW row rather than editing the old one, so
the sequence of ACL changes stays reconstructible. That property is the point —
an audit trail you can rewrite answers a different question from the one asked
after an incident.

``granted_to_email`` is stored alongside ``granted_to`` deliberately, even
though it is redundant with the user UUID at the moment of the grant. The email
is what the caller actually typed, and an email can later be reassigned to a
different person or the account deleted outright; recording the UUID alone
would leave "who was this granted to" unanswerable, and recording the email
alone would leave it ambiguous. Both, frozen at grant time.

No foreign keys. ``granted_to`` / ``granted_by`` are Supabase user UUIDs that
live in a different database entirely, and ``project_id`` must remain readable
after a project row is gone — an audit row that disappears with the thing it
describes is not an audit row.

Revision ID: 032
Revises: 031
Create Date: 2026-08-01
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "032"
down_revision = "031"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "project_grant_audit",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("granted_to", sa.String(64), nullable=False),
        sa.Column("granted_to_email", sa.String(320), nullable=True),
        # Nullable: a grant made by an unattributable caller must still be
        # recorded. "We do not know who did this" is itself worth logging, and
        # is strictly better than dropping the row.
        sa.Column("granted_by", sa.String(64), nullable=True),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("surface", sa.String(32), nullable=False),
        sa.Column(
            "granted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_check_constraint(
        "ck_project_grant_audit_role",
        "project_grant_audit",
        "role IN ('member', 'admin')",
    )
    # "show me this project's ACL history" is the query an incident starts
    # with; "show me everything this org granted" is the second one.
    op.create_index(
        "ix_project_grant_audit_project",
        "project_grant_audit",
        ["project_id", "granted_at"],
    )
    op.create_index(
        "ix_project_grant_audit_org",
        "project_grant_audit",
        ["org_id", "granted_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_project_grant_audit_org", table_name="project_grant_audit")
    op.drop_index("ix_project_grant_audit_project", table_name="project_grant_audit")
    op.drop_table("project_grant_audit")
