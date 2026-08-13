"""Four project roles: read, write, manage, admin.

031 stored ``member`` (see the project) and ``admin`` (change who sees it).
Writes were not gated on that column — any org key could persist into a
project. This revision makes the stored role mean what the Access modal
says:

* ``read``   — see / recall; cannot persist into the project brain
* ``write``  — persist (today's implicit ``member``)
* ``manage`` — grant/revoke read and write
* ``admin``  — grant/revoke any role; last admin is irremovable

Existing ``member`` rows become ``write`` so nobody who could already write
via an agent is silently locked to read-only. Audit rows are remapped the
same way so history uses the current vocabulary.

Revision ID: 040
Revises: 039
Create Date: 2026-08-13
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

revision: str = "040"
down_revision: Union[str, None] = "039"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

NEW_ROLES = "role IN ('read', 'write', 'manage', 'admin')"
OLD_ROLES = "role IN ('member', 'admin')"


def upgrade() -> None:
    op.drop_constraint("ck_project_members_role", "project_members", type_="check")
    op.drop_constraint(
        "ck_project_grant_audit_role", "project_grant_audit", type_="check"
    )

    op.execute(
        """
        UPDATE project_members
        SET role = 'write'
        WHERE role = 'member'
        """
    )
    op.execute(
        """
        UPDATE project_grant_audit
        SET role = 'write'
        WHERE role = 'member'
        """
    )

    op.create_check_constraint(
        "ck_project_members_role",
        "project_members",
        NEW_ROLES,
    )
    op.create_check_constraint(
        "ck_project_grant_audit_role",
        "project_grant_audit",
        NEW_ROLES,
    )


def downgrade() -> None:
    # Lossy: read and manage collapse back to member. Admin stays admin.
    op.drop_constraint("ck_project_members_role", "project_members", type_="check")
    op.drop_constraint(
        "ck_project_grant_audit_role", "project_grant_audit", type_="check"
    )

    op.execute(
        """
        UPDATE project_members
        SET role = CASE
            WHEN role = 'admin' THEN 'admin'
            ELSE 'member'
        END
        """
    )
    op.execute(
        """
        UPDATE project_grant_audit
        SET role = CASE
            WHEN role = 'admin' THEN 'admin'
            ELSE 'member'
        END
        """
    )

    op.create_check_constraint(
        "ck_project_members_role",
        "project_members",
        OLD_ROLES,
    )
    op.create_check_constraint(
        "ck_project_grant_audit_role",
        "project_grant_audit",
        OLD_ROLES,
    )
