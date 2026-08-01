"""User silo phase 1: a real user principal, project ACLs, provenance.

ADDITIVE ONLY. No column is dropped, no row is made less visible, and no
request behaves differently after this runs. Phase 2 is what starts routing
writes to the new scope; this migration only makes that possible.

``session_memories.user_id`` is NULLABLE on purpose. NULL means "we could not
attribute this write to a person", which is a legal, permanent state for the
37,605 rows that predate this migration and for any caller we cannot resolve.
A NOT NULL column would force us to invent an owner, and inventing an owner
for a private scope is how you leak one person's memories to another.

``scope`` had NO check constraint before this. A typo'd scope therefore created
rows that every query silently excluded — invisible, undeletable-by-UI, and
counted against quota. Introducing 'private' is the right moment to close that,
because the failure mode of a typo goes from "row is invisible" to "row that
should be private is visible".

**Provenance is called ``provenance``, not ``confidence``.** The spec asks for
a CHECK-constrained varchar named ``confidence``, but that name is already
taken: migration 016 created ``session_memories.confidence`` as a nullable
DOUBLE PRECISION. ``store_memory()`` writes a float into it on every write
path, ``_row_to_record`` reads it into ``MemoryRecord.confidence: float | None``
and ``mcp_app`` returns it verbatim in the ``recall`` response. Adding a second
column of that name is impossible; converting the existing one would destroy
data and change an observable API payload type, which phase 1 is not allowed to
do. The three labels, the CHECK constraint, the EXTRACTED default and the
curator backfill are exactly as specified — only the name moved.

**Backfill.** Exactly one: ``provenance``. Rows authored by the curator become
INFERRED, everything else EXTRACTED. This is safe because it adds information
to rows without changing who can read them. Memories are NOT backfilled to
private — see decision 6 in the spec: retroactively hiding a team's accumulated
brain is a worse surprise than leaving it.

Revision ID: 031
Revises: 030
Create Date: 2026-08-01
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "031"
down_revision = "030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---- identity -------------------------------------------------------
    op.add_column(
        "session_memories",
        sa.Column("user_id", sa.String(64), nullable=True),
    )

    # ---- provenance -----------------------------------------------------
    # Server default so existing rows land on EXTRACTED without a table
    # rewrite, and so a writer that forgets the column still produces a
    # truthful row rather than a NULL that reads as "unknown".
    op.add_column(
        "session_memories",
        sa.Column(
            "provenance",
            sa.String(16),
            nullable=False,
            server_default="EXTRACTED",
        ),
    )

    # The one backfill. Curator output is an LLM deduction; leaving it
    # indistinguishable from something a user stated is the exact failure
    # external review flagged ("if it starts driving inferences that aren't
    # correct").
    op.execute(
        """
        UPDATE session_memories
        SET provenance = 'INFERRED'
        WHERE metadata->>'source' = 'curator'
        """
    )

    # ---- constraints ----------------------------------------------------
    op.create_check_constraint(
        "ck_session_memories_scope",
        "session_memories",
        "scope IN ('private', 'project', 'org', 'session', 'agent')",
    )
    op.create_check_constraint(
        "ck_session_memories_provenance",
        "session_memories",
        "provenance IN ('EXTRACTED', 'INFERRED', 'AMBIGUOUS')",
    )

    # A private row with no owner is unreachable by every caller including its
    # author, so it is a bug by construction rather than a state to handle.
    op.create_check_constraint(
        "ck_session_memories_private_has_owner",
        "session_memories",
        "scope <> 'private' OR user_id IS NOT NULL",
    )

    # ---- recall index ---------------------------------------------------
    # Partial: private is a small slice today and the predicate is always
    # present on the private read path.
    op.execute(
        """
        CREATE INDEX ix_session_memories_private
        ON session_memories (org_id, user_id, created_at DESC)
        WHERE scope = 'private'
        """
    )

    # ---- project ACL ----------------------------------------------------
    op.create_table(
        "project_members",
        sa.Column("project_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("role", sa.String(32), nullable=False, server_default="member"),
        sa.Column("granted_by", sa.String(64), nullable=True),
        sa.Column(
            "granted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint("project_id", "user_id"),
    )
    op.create_check_constraint(
        "ck_project_members_role",
        "project_members",
        "role IN ('member', 'admin')",
    )
    # "which projects can this user read" is the Phase 2 recall predicate.
    op.create_index("ix_project_members_user", "project_members", ["user_id"])

    # ---- per-user curator cursor ---------------------------------------
    # The curator becomes per-user in Phase 2. Widening the PK now means Phase
    # 2 is a code change with no migration. Existing rows get '' rather than
    # NULL because NULL is not comparable in a PK.
    op.add_column(
        "brain_curator_state",
        sa.Column("user_id", sa.String(64), nullable=False, server_default=""),
    )
    op.drop_constraint(
        "brain_curator_state_pkey", "brain_curator_state", type_="primary"
    )
    op.create_primary_key(
        "brain_curator_state_pkey", "brain_curator_state", ["org_id", "user_id"]
    )


def downgrade() -> None:
    op.drop_constraint(
        "brain_curator_state_pkey", "brain_curator_state", type_="primary"
    )
    op.create_primary_key("brain_curator_state_pkey", "brain_curator_state", ["org_id"])
    op.drop_column("brain_curator_state", "user_id")

    op.drop_index("ix_project_members_user", table_name="project_members")
    op.drop_table("project_members")

    op.execute("DROP INDEX IF EXISTS ix_session_memories_private")
    op.drop_constraint(
        "ck_session_memories_private_has_owner", "session_memories", type_="check"
    )
    op.drop_constraint(
        "ck_session_memories_provenance", "session_memories", type_="check"
    )
    op.drop_constraint("ck_session_memories_scope", "session_memories", type_="check")
    op.drop_column("session_memories", "provenance")
    op.drop_column("session_memories", "user_id")
