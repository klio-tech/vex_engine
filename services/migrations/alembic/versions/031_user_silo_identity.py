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

**Rolling back must leave a database this migration can re-apply.** The first
version of this file was a one-way door: the downgrade dropped ``user_id``
while leaving ``scope = 'private'`` rows behind, so the next roll-forward died
on ``ck_session_memories_private_has_owner`` and alembic was stuck at 030 until
someone hand-edited production rows. Three states had to be made survivable,
and each is handled by preserving information rather than by relaxing a
constraint:

1. **Private ownership.** The downgrade copies every non-NULL ``user_id`` into
   ``session_memories_user_backup`` before dropping the column, and the upgrade
   restores from it and then drops the backup. ``scope`` is left as
   ``'private'`` on purpose: 030 has no scope constraint, and no 030-era query
   selects ``scope = 'private'``, so the rows stay unreadable exactly as they
   were. Rewriting them to ``'org'`` would make one person's memories visible
   to their whole team — a downgrade must not widen access.

2. **Typo'd scopes.** The docstring below has always said these exist in
   production, and the upgrade used to abort on them. It now quarantines them:
   the whole row is copied into ``session_memories_quarantine`` as JSON and
   removed from the live table. These rows are already unreachable by every
   query — that is what the missing constraint caused — so setting them aside
   loses no reachable data, while rewriting their scope to a valid value would
   suddenly surface content nobody has read in months.

3. **Per-user curator rows.** The downgrade stashes every
   ``brain_curator_state`` row and keeps one per org, chosen deterministically
   (furthest-advanced cursor, ties broken by ``user_id``), so restoring the
   single-column primary key cannot raise ``UniqueViolation``. The upgrade
   re-labels the survivor with its original owner and re-inserts the rest.

The round trip is therefore idempotent and repeatable: 035 -> 030 -> 035 any
number of times converges on the same rows.

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

VALID_SCOPES = ("private", "project", "org", "session", "agent")

#: Forensic table. Rows land here when the upgrade cannot legally keep them in
#: ``session_memories``. Never written to by application code, never read on a
#: request path — it exists so that "we dropped it" is never the answer.
QUARANTINE_TABLE = "session_memories_quarantine"

#: Transient. Written by :func:`downgrade`, consumed and dropped by
#: :func:`upgrade`. Its lifetime is exactly one roll-back window.
MEMORY_USER_BACKUP = "session_memories_user_backup"
CURATOR_USER_BACKUP = "brain_curator_state_user_backup"

#: One row per org survives the downgrade. Which one is not a coin flip: the
#: furthest-advanced cursor wins, because resuming from an earlier cursor
#: re-processes memories while resuming from a later one skips them, and
#: re-processing is the recoverable mistake.
_CURATOR_WINNER = (
    "ROW_NUMBER() OVER ("
    " PARTITION BY org_id"
    " ORDER BY last_processed_at DESC NULLS LAST,"
    "          last_run_at DESC,"
    "          processed_count DESC,"
    "          user_id ASC"
    ") = 1"
)


def _table_exists(name: str) -> bool:
    return bool(
        op.get_bind()
        .execute(sa.text("SELECT to_regclass(:n) IS NOT NULL"), {"n": name})
        .scalar()
    )


def _table_has_rows(name: str) -> bool:
    return bool(
        op.get_bind().execute(sa.text(f"SELECT EXISTS (SELECT 1 FROM {name})")).scalar()
    )


def _quarantine(where: str, reason: str) -> None:
    """Move every ``session_memories`` row matching ``where`` out of the way.

    The full row is preserved as JSON, so a DBA can reconstruct or replay it;
    only its presence in the constrained table is given up.
    """
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {QUARANTINE_TABLE} (
            quarantine_id  BIGSERIAL PRIMARY KEY,
            memory_id      UUID NOT NULL,
            reason         TEXT NOT NULL,
            quarantined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            payload        JSONB NOT NULL
        )
        """
    )
    op.execute(
        f"""
        INSERT INTO {QUARANTINE_TABLE} (memory_id, reason, payload)
        SELECT m.id, '{reason}', to_jsonb(m)
        FROM session_memories m
        WHERE {where}
        """
    )
    op.execute(f"DELETE FROM session_memories m WHERE {where}")


def upgrade() -> None:
    # ---- identity -------------------------------------------------------
    op.add_column(
        "session_memories",
        sa.Column("user_id", sa.String(64), nullable=True),
    )

    # Re-roll-forward after a rollback: give the private memories their owners
    # back before anything asserts that they have one. Absent on a first run.
    if _table_exists(MEMORY_USER_BACKUP):
        op.execute(
            f"""
            UPDATE session_memories m
            SET user_id = b.user_id
            FROM {MEMORY_USER_BACKUP} b
            WHERE m.id = b.memory_id AND m.user_id IS NULL
            """
        )
        op.execute(f"DROP TABLE {MEMORY_USER_BACKUP}")

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

    # ---- quarantine before constraining --------------------------------
    # A migration that aborts on pre-existing bad rows is a migration that
    # never runs. Both classes below are already unreachable by every query,
    # so moving them aside changes nothing observable — and unlike rewriting
    # their scope to something valid, it cannot make hidden content visible.
    _quarantine(
        "m.scope NOT IN (" + ", ".join(f"'{s}'" for s in VALID_SCOPES) + ")",
        "unknown_scope",
    )
    _quarantine(
        "m.scope = 'private' AND m.user_id IS NULL",
        "private_without_owner",
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

    # Re-roll-forward after a rollback. The row that survived the downgrade may
    # have advanced its cursor while the schema was at 030, so it is re-labelled
    # in place rather than overwritten by the stashed copy; the rows that were
    # collapsed away are re-inserted verbatim.
    if _table_exists(CURATOR_USER_BACKUP):
        op.execute(
            f"""
            UPDATE brain_curator_state s
            SET user_id = b.user_id
            FROM {CURATOR_USER_BACKUP} b
            WHERE b.collapsed_into_org_row
              AND s.org_id = b.org_id
              AND s.user_id = ''
            """
        )
        op.execute(
            f"""
            INSERT INTO brain_curator_state
                (org_id, user_id, last_processed_at, last_processed_id,
                 last_run_at, processed_count)
            SELECT org_id, user_id, last_processed_at, last_processed_id,
                   last_run_at, processed_count
            FROM {CURATOR_USER_BACKUP}
            WHERE NOT collapsed_into_org_row
            ON CONFLICT (org_id, user_id) DO NOTHING
            """
        )
        op.execute(f"DROP TABLE {CURATOR_USER_BACKUP}")


def downgrade() -> None:
    # ---- per-user curator cursor ---------------------------------------
    # Restoring a (org_id) primary key over per-user rows raises
    # UniqueViolation the moment Phase 2 has written a second row for any org.
    # Stash every row, keep the furthest-advanced one per org, and let the
    # upgrade put the rest back.
    op.execute(f"DROP TABLE IF EXISTS {CURATOR_USER_BACKUP}")
    op.execute(
        f"""
        CREATE TABLE {CURATOR_USER_BACKUP} AS
        SELECT org_id, user_id, last_processed_at, last_processed_id,
               last_run_at, processed_count,
               {_CURATOR_WINNER} AS collapsed_into_org_row
        FROM brain_curator_state
        """
    )
    op.execute(
        f"""
        ALTER TABLE {CURATOR_USER_BACKUP}
        ADD PRIMARY KEY (org_id, user_id)
        """
    )
    op.execute(
        f"""
        DELETE FROM brain_curator_state s
        USING {CURATOR_USER_BACKUP} b
        WHERE s.org_id = b.org_id
          AND s.user_id = b.user_id
          AND NOT b.collapsed_into_org_row
        """
    )

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

    # ---- who owned the private memories --------------------------------
    # Dropping user_id without this is what wedged the chain: the private rows
    # survive, their owners do not, and the next roll-forward dies on
    # ck_session_memories_private_has_owner with no way to repair it from
    # inside the database. The rows keep scope = 'private' — 030 has no scope
    # constraint and no 030-era query selects it, so they stay exactly as
    # unreadable as they were, which is the only safe direction for a rollback
    # to move access in.
    op.execute(f"DROP TABLE IF EXISTS {MEMORY_USER_BACKUP}")
    op.execute(
        f"""
        CREATE TABLE {MEMORY_USER_BACKUP} AS
        SELECT id AS memory_id, user_id, scope
        FROM session_memories
        WHERE user_id IS NOT NULL
        """
    )
    op.execute(f"ALTER TABLE {MEMORY_USER_BACKUP} ADD PRIMARY KEY (memory_id)")

    op.drop_column("session_memories", "user_id")

    # The quarantine table is forensic and outlives the migration that filled
    # it — dropping evidence on the way down is how the evidence stops being
    # worth keeping. An empty one carries no information, so it goes.
    if _table_exists(QUARANTINE_TABLE) and not _table_has_rows(QUARANTINE_TABLE):
        op.execute(f"DROP TABLE {QUARANTINE_TABLE}")
