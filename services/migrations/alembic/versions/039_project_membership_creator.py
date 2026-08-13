"""Projects are visible only to their members: record the creator, backfill.

Ruling, 2026-08-12 (see ``docs/superpowers/specs/2026-08-12-project-listing-
visibility-design.md``): *"Private by default — everything."* A project is
visible only to its members. The spec's git-remote exception (§4 — "remote-
backed projects stay org-visible") was considered and REJECTED. There is no
org-admin bypass in the engine.

That ruling is unimplementable today for one reason: **nobody's membership
exists**. ``project_members`` (migration 031) holds exactly 3 rows across the
whole production database, and ``projects`` (migration 020) has no column
that says who caused a row to exist. Ship the listing change without this
migration and every project — all ~118 of them in the org that motivated this
work — disappears for everyone, including the person who has been using it
daily. That is not a stricter product, it is an outage shaped like a feature.

This migration does two things, and the second is the one that keeps the
first from being a foot-gun:

1. **Add ``projects.created_by``** (nullable — a pre-existing row may have no
   honest answer; NEVER invent one). The engine (``shared.brain.ensure_project``,
   a companion code change alongside this migration) now sets it at INSERT
   time going forward, first-writer-wins.

2. **Backfill.** For every existing project, ``created_by`` becomes the
   ``user_id`` of its EARLIEST attributed ``session_memories`` row (NULL
   ``user_id`` rows are ignored when picking; a project with no attributed
   memory at all keeps ``created_by = NULL`` — this is a heuristic, not a
   fact, and a project nobody can be honestly attributed to must not get an
   invented owner). Every project that DOES get a ``created_by`` also gets a
   ``project_members`` row for that user with role ``'admin'`` — otherwise
   the backfilled ``created_by`` is decorative and the project is still
   invisible to the very person the column says created it.

   ``ON CONFLICT (project_id, user_id) DO NOTHING`` on that insert is
   load-bearing: the 3 pre-existing membership rows must never be touched —
   not clobbered, not demoted from whatever role they already hold — by a
   heuristic derived from memory authorship they may not agree with.

**A project with a NULL ``created_by`` gets no membership row and stays
invisible to everyone once the listing change ships.** That is the correct,
conservative outcome for a project this backfill cannot honestly attribute —
not a bug to work around by widening the guess.

**Downgrade** drops the column. It deliberately does NOT try to remove the
``project_members`` rows the backfill inserted: by the time anyone rolls this
back, those rows may be exactly the access grant a real person is relying on
(they may have used ``project_grant`` to add teammates on top of the
backfilled admin seat), and silently revoking access is not what "undo this
migration" should mean for a live ACL table — see 031/032/033's stance that a
downgrade must never be the thing that surprises someone about who can see
what. Re-running ``upgrade()`` after a downgrade re-derives the same
``created_by`` values from the same ``session_memories`` rows and re-inserts
via the same ``ON CONFLICT DO NOTHING``, so the round trip is idempotent:
nothing is duplicated and nothing already-admitted is downgraded.

Revision ID: 039
Revises: 038
Create Date: 2026-08-12
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "039"
down_revision: Union[str, None] = "038"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("created_by", sa.String(64), nullable=True),
    )

    # ---- backfill: earliest attributed session_memories.user_id per project
    #
    # DISTINCT ON (project_id) ... ORDER BY project_id, created_at ASC, id ASC
    # picks exactly one row per project — the earliest by wall-clock time,
    # with the row id as a deterministic tiebreak for same-instant writes.
    # NULL user_id rows are excluded by the WHERE, matching "ignore NULL
    # user_ids when picking" — a project written to only by unattributed
    # callers gets no candidate here and its created_by stays NULL, which the
    # UPDATE's WHERE-guard (created_by IS NULL, trivially true pre-backfill)
    # then simply never sets.
    op.execute(
        """
        UPDATE projects p
        SET created_by = earliest.user_id
        FROM (
            SELECT DISTINCT ON (project_id) project_id, user_id
            FROM session_memories
            WHERE project_id IS NOT NULL AND user_id IS NOT NULL
            ORDER BY project_id, created_at ASC, id ASC
        ) AS earliest
        WHERE p.id = earliest.project_id
          AND p.created_by IS NULL
        """
    )

    # ---- backfill: seat the derived creator as admin
    #
    # ON CONFLICT DO NOTHING is the whole safety property: the 3 rows that
    # already exist in project_members are never touched by this heuristic,
    # whatever role they currently hold.
    op.execute(
        """
        INSERT INTO project_members (project_id, user_id, role, granted_by)
        SELECT id, created_by, 'admin', created_by
        FROM projects
        WHERE created_by IS NOT NULL
        ON CONFLICT (project_id, user_id) DO NOTHING
        """
    )


def downgrade() -> None:
    # The project_members rows this migration's backfill inserted are left in
    # place on purpose — see the module docstring. Only the structural change
    # (the column) is reversed.
    op.drop_column("projects", "created_by")
