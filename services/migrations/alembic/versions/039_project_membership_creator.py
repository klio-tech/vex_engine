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

**Amended 2026-08-12, same day, before this migration ever ran anywhere.**
The above was ruled unacceptable: a read-only count against PRODUCTION with
this exact predicate found 134 projects, only 43 (32%) get a ``created_by``
from earliest-attributed-memory — **91 get none**, 75 of which still hold
memories. Older writes came from API keys issued before keys carried a user,
so those ``session_memories`` rows have ``user_id NULL`` and there is nothing
to attribute from. Shipping members-only listing on top of that makes those
91 projects vanish for everyone, including the workspace owner who has been
using one daily. That is an outage shaped like a feature, not a stricter
product — the same judgment that motivated this migration in the first
place, now applied one layer deeper.

**Abhishek's ruling, same day: a project with no attributable creator is
assigned to the workspace's owner**, not left permanently orphaned. This is
still not an invented fact about *who wrote the memories* — it is a
deliberate, disclosed policy decision about *who is accountable for a
project nobody else can be honestly attributed to*, and it only fires when
step 2 above yields nothing.

3. **Fallback: the workspace owner.** For every project that steps 1–2 left
   with ``created_by IS NULL``, resolve the owner of the Supabase account
   backing the project's ``org_id`` and use that as ``created_by`` instead.
   "Owner" here is the SAME notion of ownership the rest of the engine
   already uses for this exact account row — see
   ``services/mcp-server/app/directory.py`` and
   ``services/shared/shared/auth.py``'s ``_query_plan_from_supabase``, both
   of which prefer the immutable ``accounts.id`` (matched against ``org_id``
   when it parses as a UUID — most orgs today) and fall back to the mutable
   ``accounts.slug`` (via ``organizations.account_slug``, for the legacy
   orgs migration 003 backfilled with ``account_slug = org_id``) for a
   non-personal account. Within that account row, MakerKit's schema already
   names one canonical owner — ``accounts.primary_owner_user_id`` — so no
   further tie-break is needed in the common case. On the rare account whose
   ``primary_owner_user_id`` is unset (e.g. a deleted owner user), the
   deterministic fallback is the *earliest* ``accounts_memberships`` row for
   that account (by ``created_at``, then ``user_id`` as a tiebreak for a
   same-instant insert) — earliest membership approximates "the person who
   has been here longest," the same intuition migration 039's own
   session-memories heuristic uses for "earliest write wins."

   This lookup reaches into a second database (``SUPABASE_DATABASE_URL``),
   which no other migration has needed before. It fails closed at every
   step, matching every other cross-database read in this codebase: no
   ``SUPABASE_DATABASE_URL`` configured, no matching account row, or no
   resolvable owner on that account (no ``primary_owner_user_id`` AND no
   ``accounts_memberships`` rows) all mean the project's ``created_by``
   stays NULL — the org's owner cannot be honestly identified, so no owner
   is invented. That project remains an orphan, same as before this
   amendment, and is reported (not silently swallowed) by the caller
   inspecting the still-NULL count after ``upgrade()`` runs.

   The fallback seats its resolved owner exactly as step 2 seats a derived
   creator: one ``project_members`` row, role ``'admin'``, via the SAME
   ``ON CONFLICT (project_id, user_id) DO NOTHING`` insert — so a project
   that already has a membership row from ANY source (a pre-existing row,
   or a step-2 seat for the SAME user under a different path) is never
   touched twice, and a correctly-attributed project (step 1 succeeded) is
   never revisited by this fallback at all, because it never has
   ``created_by IS NULL`` in the first place.

**A project with a NULL ``created_by`` after both the derivation and the
fallback gets no membership row and stays invisible to everyone once the
listing change ships.** That is the correct, conservative outcome for a
project this migration truly cannot attribute to anyone, honestly or by
policy — not a bug to work around by widening the guess further.

**Downgrade** drops the column. It deliberately does NOT try to remove the
``project_members`` rows the backfill inserted (by either step 2 or step
3): by the time anyone rolls this back, those rows may be exactly the
access grant a real person is relying on (they may have used
``project_grant`` to add teammates on top of the backfilled admin seat), and
silently revoking access is not what "undo this migration" should mean for
a live ACL table — see 031/032/033's stance that a downgrade must never be
the thing that surprises someone about who can see what. Re-running
``upgrade()`` after a downgrade re-derives the same ``created_by`` values
from the same ``session_memories`` rows and the same Supabase owner lookup,
and re-inserts via the same ``ON CONFLICT DO NOTHING``, so the round trip is
idempotent: nothing is duplicated and nothing already-admitted is
downgraded.

Revision ID: 039
Revises: 038
Create Date: 2026-08-12
"""

import os
import uuid
from collections.abc import Sequence
from typing import Optional, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import create_engine, text

revision: str = "039"
down_revision: Union[str, None] = "038"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _is_uuid(value: Optional[str]) -> bool:
    """Mirrors ``shared.auth._is_uuid``: guards the ``accounts.id`` join.

    ``org_id`` is a free-form ``varchar(64)``; legacy rows hold slug-like
    values Postgres would reject when cast to ``uuid``.
    """
    if not value:
        return False
    try:
        uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def _resolve_workspace_owners(bind, org_ids: list[str]) -> dict[str, str]:
    """Resolve each of ``org_ids`` to its Supabase workspace owner's user id.

    Fails closed at every step (see the module docstring's step 3): a
    missing ``SUPABASE_DATABASE_URL``, an org with no ``account_slug``, an
    account row that cannot be found, or an account with neither a
    ``primary_owner_user_id`` nor any ``accounts_memberships`` row all
    simply omit that ``org_id`` from the returned mapping — never raise,
    never invent a value.
    """
    if not org_ids:
        return {}

    supabase_url = os.environ.get("SUPABASE_DATABASE_URL")
    if not supabase_url:
        return {}

    slug_rows = bind.execute(
        text(
            "SELECT org_id, account_slug FROM organizations "
            "WHERE org_id = ANY(:ids)"
        ),
        {"ids": list(org_ids)},
    ).fetchall()
    slug_by_org = {row.org_id: row.account_slug for row in slug_rows if row.account_slug}
    if not slug_by_org:
        return {}

    owners: dict[str, str] = {}
    engine = create_engine(supabase_url)
    try:
        with engine.connect() as sconn:
            for org_id, slug in slug_by_org.items():
                account = None
                if _is_uuid(org_id):
                    account = sconn.execute(
                        text(
                            "SELECT id, primary_owner_user_id FROM accounts "
                            "WHERE id = CAST(:account_id AS uuid) "
                            "AND is_personal_account = false LIMIT 1"
                        ),
                        {"account_id": org_id},
                    ).fetchone()
                if account is None:
                    account = sconn.execute(
                        text(
                            "SELECT id, primary_owner_user_id FROM accounts "
                            "WHERE slug = :slug AND is_personal_account = false "
                            "LIMIT 1"
                        ),
                        {"slug": slug},
                    ).fetchone()
                if account is None:
                    continue

                account_id, primary_owner = account[0], account[1]
                if primary_owner:
                    owners[org_id] = str(primary_owner)
                    continue

                # No primary owner recorded (e.g. a deleted owner user): the
                # deterministic fallback is the earliest membership on this
                # account, mirroring the "earliest wins" logic step 1 above
                # already uses for session_memories authorship.
                earliest_member = sconn.execute(
                    text(
                        "SELECT user_id FROM accounts_memberships "
                        "WHERE account_id = :account_id "
                        "ORDER BY created_at ASC, user_id ASC LIMIT 1"
                    ),
                    {"account_id": account_id},
                ).fetchone()
                if earliest_member is not None and earliest_member[0]:
                    owners[org_id] = str(earliest_member[0])
    finally:
        engine.dispose()

    return owners


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

    # ---- fallback: workspace owner, for whatever step 1 left unattributed
    #
    # Deliberately a SEPARATE pass over ONLY the still-NULL rows: step 1's
    # UPDATE already committed its results, so a project that resolved there
    # is never reconsidered here, however this fallback's own heuristic
    # would have answered. See the module docstring's step 3 for why this
    # exists and how "owner" is defined.
    bind = op.get_bind()
    orphans = bind.execute(
        text("SELECT id, org_id FROM projects WHERE created_by IS NULL")
    ).fetchall()
    if orphans:
        org_ids = sorted({row.org_id for row in orphans if row.org_id})
        owners = _resolve_workspace_owners(bind, org_ids)
        for row in orphans:
            owner = owners.get(row.org_id)
            if not owner:
                continue
            bind.execute(
                text(
                    "UPDATE projects SET created_by = :owner "
                    "WHERE id = :id AND created_by IS NULL"
                ),
                {"owner": owner, "id": row.id},
            )

    # ---- seat the resolved creator (step 1 OR the step-3 fallback) as admin
    #
    # ON CONFLICT DO NOTHING is the whole safety property: the 3 rows that
    # already existed in project_members before this migration first ran are
    # never touched by this heuristic, whatever role they currently hold —
    # and neither is a row a prior run of this same migration already
    # inserted, which is what makes re-running upgrade() idempotent.
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
