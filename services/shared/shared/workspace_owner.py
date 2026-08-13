"""Resolve a Vex org to its Supabase workspace owner, and its live membership.

This is the ONE place that knows how to turn ``organizations.org_id`` /
``account_slug`` into a Supabase ``accounts`` row, and from there into "who
owns this workspace" and "who currently belongs to it". Two call sites need
exactly this, and neither may drift from the other:

* **Migration 039** (``services/migrations/alembic/versions/
  039_project_membership_creator.py``) used this to backfill ``created_by``
  for the ~91 projects migration 039 could not attribute to a memory author.
  It now imports :func:`resolve_workspace_owners` from here instead of
  carrying its own copy.
* **The project-ownership-transfer sweep** (``services/async-worker/app/
  ownership_sweep.py``) uses both functions here: :func:`resolve_workspace_owners`
  to find who a departed sole owner's project transfers to, and
  :func:`fetch_workspace_membership` to learn who has actually departed in the
  first place.

Two definitions of "owner" that disagree would mean a project could be
attributed to one person by the migration and transferred to a different
person by the sweep — an inconsistency nobody could explain from the data
alone. Hence one function, imported twice, not two functions maintained in
parallel.

**"Owner" is the SAME notion the rest of the engine already uses for this
exact account row** — see ``services/mcp-server/app/directory.py`` and
``services/shared/shared/auth.py``'s ``_query_plan_from_supabase``: prefer the
immutable ``accounts.id`` (matched against ``org_id`` when it parses as a
UUID — most orgs today) and fall back to the mutable ``accounts.slug`` (via
``organizations.account_slug``, for the legacy orgs migration 003 backfilled
with ``account_slug = org_id``), restricted to a non-personal account.
MakerKit's schema already names one canonical owner —
``accounts.primary_owner_user_id`` — so no further tie-break is needed in the
common case. On the rare account whose ``primary_owner_user_id`` is unset
(e.g. a deleted owner user), the deterministic fallback is the *earliest*
``accounts_memberships`` row for that account (by ``created_at``, then
``user_id`` as a tiebreak for a same-instant insert).

Both functions fail closed at every step, matching every other cross-database
read in this codebase (``app/directory.py``, ``shared/auth.py``): no
``SUPABASE_DATABASE_URL``/engine, no matching account row, or no resolvable
owner all mean "cannot answer" — never an invented value.
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine


class WorkspaceLookupError(Exception):
    """The Supabase query itself failed (connection, timeout, syntax, ...).

    Distinct from "the query succeeded and found nothing" — see
    :func:`fetch_workspace_membership`'s docstring for why callers must treat
    the two differently. A caller that catches this must skip the org
    entirely, never proceed as though membership were checked and came back
    empty.
    """


def _is_uuid(value: Optional[str]) -> bool:
    """Return True when ``value`` is a well-formed UUID string.

    Guards the ``accounts.id`` join: ``org_id`` is a free-form ``varchar(64)``
    and legacy rows hold slug-like values, which Postgres would reject when
    compared against a ``uuid`` column. Mirrors ``shared.auth._is_uuid`` and
    the migration-039 copy this module replaces; kept local rather than
    imported so this module has no dependency on ``shared.auth`` for a
    ten-line format check.
    """
    if not value:
        return False
    try:
        uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def _find_account(conn: Connection, *, org_id: Optional[str], slug: Optional[str]):
    """Return the ``(id, primary_owner_user_id)`` row for this org's Supabase
    account, or ``None`` when no non-personal account matches either the
    UUID-preferring ``id`` join or the ``slug`` fallback.
    """
    account = None
    if _is_uuid(org_id):
        account = conn.execute(
            text(
                "SELECT id, primary_owner_user_id FROM accounts "
                "WHERE id = CAST(:account_id AS uuid) "
                "AND is_personal_account = false LIMIT 1"
            ),
            {"account_id": org_id},
        ).fetchone()
    if account is None and slug:
        account = conn.execute(
            text(
                "SELECT id, primary_owner_user_id FROM accounts "
                "WHERE slug = :slug AND is_personal_account = false LIMIT 1"
            ),
            {"slug": slug},
        ).fetchone()
    return account


def _supabase_engine_from_env() -> Optional[Engine]:
    supabase_url = os.environ.get("SUPABASE_DATABASE_URL")
    if not supabase_url:
        return None
    return create_engine(supabase_url)


def resolve_workspace_owners(
    bind, org_ids: list[str], *, supabase_engine: Optional[Engine] = None
) -> dict[str, str]:
    """Resolve each of ``org_ids`` to its Supabase workspace owner's user id.

    ``bind`` is a connection/session against the MAIN database — it is used
    only to read ``organizations.account_slug`` for the given ``org_ids``.
    ``supabase_engine``, when given, is reused as-is (the sweep builds one
    engine for a whole run rather than one per call). When omitted, an engine
    is built from ``SUPABASE_DATABASE_URL`` and disposed before returning —
    migration 039's original, one-shot calling convention.

    Fails closed at every step: a missing ``SUPABASE_DATABASE_URL`` (when no
    ``supabase_engine`` is supplied), an org with no ``account_slug``, an
    account row that cannot be found, or an account with neither a
    ``primary_owner_user_id`` nor any ``accounts_memberships`` row all simply
    omit that ``org_id`` from the returned mapping — never raise, never
    invent a value.
    """
    if not org_ids:
        return {}

    engine = supabase_engine
    owns_engine = False
    if engine is None:
        engine = _supabase_engine_from_env()
        if engine is None:
            return {}
        owns_engine = True

    slug_rows = bind.execute(
        text("SELECT org_id, account_slug FROM organizations WHERE org_id = ANY(:ids)"),
        {"ids": list(org_ids)},
    ).fetchall()
    slug_by_org = {
        row.org_id: row.account_slug for row in slug_rows if row.account_slug
    }
    if not slug_by_org:
        return {}

    owners: dict[str, str] = {}
    try:
        with engine.connect() as sconn:
            for org_id, slug in slug_by_org.items():
                account = _find_account(sconn, org_id=org_id, slug=slug)
                if account is None:
                    continue

                account_id, primary_owner = account[0], account[1]
                if primary_owner:
                    owners[org_id] = str(primary_owner)
                    continue

                # No primary owner recorded (e.g. a deleted owner user): the
                # deterministic fallback is the earliest membership on this
                # account, mirroring "earliest wins" elsewhere in this file.
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
        if owns_engine:
            engine.dispose()

    return owners


def fetch_workspace_membership(
    supabase_engine: Engine, *, org_id: Optional[str], account_slug: Optional[str]
) -> set[str]:
    """Return the CURRENT set of Supabase user ids belonging to this org's
    workspace account.

    This is the departure oracle for the ownership-transfer sweep: a
    ``project_members.user_id`` absent from this set has left the workspace.
    Getting this wrong in the "everyone left" direction is the failure mode
    the sweep's whole safety design exists to prevent, so the two ways this
    can go wrong are deliberately NOT collapsed into one return value:

    * The query itself fails (network blip, bad credentials, Supabase down)
      -> raises :class:`WorkspaceLookupError`. The caller must skip the org
      entirely — an error is not evidence that nobody is a member.
    * The query succeeds but finds no account, or an account with zero
      ``accounts_memberships`` rows -> returns an EMPTY set. The caller must
      treat this the same as a failure for the purpose of the sweep (skip the
      org, log loudly): a real workspace always has at least its owner as a
      member, so an empty result means something is wrong upstream (a
      mis-provisioned account, a bad slug), not that every member departed at
      once.

    Raises :class:`WorkspaceLookupError` rather than returning ``set()`` on
    failure specifically so a caller cannot accidentally treat "the query
    blew up" the same as "the query answered zero" via a shared code path —
    the two must be handled by explicitly different branches.
    """
    try:
        with supabase_engine.connect() as sconn:
            account = _find_account(sconn, org_id=org_id, slug=account_slug)
            if account is None:
                return set()
            account_id = account[0]
            rows = sconn.execute(
                text(
                    "SELECT user_id FROM accounts_memberships "
                    "WHERE account_id = :account_id"
                ),
                {"account_id": account_id},
            ).fetchall()
            return {str(row[0]) for row in rows if row[0]}
    except Exception as exc:
        raise WorkspaceLookupError(
            f"Supabase membership query failed for org_id={org_id!r}: {exc}"
        ) from exc
