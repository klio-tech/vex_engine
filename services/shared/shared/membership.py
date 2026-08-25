"""May this user act in this org?

Once ``org_id`` is selected by the caller rather than derived from their key,
this is the tenancy boundary, and it has to be asked on every call rather than
once at switch time — a session outlives a membership revocation.

The engine holds no local users or memberships table on purpose: identity lives
in Supabase, and a local copy would be a cache with no invalidation.

**Two databases, joined in Python.** ``public.accounts`` and
``public.accounts_memberships`` live in Supabase (``SUPABASE_DATABASE_URL``);
``organizations`` lives in the engine's own database (``DATABASE_URL``). They
are separate servers, so a single SQL statement naming both cannot run —
however well it works against a test database that happens to hold both. This
mirrors ``app.oauth.GrantResolver._lookup``, which asks Supabase for
membership, asks the engine for the org, and joins the two answers itself.

Unlike every other engine query against ``accounts``, this one does NOT filter
``is_personal_account = false``. A user's personal account is their personal
node, and excluding it is exactly what made the engine org-rooted.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

#: Supabase side: which accounts is this user a member of?
_SLUGS_SQL = text(
    """
    SELECT a.slug AS slug,
           a.is_personal_account AS is_personal
    FROM public.accounts_memberships m
    JOIN public.accounts a ON a.id = m.account_id
    WHERE m.user_id = :user_id
    """
)

#: Engine side: which of those slugs have an org row?
_ORGS_SQL = text(
    """
    SELECT org_id, account_slug
    FROM organizations
    WHERE account_slug = ANY(:slugs)
    """
)


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def reachable_orgs(supabase: Any, tsdb: Any, *, user_id: Optional[str]) -> list[dict]:
    """Every org node this user may act in, personal node first.

    Args:
        supabase: A connection to the Supabase database (accounts, memberships).
        tsdb: A connection to the engine's own database (organizations).
        user_id: The Supabase user UUID, or ``None``.

    Returns:
        ``[{"org_id", "account_slug", "is_personal"}]``, personal node first
        then by slug. An account with no ``organizations`` row is omitted: the
        engine cannot store memory for a node it has no id for.

    An unattributable caller reaches nothing. Returning "everything" for a
    blank user id would turn a failure to identify someone into access to
    every tenant.
    """
    cleaned = _clean(user_id)
    if cleaned is None:
        return []

    rows = supabase.execute(_SLUGS_SQL, {"user_id": cleaned}).fetchall()
    if not rows:
        return []
    personal_by_slug = {r.slug: bool(r.is_personal) for r in rows}

    org_rows = tsdb.execute(_ORGS_SQL, {"slugs": list(personal_by_slug)}).fetchall()

    resolved = [
        {
            "org_id": o.org_id,
            "account_slug": o.account_slug,
            "is_personal": personal_by_slug.get(o.account_slug, False),
        }
        for o in org_rows
    ]
    # Personal node first, then by slug — a stable order so a caller listing
    # their nodes sees the same sequence every time.
    resolved.sort(key=lambda o: (not o["is_personal"], o["account_slug"]))
    return resolved


def may_reach(
    supabase: Any, tsdb: Any, *, user_id: Optional[str], org_id: Optional[str]
) -> bool:
    """Whether ``user_id`` may act in ``org_id``.

    Deliberately a fresh lookup rather than a cached set: this is the boundary,
    and the cost of it being stale is one tenant reading another's memory.
    """
    cleaned_user = _clean(user_id)
    cleaned_org = _clean(org_id)
    if cleaned_user is None or cleaned_org is None:
        return False
    return any(
        o["org_id"] == cleaned_org
        for o in reachable_orgs(supabase, tsdb, user_id=cleaned_user)
    )
