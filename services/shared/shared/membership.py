"""May this user act in this org?

Once ``org_id`` is selected by the caller rather than derived from their key,
this is the tenancy boundary, and it has to be asked on every call rather than
once at switch time -- a session outlives a membership revocation.

The query is the one already used by ``app.oauth.GrantResolver._lookup``,
lifted out so the key path and the OAuth path share one answer. The engine
holds no local users or memberships table on purpose: identity lives in
Supabase, and a local copy would be a cache with no invalidation.

Unlike every other engine query against ``accounts``, this one does NOT filter
``is_personal_account = false``. A user's personal account is their personal
node, and excluding it is exactly what made the engine org-rooted.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

_LIST_SQL = text(
    """
    SELECT o.org_id AS org_id,
           a.slug AS account_slug,
           a.is_personal_account AS is_personal
    FROM public.accounts_memberships m
    JOIN public.accounts a ON a.id = m.account_id
    JOIN organizations o ON o.account_slug = a.slug
    WHERE m.user_id = :user_id
    ORDER BY a.is_personal_account DESC, a.slug ASC
    """
)


def _clean(value: Optional[str]) -> Optional[str]:
    """Normalise an identifier, collapsing every unusable form to ``None``."""
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def reachable_orgs(db: Any, *, user_id: Optional[str]) -> list[dict]:
    """Every org node this user may act in, personal node first.

    An unattributable caller reaches nothing. Returning "everything" for a
    blank user id would turn a failure to identify someone into access to
    every tenant.
    """
    cleaned = _clean(user_id)
    if cleaned is None:
        logger.debug("membership: blank user id reaches no org")
        return []
    rows = db.execute(_LIST_SQL, {"user_id": cleaned}).fetchall()
    return [
        {
            "org_id": r.org_id,
            "account_slug": r.account_slug,
            "is_personal": bool(r.is_personal),
        }
        for r in rows
    ]


def may_reach(db: Any, *, user_id: Optional[str], org_id: Optional[str]) -> bool:
    """Whether ``user_id`` may act in ``org_id``.

    Deliberately a fresh query rather than a cached set: this is the boundary,
    and the cost of it being stale is one tenant reading another's memory.
    """
    cleaned_user = _clean(user_id)
    cleaned_org = _clean(org_id)
    if cleaned_user is None or cleaned_org is None:
        return False
    return any(
        org["org_id"] == cleaned_org for org in reachable_orgs(db, user_id=cleaned_user)
    )
