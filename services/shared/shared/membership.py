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


#: Is there already a personal node for this user?
_PERSONAL_BY_OWNER_SQL = text(
    """
    SELECT org_id
    FROM organizations
    WHERE personal_owner_user_id = :user_id
    LIMIT 1
    """
)

#: ``account_slug`` is UNIQUE, so a row may already exist for this slug from
#: some earlier path. Look before inserting rather than catching IntegrityError.
_ORG_BY_SLUG_SQL = text(
    """
    SELECT org_id, personal_owner_user_id
    FROM organizations
    WHERE account_slug = :account_slug
    LIMIT 1
    """
)

#: Adopt an existing, unowned org row for this slug as the user's personal node.
#: The ``IS NULL`` guard is repeated in SQL so a concurrent claim loses rather
#: than being overwritten between the SELECT and the UPDATE.
_ADOPT_PERSONAL_SQL = text(
    """
    UPDATE organizations
       SET personal_owner_user_id = :user_id,
           updated_at = now()
     WHERE org_id = :org_id
       AND personal_owner_user_id IS NULL
    RETURNING org_id
    """
)

_ENSURE_PERSONAL_SQL = text(
    """
    INSERT INTO organizations (org_id, name, account_slug, personal_owner_user_id)
    VALUES (:org_id, :name, :account_slug, :user_id)
    ON CONFLICT (org_id) DO UPDATE
        SET personal_owner_user_id = EXCLUDED.personal_owner_user_id
    RETURNING org_id
    """
)


def _personal_org_id(user_id: str) -> str:
    """The org_id a personal node is minted at.

    One definition because the adoption guard and the insert must agree: if
    they drift, the guard stops recognising rows this module created and
    provisioning fails for the users it is supposed to serve.

    The whole user id, not a prefix -- 8 hex chars collide at birthday odds
    around 65k users, and org_id is String(64) while "personal_" plus a UUID
    is 45 characters.
    """
    return f"personal_{user_id}"


def ensure_personal_org(tsdb: Any, *, user_id: str, account_slug: str) -> str:
    """The org row backing this user's personal node, created if absent.

    Takes the engine connection only: it writes ``organizations``, which is
    engine-side. Membership lives in Supabase and is not consulted here — the
    caller has already established who the user is.

    Idempotent, and enforced as such by the partial unique index migration 048
    puts on ``personal_owner_user_id``. A second personal node for one user
    would split their own memory in two, which is invisible until they notice
    something they wrote is gone.

    Args:
        tsdb: A connection to the engine's own database.
        user_id: The Supabase user UUID.
        account_slug: The slug of that user's personal Supabase account.

    Returns:
        The ``org_id`` of the personal node.

    Raises:
        ValueError: If either identifier is blank, or if the slug already
            belongs to somebody else's personal node.
    """
    cleaned_user = _clean(user_id)
    cleaned_slug = _clean(account_slug)
    if cleaned_user is None or cleaned_slug is None:
        raise ValueError("ensure_personal_org needs a user_id and an account_slug")

    existing = tsdb.execute(_PERSONAL_BY_OWNER_SQL, {"user_id": cleaned_user}).first()
    if existing is not None:
        return existing.org_id

    # A row may already exist for this slug — account_slug is UNIQUE, so
    # inserting a second one would fail outright. Adopt it if it is unowned;
    # refuse if it is somebody else's, because handing this caller that row
    # would hand them another person's memory.
    by_slug = tsdb.execute(_ORG_BY_SLUG_SQL, {"account_slug": cleaned_slug}).first()
    if by_slug is not None:
        owner = _clean(by_slug.personal_owner_user_id)
        if owner is not None and owner != cleaned_user:
            raise ValueError(
                f"account_slug {cleaned_slug!r} is already another user's personal node"
            )
        # An unowned row is not automatically ours to take. Every org that
        # predates the personal_owner_user_id column has it NULL, team orgs
        # included, so adopting on "unowned" alone would stamp a company's
        # memory as one person's personal node. Only a row already sitting at
        # the id we mint can be one of ours -- anything else is a real org that
        # happens to share the slug.
        if owner is None and by_slug.org_id != _personal_org_id(cleaned_user):
            raise ValueError(
                f"account_slug {cleaned_slug!r} already belongs to org "
                f"{by_slug.org_id!r}, which is not a personal node"
            )
        adopted = tsdb.execute(
            _ADOPT_PERSONAL_SQL,
            {"org_id": by_slug.org_id, "user_id": cleaned_user},
        ).first()
        if adopted is None:
            raise ValueError(
                f"account_slug {cleaned_slug!r} was claimed by another user concurrently"
            )
        return adopted.org_id

    # The whole user id, not a prefix: "personal_" + a 36-char UUID is 45
    # characters and org_id is String(64), so there is no reason to truncate.
    # An 8-hex-char prefix collides at birthday odds around 65k users, and the
    # partial unique index would then reject the second user's personal node --
    # a signup failure with no obvious cause.
    org_id = _personal_org_id(cleaned_user)
    row = tsdb.execute(
        _ENSURE_PERSONAL_SQL,
        {
            "org_id": org_id,
            "name": cleaned_slug,
            "account_slug": cleaned_slug,
            "user_id": cleaned_user,
        },
    ).first()
    return row.org_id
