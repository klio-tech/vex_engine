"""May this user act in this org?

Once ``org_id`` is selected by the caller rather than derived from their key,
this is the tenancy boundary, and it has to be asked on every call rather than
once at switch time — a session outlives a membership revocation.

The engine holds no local users or memberships table on purpose: identity lives
in the workspace DB (Plane), and a local copy would be a cache with no
invalidation.

**Two databases, joined in Python.** ``workspaces`` and ``workspace_members``
live in the workspace DB (``WORKSPACE_DATABASE_URL``); ``organizations`` lives
in the engine's own database (``DATABASE_URL``). They are separate servers, so
a single SQL statement naming both cannot run — however well it works against a
test database that happens to hold both.

The join key is ``organizations.workspace_id``, never the slug: slug is the
*address* (username / org handle) and mutable — Plane's soft-delete rewrites
it — while the workspace id is the identity.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

#: Workspace DB side: which living workspaces is this user an active member of?
_WORKSPACES_SQL = text(
    """
    SELECT w.id AS workspace_id,
           w.slug AS slug,
           w.is_personal AS is_personal,
           w.owner_id AS owner_id
    FROM workspace_members m
    JOIN workspaces w ON w.id = m.workspace_id
    WHERE m.member_id = :user_id
      AND m.is_active = true
      AND m.deleted_at IS NULL
      AND w.deleted_at IS NULL
    """
)

#: Engine side: which of those workspaces have an org row? Joined by the
#: immutable id -- slug is the address, not the identity.
_ORGS_SQL = text(
    """
    SELECT org_id, workspace_id
    FROM organizations
    WHERE workspace_id = ANY(CAST(:workspace_ids AS uuid[]))
    """
)


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def member_workspaces(workspace_db: Any, *, user_id: Optional[str]) -> list[dict]:
    """Every living workspace this user is an active member of.

    The workspace-DB half of :func:`reachable_orgs`, exported on its own
    because the provisioning path needs it too: ``reachable_orgs`` omits a
    workspace with no ``organizations`` row, and the provisioning moment is
    exactly a workspace in that state.

    Args:
        workspace_db: A connection to the workspace database.
        user_id: The user UUID, or ``None``.

    Returns:
        ``[{"workspace_id", "slug", "is_personal", "owner_user_id"}]``, ids
        stringified. An unattributable caller gets ``[]`` — same fail-closed
        rule as :func:`reachable_orgs`.
    """
    cleaned = _clean(user_id)
    if cleaned is None:
        return []
    rows = workspace_db.execute(_WORKSPACES_SQL, {"user_id": cleaned}).fetchall()
    return [
        {
            "workspace_id": str(r.workspace_id),
            "slug": r.slug,
            "is_personal": bool(r.is_personal),
            "owner_user_id": str(r.owner_id) if r.owner_id is not None else None,
        }
        for r in rows
    ]


def reachable_orgs(
    workspace_db: Any, tsdb: Any, *, user_id: Optional[str]
) -> list[dict]:
    """Every org node this user may act in, personal node first.

    Args:
        workspace_db: A connection to the workspace database (workspaces,
            workspace_members).
        tsdb: A connection to the engine's own database (organizations).
        user_id: The user UUID (the same value in both systems), or ``None``.

    Returns:
        ``[{"org_id", "workspace_id", "slug", "is_personal"}]``, personal node
        first then by slug. A workspace with no ``organizations`` row is
        omitted: the engine cannot store memory for a node it has no id for.

    An unattributable caller reaches nothing. Returning "everything" for a
    blank user id would turn a failure to identify someone into access to
    every tenant.
    """
    rows = member_workspaces(workspace_db, user_id=user_id)
    if not rows:
        return []
    # Keyed by the stringified id so uuid.UUID rows from one side and str rows
    # from the other meet on equal terms.
    info_by_id = {r["workspace_id"]: r for r in rows}

    org_rows = tsdb.execute(
        _ORGS_SQL, {"workspace_ids": [r["workspace_id"] for r in rows]}
    ).fetchall()

    resolved = []
    for o in org_rows:
        info = info_by_id.get(str(o.workspace_id))
        if info is None:  # pragma: no cover - ANY() already filtered
            continue
        resolved.append(
            {
                "org_id": o.org_id,
                "workspace_id": str(o.workspace_id),
                "slug": info["slug"],
                "is_personal": info["is_personal"],
            }
        )
    # Personal node first, then by slug — a stable order so a caller listing
    # their nodes sees the same sequence every time.
    resolved.sort(key=lambda o: (not o["is_personal"], o["slug"]))
    return resolved


def may_reach(
    workspace_db: Any, tsdb: Any, *, user_id: Optional[str], org_id: Optional[str]
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
        for o in reachable_orgs(workspace_db, tsdb, user_id=cleaned_user)
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
    engine-side. Membership lives in the workspace DB and is not consulted
    here — the caller has already established who the user is.

    Idempotent, and enforced as such by the partial unique index migration 048
    puts on ``personal_owner_user_id``. A second personal node for one user
    would split their own memory in two, which is invisible until they notice
    something they wrote is gone.

    Args:
        tsdb: A connection to the engine's own database.
        user_id: The user UUID — the same value in the workspace DB and here.
        account_slug: The slug of that user's personal workspace (their
            username).

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


#: The org row linked to this workspace, if any. The id is the identity, so
#: this lookup wins over anything the slug says.
_ORG_BY_WORKSPACE_SQL = text(
    """
    SELECT org_id
    FROM organizations
    WHERE workspace_id = :workspace_id
    LIMIT 1
    """
)

#: The slug's current holder, link state included. ``account_slug`` is UNIQUE,
#: so at most one row answers.
_ORG_LINK_BY_SLUG_SQL = text(
    """
    SELECT org_id, workspace_id, personal_owner_user_id
    FROM organizations
    WHERE account_slug = :account_slug
    LIMIT 1
    """
)

#: Adopt a legacy (pre-handshake) org for this workspace. The
#: ``workspace_id IS NULL`` guard is repeated in SQL so a concurrent claim
#: loses rather than overwriting a link established between the SELECT and
#: this UPDATE. COALESCE keeps an already-set personal owner.
_ADOPT_WORKSPACE_SQL = text(
    """
    UPDATE organizations
       SET workspace_id = :workspace_id,
           personal_owner_user_id
               = COALESCE(personal_owner_user_id, :owner_user_id),
           updated_at = now()
     WHERE org_id = :org_id
       AND workspace_id IS NULL
    RETURNING org_id
    """
)

#: Fresh insert, linked from birth. ``ON CONFLICT DO NOTHING`` with no target
#: covers every arbiter (org_id PK, account_slug UNIQUE, the partial unique on
#: workspace_id): a concurrent ensure simply gets no row back and re-reads,
#: instead of aborting the caller's transaction with an IntegrityError.
_INSERT_WORKSPACE_ORG_SQL = text(
    """
    INSERT INTO organizations
        (org_id, name, account_slug, workspace_id, personal_owner_user_id)
    VALUES (:org_id, :name, :account_slug, :workspace_id, :owner_user_id)
    ON CONFLICT DO NOTHING
    RETURNING org_id
    """
)


def _workspace_org_id(workspace_id: str) -> str:
    """The org_id a workspace-provisioned node is minted at.

    One definition for the same reason as :func:`_personal_org_id`: the insert
    and every caller predicting the id must agree. "ws_" plus a 36-char UUID
    is 39 characters against org_id's String(64) — no truncation, so no
    birthday collisions.
    """
    return f"ws_{workspace_id}"


def ensure_org_for_workspace(
    tsdb: Any,
    *,
    workspace_id: str,
    slug: str,
    is_personal: bool,
    owner_user_id: Optional[str],
) -> str:
    """The org row backing this workspace, created or adopted if absent.

    Selection is the provisioning moment: the first time a member selects a
    workspace that has no engine org, this mints one. Takes the engine
    connection only — membership lives in the workspace DB and has already
    been established by the caller.

    Every path stamps ``organizations.workspace_id``. That is not a nicety:
    :func:`reachable_orgs` joins by ``workspace_id``, so an org row without
    the stamp is a node nobody can ever select.

    Args:
        tsdb: A connection to the engine's own database.
        workspace_id: The workspace UUID (the identity; the join key).
        slug: The workspace's current slug (the address; never a join key).
        is_personal: Whether the workspace is its owner's personal one.
        owner_user_id: The owning user's UUID, for personal mirroring.

    Returns:
        The ``org_id`` of the linked node.

    Raises:
        ValueError: When an identifier is blank; when the slug is already held
            by an org linked to a DIFFERENT workspace (a link is never
            re-pointed); when the slug's org is somebody else's personal node;
            or when the owner already has a different personal org (048's
            one-personal-node invariant, refused here rather than tripped as
            an IntegrityError).
    """
    cleaned_workspace = _clean(workspace_id)
    cleaned_slug = _clean(slug)
    if cleaned_workspace is None or cleaned_slug is None:
        raise ValueError("ensure_org_for_workspace needs a workspace_id and a slug")
    cleaned_owner = _clean(owner_user_id)
    mirrored_owner = cleaned_owner if (is_personal and cleaned_owner) else None

    # Already linked: return it, touch nothing. The id wins over the slug —
    # Plane soft-delete rewrites slugs, the workspace id never moves.
    linked = tsdb.execute(
        _ORG_BY_WORKSPACE_SQL, {"workspace_id": cleaned_workspace}
    ).first()
    if linked is not None:
        return linked.org_id

    by_slug = tsdb.execute(
        _ORG_LINK_BY_SLUG_SQL, {"account_slug": cleaned_slug}
    ).first()

    if mirrored_owner is not None:
        existing_personal = tsdb.execute(
            _PERSONAL_BY_OWNER_SQL, {"user_id": mirrored_owner}
        ).first()
        if existing_personal is not None and (
            by_slug is None or existing_personal.org_id != by_slug.org_id
        ):
            # Minting (or adopting) anything else would give this owner a
            # second personal node and split their memory in two.
            raise ValueError(
                f"user {mirrored_owner!r} already has a personal node "
                f"({existing_personal.org_id!r}); refusing to provision a second"
            )

    if by_slug is not None:
        return _adopt_org_for_workspace(
            tsdb,
            by_slug=by_slug,
            workspace_id=cleaned_workspace,
            slug=cleaned_slug,
            mirrored_owner=mirrored_owner,
        )

    inserted = tsdb.execute(
        _INSERT_WORKSPACE_ORG_SQL,
        {
            "org_id": _workspace_org_id(cleaned_workspace),
            "name": cleaned_slug,
            "account_slug": cleaned_slug,
            "workspace_id": cleaned_workspace,
            "owner_user_id": mirrored_owner,
        },
    ).first()
    if inserted is not None:
        return inserted.org_id

    # ON CONFLICT DO NOTHING returned no row: a concurrent ensure got there
    # first. If it linked this same workspace, its row is the answer.
    winner = tsdb.execute(
        _ORG_BY_WORKSPACE_SQL, {"workspace_id": cleaned_workspace}
    ).first()
    if winner is not None:
        return winner.org_id
    raise ValueError(
        f"workspace {cleaned_workspace!r} could not be provisioned: its slug "
        f"or org id was claimed concurrently by another workspace"
    )


def _adopt_org_for_workspace(
    tsdb: Any,
    *,
    by_slug: Any,
    workspace_id: str,
    slug: str,
    mirrored_owner: Optional[str],
) -> str:
    """Stamp a pre-handshake org row with its workspace id.

    This is how every existing MoonForge-era org joins the handshake without a
    data migration. Split out of :func:`ensure_org_for_workspace` so each half
    stays small enough to read whole.
    """
    existing_link = by_slug.workspace_id
    if existing_link is not None:
        if str(existing_link) == workspace_id:
            # A concurrent ensure for this same workspace won between our two
            # reads. Its row is ours.
            return by_slug.org_id
        # Never re-point a link: the team-org-hijack lesson from 6ad63ae in a
        # new coat. The workspace id is the identity, and identities do not
        # transfer over a slug collision.
        raise ValueError(
            f"account_slug {slug!r} is already linked to workspace "
            f"{existing_link!s}, not {workspace_id!r}; a link is never re-pointed"
        )

    slug_owner = _clean(by_slug.personal_owner_user_id)
    if slug_owner is not None and slug_owner != mirrored_owner:
        # A slug collision with somebody's personal node must not hand this
        # workspace their private memories.
        raise ValueError(
            f"account_slug {slug!r} is another user's personal node; "
            f"refusing to link workspace {workspace_id!r} to it"
        )

    adopted = tsdb.execute(
        _ADOPT_WORKSPACE_SQL,
        {
            "org_id": by_slug.org_id,
            "workspace_id": workspace_id,
            "owner_user_id": mirrored_owner,
        },
    ).first()
    if adopted is not None:
        return adopted.org_id

    # Lost the race: somebody stamped the row between our read and the guarded
    # UPDATE. Re-read and return the winner only if it is us.
    now = tsdb.execute(_ORG_LINK_BY_SLUG_SQL, {"account_slug": slug}).first()
    if now is not None and str(now.workspace_id) == workspace_id:
        return now.org_id
    raise ValueError(
        f"account_slug {slug!r} was claimed by another workspace concurrently"
    )
