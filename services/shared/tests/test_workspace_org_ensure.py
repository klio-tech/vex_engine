"""Selecting a workspace provisions its engine org — the ensure's contract.

``ensure_org_for_workspace`` is the handshake's write path: the moment a
member first selects a workspace, the engine mints (or adopts) the org row
that backs it. Every rule from the plan is one test here, and the one that
closes the loop for real is the round-trip: the org the ensure returns must be
immediately visible in ``reachable_orgs`` for a member of that workspace. An
ensure that forgets to stamp ``organizations.workspace_id`` provisions a node
nobody can select — the join is by ``workspace_id``, so the row simply never
appears — and only the round-trip assertion can catch that.

The fixture creates the two workspace-DB tables for real inside a transaction
that is always rolled back, exactly as ``test_membership_resolution.py`` does;
``organizations`` is the engine's own real table, covered by the same
rollback.
"""

import pytest
from sqlalchemy import create_engine, text

from shared.membership import ensure_org_for_workspace, reachable_orgs
from shared.testing import database_url, requires_db

DB_URL = database_url()

OWNER_USER = "77777777-7777-7777-7777-777777777777"
OTHER_USER = "88888888-8888-8888-8888-888888888888"

WORKSPACE_ID = "cccccccc-0000-0000-0000-000000000001"
OTHER_WORKSPACE_ID = "cccccccc-0000-0000-0000-000000000002"

SLUG = "ensure-me"

needs_db = requires_db("the workspace org ensure tests")


@pytest.fixture()
def db():
    """A rolled-back transaction against the engine database.

    Postgres DDL is transactional, so the workspace-DB-shaped tables the
    round-trip test creates vanish with the data on rollback.
    """
    engine = create_engine(DB_URL)
    conn = engine.connect()
    trans = conn.begin()
    try:
        yield conn
    finally:
        trans.rollback()
        conn.close()
        engine.dispose()


def _org_row(db, org_id):
    return db.execute(
        text(
            "SELECT org_id, name, account_slug, workspace_id, "
            "personal_owner_user_id FROM organizations WHERE org_id = :org_id"
        ),
        {"org_id": org_id},
    ).first()


def _insert_org(db, *, org_id, slug, workspace_id=None, personal_owner=None):
    db.execute(
        text(
            "INSERT INTO organizations "
            "(org_id, name, account_slug, workspace_id, personal_owner_user_id) "
            "VALUES (:org_id, :org_id, :slug, :workspace_id, :owner)"
        ),
        {
            "org_id": org_id,
            "slug": slug,
            "workspace_id": workspace_id,
            "owner": personal_owner,
        },
    )


@needs_db
def test_nothing_exists_inserts_a_linked_org(db):
    org_id = ensure_org_for_workspace(
        db,
        workspace_id=WORKSPACE_ID,
        slug=SLUG,
        is_personal=False,
        owner_user_id=None,
    )
    assert org_id == f"ws_{WORKSPACE_ID}"
    row = _org_row(db, org_id)
    assert row is not None
    assert row.account_slug == SLUG
    assert row.name == SLUG
    # The stamp is the whole point: reachable_orgs joins by workspace_id, so
    # an org without it is a node nobody can select.
    assert str(row.workspace_id) == WORKSPACE_ID
    assert row.personal_owner_user_id is None


@needs_db
def test_already_linked_is_idempotent(db):
    first = ensure_org_for_workspace(
        db, workspace_id=WORKSPACE_ID, slug=SLUG, is_personal=False, owner_user_id=None
    )
    second = ensure_org_for_workspace(
        db, workspace_id=WORKSPACE_ID, slug=SLUG, is_personal=False, owner_user_id=None
    )
    assert second == first
    count = db.execute(
        text("SELECT count(*) FROM organizations WHERE workspace_id = :wid"),
        {"wid": WORKSPACE_ID},
    ).scalar()
    assert count == 1


@needs_db
def test_already_linked_wins_even_when_the_slug_moved(db):
    """The id is the identity; the slug is only the address.

    Once linked, the org is returned by its workspace_id even if the caller
    now presents a different slug (Plane soft-delete rewrites slugs).
    """
    _insert_org(db, org_id="org_linked", slug="old-address", workspace_id=WORKSPACE_ID)
    got = ensure_org_for_workspace(
        db,
        workspace_id=WORKSPACE_ID,
        slug="new-address",
        is_personal=False,
        owner_user_id=None,
    )
    assert got == "org_linked"
    # Touch nothing: the slug is not rewritten by the ensure.
    assert _org_row(db, "org_linked").account_slug == "old-address"


@needs_db
def test_legacy_adoption_stamps_workspace_id(db):
    # A MoonForge-era org: right slug, no workspace link yet.
    _insert_org(db, org_id="org_legacy", slug=SLUG, workspace_id=None)
    got = ensure_org_for_workspace(
        db, workspace_id=WORKSPACE_ID, slug=SLUG, is_personal=False, owner_user_id=None
    )
    assert got == "org_legacy"
    assert str(_org_row(db, "org_legacy").workspace_id) == WORKSPACE_ID


@needs_db
def test_a_slug_held_by_a_different_workspace_is_refused(db):
    # Never re-point a link: the team-org-hijack lesson in a new coat.
    _insert_org(db, org_id="org_taken", slug=SLUG, workspace_id=OTHER_WORKSPACE_ID)
    with pytest.raises(ValueError):
        ensure_org_for_workspace(
            db,
            workspace_id=WORKSPACE_ID,
            slug=SLUG,
            is_personal=False,
            owner_user_id=None,
        )
    # And the loser's claim changed nothing.
    assert str(_org_row(db, "org_taken").workspace_id) == OTHER_WORKSPACE_ID


@needs_db
def test_personal_mirroring_on_a_fresh_insert(db):
    org_id = ensure_org_for_workspace(
        db,
        workspace_id=WORKSPACE_ID,
        slug=SLUG,
        is_personal=True,
        owner_user_id=OWNER_USER,
    )
    assert _org_row(db, org_id).personal_owner_user_id == OWNER_USER


@needs_db
def test_personal_mirroring_on_adoption(db):
    _insert_org(db, org_id=f"personal_{OWNER_USER}", slug=SLUG, workspace_id=None)
    got = ensure_org_for_workspace(
        db,
        workspace_id=WORKSPACE_ID,
        slug=SLUG,
        is_personal=True,
        owner_user_id=OWNER_USER,
    )
    assert got == f"personal_{OWNER_USER}"
    row = _org_row(db, got)
    assert str(row.workspace_id) == WORKSPACE_ID
    assert row.personal_owner_user_id == OWNER_USER


@needs_db
def test_an_owner_with_a_different_personal_org_is_refused(db):
    """048's invariant, respected rather than tripped.

    A second personal node would split one person's memory in two; the ensure
    must refuse before the partial unique index turns it into an
    IntegrityError with no obvious cause.
    """
    _insert_org(
        db,
        org_id="org_their_personal",
        slug="their-old-home",
        workspace_id=None,
        personal_owner=OWNER_USER,
    )
    with pytest.raises(ValueError):
        ensure_org_for_workspace(
            db,
            workspace_id=WORKSPACE_ID,
            slug=SLUG,
            is_personal=True,
            owner_user_id=OWNER_USER,
        )


@needs_db
def test_somebody_elses_personal_node_is_never_adopted(db):
    # A slug collision with another person's personal node must not hand this
    # workspace their private memories.
    _insert_org(
        db,
        org_id="org_alice_personal",
        slug=SLUG,
        workspace_id=None,
        personal_owner=OTHER_USER,
    )
    for is_personal, owner in ((False, None), (True, OWNER_USER)):
        with pytest.raises(ValueError):
            ensure_org_for_workspace(
                db,
                workspace_id=WORKSPACE_ID,
                slug=SLUG,
                is_personal=is_personal,
                owner_user_id=owner,
            )


@needs_db
def test_blank_identifiers_are_refused(db):
    for workspace_id, slug in (("", SLUG), (WORKSPACE_ID, ""), ("  ", "  ")):
        with pytest.raises(ValueError):
            ensure_org_for_workspace(
                db,
                workspace_id=workspace_id,
                slug=slug,
                is_personal=False,
                owner_user_id=None,
            )


class _RacingDb:
    """A connection wrapper that lets a competitor act between SELECT and UPDATE.

    The adoption stamp's ``WHERE workspace_id IS NULL`` guard only matters in
    the window after the ensure has read the row and before it writes it; this
    wrapper runs the competitor's stamp exactly there, which no sequential test
    can reach otherwise.
    """

    def __init__(self, conn, competitor_workspace_id):
        self._conn = conn
        self._competitor_workspace_id = competitor_workspace_id
        self._fired = False

    def execute(self, statement, *args, **kwargs):
        sql = str(statement)
        if not self._fired and "workspace_id IS NULL" in sql and "UPDATE" in sql:
            self._fired = True
            self._conn.execute(
                text(
                    "UPDATE organizations SET workspace_id = :wid "
                    "WHERE account_slug = :slug"
                ),
                {"wid": self._competitor_workspace_id, "slug": SLUG},
            )
        return self._conn.execute(statement, *args, **kwargs)


@needs_db
def test_a_lost_adoption_race_against_ourselves_returns_the_winner(db):
    # The competitor stamped the SAME workspace id: the row is ours after all.
    _insert_org(db, org_id="org_raced", slug=SLUG, workspace_id=None)
    racing = _RacingDb(db, WORKSPACE_ID)
    got = ensure_org_for_workspace(
        racing,
        workspace_id=WORKSPACE_ID,
        slug=SLUG,
        is_personal=False,
        owner_user_id=None,
    )
    assert got == "org_raced"


@needs_db
def test_a_lost_adoption_race_against_a_stranger_is_refused(db):
    # The competitor stamped a DIFFERENT workspace: losing must not overwrite.
    _insert_org(db, org_id="org_raced", slug=SLUG, workspace_id=None)
    racing = _RacingDb(db, OTHER_WORKSPACE_ID)
    with pytest.raises(ValueError):
        ensure_org_for_workspace(
            racing,
            workspace_id=WORKSPACE_ID,
            slug=SLUG,
            is_personal=False,
            owner_user_id=None,
        )
    assert str(_org_row(db, "org_raced").workspace_id) == OTHER_WORKSPACE_ID


def _build_workspace_tables(db):
    db.execute(
        text(
            """
            CREATE TABLE workspaces (
                id uuid PRIMARY KEY,
                slug text UNIQUE,
                owner_id uuid,
                is_personal boolean NOT NULL DEFAULT false,
                deleted_at timestamptz
            )
            """
        )
    )
    db.execute(
        text(
            """
            CREATE TABLE workspace_members (
                workspace_id uuid NOT NULL REFERENCES workspaces(id),
                member_id uuid NOT NULL,
                is_active boolean NOT NULL DEFAULT true,
                deleted_at timestamptz
            )
            """
        )
    )


@needs_db
def test_the_ensured_org_is_immediately_reachable(db):
    """The loop, closed for real.

    ``reachable_orgs`` joins by ``workspace_id``; an ensure that forgets the
    stamp on ANY path provisions a node no member can ever select. Both the
    fresh-insert path and the legacy-adoption path must land in the very next
    membership resolution.
    """
    _build_workspace_tables(db)
    db.execute(
        text(
            "INSERT INTO workspaces (id, slug, owner_id, is_personal) "
            "VALUES (:id, :slug, :owner, true)"
        ),
        {"id": WORKSPACE_ID, "slug": SLUG, "owner": OWNER_USER},
    )
    db.execute(
        text(
            "INSERT INTO workspace_members (workspace_id, member_id) "
            "VALUES (:wid, :member)"
        ),
        {"wid": WORKSPACE_ID, "member": OWNER_USER},
    )

    # Fresh insert path.
    org_id = ensure_org_for_workspace(
        db,
        workspace_id=WORKSPACE_ID,
        slug=SLUG,
        is_personal=True,
        owner_user_id=OWNER_USER,
    )
    orgs = reachable_orgs(db, db, user_id=OWNER_USER)
    assert [o["org_id"] for o in orgs] == [org_id]
    assert orgs[0]["workspace_id"] == WORKSPACE_ID

    # Legacy adoption path: a second workspace whose org predates the link.
    db.execute(
        text(
            "INSERT INTO workspaces (id, slug, owner_id, is_personal) "
            "VALUES (:id, 'legacy-team', :owner, false)"
        ),
        {"id": OTHER_WORKSPACE_ID, "owner": OWNER_USER},
    )
    db.execute(
        text(
            "INSERT INTO workspace_members (workspace_id, member_id) "
            "VALUES (:wid, :member)"
        ),
        {"wid": OTHER_WORKSPACE_ID, "member": OWNER_USER},
    )
    _insert_org(db, org_id="org_legacy_team", slug="legacy-team", workspace_id=None)
    adopted = ensure_org_for_workspace(
        db,
        workspace_id=OTHER_WORKSPACE_ID,
        slug="legacy-team",
        is_personal=False,
        owner_user_id=None,
    )
    got = {o["org_id"] for o in reachable_orgs(db, db, user_id=OWNER_USER)}
    assert adopted == "org_legacy_team"
    assert got == {org_id, "org_legacy_team"}
