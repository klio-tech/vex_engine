"""A user's personal account is their personal node, and it has an org row.

Every other engine query filters ``is_personal_account = false``, which is why
a personal account resolves to no org today. The personal node is the one part
of the model that must always exist: it is where solo agent work accumulates
before any team is involved.

The fixture follows ``test_membership_resolution.py`` and creates the two
Supabase-side tables (``public.accounts``, ``public.accounts_memberships``) for
real inside a transaction that is always rolled back. A stubbed session would
match SQL by substring and return canned rows, which cannot catch the thing
this file exists to catch -- a second call minting a second personal node.

``reachable_orgs`` takes two connections because the accounts tables live
behind ``SUPABASE_DATABASE_URL`` in production while ``organizations`` lives
behind ``DATABASE_URL``. Locally one connection stands in for both.
"""

import pytest
from sqlalchemy import create_engine, text

from shared.membership import ensure_personal_org, reachable_orgs
from shared.testing import database_url, requires_db

DB_URL = database_url()

SUBJECT_USER = "33333333-3333-3333-3333-333333333333"
STRANGER_USER = "88888888-8888-8888-8888-888888888888"

PERSONAL_SLUG = "solo-subject"
TEAM_SLUG = "team-beta"

TEAM_ORG = "org_team_beta"

PERSONAL_ACCOUNT_ID = "bbbbbbbb-0000-0000-0000-000000000001"
TEAM_ACCOUNT_ID = "bbbbbbbb-0000-0000-0000-000000000002"


class _Personal:
    """Unpacks as ``(db, user_id, slug)`` -- the triple every test wants."""

    def __init__(self, db, user_id, slug):
        self.db = db
        self.user_id = user_id
        self.slug = slug

    def __iter__(self):
        return iter((self.db, self.user_id, self.slug))


@pytest.fixture()
def personal_fixture():
    """Real Supabase-shaped tables, in a transaction that is always rolled back.

    Postgres DDL is transactional, so both the CREATE TABLEs and every row
    written by the code under test vanish on rollback -- including any
    ``organizations`` row ``ensure_personal_org`` inserts.
    """
    engine = create_engine(DB_URL)
    conn = engine.connect()
    trans = conn.begin()
    try:
        conn.execute(
            text(
                """
                CREATE TABLE public.accounts (
                    id uuid PRIMARY KEY,
                    slug text UNIQUE,
                    is_personal_account boolean NOT NULL DEFAULT false
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE public.accounts_memberships (
                    account_id uuid NOT NULL REFERENCES public.accounts(id),
                    user_id text NOT NULL
                )
                """
            )
        )

        for account_id, slug, is_personal in (
            (PERSONAL_ACCOUNT_ID, PERSONAL_SLUG, True),
            (TEAM_ACCOUNT_ID, TEAM_SLUG, False),
        ):
            conn.execute(
                text(
                    "INSERT INTO public.accounts (id, slug, is_personal_account) "
                    "VALUES (:id, :slug, :personal)"
                ),
                {"id": account_id, "slug": slug, "personal": is_personal},
            )
        for account_id in (PERSONAL_ACCOUNT_ID, TEAM_ACCOUNT_ID):
            conn.execute(
                text(
                    "INSERT INTO public.accounts_memberships (account_id, user_id) "
                    "VALUES (:account_id, :user_id)"
                ),
                {"account_id": account_id, "user_id": SUBJECT_USER},
            )

        # The team already has an org row; the personal account deliberately
        # does not, because provisioning it is what is under test.
        conn.execute(
            text(
                "INSERT INTO organizations (org_id, name, account_slug) "
                "VALUES (:org_id, :name, :slug)"
            ),
            {"org_id": TEAM_ORG, "name": TEAM_ORG, "slug": TEAM_SLUG},
        )

        yield _Personal(conn, SUBJECT_USER, PERSONAL_SLUG)
    finally:
        trans.rollback()
        conn.close()
        engine.dispose()


@requires_db("the personal node tests")
def test_a_personal_account_gets_an_org_row(personal_fixture):
    db, user_id, slug = personal_fixture
    org_id = ensure_personal_org(db, user_id=user_id, account_slug=slug)
    assert org_id
    row = db.execute(
        text(
            "SELECT account_slug, personal_owner_user_id FROM organizations "
            "WHERE org_id = :org_id"
        ),
        {"org_id": org_id},
    ).first()
    assert row is not None, "ensure_personal_org returned an id with no row behind it"
    assert row.account_slug == slug
    assert row.personal_owner_user_id == user_id


@requires_db("the personal node tests")
def test_it_is_idempotent(personal_fixture):
    db, user_id, slug = personal_fixture
    first = ensure_personal_org(db, user_id=user_id, account_slug=slug)
    second = ensure_personal_org(db, user_id=user_id, account_slug=slug)
    assert first == second, "a second call minted a second personal node"
    count = db.execute(
        text(
            "SELECT count(*) AS n FROM organizations "
            "WHERE personal_owner_user_id = :user_id"
        ),
        {"user_id": user_id},
    ).first()
    assert count.n == 1, "a person must have exactly one personal node"


@requires_db("the personal node tests")
def test_the_whole_user_id_is_in_the_org_id(personal_fixture):
    """Not a prefix.

    An 8-hex-char prefix collides at birthday odds around 65k users, and the
    partial unique index would then reject the second user's personal node --
    a signup failure with no obvious cause. ``"personal_"`` plus a 36-char
    UUID is 45 characters and ``org_id`` is ``String(64)``.
    """
    db, user_id, slug = personal_fixture
    org_id = ensure_personal_org(db, user_id=user_id, account_slug=slug)
    assert user_id in org_id
    assert len(org_id) <= 64


@requires_db("the personal node tests")
def test_the_personal_node_is_reachable_and_sorts_first(personal_fixture):
    db, user_id, slug = personal_fixture
    org_id = ensure_personal_org(db, user_id=user_id, account_slug=slug)
    orgs = reachable_orgs(db, db, user_id=user_id)
    assert orgs, "personal node did not appear in reachable_orgs"
    assert orgs[0]["is_personal"] is True
    assert orgs[0]["org_id"] == org_id
    # The team node is still reachable; provisioning the personal one does not
    # displace it.
    assert TEAM_ORG in {o["org_id"] for o in orgs}


@requires_db("the personal node tests")
def test_a_personal_node_belongs_to_exactly_one_user(personal_fixture):
    db, user_id, slug = personal_fixture
    org_id = ensure_personal_org(db, user_id=user_id, account_slug=slug)
    other = reachable_orgs(db, db, user_id=STRANGER_USER)
    assert org_id not in {o["org_id"] for o in other}


@requires_db("the personal node tests")
def test_a_second_user_cannot_take_over_an_existing_personal_node(personal_fixture):
    """The node's owner is not reassignable by asking for the same slug.

    Whoever passes the slug second would otherwise be handed the first user's
    memory, which is the worst failure available here.
    """
    db, user_id, slug = personal_fixture
    ensure_personal_org(db, user_id=user_id, account_slug=slug)
    with pytest.raises(ValueError):
        ensure_personal_org(db, user_id=STRANGER_USER, account_slug=slug)


@requires_db("the personal node tests")
def test_a_blank_identity_provisions_nothing(personal_fixture):
    db, user_id, slug = personal_fixture
    for blank in (None, "", "   "):
        with pytest.raises(ValueError):
            ensure_personal_org(db, user_id=blank, account_slug=slug)
        with pytest.raises(ValueError):
            ensure_personal_org(db, user_id=user_id, account_slug=blank)
    count = db.execute(
        text("SELECT count(*) AS n FROM organizations WHERE account_slug = :slug"),
        {"slug": slug},
    ).first()
    assert count.n == 0, "a blank identity created an org row"


def test_an_existing_team_org_is_never_adopted(personal_fixture):
    """A team org must not become somebody's personal node.

    Every pre-existing org has personal_owner_user_id IS NULL, so "adopt any
    unowned row for this slug" would stamp a whole company's memory as one
    person's private node. Unreachable today only because Supabase slugs are
    unique -- an invariant held in a different database, which is not a thing
    to depend on at a tenancy boundary.
    """
    db = personal_fixture.db
    db.execute(
        text(
            "INSERT INTO organizations (org_id, name, account_slug) "
            "VALUES ('org_real_team', 'Real Team', 'shared-slug')"
        )
    )
    with pytest.raises(ValueError):
        ensure_personal_org(db, user_id=SUBJECT_USER, account_slug="shared-slug")

    still_unowned = db.execute(
        text(
            "SELECT personal_owner_user_id FROM organizations "
            "WHERE org_id = 'org_real_team'"
        )
    ).scalar()
    assert still_unowned is None, "a team org was stamped as a personal node"
