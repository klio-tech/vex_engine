"""Membership is the tenancy boundary once org_id is selected, not derived.

The check that matters is the negative one: a user who is not a member of an
org must not reach it, however the org id arrives. A test that only asserts a
member CAN reach their own org passes against a function that returns True
unconditionally -- so the fixture builds a second org, owned by a second user,
that the subject must never see, and the negative assertions are the ones that
carry this file.

The two Supabase-side tables this reads (``public.accounts`` and
``public.accounts_memberships``) do not exist in the engine's database, so the
fixture creates them for real inside a transaction and rolls it back on
teardown. Stubbing the session instead would match SQL by substring and return
canned rows, which cannot catch a wrong WHERE or JOIN clause -- the single
failure this module exists to prevent.
"""

import pytest
from sqlalchemy import create_engine, text

from shared.membership import may_reach, reachable_orgs
from shared.testing import database_url, requires_db

DB_URL = database_url()

SUBJECT_USER = "11111111-1111-1111-1111-111111111111"
OTHER_USER = "22222222-2222-2222-2222-222222222222"
STRANGER_USER = "99999999-9999-9999-9999-999999999999"

PERSONAL_SLUG = "subject-personal"
TEAM_SLUG = "team-alpha"
OTHER_SLUG = "someone-else"

PERSONAL_ORG = "org_subject_personal"
TEAM_ORG = "org_team_alpha"
OTHER_ORG = "org_someone_else"


class _Membership:
    """The fixture's return value.

    Unpacks as ``(user_id, org_id, slug)`` and carries the live connection as
    ``.db`` -- the connection is not part of the tuple because every test that
    unpacks it wants the identity triple, not the session.
    """

    def __init__(self, db, user_id, org_id, slug):
        self.db = db
        self.user_id = user_id
        self.org_id = org_id
        self.slug = slug

    def __iter__(self):
        return iter((self.user_id, self.org_id, self.slug))


@pytest.fixture()
def membership_fixture():
    """Real Supabase-shaped tables, in a transaction that is always rolled back.

    Postgres DDL is transactional, so the CREATE TABLEs and the rows both
    vanish on rollback and the test is repeatable against a database that is
    otherwise left untouched.
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

        accounts = (
            ("aaaaaaaa-0000-0000-0000-000000000001", PERSONAL_SLUG, True),
            ("aaaaaaaa-0000-0000-0000-000000000002", TEAM_SLUG, False),
            ("aaaaaaaa-0000-0000-0000-000000000003", OTHER_SLUG, False),
        )
        for account_id, slug, is_personal in accounts:
            conn.execute(
                text(
                    "INSERT INTO public.accounts (id, slug, is_personal_account) "
                    "VALUES (:id, :slug, :personal)"
                ),
                {"id": account_id, "slug": slug, "personal": is_personal},
            )

        # The subject belongs to their personal node and one team; the third
        # account belongs to someone else entirely.
        for account_id, user_id in (
            ("aaaaaaaa-0000-0000-0000-000000000001", SUBJECT_USER),
            ("aaaaaaaa-0000-0000-0000-000000000002", SUBJECT_USER),
            ("aaaaaaaa-0000-0000-0000-000000000003", OTHER_USER),
        ):
            conn.execute(
                text(
                    "INSERT INTO public.accounts_memberships (account_id, user_id) "
                    "VALUES (:account_id, :user_id)"
                ),
                {"account_id": account_id, "user_id": user_id},
            )

        for org_id, slug in (
            (PERSONAL_ORG, PERSONAL_SLUG),
            (TEAM_ORG, TEAM_SLUG),
            (OTHER_ORG, OTHER_SLUG),
        ):
            conn.execute(
                text(
                    "INSERT INTO organizations (org_id, name, account_slug) "
                    "VALUES (:org_id, :name, :slug)"
                ),
                {"org_id": org_id, "name": org_id, "slug": slug},
            )

        yield _Membership(conn, SUBJECT_USER, TEAM_ORG, TEAM_SLUG)
    finally:
        trans.rollback()
        conn.close()
        engine.dispose()


@requires_db("the membership resolution tests")
def test_a_member_reaches_their_org(membership_fixture):
    user_id, org_id, _slug = membership_fixture
    assert may_reach(membership_fixture.db, user_id=user_id, org_id=org_id) is True


@requires_db("the membership resolution tests")
def test_a_non_member_does_not_reach_it(membership_fixture):
    _user_id, org_id, _slug = membership_fixture
    assert (
        may_reach(
            membership_fixture.db,
            user_id=STRANGER_USER,
            org_id=org_id,
        )
        is False
    )
    # ...and the org that really belongs to somebody else is just as closed,
    # even though it exists and has a membership row of its own.
    assert (
        may_reach(membership_fixture.db, user_id=SUBJECT_USER, org_id=OTHER_ORG)
        is False
    )


@requires_db("the membership resolution tests")
def test_an_unknown_org_is_not_reachable(membership_fixture):
    user_id, _org_id, _slug = membership_fixture
    assert (
        may_reach(membership_fixture.db, user_id=user_id, org_id="org_does_not_exist")
        is False
    )


@requires_db("the membership resolution tests")
def test_a_blank_user_reaches_nothing(membership_fixture):
    # An unattributable caller must not fall through to "everything".
    _user_id, org_id, _slug = membership_fixture
    for blank in (None, "", "   "):
        assert may_reach(membership_fixture.db, user_id=blank, org_id=org_id) is False
        assert reachable_orgs(membership_fixture.db, user_id=blank) == []


@requires_db("the membership resolution tests")
def test_reachable_orgs_lists_only_the_users_own(membership_fixture):
    user_id, org_id, _slug = membership_fixture
    orgs = reachable_orgs(membership_fixture.db, user_id=user_id)
    got = [o["org_id"] for o in orgs]
    assert org_id in got
    assert OTHER_ORG not in got
    # The personal node is included -- this is the one query against accounts
    # that must not filter is_personal_account -- and it sorts first.
    assert got == [PERSONAL_ORG, TEAM_ORG]
    assert orgs[0]["is_personal"] is True
    assert orgs[0]["account_slug"] == PERSONAL_SLUG
    assert orgs[1]["is_personal"] is False


@requires_db("the membership resolution tests")
def test_may_reach_agrees_with_reachable_orgs(membership_fixture):
    # The two must never disagree: one is the list, the other the predicate.
    user_id, _org_id, _slug = membership_fixture
    for org in reachable_orgs(membership_fixture.db, user_id=user_id):
        assert may_reach(membership_fixture.db, user_id=user_id, org_id=org["org_id"])
