"""Membership is the tenancy boundary once org_id is selected, not derived.

The check that matters is the negative one: a user who is not a member of an
org must not reach it, however the org id arrives. A test that only asserts a
member CAN reach their own org passes against a function that returns True
unconditionally -- so the fixture builds a second org, owned by a second user,
that the subject must never see, and the negative assertions are the ones that
carry this file.

The two workspace-DB tables this reads (``workspaces`` and
``workspace_members``) do not exist in the engine's database, so the fixture
creates them for real inside a transaction and rolls it back on teardown.
Stubbing the session instead would match SQL by substring and return canned
rows, which cannot catch a wrong WHERE or JOIN clause -- the single failure
this module exists to prevent.
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

PERSONAL_WORKSPACE = "aaaaaaaa-0000-0000-0000-000000000001"
TEAM_WORKSPACE = "aaaaaaaa-0000-0000-0000-000000000002"
OTHER_WORKSPACE = "aaaaaaaa-0000-0000-0000-000000000003"


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
    """Real workspace-DB-shaped tables, in a transaction always rolled back.

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
                CREATE TABLE workspaces (
                    id uuid PRIMARY KEY,
                    slug text UNIQUE,
                    is_personal boolean NOT NULL DEFAULT false,
                    deleted_at timestamptz
                )
                """
            )
        )
        conn.execute(
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

        workspaces = (
            (PERSONAL_WORKSPACE, PERSONAL_SLUG, True),
            (TEAM_WORKSPACE, TEAM_SLUG, False),
            (OTHER_WORKSPACE, OTHER_SLUG, False),
        )
        for workspace_id, slug, is_personal in workspaces:
            conn.execute(
                text(
                    "INSERT INTO workspaces (id, slug, is_personal) "
                    "VALUES (:id, :slug, :personal)"
                ),
                {"id": workspace_id, "slug": slug, "personal": is_personal},
            )

        # The subject belongs to their personal workspace and one team; the
        # third workspace belongs to someone else entirely.
        for workspace_id, member_id in (
            (PERSONAL_WORKSPACE, SUBJECT_USER),
            (TEAM_WORKSPACE, SUBJECT_USER),
            (OTHER_WORKSPACE, OTHER_USER),
        ):
            conn.execute(
                text(
                    "INSERT INTO workspace_members (workspace_id, member_id) "
                    "VALUES (:workspace_id, :member_id)"
                ),
                {"workspace_id": workspace_id, "member_id": member_id},
            )

        for org_id, slug, workspace_id in (
            (PERSONAL_ORG, PERSONAL_SLUG, PERSONAL_WORKSPACE),
            (TEAM_ORG, TEAM_SLUG, TEAM_WORKSPACE),
            (OTHER_ORG, OTHER_SLUG, OTHER_WORKSPACE),
        ):
            conn.execute(
                text(
                    "INSERT INTO organizations "
                    "(org_id, name, account_slug, workspace_id) "
                    "VALUES (:org_id, :name, :slug, :workspace_id)"
                ),
                {
                    "org_id": org_id,
                    "name": org_id,
                    "slug": slug,
                    "workspace_id": workspace_id,
                },
            )

        yield _Membership(conn, SUBJECT_USER, TEAM_ORG, TEAM_SLUG)
    finally:
        trans.rollback()
        conn.close()
        engine.dispose()


@requires_db("the membership resolution tests")
def test_a_member_reaches_their_org(membership_fixture):
    user_id, org_id, _slug = membership_fixture
    assert (
        may_reach(
            membership_fixture.db, membership_fixture.db, user_id=user_id, org_id=org_id
        )
        is True
    )


@requires_db("the membership resolution tests")
def test_a_non_member_does_not_reach_it(membership_fixture):
    _user_id, org_id, _slug = membership_fixture
    assert (
        may_reach(
            membership_fixture.db,
            membership_fixture.db,
            user_id=STRANGER_USER,
            org_id=org_id,
        )
        is False
    )
    # ...and the org that really belongs to somebody else is just as closed,
    # even though it exists and has a membership row of its own.
    assert (
        may_reach(
            membership_fixture.db,
            membership_fixture.db,
            user_id=SUBJECT_USER,
            org_id=OTHER_ORG,
        )
        is False
    )


@requires_db("the membership resolution tests")
def test_an_unknown_org_is_not_reachable(membership_fixture):
    user_id, _org_id, _slug = membership_fixture
    assert (
        may_reach(
            membership_fixture.db,
            membership_fixture.db,
            user_id=user_id,
            org_id="org_does_not_exist",
        )
        is False
    )


@requires_db("the membership resolution tests")
def test_a_blank_user_reaches_nothing(membership_fixture):
    # An unattributable caller must not fall through to "everything".
    _user_id, org_id, _slug = membership_fixture
    for blank in (None, "", "   "):
        assert (
            may_reach(
                membership_fixture.db,
                membership_fixture.db,
                user_id=blank,
                org_id=org_id,
            )
            is False
        )
        assert (
            reachable_orgs(membership_fixture.db, membership_fixture.db, user_id=blank)
            == []
        )


@requires_db("the membership resolution tests")
def test_reachable_orgs_lists_only_the_users_own(membership_fixture):
    user_id, org_id, _slug = membership_fixture
    orgs = reachable_orgs(membership_fixture.db, membership_fixture.db, user_id=user_id)
    got = [o["org_id"] for o in orgs]
    assert org_id in got
    assert OTHER_ORG not in got
    # The personal workspace is included and sorts first.
    assert got == [PERSONAL_ORG, TEAM_ORG]
    assert orgs[0]["is_personal"] is True
    assert orgs[0]["slug"] == PERSONAL_SLUG
    assert orgs[0]["workspace_id"] == PERSONAL_WORKSPACE
    assert orgs[1]["is_personal"] is False
    assert orgs[1]["workspace_id"] == TEAM_WORKSPACE


@requires_db("the membership resolution tests")
def test_may_reach_agrees_with_reachable_orgs(membership_fixture):
    # The two must never disagree: one is the list, the other the predicate.
    user_id, _org_id, _slug = membership_fixture
    for org in reachable_orgs(
        membership_fixture.db, membership_fixture.db, user_id=user_id
    ):
        assert may_reach(
            membership_fixture.db,
            membership_fixture.db,
            user_id=user_id,
            org_id=org["org_id"],
        )


# --- The kill switches -------------------------------------------------------
#
# Revocation is not a delete in Plane: memberships are deactivated
# (``is_active = false``) or soft-deleted (``deleted_at`` set), and workspaces
# are soft-deleted. Each of those must remove the workspace from
# ``reachable_orgs`` on the very next call -- a filter dropped from the WHERE
# clause turns "revoked" into "still inside", which is the exact failure this
# module exists to prevent.


@requires_db("the membership resolution tests")
def test_a_deactivated_membership_is_gone_on_the_next_call(membership_fixture):
    user_id, org_id, _slug = membership_fixture
    db = membership_fixture.db
    assert may_reach(db, db, user_id=user_id, org_id=org_id) is True

    db.execute(
        text(
            "UPDATE workspace_members SET is_active = false "
            "WHERE workspace_id = :ws AND member_id = :member"
        ),
        {"ws": TEAM_WORKSPACE, "member": user_id},
    )

    assert may_reach(db, db, user_id=user_id, org_id=org_id) is False
    assert org_id not in {o["org_id"] for o in reachable_orgs(db, db, user_id=user_id)}


@requires_db("the membership resolution tests")
def test_a_soft_deleted_workspace_is_gone_on_the_next_call(membership_fixture):
    user_id, org_id, _slug = membership_fixture
    db = membership_fixture.db
    assert may_reach(db, db, user_id=user_id, org_id=org_id) is True

    db.execute(
        text("UPDATE workspaces SET deleted_at = now() WHERE id = :ws"),
        {"ws": TEAM_WORKSPACE},
    )

    assert may_reach(db, db, user_id=user_id, org_id=org_id) is False
    assert org_id not in {o["org_id"] for o in reachable_orgs(db, db, user_id=user_id)}


@requires_db("the membership resolution tests")
def test_a_soft_deleted_membership_is_gone_on_the_next_call(membership_fixture):
    user_id, org_id, _slug = membership_fixture
    db = membership_fixture.db
    assert may_reach(db, db, user_id=user_id, org_id=org_id) is True

    db.execute(
        text(
            "UPDATE workspace_members SET deleted_at = now() "
            "WHERE workspace_id = :ws AND member_id = :member"
        ),
        {"ws": TEAM_WORKSPACE, "member": user_id},
    )

    assert may_reach(db, db, user_id=user_id, org_id=org_id) is False
    assert org_id not in {o["org_id"] for o in reachable_orgs(db, db, user_id=user_id)}


class _RecordingConn:
    """A connection that records the SQL it is handed and returns fixed rows."""

    def __init__(self, rows):
        self._rows = rows
        self.seen: list[str] = []

    def execute(self, sql, params=None):
        self.seen.append(str(sql))
        rows = self._rows

        class _Result:
            def fetchall(self_inner):
                return rows

        return _Result()


class _Row:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_the_two_databases_are_queried_separately():
    """The workspace DB and the engine are different servers; one JOIN cannot span them.

    ``workspaces`` and ``workspace_members`` live behind
    ``WORKSPACE_DATABASE_URL``; ``organizations`` lives behind ``DATABASE_URL``.
    A single statement naming both parses and runs happily against a test
    database that holds both, and fails in production with 'relation does not
    exist'. This is the only test that can catch that, because the local
    fixture legitimately puts both in one database.
    """
    workspace_db = _RecordingConn(
        [_Row(workspace_id="ws-1", slug="acme", is_personal=False)]
    )
    tsdb = _RecordingConn([_Row(org_id="org_acme", workspace_id="ws-1")])

    reachable_orgs(workspace_db, tsdb, user_id="user-1")

    workspace_sql = " ".join(workspace_db.seen).lower()
    tsdb_sql = " ".join(tsdb.seen).lower()

    assert workspace_db.seen, "the workspace DB side was never queried"
    assert tsdb.seen, "the engine side was never queried"
    assert "organizations" not in workspace_sql, (
        "the workspace DB connection was asked about organizations, which does "
        "not exist there"
    )
    assert "workspaces" not in tsdb_sql, (
        "the engine connection was asked about workspaces, which does not exist there"
    )
    assert "workspace_members" not in tsdb_sql, (
        "the engine connection was asked about workspace_members, which does "
        "not exist there"
    )
