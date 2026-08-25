"""The selected node is re-checked on every call, not at switch time.

This is the test that has to survive: a session outlives a membership
revocation, so checking once when the caller switches and trusting it
afterwards is the same as not checking.

Every selection test here names a node that is NOT the key's own org. Selecting
the key's own org short-circuits before membership is consulted -- correctly,
since that is today's behaviour for every existing caller -- so a revocation
test written against it would pass against a resolver that never checks
anything at all.

The two workspace-DB tables (``workspaces``, ``workspace_members``) do not
exist in the engine's database, so the fixture creates them for real inside a
transaction that is always rolled back, exactly as
``services/shared/tests/test_membership_resolution.py`` does. Stubbing the
session instead would match SQL by substring and could not catch a wrong JOIN.
"""

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, text

from shared.testing import database_url, requires_db

from app import identity as identity_module

DB_URL = database_url()

SUBJECT_USER = "44444444-4444-4444-4444-444444444444"
OTHER_USER = "55555555-5555-5555-5555-555555555555"

KEY_SLUG = "node-team"
OTHER_NODE_SLUG = "node-personal"
STRANGER_SLUG = "node-stranger"

KEY_ORG = "org_node_team"
OTHER_NODE_ORG = "org_node_personal"
STRANGER_ORG = "org_node_stranger"

KEY_WORKSPACE_ID = "bbbbbbbb-0000-0000-0000-000000000001"
PERSONAL_WORKSPACE_ID = "bbbbbbbb-0000-0000-0000-000000000002"
STRANGER_WORKSPACE_ID = "bbbbbbbb-0000-0000-0000-000000000003"

# A workspace the subject can reach that has NO org row yet: the provisioning
# moment. And one of the stranger's in the same state, to prove selection by
# workspace id does not bypass membership.
ORGLESS_SLUG = "node-orgless"
ORGLESS_WORKSPACE_ID = "bbbbbbbb-0000-0000-0000-000000000004"
STRANGER_ORGLESS_WORKSPACE_ID = "bbbbbbbb-0000-0000-0000-000000000005"


class _State:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Req:
    def __init__(self, headers=None, state=None):
        self.headers = headers or {}
        self.state = state or _State()


class _Node:
    """Unpacks as ``(user_id, key_org, other_org)``; carries the connection."""

    def __init__(self, db, user_id, key_org, other_org):
        self.db = db
        self.user_id = user_id
        self.key_org = key_org
        self.other_org = other_org

    def __iter__(self):
        return iter((self.user_id, self.key_org, self.other_org))


@pytest.fixture()
def node_fixture(monkeypatch):
    """Real workspace-DB-shaped tables plus a patched principal, rolled back after.

    The principal is patched rather than minted because this suite is about
    what happens AFTER identity is established: whose key it is and who the
    human is are inputs here, not the thing under test.
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
                    owner_id uuid,
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

        for workspace_id, slug, is_personal, owner_id in (
            (KEY_WORKSPACE_ID, KEY_SLUG, False, SUBJECT_USER),
            (PERSONAL_WORKSPACE_ID, OTHER_NODE_SLUG, True, SUBJECT_USER),
            (STRANGER_WORKSPACE_ID, STRANGER_SLUG, False, OTHER_USER),
            (ORGLESS_WORKSPACE_ID, ORGLESS_SLUG, False, SUBJECT_USER),
            (STRANGER_ORGLESS_WORKSPACE_ID, "node-stranger-orgless", False, OTHER_USER),
        ):
            conn.execute(
                text(
                    "INSERT INTO workspaces (id, slug, owner_id, is_personal) "
                    "VALUES (:id, :slug, :owner, :personal)"
                ),
                {
                    "id": workspace_id,
                    "slug": slug,
                    "owner": owner_id,
                    "personal": is_personal,
                },
            )

        for workspace_id, member_id in (
            (KEY_WORKSPACE_ID, SUBJECT_USER),
            (PERSONAL_WORKSPACE_ID, SUBJECT_USER),
            (STRANGER_WORKSPACE_ID, OTHER_USER),
            (ORGLESS_WORKSPACE_ID, SUBJECT_USER),
            (STRANGER_ORGLESS_WORKSPACE_ID, OTHER_USER),
        ):
            conn.execute(
                text(
                    "INSERT INTO workspace_members (workspace_id, member_id) "
                    "VALUES (:workspace_id, :member_id)"
                ),
                {"workspace_id": workspace_id, "member_id": member_id},
            )

        for org_id, slug, workspace_id in (
            (KEY_ORG, KEY_SLUG, KEY_WORKSPACE_ID),
            (OTHER_NODE_ORG, OTHER_NODE_SLUG, PERSONAL_WORKSPACE_ID),
            (STRANGER_ORG, STRANGER_SLUG, STRANGER_WORKSPACE_ID),
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

        monkeypatch.setattr(
            identity_module,
            "resolve_principal_plan",
            lambda request: (KEY_ORG, "agent-1", "free", None),
        )
        monkeypatch.setattr(
            identity_module,
            "resolve_request_user_id",
            lambda request: SUBJECT_USER,
        )

        yield _Node(conn, SUBJECT_USER, KEY_ORG, OTHER_NODE_ORG)
    finally:
        trans.rollback()
        conn.close()
        engine.dispose()


@pytest.fixture()
def revoke():
    """Deactivate the membership row backing a node, mid-test.

    ``is_active = false`` rather than DELETE because that is what Plane does:
    revocation is a deactivation, and the resolver must treat it as gone.
    """

    def _revoke(db, user_id, workspace_id):
        db.execute(
            text(
                "UPDATE workspace_members SET is_active = false "
                "WHERE member_id = :member_id AND workspace_id = :workspace_id"
            ),
            {"member_id": user_id, "workspace_id": workspace_id},
        )

    return _revoke


@pytest.fixture()
def unattributable(monkeypatch):
    """Make the caller resolve to no human at all."""
    monkeypatch.setattr(
        identity_module, "resolve_request_user_id", lambda request: None
    )


needs_db = requires_db("the node selection tests")


@needs_db
def test_without_a_selection_the_keys_org_is_used(node_fixture):
    # Preserves today's behaviour exactly for every existing caller.
    _user_id, key_org, _other = node_fixture
    got = identity_module.resolve_active_org(_Req({}), node_fixture.db)
    assert got == key_org


@needs_db
def test_without_a_selection_the_workspace_db_is_never_consulted(node_fixture):
    """The unselected path must not acquire a workspace DB connection at all.

    A deployment with no WORKSPACE_DATABASE_URL is the common case today, and
    it must keep working untouched.
    """

    def _explode():  # pragma: no cover - only runs if the contract breaks
        raise AssertionError("the unselected path opened a workspace DB connection")

    import app.workspace_db as workspace_db_module

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(workspace_db_module, "workspace_engine", lambda: _explode())
        assert (
            identity_module.resolve_active_org(_Req({}), node_fixture.db)
            == node_fixture.key_org
        )


@needs_db
def test_selecting_the_keys_own_org_is_a_no_op(node_fixture):
    _user_id, key_org, _other = node_fixture
    got = identity_module.resolve_active_org(
        _Req({"X-Klio-Node": key_org}), node_fixture.db, workspace_db=node_fixture.db
    )
    assert got == key_org


@needs_db
def test_a_reachable_node_may_be_selected(node_fixture):
    # A node that is NOT the key's org, so membership is genuinely consulted.
    _user_id, _key_org, other_org = node_fixture
    got = identity_module.resolve_active_org(
        _Req({"X-Klio-Node": other_org}), node_fixture.db, workspace_db=node_fixture.db
    )
    assert got == other_org


@needs_db
def test_the_legacy_vex_header_selects_too(node_fixture):
    _user_id, _key_org, other_org = node_fixture
    got = identity_module.resolve_active_org(
        _Req({"X-Vex-Node": other_org}), node_fixture.db, workspace_db=node_fixture.db
    )
    assert got == other_org


@needs_db
def test_an_unreachable_node_is_not_found(node_fixture):
    with pytest.raises(HTTPException) as exc:
        identity_module.resolve_active_org(
            _Req({"X-Klio-Node": STRANGER_ORG}),
            node_fixture.db,
            workspace_db=node_fixture.db,
        )
    # not_found, not forbidden: 403 confirms the org exists, which is an
    # existence oracle. The project tools already answer this way.
    assert exc.value.status_code == 404


@needs_db
def test_revocation_takes_effect_on_the_very_next_call(node_fixture, revoke):
    user_id, _key_org, other_org = node_fixture
    req = _Req({"X-Klio-Node": other_org})
    assert (
        identity_module.resolve_active_org(
            req, node_fixture.db, workspace_db=node_fixture.db
        )
        == other_org
    )

    revoke(node_fixture.db, user_id, PERSONAL_WORKSPACE_ID)

    # Same request object: nothing memoized on it may be trusted either.
    with pytest.raises(HTTPException) as exc:
        identity_module.resolve_active_org(
            req, node_fixture.db, workspace_db=node_fixture.db
        )
    assert exc.value.status_code == 404


@needs_db
def test_an_unattributable_caller_cannot_select_a_node(node_fixture, unattributable):
    _user_id, _key_org, other_org = node_fixture
    with pytest.raises(HTTPException) as exc:
        identity_module.resolve_active_org(
            _Req({"X-Klio-Node": other_org}),
            node_fixture.db,
            workspace_db=node_fixture.db,
        )
    assert exc.value.status_code == 404


@needs_db
def test_an_unconfigured_workspace_db_fails_closed(node_fixture, monkeypatch):
    """No workspace DB means the membership check cannot run: nothing is reachable.

    Failing open here would let any caller select any node simply by pointing
    the engine at a deployment with no workspace DB configured.
    """
    import app.workspace_db as workspace_db_module

    monkeypatch.setattr(workspace_db_module, "workspace_engine", lambda: None)
    with pytest.raises(HTTPException) as exc:
        identity_module.resolve_active_org(
            _Req({"X-Klio-Node": node_fixture.other_org}), node_fixture.db
        )
    assert exc.value.status_code == 404


def _org_row(db, org_id):
    return db.execute(
        text(
            "SELECT org_id, account_slug, workspace_id FROM organizations "
            "WHERE org_id = :org_id"
        ),
        {"org_id": org_id},
    ).first()


@needs_db
def test_selecting_an_orgless_workspace_provisions_its_org(node_fixture):
    """Selection is the provisioning moment.

    The header names a workspace the caller can reach that has no org row;
    the resolver must mint one and return it — linked, so the very next
    membership resolution can see it.
    """
    got = identity_module.resolve_active_org(
        _Req({"X-Klio-Node": ORGLESS_WORKSPACE_ID}),
        node_fixture.db,
        workspace_db=node_fixture.db,
    )
    assert got == f"ws_{ORGLESS_WORKSPACE_ID}"
    row = _org_row(node_fixture.db, got)
    assert row is not None
    assert row.account_slug == ORGLESS_SLUG
    assert str(row.workspace_id) == ORGLESS_WORKSPACE_ID


@needs_db
def test_the_workspace_may_be_named_by_its_future_org_id(node_fixture):
    got = identity_module.resolve_active_org(
        _Req({"X-Klio-Node": f"ws_{ORGLESS_WORKSPACE_ID}"}),
        node_fixture.db,
        workspace_db=node_fixture.db,
    )
    assert got == f"ws_{ORGLESS_WORKSPACE_ID}"


@needs_db
def test_the_workspace_may_be_named_by_its_slug(node_fixture):
    # The slug is the address the caller actually knows. It is matched against
    # the caller's OWN memberships (fetched from the workspace DB), never used
    # as a cross-database join key.
    got = identity_module.resolve_active_org(
        _Req({"X-Klio-Node": ORGLESS_SLUG}),
        node_fixture.db,
        workspace_db=node_fixture.db,
    )
    assert got == f"ws_{ORGLESS_WORKSPACE_ID}"


@needs_db
def test_provisioning_adopts_a_legacy_org_by_slug(node_fixture):
    # An org row with the workspace's slug but no link joins the handshake.
    node_fixture.db.execute(
        text(
            "INSERT INTO organizations (org_id, name, account_slug) "
            "VALUES ('org_legacy_node', 'org_legacy_node', :slug)"
        ),
        {"slug": ORGLESS_SLUG},
    )
    got = identity_module.resolve_active_org(
        _Req({"X-Klio-Node": ORGLESS_WORKSPACE_ID}),
        node_fixture.db,
        workspace_db=node_fixture.db,
    )
    assert got == "org_legacy_node"
    assert str(_org_row(node_fixture.db, got).workspace_id) == ORGLESS_WORKSPACE_ID


@needs_db
def test_a_strangers_orgless_workspace_is_not_provisioned(node_fixture):
    # Membership gates provisioning exactly as it gates selection.
    for named in (
        STRANGER_ORGLESS_WORKSPACE_ID,
        f"ws_{STRANGER_ORGLESS_WORKSPACE_ID}",
        "node-stranger-orgless",
    ):
        with pytest.raises(HTTPException) as exc:
            identity_module.resolve_active_org(
                _Req({"X-Klio-Node": named}),
                node_fixture.db,
                workspace_db=node_fixture.db,
            )
        assert exc.value.status_code == 404
    # And nothing was minted for it.
    assert _org_row(node_fixture.db, f"ws_{STRANGER_ORGLESS_WORKSPACE_ID}") is None


@needs_db
def test_a_refused_ensure_is_a_404_not_a_500(node_fixture):
    # The workspace's slug is already held by an org linked elsewhere: the
    # ensure refuses, and the caller sees the same shape as any other refusal.
    node_fixture.db.execute(
        text(
            "INSERT INTO organizations (org_id, name, account_slug, workspace_id) "
            "VALUES ('org_conflicting', 'org_conflicting', :slug, "
            "'bbbbbbbb-0000-0000-0000-00000000ffff')"
        ),
        {"slug": ORGLESS_SLUG},
    )
    with pytest.raises(HTTPException) as exc:
        identity_module.resolve_active_org(
            _Req({"X-Klio-Node": ORGLESS_WORKSPACE_ID}),
            node_fixture.db,
            workspace_db=node_fixture.db,
        )
    assert exc.value.status_code == 404
