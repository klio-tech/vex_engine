"""The selected node is re-checked on every call, not at switch time.

This is the test that has to survive: a session outlives a membership
revocation, so checking once when the caller switches and trusting it
afterwards is the same as not checking.

Every selection test here names a node that is NOT the key's own org. Selecting
the key's own org short-circuits before membership is consulted -- correctly,
since that is today's behaviour for every existing caller -- so a revocation
test written against it would pass against a resolver that never checks
anything at all.

The two Supabase-side tables (``public.accounts``, ``public.accounts_memberships``)
do not exist in the engine's database, so the fixture creates them for real
inside a transaction that is always rolled back, exactly as
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

PERSONAL_ACCOUNT_ID = "bbbbbbbb-0000-0000-0000-000000000002"


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
    """Real Supabase-shaped tables plus a patched principal, rolled back after.

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
            ("bbbbbbbb-0000-0000-0000-000000000001", KEY_SLUG, False),
            (PERSONAL_ACCOUNT_ID, OTHER_NODE_SLUG, True),
            ("bbbbbbbb-0000-0000-0000-000000000003", STRANGER_SLUG, False),
        ):
            conn.execute(
                text(
                    "INSERT INTO public.accounts (id, slug, is_personal_account) "
                    "VALUES (:id, :slug, :personal)"
                ),
                {"id": account_id, "slug": slug, "personal": is_personal},
            )

        for account_id, user_id in (
            ("bbbbbbbb-0000-0000-0000-000000000001", SUBJECT_USER),
            (PERSONAL_ACCOUNT_ID, SUBJECT_USER),
            ("bbbbbbbb-0000-0000-0000-000000000003", OTHER_USER),
        ):
            conn.execute(
                text(
                    "INSERT INTO public.accounts_memberships (account_id, user_id) "
                    "VALUES (:account_id, :user_id)"
                ),
                {"account_id": account_id, "user_id": user_id},
            )

        for org_id, slug in (
            (KEY_ORG, KEY_SLUG),
            (OTHER_NODE_ORG, OTHER_NODE_SLUG),
            (STRANGER_ORG, STRANGER_SLUG),
        ):
            conn.execute(
                text(
                    "INSERT INTO organizations (org_id, name, account_slug) "
                    "VALUES (:org_id, :name, :slug)"
                ),
                {"org_id": org_id, "name": org_id, "slug": slug},
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
    """Delete the membership row backing a node, mid-test."""

    def _revoke(db, user_id, account_id):
        db.execute(
            text(
                "DELETE FROM public.accounts_memberships "
                "WHERE user_id = :user_id AND account_id = :account_id"
            ),
            {"user_id": user_id, "account_id": account_id},
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
def test_without_a_selection_supabase_is_never_consulted(node_fixture):
    """The unselected path must not acquire a Supabase connection at all.

    A deployment with no SUPABASE_DATABASE_URL is the common case today, and it
    must keep working untouched.
    """

    def _explode():  # pragma: no cover - only runs if the contract breaks
        raise AssertionError("the unselected path opened a Supabase connection")

    import app.directory as directory_module

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(directory_module, "_engines", lambda: _explode())
        assert (
            identity_module.resolve_active_org(_Req({}), node_fixture.db)
            == node_fixture.key_org
        )


@needs_db
def test_selecting_the_keys_own_org_is_a_no_op(node_fixture):
    _user_id, key_org, _other = node_fixture
    got = identity_module.resolve_active_org(
        _Req({"X-Klio-Node": key_org}), node_fixture.db, supabase=node_fixture.db
    )
    assert got == key_org


@needs_db
def test_a_reachable_node_may_be_selected(node_fixture):
    # A node that is NOT the key's org, so membership is genuinely consulted.
    _user_id, _key_org, other_org = node_fixture
    got = identity_module.resolve_active_org(
        _Req({"X-Klio-Node": other_org}), node_fixture.db, supabase=node_fixture.db
    )
    assert got == other_org


@needs_db
def test_the_legacy_vex_header_selects_too(node_fixture):
    _user_id, _key_org, other_org = node_fixture
    got = identity_module.resolve_active_org(
        _Req({"X-Vex-Node": other_org}), node_fixture.db, supabase=node_fixture.db
    )
    assert got == other_org


@needs_db
def test_an_unreachable_node_is_not_found(node_fixture):
    with pytest.raises(HTTPException) as exc:
        identity_module.resolve_active_org(
            _Req({"X-Klio-Node": STRANGER_ORG}),
            node_fixture.db,
            supabase=node_fixture.db,
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
            req, node_fixture.db, supabase=node_fixture.db
        )
        == other_org
    )

    revoke(node_fixture.db, user_id, PERSONAL_ACCOUNT_ID)

    # Same request object: nothing memoized on it may be trusted either.
    with pytest.raises(HTTPException) as exc:
        identity_module.resolve_active_org(
            req, node_fixture.db, supabase=node_fixture.db
        )
    assert exc.value.status_code == 404


@needs_db
def test_an_unattributable_caller_cannot_select_a_node(node_fixture, unattributable):
    _user_id, _key_org, other_org = node_fixture
    with pytest.raises(HTTPException) as exc:
        identity_module.resolve_active_org(
            _Req({"X-Klio-Node": other_org}),
            node_fixture.db,
            supabase=node_fixture.db,
        )
    assert exc.value.status_code == 404


@needs_db
def test_an_unconfigured_directory_fails_closed(node_fixture, monkeypatch):
    """No Supabase means the membership check cannot run, so nothing is reachable.

    Failing open here would let any caller select any node simply by pointing
    the engine at a deployment with no directory configured.
    """
    import app.directory as directory_module

    monkeypatch.setattr(directory_module, "_engines", lambda: (None, None))
    with pytest.raises(HTTPException) as exc:
        identity_module.resolve_active_org(
            _Req({"X-Klio-Node": node_fixture.other_org}), node_fixture.db
        )
    assert exc.value.status_code == 404
