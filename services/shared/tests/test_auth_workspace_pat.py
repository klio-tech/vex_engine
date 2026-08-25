"""Workspace personal access tokens are engine credentials.

A PAT (Plane's ``api_tokens`` row, plaintext ``plane_api_<32hex>``) is a USER
credential: ``KeyInfo.user_id`` is the token's workspace user and the org is
the token's workspace org (or the user's personal workspace for a
workspace-less token). The engine-keys lookup stays FIRST; the PAT arm runs
only on an engine-keys miss, and an unset workspace engine means PATs simply
do not validate while engine keys are untouched.

The workspace tables (``api_tokens``, ``workspaces``, ``workspace_members``)
do not exist in the engine's database, so the fixture creates real,
workspace-shaped tables inside a transaction that is always rolled back —
the ``test_membership_resolution`` pattern — and hands the validator two
RECORDING engines that proxy that one connection. Recording matters twice:
it is how the short-circuit test proves the workspace side was never asked,
and how the connection-guard test proves neither side was asked about the
other's tables (the two-databases-joined-in-Python contract that a single
local Postgres cannot enforce by itself).
"""

import hashlib
import json
import threading
import uuid
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text

from shared.auth import AuthError, KeyValidator
from shared.testing import database_url, requires_db

DB_URL = database_url()

USER = "3f7f5a2e-0000-4000-8000-0000000000aa"
OTHER_USER = "3f7f5a2e-0000-4000-8000-0000000000bb"

TEAM_WS = "beadfeed-0000-4000-8000-000000000001"
PERSONAL_WS = "beadfeed-0000-4000-8000-000000000002"

TEAM_SLUG = "pat-team-alpha"
PERSONAL_SLUG = "pat-personal-owner"


def _pat() -> str:
    return f"plane_api_{uuid.uuid4().hex}"


class _RecordingConn:
    """Proxy one live connection, recording every statement it is asked."""

    def __init__(self, conn, statements, fail_on=None):
        self._conn = conn
        self._statements = statements
        self._fail_on = fail_on

    def execute(self, sql, params=None):
        stmt = str(sql)
        self._statements.append(stmt)
        if self._fail_on is not None and self._fail_on in stmt:
            raise RuntimeError(f"injected failure on {self._fail_on!r}")
        return self._conn.execute(sql, params or {})

    def commit(self):
        # The outer test transaction owns commit/rollback.
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _RecordingEngine:
    """An Engine-shaped proxy over the fixture's single rolled-back connection."""

    def __init__(self, conn):
        self._conn = conn
        self.statements: list[str] = []
        self.fail_on: str | None = None

    def connect(self):
        return _RecordingConn(self._conn, self.statements, self.fail_on)

    def dispose(self):
        pass


class _PatFixture:
    def __init__(self, conn):
        self.conn = conn
        self.engine_side = _RecordingEngine(conn)
        self.workspace_side = _RecordingEngine(conn)


@pytest.fixture()
def pat_fixture():
    """Real workspace-shaped tables in a transaction always rolled back."""
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
        conn.execute(
            text(
                """
                CREATE TABLE api_tokens (
                    id uuid PRIMARY KEY,
                    token varchar UNIQUE NOT NULL,
                    user_id uuid NOT NULL,
                    workspace_id uuid,
                    is_active boolean NOT NULL DEFAULT true,
                    expired_at timestamptz,
                    last_used timestamptz,
                    deleted_at timestamptz
                )
                """
            )
        )
        for ws_id, slug, owner, is_personal in (
            (TEAM_WS, TEAM_SLUG, USER, False),
            (PERSONAL_WS, PERSONAL_SLUG, USER, True),
        ):
            conn.execute(
                text(
                    "INSERT INTO workspaces (id, slug, owner_id, is_personal) "
                    "VALUES (:id, :slug, :owner, :personal)"
                ),
                {"id": ws_id, "slug": slug, "owner": owner, "personal": is_personal},
            )
            conn.execute(
                text(
                    "INSERT INTO workspace_members (workspace_id, member_id) "
                    "VALUES (:ws, :uid)"
                ),
                {"ws": ws_id, "uid": USER},
            )
        yield _PatFixture(conn)
    finally:
        trans.rollback()
        conn.close()
        engine.dispose()


def _insert_token(
    conn,
    *,
    token: str,
    user_id: str = USER,
    workspace_id: str | None = TEAM_WS,
    is_active: bool = True,
    expired_at: str | None = None,
    deleted_at: str | None = None,
) -> str:
    token_id = str(uuid.uuid4())
    conn.execute(
        text(
            "INSERT INTO api_tokens "
            "(id, token, user_id, workspace_id, is_active, expired_at, deleted_at) "
            "VALUES (:id, :token, :uid, :ws, :active, "
            "CAST(:expired AS timestamptz), CAST(:deleted AS timestamptz))"
        ),
        {
            "id": token_id,
            "token": token,
            "uid": user_id,
            "ws": workspace_id,
            "active": is_active,
            "expired": expired_at,
            "deleted": deleted_at,
        },
    )
    return token_id


def _validator(fx: _PatFixture, *, workspace_engine: bool = True) -> KeyValidator:
    v = KeyValidator.__new__(KeyValidator)
    v._required_scope = "memory"
    v._cache = {}
    v._cache_ttl_s = 0
    v._cache_lock = threading.Lock()
    v._warn_lock = threading.Lock()
    v._warned_supabase_unconfigured = False
    v._warned_workspace_unconfigured = False
    v._engine = fx.engine_side
    v._supabase_engine = None
    v._workspace_engine = fx.workspace_side if workspace_engine else None
    # neutralize the enforcement stages that need infra
    v._check_rate_limit = MagicMock()
    v._check_quota = MagicMock()
    v._check_agent_limit = MagicMock()
    v._track_usage = MagicMock()
    v._usage_lock = MagicMock()
    return v


# ---------------------------------------------------------------------------
# Rule 1: a valid active token authenticates as the workspace's org
# ---------------------------------------------------------------------------


@requires_db("the workspace PAT tests")
def test_active_workspace_token_authenticates_and_ensures_the_org(pat_fixture):
    raw = _pat()
    _insert_token(pat_fixture.conn, token=raw, workspace_id=TEAM_WS)
    v = _validator(pat_fixture)

    info = v.validate(raw)

    assert info.org_id == f"ws_{TEAM_WS}"
    row = pat_fixture.conn.execute(
        text("SELECT org_id, account_slug FROM organizations WHERE workspace_id = :ws"),
        {"ws": TEAM_WS},
    ).fetchone()
    assert row is not None
    assert row.org_id == info.org_id
    assert row.account_slug == TEAM_SLUG


@requires_db("the workspace PAT tests")
def test_already_linked_org_is_reused_not_reminted(pat_fixture):
    pat_fixture.conn.execute(
        text(
            "INSERT INTO organizations (org_id, name, account_slug, workspace_id) "
            "VALUES ('pat-existing-org', :slug, :slug, CAST(:ws AS uuid))"
        ),
        {"slug": TEAM_SLUG, "ws": TEAM_WS},
    )
    raw = _pat()
    _insert_token(pat_fixture.conn, token=raw, workspace_id=TEAM_WS)

    info = _validator(pat_fixture).validate(raw)

    assert info.org_id == "pat-existing-org"


# ---------------------------------------------------------------------------
# Rule 2: workspace-less token resolves the user's personal workspace
# ---------------------------------------------------------------------------


@requires_db("the workspace PAT tests")
def test_workspace_less_token_lands_on_the_personal_workspace_org(pat_fixture):
    raw = _pat()
    _insert_token(pat_fixture.conn, token=raw, workspace_id=None)

    info = _validator(pat_fixture).validate(raw)

    assert info.org_id == f"ws_{PERSONAL_WS}"


@requires_db("the workspace PAT tests")
def test_workspace_less_token_with_no_personal_workspace_is_401(pat_fixture):
    raw = _pat()
    _insert_token(pat_fixture.conn, token=raw, user_id=OTHER_USER, workspace_id=None)

    with pytest.raises(AuthError) as e:
        _validator(pat_fixture).validate(raw)
    assert e.value.status_code == 401
    assert e.value.detail == "Invalid API key"


# ---------------------------------------------------------------------------
# Rule 3: dead tokens are the SAME 401 as unknown keys — no oracle
# ---------------------------------------------------------------------------


@requires_db("the workspace PAT tests")
@pytest.mark.parametrize(
    "kwargs",
    [
        {"is_active": False},
        {"expired_at": "2020-01-01T00:00:00+00:00"},
        {"deleted_at": "2020-01-01T00:00:00+00:00"},
    ],
    ids=["inactive", "expired", "soft-deleted"],
)
def test_dead_token_is_indistinguishable_from_unknown(pat_fixture, kwargs):
    raw = _pat()
    _insert_token(pat_fixture.conn, token=raw, **kwargs)
    v = _validator(pat_fixture)

    with pytest.raises(AuthError) as dead:
        v.validate(raw)
    with pytest.raises(AuthError) as unknown:
        v.validate(_pat())

    assert dead.value.status_code == 401
    assert dead.value.status_code == unknown.value.status_code
    assert dead.value.detail == unknown.value.detail


@requires_db("the workspace PAT tests")
def test_future_expiry_still_validates(pat_fixture):
    raw = _pat()
    _insert_token(pat_fixture.conn, token=raw, expired_at="2099-01-01T00:00:00+00:00")
    info = _validator(pat_fixture).validate(raw)
    assert info.org_id == f"ws_{TEAM_WS}"


# ---------------------------------------------------------------------------
# Rule 4: the KeyInfo shape — user, scopes, key_id, plan path
# ---------------------------------------------------------------------------


@requires_db("the workspace PAT tests")
def test_keyinfo_carries_user_scopes_token_id_and_plan(pat_fixture):
    raw = _pat()
    token_id = _insert_token(pat_fixture.conn, token=raw, workspace_id=TEAM_WS)

    info = _validator(pat_fixture).validate(raw)

    assert info.user_id == USER
    assert info.scopes == ["memory", "read"]
    assert info.key_id == token_id
    # No Supabase engine configured -> the existing org-plan path's
    # documented fallback.
    assert info.plan == "free"


# ---------------------------------------------------------------------------
# Rule 5: engine-keys lookup stays first; PAT only on miss
# ---------------------------------------------------------------------------


@requires_db("the workspace PAT tests")
def test_engine_key_hit_short_circuits_the_pat_arm(pat_fixture):
    raw = "ag_live_short_circuit_" + uuid.uuid4().hex
    key_hash = hashlib.sha256(raw.encode()).hexdigest()
    entry = {
        "id": str(uuid.uuid4()),
        "key_hash": key_hash,
        "scopes": ["memory"],
        "revoked": False,
    }
    pat_fixture.conn.execute(
        text(
            "INSERT INTO organizations (org_id, name, account_slug, api_keys) "
            "VALUES ('pat-engine-key-org', 'n', 'pat-engine-key-org', "
            "CAST(:keys AS jsonb))"
        ),
        {"keys": json.dumps([entry])},
    )
    # The same raw string also exists as a workspace token; the engine key
    # must win without the workspace side ever being consulted.
    _insert_token(pat_fixture.conn, token=raw, workspace_id=TEAM_WS)

    info = _validator(pat_fixture).validate(raw)

    assert info.org_id == "pat-engine-key-org"
    assert pat_fixture.workspace_side.statements == []


# ---------------------------------------------------------------------------
# Rule 6: no workspace engine -> PAT is 401, engine keys unaffected
# ---------------------------------------------------------------------------


@requires_db("the workspace PAT tests")
def test_without_a_workspace_engine_pats_401_and_engine_keys_survive(pat_fixture):
    pat_raw = _pat()
    _insert_token(pat_fixture.conn, token=pat_raw, workspace_id=TEAM_WS)

    key_raw = "ag_live_no_ws_engine_" + uuid.uuid4().hex
    entry = {
        "id": str(uuid.uuid4()),
        "key_hash": hashlib.sha256(key_raw.encode()).hexdigest(),
        "scopes": ["memory"],
        "revoked": False,
    }
    pat_fixture.conn.execute(
        text(
            "INSERT INTO organizations (org_id, name, account_slug, api_keys) "
            "VALUES ('pat-no-ws-org', 'n', 'pat-no-ws-org', CAST(:keys AS jsonb))"
        ),
        {"keys": json.dumps([entry])},
    )

    v = _validator(pat_fixture, workspace_engine=False)

    with pytest.raises(AuthError) as e:
        v.validate(pat_raw)
    assert e.value.status_code == 401

    assert v.validate(key_raw).org_id == "pat-no-ws-org"


# ---------------------------------------------------------------------------
# Rule 7: neither connection is asked about the other side's tables
# ---------------------------------------------------------------------------


@requires_db("the workspace PAT tests")
def test_each_connection_only_sees_its_own_tables(pat_fixture):
    raw = _pat()
    _insert_token(pat_fixture.conn, token=raw, workspace_id=TEAM_WS)

    _validator(pat_fixture).validate(raw)

    workspace_tables = ("api_tokens", "workspaces", "workspace_members")
    engine_tables = ("organizations",)

    assert pat_fixture.workspace_side.statements, "PAT arm never ran"
    for stmt in pat_fixture.workspace_side.statements:
        for table in engine_tables:
            assert table not in stmt, (
                f"workspace connection was asked about engine table {table}: {stmt}"
            )
    for stmt in pat_fixture.engine_side.statements:
        for table in workspace_tables:
            assert table not in stmt, (
                f"engine connection was asked about workspace table {table}: {stmt}"
            )


# ---------------------------------------------------------------------------
# Rule 8: last_used is best-effort — a failing UPDATE never fails validation
# ---------------------------------------------------------------------------


@requires_db("the workspace PAT tests")
def test_last_used_is_recorded(pat_fixture):
    raw = _pat()
    token_id = _insert_token(pat_fixture.conn, token=raw, workspace_id=TEAM_WS)

    _validator(pat_fixture).validate(raw)

    row = pat_fixture.conn.execute(
        text("SELECT last_used FROM api_tokens WHERE id = :id"), {"id": token_id}
    ).fetchone()
    assert row.last_used is not None


@requires_db("the workspace PAT tests")
def test_failing_last_used_update_never_fails_validation(pat_fixture):
    raw = _pat()
    _insert_token(pat_fixture.conn, token=raw, workspace_id=TEAM_WS)
    pat_fixture.workspace_side.fail_on = "UPDATE api_tokens"

    info = _validator(pat_fixture).validate(raw)

    assert info.org_id == f"ws_{TEAM_WS}"
