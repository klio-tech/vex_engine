"""The REST capture and proposals routes act in the SELECTED node.

Task B2 of the context-network wiring: the routes in ``app.capture`` and
``app.proposals`` resolve their principal through
``resolve_active_principal(request, db)`` — the same single resolver the MCP
tool bodies adopted in B1 — so an ``X-Klio-Node`` header redirects the whole
request into the selected node, an absent header keeps today's behaviour
byte-identical, and an unreachable node is a clean 404 with NOTHING written.

The sensitive shape is hook capture: a ``Stop`` hook firing with a node
header must capture INTO that node. A transcript landing in the key's org
while the caller believes they selected another is silent cross-tenant data
placement, which is why the zero-write assertion on the 404 path is part of
the contract, not a nicety.

Seams mirror ``test_active_principal_wiring.py``: identity is patched at
``app.identity`` so the REAL membership check runs against real
workspace-shaped tables inside a rolled-back transaction, while storage is
patched at the route modules and asserted on the ``org_id`` it was handed —
the binding this task exists to move.
"""

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from shared.auth import KeyInfo
from shared.testing import database_url, requires_db

from app import capture as capture_module
from app import identity as identity_module
from app import proposals as proposals_module
from app.auth import verify_api_key
from app.main import create_app

DB_URL = database_url()

SUBJECT_USER = "44444444-4444-4444-4444-444444444444"
OTHER_USER = "55555555-5555-5555-5555-555555555555"

KEY_ORG = "org_capture_team"
OTHER_NODE_ORG = "org_capture_personal"
STRANGER_ORG = "org_capture_stranger"

KEY_WORKSPACE_ID = "dddddddd-0000-0000-0000-000000000001"
PERSONAL_WORKSPACE_ID = "dddddddd-0000-0000-0000-000000000002"
STRANGER_WORKSPACE_ID = "dddddddd-0000-0000-0000-000000000003"


class _KeepOpen:
    """The fixture connection with ``close()`` neutered.

    The route bodies close the session they open — correct per-request, fatal
    for a fixture-owned connection the rollback still needs.
    """

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):  # noqa: D102 - the whole point of the wrapper
        pass


def _fake_key_info() -> KeyInfo:
    """The dependency-injected key: key-org tenant, deliberately unowned.

    ``user_id=None`` keeps the hook-scope routing on its unattributable arm,
    so the capture routes exercise none of the project-scope machinery — the
    org binding is the only thing under test here.
    """
    return KeyInfo(
        org_id=KEY_ORG,
        key_id="test-key",
        scopes=["ingest", "memory"],
        plan="free",
        user_id=None,
    )


@pytest.fixture()
def wired_client(monkeypatch):
    """TestClient over real workspace-shaped tables, rolled back after.

    Identity is patched where :mod:`app.identity` looks it up, so the
    membership check inside ``resolve_active_org`` runs for real; the route
    dependencies keep their normal test seam (``dependency_overrides`` on
    ``verify_api_key``, exactly as ``conftest.client`` does).
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
        for workspace_id, slug, owner in (
            (KEY_WORKSPACE_ID, "capture-team", SUBJECT_USER),
            (PERSONAL_WORKSPACE_ID, "capture-personal", SUBJECT_USER),
            (STRANGER_WORKSPACE_ID, "capture-stranger", OTHER_USER),
        ):
            conn.execute(
                text(
                    "INSERT INTO workspaces (id, slug, owner_id) "
                    "VALUES (:id, :slug, :owner)"
                ),
                {"id": workspace_id, "slug": slug, "owner": owner},
            )
        for workspace_id, member in (
            (KEY_WORKSPACE_ID, SUBJECT_USER),
            (PERSONAL_WORKSPACE_ID, SUBJECT_USER),
            (STRANGER_WORKSPACE_ID, OTHER_USER),
        ):
            conn.execute(
                text(
                    "INSERT INTO workspace_members (workspace_id, member_id) "
                    "VALUES (:w, :m)"
                ),
                {"w": workspace_id, "m": member},
            )
        for org_id, slug, workspace_id in (
            (KEY_ORG, "capture-team", KEY_WORKSPACE_ID),
            (OTHER_NODE_ORG, "capture-personal", PERSONAL_WORKSPACE_ID),
            (STRANGER_ORG, "capture-stranger", STRANGER_WORKSPACE_ID),
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
            identity_module, "resolve_request_user_id", lambda request: SUBJECT_USER
        )

        @contextlib.contextmanager
        def _fixture_workspace_db(workspace_db=None):
            yield conn

        monkeypatch.setattr(
            identity_module, "_workspace_connection", _fixture_workspace_db
        )
        keep_open = MagicMock(return_value=_KeepOpen(conn))
        monkeypatch.setattr(capture_module, "SessionLocal", keep_open)
        monkeypatch.setattr(proposals_module, "SessionLocal", keep_open)

        app = create_app()
        app.dependency_overrides[verify_api_key] = _fake_key_info
        yield TestClient(app)
    finally:
        trans.rollback()
        conn.close()
        engine.dispose()


_JOB_ID = "77777777-7777-7777-7777-777777777777"


def _transcript_patches(write_mock, create_mock=None):
    """Neuter everything a transcript capture does EXCEPT the org binding.

    ``create_mock`` is passed in rather than made here so the caller still
    holds it once the patch context has exited.
    """
    create_mock = (
        create_mock if create_mock is not None else MagicMock(return_value=_JOB_ID)
    )
    return (
        patch.object(capture_module, "_ensure_project_id", return_value=None),
        patch.object(capture_module, "_check_memory_quota_402", return_value=False),
        patch.object(capture_module, "record_usage"),
        patch.object(capture_module, "create_job", new=create_mock),
        patch.object(capture_module, "publish_job", new=AsyncMock()),
        patch.object(capture_module, "build_captured_payload", return_value=None),
        patch.object(capture_module.brain, "write_scoped_memory", new=write_mock),
    )


_TRANSCRIPT_BODY = {"messages": [{"role": "user", "content": "we shipped the fix"}]}


needs_db = requires_db("the capture/proposals node-wiring tests")


@needs_db
def test_hook_transcript_captures_into_the_selected_node(wired_client):
    """The org binding is resolved in the REQUEST and carried on the job.

    Distillation runs in another process now, so this is the last point at
    which the selected node is known -- if the wrong org reaches the job row,
    every memory the worker writes from it lands in the wrong tenant.
    """
    write_mock = AsyncMock(return_value={"stored": True, "id": "m1"})
    create_mock = MagicMock(return_value=_JOB_ID)
    p = _transcript_patches(write_mock, create_mock)
    with p[0], p[1], p[2], p[3], p[4], p[5], p[6]:
        resp = wired_client.post(
            "/capture/transcript",
            json=_TRANSCRIPT_BODY,
            headers={"X-Klio-Node": OTHER_NODE_ORG},
        )
    assert resp.status_code == 200
    create_mock.assert_called_once()
    assert create_mock.call_args.kwargs["org_id"] == OTHER_NODE_ORG
    write_mock.assert_not_awaited()


@needs_db
def test_hook_transcript_without_a_header_stays_in_the_keys_org(wired_client):
    write_mock = AsyncMock(return_value={"stored": True, "id": "m1"})
    create_mock = MagicMock(return_value=_JOB_ID)
    p = _transcript_patches(write_mock, create_mock)
    with p[0], p[1], p[2], p[3], p[4], p[5], p[6]:
        resp = wired_client.post("/capture/transcript", json=_TRANSCRIPT_BODY)
    assert resp.status_code == 200
    create_mock.assert_called_once()
    assert create_mock.call_args.kwargs["org_id"] == KEY_ORG


@needs_db
def test_hook_transcript_to_an_unreachable_node_is_404_and_writes_nothing(
    wired_client,
):
    """The wrong-org outcome this task exists to prevent.

    A refused selection must be the 404-shaped refusal AND a zero-write:
    landing the transcript in the key's org instead would be silent
    cross-tenant data placement, reported to the caller as success.
    """
    write_mock = AsyncMock(return_value={"stored": True, "id": "m1"})
    p = _transcript_patches(write_mock)
    with p[0], p[1], p[2], p[3], p[4], p[5], p[6]:
        resp = wired_client.post(
            "/capture/transcript",
            json=_TRANSCRIPT_BODY,
            headers={"X-Klio-Node": STRANGER_ORG},
        )
    assert resp.status_code == 404
    assert "unknown node" in str(resp.json())
    write_mock.assert_not_awaited()


@needs_db
def test_proposals_list_reads_the_selected_node(wired_client):
    list_mock = MagicMock(return_value=[])
    with patch.object(proposals_module, "_list_proposals", new=list_mock):
        resp = wired_client.get("/proposals", headers={"X-Klio-Node": OTHER_NODE_ORG})
    assert resp.status_code == 200
    list_mock.assert_called_once()
    assert list_mock.call_args.kwargs["org_id"] == OTHER_NODE_ORG


@needs_db
def test_proposals_list_without_a_header_reads_the_keys_org(wired_client):
    list_mock = MagicMock(return_value=[])
    with patch.object(proposals_module, "_list_proposals", new=list_mock):
        resp = wired_client.get("/proposals")
    assert resp.status_code == 200
    list_mock.assert_called_once()
    assert list_mock.call_args.kwargs["org_id"] == KEY_ORG


@needs_db
def test_proposals_list_to_an_unreachable_node_is_404_and_reads_nothing(
    wired_client,
):
    list_mock = MagicMock(return_value=[])
    with patch.object(proposals_module, "_list_proposals", new=list_mock):
        resp = wired_client.get("/proposals", headers={"X-Klio-Node": STRANGER_ORG})
    assert resp.status_code == 404
    assert "unknown node" in str(resp.json())
    list_mock.assert_not_called()
