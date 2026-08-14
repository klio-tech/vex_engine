"""Tests for the proposals REST review surface (``app.proposals``).

Mock-based, mirroring ``tests/test_capture.py``: no live Postgres. The
``client`` fixture (``conftest.py``) overrides ``verify_api_key`` with an
unattributed key (``org_id='test-org'``, no ``user_id``) — tests that need
an attributable caller build their own client with a user-bearing
``KeyInfo`` override, matching ``tests/test_identity.py``'s pattern.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import app.proposals as proposals
from app.auth import verify_api_key
from app.main import create_app
from fastapi.testclient import TestClient
from shared.auth import KeyInfo

USER = "11111111-1111-1111-1111-111111111111"
PROPOSAL_ID = "22222222-2222-2222-2222-222222222222"
MEMORY_ID = "33333333-3333-3333-3333-333333333333"


def _attributed_client() -> TestClient:
    """A TestClient whose key resolves to an attributable user."""
    app = create_app()
    app.dependency_overrides[verify_api_key] = lambda: KeyInfo(
        org_id="test-org", key_id="test-key", scopes=["ingest"], user_id=USER,
    )
    return TestClient(app)


def _patch_db():
    session = MagicMock()
    return patch("app.proposals.SessionLocal", MagicMock(return_value=session)), session


def _open_proposal_row(kind="add", target_memory_id=None, scope="org"):
    return {
        "id": PROPOSAL_ID, "org_id": "test-org", "scope": scope, "kind": kind,
        "target_memory_id": target_memory_id, "diff": "proposed change",
        "status": "open",
    }


# ---------------------------------------------------------------------------
# GET /proposals
# ---------------------------------------------------------------------------


def test_list_proposals_defaults_to_open_and_is_org_scoped(client):
    db_patch, session = _patch_db()
    with db_patch, patch.object(
        proposals, "_list_proposals", return_value=[{"id": PROPOSAL_ID, "status": "open"}]
    ) as list_fn:
        resp = client.get("/proposals")

    assert resp.status_code == 200
    assert resp.json()["proposals"] == [{"id": PROPOSAL_ID, "status": "open"}]
    assert list_fn.call_args.kwargs["org_id"] == "test-org"
    assert list_fn.call_args.kwargs["status"] == "open"
    session.close.assert_called_once()


def test_list_proposals_honors_status_query_param(client):
    db_patch, _session = _patch_db()
    with db_patch, patch.object(proposals, "_list_proposals", return_value=[]) as list_fn:
        resp = client.get("/proposals?status=rejected")

    assert resp.status_code == 200
    assert list_fn.call_args.kwargs["status"] == "rejected"


# ---------------------------------------------------------------------------
# POST /proposals/{id}/approve
# ---------------------------------------------------------------------------


def test_approve_without_attributable_user_is_403(client):
    """The default `client` fixture's key has no user_id."""
    resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")
    assert resp.status_code == 403


def test_approve_malformed_id_is_404():
    client = _attributed_client()
    resp = client.post("/proposals/not-a-uuid/approve")
    assert resp.status_code == 404


def test_approve_unknown_proposal_is_404():
    client = _attributed_client()
    db_patch, _session = _patch_db()
    with db_patch, patch.object(proposals, "_load_proposal", return_value=None):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")
    assert resp.status_code == 404


def test_approve_already_decided_is_409():
    client = _attributed_client()
    db_patch, _session = _patch_db()
    row = _open_proposal_row()
    row["status"] = "approved"
    with db_patch, patch.object(proposals, "_load_proposal", return_value=row):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")
    assert resp.status_code == 409


def test_approve_add_kind_writes_new_memory_and_records_decided_by():
    client = _attributed_client()
    db_patch, session = _patch_db()
    write = AsyncMock(return_value={"stored": True, "id": "new-mem", "scope": "org"})
    with (
        db_patch,
        patch.object(proposals, "_load_proposal", return_value=_open_proposal_row(kind="add")),
        patch.object(proposals.brain, "write_scoped_memory", write),
        patch.object(proposals, "_mark_decided") as mark,
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")

    assert resp.status_code == 200
    assert resp.json()["status"] == "approved"
    write.assert_awaited_once()
    kw = write.await_args.kwargs
    assert kw["org_id"] == "test-org"
    assert kw["content"] == "proposed change"
    assert kw["provenance"] == "INFERRED"
    mark.assert_called_once()
    assert mark.call_args.kwargs["decided_by"] == USER
    assert mark.call_args.kwargs["status"] == "approved"


def test_approve_revise_kind_applies_via_cas_with_target_revision():
    client = _attributed_client()
    db_patch, session = _patch_db()
    row = _open_proposal_row(kind="revise", target_memory_id=MEMORY_ID, scope="agent")
    with (
        db_patch,
        patch.object(proposals, "_load_proposal", return_value=row),
        patch.object(proposals, "_target_row",
                     return_value={"scope": "agent", "revision": "rev-1"}),
        patch.object(proposals, "supersede_with_revision",
                     return_value={"conflict": False, "stored": True,
                                   "new_id": "m-new", "new_revision": "rev-2"}) as sup,
        patch.object(proposals, "_mark_decided") as mark,
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")

    assert resp.status_code == 200
    sup.assert_called_once()
    kwargs = sup.call_args.kwargs
    assert kwargs["org_id"] == "test-org"
    assert kwargs["memory_id"] == MEMORY_ID
    assert kwargs["base_revision"] == "rev-1"
    assert kwargs["user_id"] == USER
    mark.assert_called_once()
    assert mark.call_args.kwargs["status"] == "approved"


def test_approve_revise_kind_conflict_returns_409_and_does_not_decide():
    client = _attributed_client()
    db_patch, _session = _patch_db()
    row = _open_proposal_row(kind="revise", target_memory_id=MEMORY_ID, scope="org")
    with (
        db_patch,
        patch.object(proposals, "_load_proposal", return_value=row),
        patch.object(proposals, "_target_row",
                     return_value={"scope": "org", "revision": "rev-1"}),
        patch.object(proposals, "supersede_with_revision",
                     return_value={"conflict": True, "stored": False,
                                   "current": {"id": MEMORY_ID}}),
        patch.object(proposals, "_mark_decided") as mark,
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")

    assert resp.status_code == 409
    mark.assert_not_called()


def test_approve_revise_kind_missing_target_row_is_409():
    client = _attributed_client()
    db_patch, _session = _patch_db()
    row = _open_proposal_row(kind="revise", target_memory_id=MEMORY_ID, scope="org")
    with (
        db_patch,
        patch.object(proposals, "_load_proposal", return_value=row),
        patch.object(proposals, "_target_row", return_value=None),
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# POST /proposals/{id}/reject
# ---------------------------------------------------------------------------


def test_reject_without_attributable_user_is_403(client):
    resp = client.post(f"/proposals/{PROPOSAL_ID}/reject")
    assert resp.status_code == 403


def test_reject_marks_decided_with_no_write(client=None):
    client = _attributed_client()
    db_patch, _session = _patch_db()
    with (
        db_patch,
        patch.object(proposals, "_load_proposal", return_value=_open_proposal_row()),
        patch.object(proposals, "_mark_decided") as mark,
        patch.object(proposals.brain, "write_scoped_memory") as write,
        patch.object(proposals, "supersede_with_revision") as sup,
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/reject")

    assert resp.status_code == 200
    assert resp.json()["status"] == "rejected"
    mark.assert_called_once()
    assert mark.call_args.kwargs["status"] == "rejected"
    assert mark.call_args.kwargs["decided_by"] == USER
    write.assert_not_called()
    sup.assert_not_called()


def test_reject_unknown_proposal_is_404():
    client = _attributed_client()
    db_patch, _session = _patch_db()
    with db_patch, patch.object(proposals, "_load_proposal", return_value=None):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/reject")
    assert resp.status_code == 404


def test_reject_already_decided_is_409():
    client = _attributed_client()
    db_patch, _session = _patch_db()
    row = _open_proposal_row()
    row["status"] = "rejected"
    with db_patch, patch.object(proposals, "_load_proposal", return_value=row):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/reject")
    assert resp.status_code == 409
