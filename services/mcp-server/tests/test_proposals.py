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


def _attributed_client(raise_server_exceptions: bool = True) -> TestClient:
    """A TestClient whose key resolves to an attributable user.

    ``raise_server_exceptions=False`` for the reverted-claim-then-500 tests:
    the TestClient's default behaviour is to re-raise an unhandled server
    exception into the TEST rather than converting it to a response, which
    is right for catching bugs by surprise but wrong here, where the 500 IS
    the expected, asserted behaviour (mirrors ``tests/test_oauth_middleware.py``'s
    own use of the same flag).
    """
    app = create_app()
    app.dependency_overrides[verify_api_key] = lambda: KeyInfo(
        org_id="test-org",
        key_id="test-key",
        scopes=["ingest"],
        user_id=USER,
    )
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def _patch_db():
    session = MagicMock()
    return patch("app.proposals.SessionLocal", MagicMock(return_value=session)), session


def _open_proposal_row(
    kind="add", target_memory_id=None, scope="org", proposed_content="proposed content"
):
    return {
        "id": PROPOSAL_ID,
        "org_id": "test-org",
        "scope": scope,
        "kind": kind,
        "target_memory_id": target_memory_id,
        "diff": "rationale prose",
        "proposed_content": proposed_content,
        "status": "open",
    }


# ---------------------------------------------------------------------------
# GET /proposals
# ---------------------------------------------------------------------------


def test_list_proposals_defaults_to_open_status_and_default_limit(client):
    db_patch, session = _patch_db()
    with (
        db_patch,
        patch.object(
            proposals,
            "_list_proposals",
            return_value=[{"id": PROPOSAL_ID, "status": "open"}],
        ) as list_fn,
    ):
        resp = client.get("/proposals")

    assert resp.status_code == 200
    assert resp.json()["proposals"] == [{"id": PROPOSAL_ID, "status": "open"}]
    assert list_fn.call_args.kwargs["org_id"] == "test-org"
    assert list_fn.call_args.kwargs["status"] == "open"
    assert list_fn.call_args.kwargs["limit"] == 50
    session.close.assert_called_once()


def test_list_proposals_honors_status_and_limit_query_params(client):
    db_patch, _session = _patch_db()
    with (
        db_patch,
        patch.object(proposals, "_list_proposals", return_value=[]) as list_fn,
    ):
        resp = client.get("/proposals?status=rejected&limit=10")

    assert resp.status_code == 200
    assert list_fn.call_args.kwargs["status"] == "rejected"
    assert list_fn.call_args.kwargs["limit"] == 10


def test_list_proposals_rejects_limit_above_cap(client):
    resp = client.get("/proposals?limit=99999")
    assert resp.status_code == 422


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


def test_approve_add_kind_without_proposed_content_is_422_needs_content():
    client = _attributed_client()
    db_patch, _session = _patch_db()
    row = _open_proposal_row(kind="add", proposed_content=None)
    with (
        db_patch,
        patch.object(proposals, "_load_proposal", return_value=row),
        patch.object(proposals, "_claim_proposal") as claim,
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "needs_content"
    claim.assert_not_called()


def test_approve_add_kind_writes_new_memory_and_records_decided_by():
    client = _attributed_client()
    db_patch, session = _patch_db()
    write = AsyncMock(return_value={"stored": True, "id": "new-mem", "scope": "org"})
    with (
        db_patch,
        patch.object(
            proposals, "_load_proposal", return_value=_open_proposal_row(kind="add")
        ),
        patch.object(proposals, "_claim_proposal", return_value=True) as claim,
        patch.object(proposals.brain, "write_scoped_memory", write),
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")

    assert resp.status_code == 200
    assert resp.json()["status"] == "approved"
    claim.assert_called_once()
    assert claim.call_args.kwargs["decided_by"] == USER
    assert claim.call_args.kwargs["status"] == "approved"
    write.assert_awaited_once()
    kw = write.await_args.kwargs
    assert kw["org_id"] == "test-org"
    assert kw["content"] == "proposed content"
    assert kw["provenance"] == "INFERRED"


def test_approve_add_kind_raises_reverts_claim_and_500s():
    client = _attributed_client(raise_server_exceptions=False)
    db_patch, _session = _patch_db()
    write = AsyncMock(side_effect=RuntimeError("boom"))
    with (
        db_patch,
        patch.object(
            proposals, "_load_proposal", return_value=_open_proposal_row(kind="add")
        ),
        patch.object(proposals, "_claim_proposal", return_value=True),
        patch.object(proposals, "_revert_claim") as revert,
        patch.object(proposals.brain, "write_scoped_memory", write),
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")
    assert resp.status_code == 500
    revert.assert_called_once()


def test_approve_add_kind_lost_claim_race_is_409():
    """Claim races: someone else decided it between the initial load and
    the atomic claim UPDATE."""
    client = _attributed_client()
    db_patch, _session = _patch_db()
    write = AsyncMock()
    with (
        db_patch,
        patch.object(
            proposals, "_load_proposal", return_value=_open_proposal_row(kind="add")
        ),
        patch.object(proposals, "_claim_proposal", return_value=False),
        patch.object(proposals.brain, "write_scoped_memory", write),
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")
    assert resp.status_code == 409
    write.assert_not_awaited()


def test_approve_revise_kind_without_target_id_is_422():
    client = _attributed_client()
    db_patch, _session = _patch_db()
    row = _open_proposal_row(kind="revise", target_memory_id=None)
    with db_patch, patch.object(proposals, "_load_proposal", return_value=row):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")
    assert resp.status_code == 422


def test_approve_revise_kind_applies_via_cas_with_target_revision_and_type():
    client = _attributed_client()
    db_patch, _session = _patch_db()
    row = _open_proposal_row(kind="revise", target_memory_id=MEMORY_ID, scope="agent")
    with (
        db_patch,
        patch.object(proposals, "_load_proposal", return_value=row),
        patch.object(
            proposals,
            "_target_row",
            return_value={"scope": "agent", "revision": "rev-1", "memory_type": "fact"},
        ),
        patch.object(proposals, "_claim_proposal", return_value=True),
        patch.object(
            proposals,
            "supersede_with_revision",
            return_value={
                "conflict": False,
                "stored": True,
                "new_id": "m-new",
                "new_revision": "rev-2",
            },
        ) as sup,
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")

    assert resp.status_code == 200
    sup.assert_called_once()
    kwargs = sup.call_args.kwargs
    assert kwargs["org_id"] == "test-org"
    assert kwargs["memory_id"] == MEMORY_ID
    assert kwargs["base_revision"] == "rev-1"
    assert kwargs["memory_type"] == "fact"
    assert kwargs["session_id"] == f"proposal-{PROPOSAL_ID}"
    assert kwargs["new_content"] == "proposed content"
    assert kwargs["user_id"] == USER


def test_approve_revise_kind_conflict_reverts_claim_and_returns_409():
    client = _attributed_client()
    db_patch, _session = _patch_db()
    row = _open_proposal_row(kind="revise", target_memory_id=MEMORY_ID, scope="org")
    with (
        db_patch,
        patch.object(proposals, "_load_proposal", return_value=row),
        patch.object(
            proposals,
            "_target_row",
            return_value={"scope": "org", "revision": "rev-1", "memory_type": "fact"},
        ),
        patch.object(proposals, "_claim_proposal", return_value=True),
        patch.object(proposals, "_revert_claim") as revert,
        patch.object(
            proposals,
            "supersede_with_revision",
            return_value={
                "conflict": True,
                "stored": False,
                "current": {"id": MEMORY_ID},
            },
        ),
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")

    assert resp.status_code == 409
    revert.assert_called_once()


def test_approve_revise_kind_raises_reverts_claim_and_500s():
    """The proposal is already claimed ('approved') by the time
    supersede_with_revision runs — an unexpected exception there must not
    leave it stuck approved with nothing written. FastAPI turns the
    re-raised exception into a 500."""
    client = _attributed_client(raise_server_exceptions=False)
    db_patch, _session = _patch_db()
    row = _open_proposal_row(kind="revise", target_memory_id=MEMORY_ID, scope="org")
    with (
        db_patch,
        patch.object(proposals, "_load_proposal", return_value=row),
        patch.object(
            proposals,
            "_target_row",
            return_value={"scope": "org", "revision": "rev-1", "memory_type": "fact"},
        ),
        patch.object(proposals, "_claim_proposal", return_value=True),
        patch.object(proposals, "_revert_claim") as revert,
        patch.object(
            proposals, "supersede_with_revision", side_effect=RuntimeError("boom")
        ),
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")

    assert resp.status_code == 500
    revert.assert_called_once()
    assert revert.call_args.kwargs["org_id"] == "test-org"
    assert revert.call_args.kwargs["proposal_id"] == PROPOSAL_ID


def test_approve_revise_kind_forwards_project_id_for_project_scoped_target():
    """M3: a project-scoped target's project_id must reach
    supersede_with_revision — _target_row's SELECT includes it and
    approve_proposal forwards it, closing the gap where
    _prepare_cas_write's ValueError (no project_id for scope='project')
    was previously masked as an opaque redaction_failed 409."""
    client = _attributed_client()
    db_patch, _session = _patch_db()
    project_id = "44444444-4444-4444-4444-444444444444"
    row = _open_proposal_row(kind="revise", target_memory_id=MEMORY_ID, scope="project")
    with (
        db_patch,
        patch.object(proposals, "_load_proposal", return_value=row),
        patch.object(
            proposals,
            "_target_row",
            return_value={
                "scope": "project",
                "revision": "rev-1",
                "memory_type": "fact",
                "project_id": project_id,
            },
        ),
        patch.object(proposals, "_claim_proposal", return_value=True),
        patch.object(
            proposals,
            "supersede_with_revision",
            return_value={
                "conflict": False,
                "stored": True,
                "new_id": "m-new",
                "new_revision": "rev-2",
            },
        ) as sup,
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")

    assert resp.status_code == 200
    assert sup.call_args.kwargs["project_id"] == project_id


def test_approve_revise_kind_forbidden_reverts_claim_and_returns_403():
    """M3: revise's forbidden result must map to 403, aligned with
    retire's own forbidden -> 403 mapping (see
    test_approve_retire_kind_forbidden_reverts_claim_and_returns_403) —
    a 409 implies a retriable conflict, but re-approving will not change
    who the caller is."""
    client = _attributed_client()
    db_patch, _session = _patch_db()
    row = _open_proposal_row(kind="revise", target_memory_id=MEMORY_ID, scope="org")
    with (
        db_patch,
        patch.object(proposals, "_load_proposal", return_value=row),
        patch.object(
            proposals,
            "_target_row",
            return_value={
                "scope": "org",
                "revision": "rev-1",
                "memory_type": "fact",
                "project_id": None,
            },
        ),
        patch.object(proposals, "_claim_proposal", return_value=True),
        patch.object(proposals, "_revert_claim") as revert,
        patch.object(
            proposals,
            "supersede_with_revision",
            return_value={"conflict": False, "stored": False, "error": "forbidden"},
        ),
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")

    assert resp.status_code == 403
    revert.assert_called_once()


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


def test_approve_retire_kind_retracts_not_supersedes():
    """HIGH fix: retire calls retract_memory, never supersede_with_revision —
    a retraction removes the belief, it does not replace it with prose."""
    client = _attributed_client()
    db_patch, _session = _patch_db()
    row = _open_proposal_row(
        kind="retire", target_memory_id=MEMORY_ID, proposed_content=None
    )
    with (
        db_patch,
        patch.object(proposals, "_load_proposal", return_value=row),
        patch.object(proposals, "_claim_proposal", return_value=True),
        patch.object(
            proposals,
            "retract_memory",
            return_value={"forgotten": True, "id": MEMORY_ID},
        ) as rm,
        patch.object(proposals, "supersede_with_revision") as sup,
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")

    assert resp.status_code == 200
    assert resp.json()["kind"] == "retire"
    rm.assert_called_once()
    kwargs = rm.call_args.kwargs
    assert kwargs["memory_id"] == MEMORY_ID
    assert kwargs["user_id"] == USER
    assert kwargs["org_id"] == "test-org"
    sup.assert_not_called()


def test_approve_retire_kind_needs_no_proposed_content():
    """Unlike add/revise, retire never 422s for missing proposed_content."""
    client = _attributed_client()
    db_patch, _session = _patch_db()
    row = _open_proposal_row(
        kind="retire", target_memory_id=MEMORY_ID, proposed_content=None
    )
    with (
        db_patch,
        patch.object(proposals, "_load_proposal", return_value=row),
        patch.object(proposals, "_claim_proposal", return_value=True),
        patch.object(
            proposals,
            "retract_memory",
            return_value={"forgotten": True, "id": MEMORY_ID},
        ),
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")
    assert resp.status_code == 200


def test_approve_retire_kind_forbidden_reverts_claim_and_returns_403():
    client = _attributed_client()
    db_patch, _session = _patch_db()
    row = _open_proposal_row(
        kind="retire", target_memory_id=MEMORY_ID, proposed_content=None
    )
    with (
        db_patch,
        patch.object(proposals, "_load_proposal", return_value=row),
        patch.object(proposals, "_claim_proposal", return_value=True),
        patch.object(proposals, "_revert_claim") as revert,
        patch.object(
            proposals,
            "retract_memory",
            return_value={"forgotten": False, "error": "forbidden"},
        ),
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")
    assert resp.status_code == 403
    revert.assert_called_once()


def test_approve_retire_kind_raises_reverts_claim_and_500s():
    client = _attributed_client(raise_server_exceptions=False)
    db_patch, _session = _patch_db()
    row = _open_proposal_row(
        kind="retire", target_memory_id=MEMORY_ID, proposed_content=None
    )
    with (
        db_patch,
        patch.object(proposals, "_load_proposal", return_value=row),
        patch.object(proposals, "_claim_proposal", return_value=True),
        patch.object(proposals, "_revert_claim") as revert,
        patch.object(proposals, "retract_memory", side_effect=RuntimeError("boom")),
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/approve")
    assert resp.status_code == 500
    revert.assert_called_once()


# ---------------------------------------------------------------------------
# POST /proposals/{id}/reject
# ---------------------------------------------------------------------------


def test_reject_without_attributable_user_is_403(client):
    resp = client.post(f"/proposals/{PROPOSAL_ID}/reject")
    assert resp.status_code == 403


def test_reject_marks_decided_with_no_write():
    client = _attributed_client()
    db_patch, _session = _patch_db()
    with (
        db_patch,
        patch.object(proposals, "_load_proposal", return_value=_open_proposal_row()),
        patch.object(proposals, "_claim_proposal", return_value=True) as claim,
        patch.object(proposals.brain, "write_scoped_memory") as write,
        patch.object(proposals, "supersede_with_revision") as sup,
        patch.object(proposals, "retract_memory") as rm,
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/reject")

    assert resp.status_code == 200
    assert resp.json()["status"] == "rejected"
    claim.assert_called_once()
    assert claim.call_args.kwargs["status"] == "rejected"
    assert claim.call_args.kwargs["decided_by"] == USER
    write.assert_not_called()
    sup.assert_not_called()
    rm.assert_not_called()


def test_reject_lost_claim_race_is_409():
    client = _attributed_client()
    db_patch, _session = _patch_db()
    with (
        db_patch,
        patch.object(proposals, "_load_proposal", return_value=_open_proposal_row()),
        patch.object(proposals, "_claim_proposal", return_value=False),
    ):
        resp = client.post(f"/proposals/{PROPOSAL_ID}/reject")
    assert resp.status_code == 409


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


# ---------------------------------------------------------------------------
# _revert_claim (H3: rollback-first, status-guarded)
# ---------------------------------------------------------------------------


def test_revert_claim_rolls_back_before_issuing_the_revert_update():
    """H3: a revert reached after an exception mid-apply must start from a
    clean transaction — db.rollback() must happen BEFORE the revert
    UPDATE, undoing whatever the failed apply left half-applied."""
    db = MagicMock()
    calls = []
    db.rollback.side_effect = lambda: calls.append("rollback")
    db.execute.side_effect = lambda *a, **k: (
        calls.append("execute") or MagicMock(rowcount=1)
    )

    proposals._revert_claim(db, org_id="test-org", proposal_id=PROPOSAL_ID)

    assert calls[0] == "rollback"
    assert "execute" in calls
    assert calls.index("rollback") < calls.index("execute")


def test_revert_claim_update_is_status_guarded():
    """The revert UPDATE only touches rows still 'approved' — it must
    never clobber a decision some other request already made between the
    failed apply and this revert."""
    db = MagicMock()
    db.execute.return_value = MagicMock(rowcount=1)

    proposals._revert_claim(db, org_id="test-org", proposal_id=PROPOSAL_ID)

    executed_sql = str(db.execute.call_args.args[0])
    assert "status = 'approved'" in executed_sql
    assert "SET status = 'open'" in executed_sql


def test_revert_claim_logs_when_rowcount_zero(caplog):
    """A guarded revert that matches zero rows (someone else already
    decided it) must be logged, not silently ignored and not raised."""
    db = MagicMock()
    db.execute.return_value = MagicMock(rowcount=0)

    with caplog.at_level("WARNING"):
        proposals._revert_claim(db, org_id="test-org", proposal_id=PROPOSAL_ID)

    assert any("no row reverted" in rec.message for rec in caplog.records)
