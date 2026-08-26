"""The internal branch-merge and proposal-review actions (``app.internal_actions``).

Mock-based, mirroring ``tests/test_proposals.py``: the routing, gating, and
composition logic is here; the SQL itself is exercised by the live-db
internal-api tests and the shared branches suite.
"""

import uuid
from unittest.mock import MagicMock

import app.internal_actions as actions
from app.db import get_db
from app.main import create_app
from fastapi.testclient import TestClient

SECRET = "internal-test-secret"
HEADER = "X-Klio-Internal"
PLANE_PROJECT = str(uuid.uuid4())
BRANCH = str(uuid.uuid4())
PROPOSAL = str(uuid.uuid4())


def _client(monkeypatch, db):
    monkeypatch.setenv("INTERNAL_API_SECRET", SECRET)
    app = create_app()

    def _test_db():
        yield db

    app.dependency_overrides[get_db] = _test_db
    return TestClient(app)


def test_no_secret_means_the_routes_do_not_exist(monkeypatch):
    monkeypatch.delenv("INTERNAL_API_SECRET", raising=False)
    client = TestClient(create_app())
    body = {"workspace_project": PLANE_PROJECT, "branch_id": BRANCH}
    assert client.post("/internal/branch-merge", json=body).status_code == 404
    assert (
        client.get(
            f"/internal/merge-proposals?workspace_project={PLANE_PROJECT}"
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/internal/proposal-decide",
            json={
                "workspace_project": PLANE_PROJECT,
                "proposal_id": PROPOSAL,
                "action": "reject",
            },
        ).status_code
        == 404
    )


def test_merge_of_an_unlinked_project_refuses(monkeypatch):
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = None
    client = _client(monkeypatch, db)

    response = client.post(
        "/internal/branch-merge",
        json={"workspace_project": PLANE_PROJECT, "branch_id": BRANCH},
        headers={HEADER: SECRET},
    )
    assert response.status_code == 200
    assert response.json() == {"merged": False, "reason": "not_linked"}


def test_merge_files_band_proposals_then_merges(monkeypatch):
    db = MagicMock()
    project = MagicMock(id=uuid.uuid4(), org_id="org-1")
    db.execute.return_value.fetchone.return_value = project

    filed = MagicMock(return_value=3)
    merged = MagicMock(return_value={"merged": True, "promoted": 5, "superseded": 1})
    monkeypatch.setattr(actions, "file_band_proposals", filed)
    import shared.branches as branches_module

    monkeypatch.setattr(branches_module, "merge_branch", merged)
    client = _client(monkeypatch, db)

    response = client.post(
        "/internal/branch-merge",
        json={
            "workspace_project": PLANE_PROJECT,
            "branch_id": BRANCH,
            "merged_by": "user-9",
        },
        headers={HEADER: SECRET},
    )

    assert response.status_code == 200
    assert response.json() == {
        "merged": True,
        "promoted": 5,
        "superseded": 1,
        "proposals_filed": 3,
    }
    # Proposals are filed BEFORE the merge — after promotion the branch rows
    # are indistinguishable from main and the band query would find nothing.
    assert filed.call_args.kwargs["branch_id"] == BRANCH
    assert merged.call_args.kwargs["branch_id"] == BRANCH
    assert merged.call_args.kwargs["merged_by"] == "user-9"


def test_decide_reject_claims_without_touching_memory(monkeypatch):
    db = MagicMock()
    project = MagicMock(id=uuid.uuid4(), org_id="org-1")
    db.execute.return_value.fetchone.return_value = project

    monkeypatch.setattr(
        actions,
        "_load_proposal",
        lambda *a, **k: {
            "id": PROPOSAL,
            "status": "open",
            "kind": "retire",
            "target_memory_id": str(uuid.uuid4()),
        },
    )
    claim = MagicMock(return_value=True)
    monkeypatch.setattr(actions, "_claim_proposal", claim)
    client = _client(monkeypatch, db)

    response = client.post(
        "/internal/proposal-decide",
        json={
            "workspace_project": PLANE_PROJECT,
            "proposal_id": PROPOSAL,
            "action": "reject",
            "decided_by": "user-9",
        },
        headers={HEADER: SECRET},
    )

    assert response.json() == {"decided": True, "status": "rejected"}
    assert claim.call_args.kwargs["status"] == "rejected"
    assert claim.call_args.kwargs["decided_by"] == "user-9"


def test_decide_approve_retracts_the_target(monkeypatch):
    db = MagicMock()
    project = MagicMock(id=uuid.uuid4(), org_id="org-1")
    target = str(uuid.uuid4())
    db.execute.return_value.fetchone.return_value = project
    db.execute.return_value.rowcount = 1

    monkeypatch.setattr(
        actions,
        "_load_proposal",
        lambda *a, **k: {
            "id": PROPOSAL,
            "status": "open",
            "kind": "retire",
            "target_memory_id": target,
        },
    )
    monkeypatch.setattr(actions, "_claim_proposal", MagicMock(return_value=True))
    client = _client(monkeypatch, db)

    response = client.post(
        "/internal/proposal-decide",
        json={
            "workspace_project": PLANE_PROJECT,
            "proposal_id": PROPOSAL,
            "action": "approve",
        },
        headers={HEADER: SECRET},
    )

    assert response.json() == {"decided": True, "status": "approved"}
    retract_calls = [
        call
        for call in db.execute.call_args_list
        if call.args and "retracted" in str(call.args[0])
    ]
    assert len(retract_calls) == 1
    assert retract_calls[0].args[1]["memory_id"] == target


def test_decide_lost_claim_is_a_clean_refusal(monkeypatch):
    db = MagicMock()
    project = MagicMock(id=uuid.uuid4(), org_id="org-1")
    db.execute.return_value.fetchone.return_value = project

    monkeypatch.setattr(
        actions,
        "_load_proposal",
        lambda *a, **k: {
            "id": PROPOSAL,
            "status": "open",
            "kind": "retire",
            "target_memory_id": str(uuid.uuid4()),
        },
    )
    monkeypatch.setattr(actions, "_claim_proposal", MagicMock(return_value=False))
    client = _client(monkeypatch, db)

    response = client.post(
        "/internal/proposal-decide",
        json={
            "workspace_project": PLANE_PROJECT,
            "proposal_id": PROPOSAL,
            "action": "approve",
        },
        headers={HEADER: SECRET},
    )
    assert response.json() == {"decided": False, "reason": "already_decided"}


def test_unknown_action_refuses(monkeypatch):
    client = _client(monkeypatch, MagicMock())
    response = client.post(
        "/internal/proposal-decide",
        json={
            "workspace_project": PLANE_PROJECT,
            "proposal_id": PROPOSAL,
            "action": "shred",
        },
        headers={HEADER: SECRET},
    )
    assert response.json() == {"decided": False, "reason": "unknown_action"}
