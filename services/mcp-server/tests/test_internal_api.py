"""The internal status API: workspace-facing read-only endpoints.

``/internal/*`` exists for the workspace (Plane) web app's agent-memory
panel. It is not tenant-authenticated: the caller is the workspace's Django
backend, holding the shared ``INTERNAL_API_SECRET``. The auth contract is
therefore deliberately a 404 shape — an unauthenticated probe learns nothing,
not even that the routes exist — and an unset secret means the feature is
OFF: every internal route 404s, no matter what header arrives.

Data comes from the live test database inside a transaction that is always
rolled back; the route's ``get_db`` dependency is overridden to yield the
test's own connection so the uncommitted fixture rows are visible.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from app.db import get_db
from app.main import create_app
from shared.testing import database_url, requires_db

DB_URL = database_url()

SECRET = "internal-test-secret"
HEADER = "X-Klio-Internal"

ORG = "internal-api-test-org"
WORKSPACE_ID = "cafecafe-0000-4000-8000-000000000001"
PLANE_PROJECT = "cafecafe-0000-4000-8000-000000000002"
UNLINKED_PLANE_PROJECT = "cafecafe-0000-4000-8000-000000000003"
UNKNOWN_WORKSPACE = "cafecafe-0000-4000-8000-00000000000f"

_ZERO_VECTOR = "[" + ",".join(["0.0"] * 1024) + "]"


@pytest.fixture()
def db_conn():
    """A live connection whose writes are always rolled back."""
    engine = create_engine(DB_URL)
    conn = engine.connect()
    trans = conn.begin()
    try:
        yield conn
    finally:
        trans.rollback()
        conn.close()
        engine.dispose()


@pytest.fixture()
def client(db_conn, monkeypatch):
    monkeypatch.setenv("INTERNAL_API_SECRET", SECRET)
    app = create_app()

    def _test_db():
        yield db_conn

    app.dependency_overrides[get_db] = _test_db
    return TestClient(app)


def _seed_linked_project(conn) -> str:
    conn.execute(
        text(
            "INSERT INTO organizations (org_id, name, account_slug, workspace_id) "
            "VALUES (:org, 'internal test', :org, CAST(:ws AS uuid))"
        ),
        {"org": ORG, "ws": WORKSPACE_ID},
    )
    project_id = str(uuid.uuid4())
    conn.execute(
        text(
            "INSERT INTO projects "
            "(id, org_id, display_name, git_remote, plane_project_id) "
            "VALUES (CAST(:id AS uuid), :org, 'Klio Engine', "
            "'github.com/klio-tech/internal-api-test', CAST(:plane AS uuid))"
        ),
        {"id": project_id, "org": ORG, "plane": PLANE_PROJECT},
    )
    return project_id


def _seed_memory(
    conn,
    *,
    scope: str,
    project_id: str | None = None,
    status: str = "active",
    branch_id: str | None = None,
) -> None:
    conn.execute(
        text(
            """
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, user_id, memory_type,
                content, embedding, scope, status, project_id, branch_id,
                metadata, created_at, updated_at
            ) VALUES (
                :id, 'internal-api-test', :org, 'agent-1', 'user-1', 'fact',
                :content, :embedding, :scope, :status, CAST(:project AS uuid),
                CAST(:branch AS uuid), '{}'::jsonb, NOW(), NOW()
            )
            """
        ),
        {
            "id": str(uuid.uuid4()),
            "org": ORG,
            "content": f"a {scope} memory",
            "embedding": _ZERO_VECTOR,
            "scope": scope,
            "status": status,
            "project": project_id,
            "branch": branch_id,
        },
    )


# ---------------------------------------------------------------------------
# Auth shapes: right secret -> data; wrong/missing -> 404; env unset -> 404
# ---------------------------------------------------------------------------


@requires_db("the internal API tests")
def test_right_secret_returns_data(client, db_conn):
    _seed_linked_project(db_conn)
    r = client.get(
        f"/internal/workspace-node?workspace={WORKSPACE_ID}",
        headers={HEADER: SECRET},
    )
    assert r.status_code == 200
    assert r.json()["exists"] is True


@requires_db("the internal API tests")
@pytest.mark.parametrize(
    "headers", [{HEADER: "wrong-secret"}, {}], ids=["wrong", "missing"]
)
def test_wrong_or_missing_secret_is_a_404_shape(client, headers):
    for path in (
        f"/internal/workspace-node?workspace={WORKSPACE_ID}",
        f"/internal/project-link?workspace_project={PLANE_PROJECT}",
    ):
        r = client.get(path, headers=headers)
        assert r.status_code == 404


@requires_db("the internal API tests")
def test_env_unset_means_every_internal_route_404s(db_conn, monkeypatch):
    monkeypatch.delenv("INTERNAL_API_SECRET", raising=False)
    app = create_app()

    def _test_db():
        yield db_conn

    app.dependency_overrides[get_db] = _test_db
    client = TestClient(app)

    # Even the RIGHT secret cannot open a feature that is off.
    for headers in ({HEADER: SECRET}, {}):
        r = client.get(
            f"/internal/workspace-node?workspace={WORKSPACE_ID}", headers=headers
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /internal/project-link
# ---------------------------------------------------------------------------


@requires_db("the internal API tests")
def test_project_link_reports_counts_and_open_branches(client, db_conn):
    project_id = _seed_linked_project(db_conn)

    _seed_memory(db_conn, scope="org", project_id=project_id)
    _seed_memory(db_conn, scope="project", project_id=project_id)
    _seed_memory(db_conn, scope="project", project_id=project_id)
    _seed_memory(db_conn, scope="private", project_id=project_id)
    # Retracted rows must not count.
    _seed_memory(db_conn, scope="project", project_id=project_id, status="retracted")

    branch_id = str(uuid.uuid4())
    db_conn.execute(
        text(
            "INSERT INTO context_branches (id, org_id, project_id, name, status) "
            "VALUES (CAST(:id AS uuid), :org, CAST(:project AS uuid), "
            "'feat/panel', 'open')"
        ),
        {"id": branch_id, "org": ORG, "project": project_id},
    )
    db_conn.execute(
        text(
            "INSERT INTO context_branches (id, org_id, project_id, name, status, closed_at) "
            "VALUES (CAST(:id AS uuid), :org, CAST(:project AS uuid), "
            "'feat/merged', 'merged', NOW())"
        ),
        {"id": str(uuid.uuid4()), "org": ORG, "project": project_id},
    )
    _seed_memory(db_conn, scope="project", project_id=project_id, branch_id=branch_id)

    r = client.get(
        f"/internal/project-link?workspace_project={PLANE_PROJECT}",
        headers={HEADER: SECRET},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["linked"] is True
    assert body["engine_project_id"] == project_id
    assert body["display_name"] == "Klio Engine"
    assert body["memory_counts"] == {"org": 1, "project": 3, "private": 1}

    assert len(body["open_branches"]) == 1
    branch = body["open_branches"][0]
    assert branch["name"] == "feat/panel"
    assert branch["unmerged_count"] == 1
    assert branch["last_activity"] is not None


@requires_db("the internal API tests")
def test_unlinked_project_reports_linked_false(client):
    r = client.get(
        f"/internal/project-link?workspace_project={UNLINKED_PLANE_PROJECT}",
        headers={HEADER: SECRET},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["linked"] is False
    assert body["engine_project_id"] is None
    assert body["display_name"] is None
    assert body["memory_counts"] == {"org": 0, "project": 0, "private": 0}
    assert body["open_branches"] == []


# ---------------------------------------------------------------------------
# GET /internal/workspace-node
# ---------------------------------------------------------------------------


@requires_db("the internal API tests")
def test_workspace_node_reports_org_and_memory_count(client, db_conn):
    _seed_linked_project(db_conn)
    _seed_memory(db_conn, scope="org")
    _seed_memory(db_conn, scope="private")
    _seed_memory(db_conn, scope="org", status="retracted")

    r = client.get(
        f"/internal/workspace-node?workspace={WORKSPACE_ID}",
        headers={HEADER: SECRET},
    )
    assert r.status_code == 200
    body = r.json()
    assert body == {"org_id": ORG, "exists": True, "memory_count": 2}


@requires_db("the internal API tests")
def test_unknown_workspace_reports_exists_false(client):
    r = client.get(
        f"/internal/workspace-node?workspace={UNKNOWN_WORKSPACE}",
        headers={HEADER: SECRET},
    )
    assert r.status_code == 200
    assert r.json() == {"org_id": None, "exists": False, "memory_count": 0}


@requires_db("the internal API tests")
def test_node_brain_counts_and_private_visibility(client, db_conn):
    """The brain shows org rows to everyone, private rows only to their owner."""
    db_conn.execute(
        text(
            "INSERT INTO organizations (org_id, name, account_slug, workspace_id) "
            "VALUES (:org, 'brain test', :org, CAST(:ws AS uuid))"
        ),
        {"org": ORG, "ws": WORKSPACE_ID},
    )
    for content, scope, user in (
        ("an org-wide decision", "org", "user-a"),
        ("user A's private note", "private", "aaaaaaaa-0000-4000-8000-000000000001"),
        ("user B's private note", "private", "bbbbbbbb-0000-4000-8000-000000000002"),
    ):
        db_conn.execute(
            text(
                """
                INSERT INTO session_memories (
                    id, session_id, org_id, agent_id, user_id, memory_type,
                    content, scope, status, metadata, created_at, updated_at
                ) VALUES (
                    :id, 'brain-test', :org, 'agent-1', :user, 'fact',
                    :content, :scope, 'active', '{}'::jsonb, NOW(), NOW()
                )
                """
            ),
            {
                "id": str(uuid.uuid4()),
                "org": ORG,
                "user": user,
                "content": content,
                "scope": scope,
            },
        )

    response = client.get(
        f"/internal/node-brain?workspace={WORKSPACE_ID}"
        "&user=aaaaaaaa-0000-4000-8000-000000000001",
        headers={HEADER: SECRET},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["exists"] is True
    assert payload["counts"] == {"org": 1, "private": 2}
    contents = {row["content"] for row in payload["recent"]}
    assert "an org-wide decision" in contents
    assert "user A's private note" in contents
    assert "user B's private note" not in contents


@requires_db("the internal API tests")
def test_node_brain_unknown_workspace_is_empty(client):
    response = client.get(
        f"/internal/node-brain?workspace={UNKNOWN_WORKSPACE}", headers={HEADER: SECRET}
    )
    assert response.json() == {
        "exists": False,
        "org_id": None,
        "counts": {},
        "recent": [],
    }
