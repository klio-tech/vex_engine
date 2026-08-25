"""Branch merge and discard (Track C, Task C3). Real Postgres.

``merge_branch`` promotes a branch's rows to main (``branch_id`` → NULL) and,
where a branch row near-duplicates a main row (embedding similarity ≥
``DEDUP_THRESHOLD``), retires the main row through the existing
``supersede_memory`` — the branch version is the newer belief — recording the
judgment as a ``supersedes`` edge, the same shape the curator writes.
``discard_branch`` retracts the branch's rows outright.

Real Postgres because the load-bearing pieces ARE SQL: the pgvector
similarity-vs-main pass, the ``branch_id = NULL`` promotion UPDATE, the
retraction metadata merge, and the FOR UPDATE + status guard that makes both
operations idempotent.

Embeddings are one-hot literals seeded directly: two rows share a vector
(cosine similarity 1.0, above any sane threshold) exactly when the test says
they near-duplicate, and are orthogonal (similarity 0.0) otherwise. No
embedding service is involved — merge reads STORED vectors only.
"""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text

from shared import branches
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the branch merge/discard tests")

EMBED_DIM = 1024
MERGER = "aaaaaaaa-1111-2222-3333-444444444444"


def _one_hot_literal(index: int) -> str:
    values = ["0"] * EMBED_DIM
    values[index % EMBED_DIM] = "1"
    return "[" + ",".join(values) + "]"


@pytest.fixture()
def db():
    """A real Connection; every row created by the test is removed after."""
    engine = create_engine(DB_URL)
    conn = engine.connect()
    orgs: set[str] = set()
    conn.created_org_ids = orgs  # type: ignore[attr-defined]
    try:
        yield conn
    finally:
        try:
            conn.rollback()
            for org_id in orgs:
                conn.execute(
                    text("DELETE FROM memory_edges WHERE org_id = :o"), {"o": org_id}
                )
                conn.execute(
                    text("DELETE FROM session_memories WHERE org_id = :o"),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM context_branches WHERE org_id = :o"),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM projects WHERE org_id = :o"), {"o": org_id}
                )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _org(db) -> str:
    org_id = f"org_bm_{uuid.uuid4().hex[:8]}"
    db.created_org_ids.add(org_id)
    return org_id


def _project(db, *, org_id: str) -> str:
    project_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO projects (id, org_id, git_remote, display_name)
            VALUES (:id, :org_id, :remote, 'branch-merge-test')
            """
        ),
        {
            "id": project_id,
            "org_id": org_id,
            "remote": f"git@example.com:{uuid.uuid4().hex[:8]}.git",
        },
    )
    db.commit()
    return project_id


def _branch(db, *, org_id: str, project_id: str, name: str = "feat/x") -> str:
    branch_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO context_branches (id, org_id, project_id, name, status)
            VALUES (:id, :org_id, :project_id, :name, 'open')
            """
        ),
        {"id": branch_id, "org_id": org_id, "project_id": project_id, "name": name},
    )
    db.commit()
    return branch_id


def _seed(
    db,
    *,
    org_id: str,
    content: str,
    project_id: str,
    embed_index: int,
    branch_id: str | None = None,
    status: str = "active",
) -> str:
    memory_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    db.execute(
        text(
            """
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, memory_type, content,
                embedding, scope, status, provenance, project_id, branch_id,
                metadata, created_at, updated_at
            ) VALUES (
                :id, :session_id, :org_id, 'agent-1', 'fact', :content,
                :embedding, 'project', :status, 'EXTRACTED', :project_id,
                :branch_id, '{}'::jsonb, :now, :now
            )
            """
        ),
        {
            "id": memory_id,
            "session_id": f"sess-{uuid.uuid4().hex[:6]}",
            "org_id": org_id,
            "content": content,
            "embedding": _one_hot_literal(embed_index),
            "status": status,
            "project_id": project_id,
            "branch_id": branch_id,
            "now": now,
        },
    )
    db.commit()
    return memory_id


def _row(db, memory_id: str):
    row = db.execute(
        text(
            "SELECT status, branch_id, superseded_by, metadata "
            "FROM session_memories WHERE id = :id"
        ),
        {"id": memory_id},
    ).first()
    assert row is not None
    return row


# ---------------------------------------------------------------------------
# Merge: promotion
# ---------------------------------------------------------------------------


def test_merge_promotes_branch_rows_to_main(db):
    """After merge every branch row is on main (branch_id NULL, still active)
    — readable without naming the branch, which no longer exists as open."""
    org = _org(db)
    project = _project(db, org_id=org)
    branch = _branch(db, org_id=org, project_id=project)
    m1 = _seed(
        db,
        org_id=org,
        content="draft a",
        project_id=project,
        embed_index=1,
        branch_id=branch,
    )
    m2 = _seed(
        db,
        org_id=org,
        content="draft b",
        project_id=project,
        embed_index=2,
        branch_id=branch,
    )

    result = branches.merge_branch(
        db, org_id=org, project_id=project, branch_id=branch, merged_by=MERGER
    )

    assert result["merged"] is True
    assert result["promoted"] == 2
    assert result["superseded"] == 0
    for memory_id in (m1, m2):
        row = _row(db, memory_id)
        assert row.branch_id is None
        assert row.status == "active"

    branch_row = branches.get_branch_by_id(db, org_id=org, branch_id=branch)
    assert branch_row["status"] == "merged"
    assert branch_row["closed_at"] is not None


def test_merge_supersedes_near_duplicate_main_row_with_edge(db):
    """A branch row whose embedding near-duplicates a main row retires the
    main row via supersede_memory, and the judgment is a supersedes edge."""
    org = _org(db)
    project = _project(db, org_id=org)
    branch = _branch(db, org_id=org, project_id=project)
    on_main = _seed(
        db,
        org_id=org,
        content="we deploy from Docker Hub",
        project_id=project,
        embed_index=5,
    )
    on_branch = _seed(
        db,
        org_id=org,
        content="we deploy from GitHub source",
        project_id=project,
        embed_index=5,
        branch_id=branch,
    )
    unrelated_main = _seed(
        db,
        org_id=org,
        content="the pooler refuses startup params",
        project_id=project,
        embed_index=9,
    )

    result = branches.merge_branch(
        db, org_id=org, project_id=project, branch_id=branch, merged_by=MERGER
    )

    assert result["merged"] is True
    assert result["superseded"] == 1

    main_row = _row(db, on_main)
    assert main_row.status == "superseded"
    assert str(main_row.superseded_by) == on_branch

    branch_row = _row(db, on_branch)
    assert branch_row.status == "active"
    assert branch_row.branch_id is None

    assert _row(db, unrelated_main).status == "active"

    edge = db.execute(
        text(
            """
            SELECT provenance FROM memory_edges
            WHERE org_id = :o AND src_id = :src AND dst_id = :dst
              AND dst_kind = 'memory' AND relation = 'supersedes'
            """
        ),
        {"o": org, "src": on_branch, "dst": on_main},
    ).first()
    assert edge is not None, "merge must record the supersession as an edge"
    assert edge.provenance == "INFERRED"


def test_merge_similarity_pass_ignores_other_branches_and_inactive_rows(db):
    """The similarity window is MAIN only: another branch's row and a retracted
    main row must never be superseded by a merge."""
    org = _org(db)
    project = _project(db, org_id=org)
    branch = _branch(db, org_id=org, project_id=project)
    other = _branch(db, org_id=org, project_id=project, name="feat/other")
    on_other_branch = _seed(
        db,
        org_id=org,
        content="other draft",
        project_id=project,
        embed_index=5,
        branch_id=other,
    )
    retracted_main = _seed(
        db,
        org_id=org,
        content="old retracted",
        project_id=project,
        embed_index=5,
        status="retracted",
    )
    _seed(
        db,
        org_id=org,
        content="branch belief",
        project_id=project,
        embed_index=5,
        branch_id=branch,
    )

    result = branches.merge_branch(
        db, org_id=org, project_id=project, branch_id=branch, merged_by=MERGER
    )

    assert result["merged"] is True
    assert result["superseded"] == 0
    assert _row(db, on_other_branch).status == "active"
    assert _row(db, retracted_main).status == "retracted"


# ---------------------------------------------------------------------------
# Merge: idempotence and boundaries
# ---------------------------------------------------------------------------


def test_merge_is_idempotent(db):
    org = _org(db)
    project = _project(db, org_id=org)
    branch = _branch(db, org_id=org, project_id=project)
    _seed(
        db,
        org_id=org,
        content="draft",
        project_id=project,
        embed_index=1,
        branch_id=branch,
    )

    first = branches.merge_branch(
        db, org_id=org, project_id=project, branch_id=branch, merged_by=MERGER
    )
    second = branches.merge_branch(
        db, org_id=org, project_id=project, branch_id=branch, merged_by=MERGER
    )

    assert first["merged"] is True
    assert second == {"merged": False, "reason": "already_merged"}


def test_merge_of_a_discarded_branch_is_a_noop(db):
    org = _org(db)
    project = _project(db, org_id=org)
    branch = _branch(db, org_id=org, project_id=project)

    branches.discard_branch(db, org_id=org, branch_id=branch, discarded_by=MERGER)
    result = branches.merge_branch(
        db, org_id=org, project_id=project, branch_id=branch, merged_by=MERGER
    )

    assert result == {"merged": False, "reason": "already_discarded"}


def test_merge_cross_org_is_not_found(db):
    """A branch id from another org merges nothing — the id alone is never
    trusted to identify a tenant's branch."""
    org_a = _org(db)
    org_b = _org(db)
    project = _project(db, org_id=org_a)
    branch = _branch(db, org_id=org_a, project_id=project)
    row_id = _seed(
        db,
        org_id=org_a,
        content="draft",
        project_id=project,
        embed_index=1,
        branch_id=branch,
    )

    result = branches.merge_branch(
        db, org_id=org_b, project_id=project, branch_id=branch, merged_by=MERGER
    )

    assert result == {"merged": False, "reason": "not_found"}
    assert str(_row(db, row_id).branch_id) == branch


# ---------------------------------------------------------------------------
# Discard
# ---------------------------------------------------------------------------


def test_discard_retracts_branch_rows_everywhere(db):
    """Discard makes the branch's rows invisible: status='retracted', which
    every read path (recall, graph, capture recency) already excludes."""
    org = _org(db)
    project = _project(db, org_id=org)
    branch = _branch(db, org_id=org, project_id=project)
    m1 = _seed(
        db,
        org_id=org,
        content="wrong turn",
        project_id=project,
        embed_index=1,
        branch_id=branch,
    )
    on_main = _seed(
        db, org_id=org, content="main truth", project_id=project, embed_index=2
    )

    result = branches.discard_branch(
        db, org_id=org, branch_id=branch, discarded_by=MERGER
    )

    assert result["discarded"] is True
    assert result["retracted"] == 1

    row = _row(db, m1)
    assert row.status == "retracted"
    assert row.metadata["retracted_via"] == "branch_discard"
    assert row.metadata["retracted_by"] == MERGER

    assert _row(db, on_main).status == "active"

    branch_row = branches.get_branch_by_id(db, org_id=org, branch_id=branch)
    assert branch_row["status"] == "discarded"
    assert branch_row["closed_at"] is not None


def test_discard_is_idempotent(db):
    org = _org(db)
    project = _project(db, org_id=org)
    branch = _branch(db, org_id=org, project_id=project)

    first = branches.discard_branch(
        db, org_id=org, branch_id=branch, discarded_by=MERGER
    )
    second = branches.discard_branch(
        db, org_id=org, branch_id=branch, discarded_by=MERGER
    )

    assert first["discarded"] is True
    assert second == {"discarded": False, "reason": "already_discarded"}


def test_discard_of_a_merged_branch_is_a_noop(db):
    org = _org(db)
    project = _project(db, org_id=org)
    branch = _branch(db, org_id=org, project_id=project)

    branches.merge_branch(
        db, org_id=org, project_id=project, branch_id=branch, merged_by=MERGER
    )
    result = branches.discard_branch(
        db, org_id=org, branch_id=branch, discarded_by=MERGER
    )

    assert result == {"discarded": False, "reason": "already_merged"}


def test_discard_cross_org_is_not_found(db):
    org_a = _org(db)
    org_b = _org(db)
    project = _project(db, org_id=org_a)
    branch = _branch(db, org_id=org_a, project_id=project)
    row_id = _seed(
        db,
        org_id=org_a,
        content="draft",
        project_id=project,
        embed_index=1,
        branch_id=branch,
    )

    result = branches.discard_branch(
        db, org_id=org_b, branch_id=branch, discarded_by=MERGER
    )

    assert result == {"discarded": False, "reason": "not_found"}
    assert _row(db, row_id).status == "active"
