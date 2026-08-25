"""Branch-aware writes: ``write_scoped_memory(branch_id=...)`` and the dedup
window. Real Postgres.

The load-bearing behaviour is the dedup predicate, and the predicate IS SQL:
``branch_id IS NOT DISTINCT FROM :branch_id`` in every arm of
:func:`shared.memory.find_similar_memory`. Two NULLs match (main dedups
against main), NULL vs a value does not (a branch NEVER dedups against main,
and main never against a branch) — so the same belief may be drafted on a
branch while main still holds the old one, which is the whole point of a
branch. Only a real engine proves ``IS NOT DISTINCT FROM`` behaves that way;
a mock would just echo the string back.

``embed_text`` is patched to a deterministic per-content vector: identical
content embeds identically, so a dedup miss can only mean the WHERE clause
excluded the row.
"""

import uuid

import pytest
from sqlalchemy import create_engine, text

from shared import memory as memory_mod
from shared.brain import write_scoped_memory
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the branch-aware write tests")

EMBED_DIM = 1024
AGENT = "agent-branch"


def _fixed_vector(seed: str) -> list[float]:
    base = (hash(seed) % 1000) / 1000.0
    return [base] * EMBED_DIM


async def _fake_embed(content, **_kwargs):
    return _fixed_vector(content)


@pytest.fixture(autouse=True)
def _patch_embedding(monkeypatch):
    # write_scoped_memory's whole path (find_similar_memory + write_memory)
    # embeds through the async embed_text; patch it at the module boundary.
    monkeypatch.setattr(memory_mod, "embed_text", _fake_embed)


@pytest.fixture()
def db():
    """A real Connection, with every row this test created removed afterwards."""
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
    org_id = f"org_bw_{uuid.uuid4().hex[:8]}"
    db.created_org_ids.add(org_id)
    return org_id


def _project(db, *, org_id: str) -> str:
    """A real projects row — session_memories.project_id is FK-constrained."""
    project_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO projects (id, org_id, git_remote, display_name)
            VALUES (:id, :org_id, :remote, 'branch-write-test')
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


def _branch_row(db, *, org_id: str, project_id: str, name: str = "feat/x") -> str:
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


async def _write(
    db,
    *,
    org_id: str,
    project_id: str,
    session_id: str,
    content: str,
    branch_id=None,
):
    return await write_scoped_memory(
        db,
        org_id=org_id,
        agent_id=AGENT,
        content=content,
        memory_type="note",
        session_id=session_id,
        scope="project",
        project_id=project_id,
        branch_id=branch_id,
    )


def _stored_branch_id(db, memory_id: str):
    row = db.execute(
        text(
            "SELECT branch_id, scope, project_id FROM session_memories WHERE id = :id"
        ),
        {"id": memory_id},
    ).first()
    assert row is not None
    return row


# ---------------------------------------------------------------------------
# A branch write lands on the branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_branch_write_lands_with_branch_id_set(db):
    org = _org(db)
    project = _project(db, org_id=org)
    branch = _branch_row(db, org_id=org, project_id=project)

    result = await _write(
        db,
        org_id=org,
        project_id=project,
        session_id="sess-1",
        content="the schema uses a partial index",
        branch_id=branch,
    )
    assert result["stored"] is True

    row = _stored_branch_id(db, result["id"])
    assert str(row.branch_id) == branch
    assert row.scope == "project"
    assert str(row.project_id) == project


@pytest.mark.asyncio
async def test_default_write_lands_on_main(db):
    """No branch_id ⇒ the column stays NULL — main is the absence of a branch."""
    org = _org(db)
    project = _project(db, org_id=org)

    result = await _write(
        db,
        org_id=org,
        project_id=project,
        session_id="sess-1",
        content="main-line fact",
    )
    assert result["stored"] is True
    assert _stored_branch_id(db, result["id"]).branch_id is None


@pytest.mark.asyncio
async def test_branch_id_requires_project_scope(db):
    """A branch always belongs to a project (global constraint): a branch_id
    on any other scope is structurally impossible and refused."""
    with pytest.raises(ValueError):
        await write_scoped_memory(
            db,
            org_id=_org(db),
            agent_id=AGENT,
            content="orphan branch write",
            memory_type="note",
            session_id="sess-1",
            scope="org",
            branch_id=str(uuid.uuid4()),
        )


# ---------------------------------------------------------------------------
# The dedup window is branch-aware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_content_on_branch_and_main_is_not_deduped(db):
    """The branch is a DRAFT of main, so identical content on both is two
    rows, not one: main write first, then the same text on the branch."""
    org = _org(db)
    project = _project(db, org_id=org)
    branch = _branch_row(db, org_id=org, project_id=project)
    content = "we will use partial indexes for branch names"

    on_main = await _write(
        db, org_id=org, project_id=project, session_id="sess-1", content=content
    )
    assert on_main["stored"] is True

    on_branch = await _write(
        db,
        org_id=org,
        project_id=project,
        session_id="sess-1",
        content=content,
        branch_id=branch,
    )
    assert on_branch["stored"] is True, "a branch write must not dedup against main"
    assert on_branch["deduplicated"] is False

    count = db.execute(
        text("SELECT COUNT(*) FROM session_memories WHERE org_id = :o"),
        {"o": org},
    ).scalar()
    assert count == 2


@pytest.mark.asyncio
async def test_same_content_on_main_after_branch_is_not_deduped(db):
    """The other direction of the same rule: a main write never dedups
    against a branch row (NULL vs value under IS NOT DISTINCT FROM)."""
    org = _org(db)
    project = _project(db, org_id=org)
    branch = _branch_row(db, org_id=org, project_id=project)
    content = "retry with exponential backoff"

    on_branch = await _write(
        db,
        org_id=org,
        project_id=project,
        session_id="sess-1",
        content=content,
        branch_id=branch,
    )
    assert on_branch["stored"] is True

    on_main = await _write(
        db, org_id=org, project_id=project, session_id="sess-1", content=content
    )
    assert on_main["stored"] is True, "a main write must not dedup against a branch"


@pytest.mark.asyncio
async def test_same_content_twice_on_one_branch_is_deduped(db):
    """Within one branch the normal dedup contract holds: the second write of
    the same content is suppressed and reports the first row's id."""
    org = _org(db)
    project = _project(db, org_id=org)
    branch = _branch_row(db, org_id=org, project_id=project)
    content = "the curator runs nightly"

    first = await _write(
        db,
        org_id=org,
        project_id=project,
        session_id="sess-1",
        content=content,
        branch_id=branch,
    )
    assert first["stored"] is True

    second = await _write(
        db,
        org_id=org,
        project_id=project,
        session_id="sess-1",
        content=content,
        branch_id=branch,
    )
    assert second["stored"] is False
    assert second["deduplicated"] is True
    assert second["id"] == first["id"]


@pytest.mark.asyncio
async def test_same_content_twice_on_main_still_dedups(db):
    """Two NULLs match under IS NOT DISTINCT FROM: the historical main-line
    dedup behaviour is byte-identical after the predicate lands."""
    org = _org(db)
    project = _project(db, org_id=org)
    content = "deploys go through railway"

    first = await _write(
        db, org_id=org, project_id=project, session_id="sess-1", content=content
    )
    assert first["stored"] is True

    second = await _write(
        db, org_id=org, project_id=project, session_id="sess-1", content=content
    )
    assert second["stored"] is False
    assert second["deduplicated"] is True
    assert second["id"] == first["id"]
