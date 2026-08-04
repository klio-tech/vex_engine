"""Does an artifact filed under a project actually stop at the project?

The unit tests in ``test_artifact_tools.py`` prove ``_write_artifact_card`` is
CALLED with ``scope='project'``. That is not the same claim as "a non-member
cannot read it": the boundary is the recall SQL, and only a real engine proves
the predicate is effective rather than merely present. So this module writes a
real card into real Postgres through the real ``brain.write_scoped_memory`` and
then reads it back through the real ``retrieve_memories_scoped`` as two
different people.

``embed_text`` is mocked so every row shares one vector: what is under test is
which rows the WHERE clause ADMITS, never how they rank. Every adversarial
assertion here is paired with a positive control in the same test, so an empty
result can never be mistaken for a boundary that works.
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

from app.mcp_app import _write_artifact_card
from shared import projects
from shared.memory import retrieve_memories_scoped
from shared.testing import database_url, requires_db

pytestmark = requires_db("the artifact project-scope boundary tests")

DB_URL = database_url()

EMBED_DIM = 1024
FIXED_EMBEDDING = [0.1] * EMBED_DIM

# Two people in one org. PRIYA files the artifact; DEV tries to read it.
PRIYA = "11111111-2222-3333-4444-555555555555"
DEV = "99999999-8888-7777-6666-555555555555"


def _mock_embed():
    return patch(
        "shared.memory.embed_text", new=AsyncMock(return_value=FIXED_EMBEDDING)
    )


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
            for org_id in orgs:
                conn.execute(
                    text(
                        """
                        DELETE FROM project_members WHERE project_id IN (
                            SELECT id FROM projects WHERE org_id = :o)
                        """
                    ),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM session_memories WHERE org_id = :o"),
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
    org_id = f"org_{uuid.uuid4().hex[:8]}"
    db.created_org_ids.add(org_id)
    return org_id


def _project(db, *, org_id: str, name: str = "apollo") -> str:
    project_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO projects (id, org_id, git_remote, display_name)
            VALUES (:id, :org_id, :remote, :name)
            """
        ),
        {
            "id": project_id,
            "org_id": org_id,
            "remote": f"git@example.com:{uuid.uuid4().hex[:8]}.git",
            "name": name,
        },
    )
    db.commit()
    return project_id


async def _put_card(db, *, org_id: str, project_id: str | None, content: str) -> None:
    """Write one artifact card exactly as ``artifact_put`` does."""
    artifact_id = str(uuid.uuid4())
    with (
        _mock_embed(),
        patch("shared.artifacts.card_content", return_value=content),
    ):
        await _write_artifact_card(
            db,
            org_id=org_id,
            agent_id="agent-1",
            session_id=f"sess-{uuid.uuid4().hex[:6]}",
            artifact_id=artifact_id,
            title="launch checklist",
            summary=content,
            kind="document",
            size_bytes=3,
            space_id=None,
            project_id=project_id,
        )


def _scope_of(db, *, org_id: str) -> list[str]:
    rows = db.execute(
        text(
            "SELECT scope FROM session_memories "
            "WHERE org_id = :o AND memory_type = 'artifact'"
        ),
        {"o": org_id},
    ).fetchall()
    return [r[0] for r in rows]


@pytest.mark.asyncio
async def test_an_artifact_filed_under_a_project_is_invisible_to_a_non_member(db):
    """Dev is in the org and can name the project. He is not a member of it.

    Before the fix the card was written ``scope='org'``, so this returned the
    artifact to Dev — the caller asked for a boundary and got none.
    """
    org = _org(db)
    project_id = _project(db, org_id=org)
    projects.add_member(
        db,
        org_id=org,
        project_id=project_id,
        user_id=PRIYA,
        role=projects.ROLE_ADMIN,
        granted_by=PRIYA,
    )

    content = "[artifact] launch checklist — apollo go/no-go (document, 3 B)"
    await _put_card(db, org_id=org, project_id=project_id, content=content)

    assert _scope_of(db, org_id=org) == ["project"]

    with _mock_embed():
        as_dev = await retrieve_memories_scoped(db, org, content, user_id=DEV, limit=50)
        # ...and again while naming the project, in case naming it unlocks it.
        as_dev_named = await retrieve_memories_scoped(
            db, org, content, user_id=DEV, project_id=project_id, limit=50
        )
        # THE POSITIVE CONTROL. Without it an empty result above proves
        # nothing: a broken fixture that wrote no row at all would pass.
        as_priya = await retrieve_memories_scoped(
            db, org, content, user_id=PRIYA, limit=50
        )

    assert as_dev == []
    assert as_dev_named == []
    assert len(as_priya) == 1
    assert as_priya[0].memory_type == "artifact"


@pytest.mark.asyncio
async def test_an_artifact_filed_under_no_project_stays_readable_org_wide(db):
    """The inverse failure, and the more damaging one.

    An implementation that narrowed unconditionally would hide every
    org-wide artifact from the org that owns it. Dev is a member of nothing
    and must still see this one.
    """
    org = _org(db)

    content = "[artifact] runbook — how we deploy (document, 3 B)"
    await _put_card(db, org_id=org, project_id=None, content=content)

    assert _scope_of(db, org_id=org) == ["org"]

    with _mock_embed():
        as_dev = await retrieve_memories_scoped(db, org, content, user_id=DEV, limit=50)

    assert len(as_dev) == 1
    assert as_dev[0].memory_type == "artifact"
