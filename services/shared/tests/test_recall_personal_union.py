"""The personal-node union in recall — per-org visibility ladder. Real Postgres.

Task B3 of the context-network wiring: ``retrieve_memories_scoped`` gains an
``extra_org_id`` that widens the TENANCY predicate to two orgs while the
visibility ladder binds PER-ORG. The dangerous direction gets the adversarial
test: a private row belonging to ANOTHER user, seeded into the extra org, must
never surface — the extra org contributes only rows the caller could read
there.

Real Postgres for the same reason as ``test_user_silo_isolation.py``: the
boundary IS the SQL. ``embed_text`` is mocked so every row shares one vector
and ranking cannot mask an admission bug — under test is which rows the WHERE
clause admits, never their order.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

from shared.brain import write_scoped_memory
from shared.memory import retrieve_memories_scoped
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the personal-node recall union tests")

EMBED_DIM = 1024
FIXED_EMBEDDING = [0.1] * EMBED_DIM
FIXED_EMBEDDING_LITERAL = "[" + ",".join(str(x) for x in FIXED_EMBEDDING) + "]"

# The caller, and the other person whose personal rows must stay theirs.
ME = "aaaaaaaa-1111-2222-3333-444444444444"
SOMEONE_ELSE = "bbbbbbbb-1111-2222-3333-444444444444"


def _unique_org(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


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


def _org(db, prefix: str = "org_union") -> str:
    org_id = _unique_org(prefix)
    db.created_org_ids.add(org_id)
    return org_id


def _seed(
    db,
    *,
    org_id: str,
    content: str,
    scope: str = "org",
    user_id: str | None = None,
    project_id: str | None = None,
    agent_id: str = "agent-1",
) -> str:
    memory_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    db.execute(
        text(
            """
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, execution_id,
                memory_type, content, embedding, scope, status,
                user_id, provenance, project_id, metadata,
                created_at, updated_at
            ) VALUES (
                :id, :session_id, :org_id, :agent_id, NULL,
                'fact', :content, :embedding, :scope, 'active',
                :user_id, 'EXTRACTED', :project_id, '{}'::jsonb,
                :created_at, :updated_at
            )
            """
        ),
        {
            "id": memory_id,
            "session_id": f"sess-{uuid.uuid4().hex[:6]}",
            "org_id": org_id,
            "agent_id": agent_id,
            "content": content,
            "embedding": FIXED_EMBEDDING_LITERAL,
            "scope": scope,
            "user_id": user_id,
            "project_id": project_id,
            "created_at": now,
            "updated_at": now,
        },
    )
    db.commit()
    return memory_id


def _project(db, *, org_id: str, name: str, members: list[str]) -> str:
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
    for member in members:
        db.execute(
            text(
                "INSERT INTO project_members (project_id, user_id, role) "
                "VALUES (:p, :u, 'write')"
            ),
            {"p": project_id, "u": member},
        )
    db.commit()
    return project_id


def _mock_embed():
    return patch(
        "shared.memory.embed_text",
        new_callable=AsyncMock,
        return_value=FIXED_EMBEDDING,
    )


QUERY = "the shared query every fixed-vector row matches"


async def _recall(db, org_id, **kwargs):
    with _mock_embed():
        return await retrieve_memories_scoped(db, org_id, QUERY, limit=50, **kwargs)


# ---------------------------------------------------------------------------
# The union works: my own personal rows join the active org's results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_union_surfaces_my_personal_rows_alongside_the_active_org(db):
    active = _org(db, "org_active")
    personal = _org(db, "org_personal")
    team_row = _seed(db, org_id=active, content="the team decided on postgres")
    mine = _seed(
        db,
        org_id=personal,
        content="my personal note about postgres",
        scope="private",
        user_id=ME,
    )

    records = await _recall(db, active, user_id=ME, extra_org_id=personal)
    got = {str(r.id) for r in records}
    assert team_row in got
    assert mine in got
    # Every record still carries the org it came from, for source tagging.
    by_id = {str(r.id): r.org_id for r in records}
    assert by_id[team_row] == active
    assert by_id[mine] == personal


@pytest.mark.asyncio
async def test_without_extra_org_id_the_personal_rows_stay_out(db):
    """No-union calls are byte-identical: one org, exactly as before."""
    active = _org(db, "org_active")
    personal = _org(db, "org_personal")
    team_row = _seed(db, org_id=active, content="the team decided on postgres")
    _seed(
        db,
        org_id=personal,
        content="my personal note about postgres",
        scope="private",
        user_id=ME,
    )

    records = await _recall(db, active, user_id=ME)
    got = {str(r.id) for r in records}
    assert got == {team_row}


# ---------------------------------------------------------------------------
# THE test: another user's private rows in the extra org must NOT surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_another_users_private_row_in_the_extra_org_never_surfaces(db):
    """The ladder binds per-org: the extra arm carries its own user_id
    predicate, so widening tenancy to a second org admits only rows the
    caller could read THERE. A mutation that drops the extra arm's ladder
    (admitting the whole extra org on tenancy alone) fails exactly here.
    """
    active = _org(db, "org_active")
    extra = _org(db, "org_extra")
    _seed(db, org_id=active, content="the team decided on postgres")
    theirs = _seed(
        db,
        org_id=extra,
        content="someone else's private thought about postgres",
        scope="private",
        user_id=SOMEONE_ELSE,
    )
    mine = _seed(
        db,
        org_id=extra,
        content="my own private thought about postgres",
        scope="private",
        user_id=ME,
    )

    records = await _recall(db, active, user_id=ME, extra_org_id=extra)
    got = {str(r.id) for r in records}
    assert theirs not in got, "another user's private row leaked through the union"
    assert mine in got


@pytest.mark.asyncio
async def test_extra_org_project_rows_bind_to_membership_in_that_org(db):
    """The project arm's membership JOIN must key on the org it is admitting
    rows FROM. Membership in an active-org project must not unlock a
    same-named project's rows in the extra org (and the reverse holds by
    symmetry of the same predicate).
    """
    active = _org(db, "org_active")
    extra = _org(db, "org_extra")
    # I am a member of a project in the ACTIVE org only.
    _project(db, org_id=active, name="apollo", members=[ME])
    # The extra org has a project I am NOT a member of, with a row in it.
    foreign_project = _project(db, org_id=extra, name="apollo", members=[SOMEONE_ELSE])
    locked = _seed(
        db,
        org_id=extra,
        content="a project-scoped row about postgres",
        scope="project",
        user_id=SOMEONE_ELSE,
        project_id=foreign_project,
    )
    # And one project of the extra org I AM a member of — that one is mine
    # to read through the union.
    my_extra_project = _project(db, org_id=extra, name="hermes", members=[ME])
    reachable = _seed(
        db,
        org_id=extra,
        content="my project row about postgres",
        scope="project",
        user_id=ME,
        project_id=my_extra_project,
    )

    records = await _recall(db, active, user_id=ME, extra_org_id=extra)
    got = {str(r.id) for r in records}
    assert locked not in got, "extra-org project row leaked on active-org membership"
    assert reachable in got


@pytest.mark.asyncio
async def test_active_org_project_rows_do_not_open_on_extra_org_membership(db):
    """The mirror direction: membership in the EXTRA org must not admit
    project rows of the ACTIVE org."""
    active = _org(db, "org_active")
    extra = _org(db, "org_extra")
    _project(db, org_id=extra, name="apollo", members=[ME])
    active_project = _project(db, org_id=active, name="apollo", members=[SOMEONE_ELSE])
    locked = _seed(
        db,
        org_id=active,
        content="an active-org project row about postgres",
        scope="project",
        user_id=SOMEONE_ELSE,
        project_id=active_project,
    )

    records = await _recall(db, active, user_id=ME, extra_org_id=extra)
    got = {str(r.id) for r in records}
    assert locked not in got, "active-org project row leaked on extra-org membership"


# ---------------------------------------------------------------------------
# Writes never fan out: a remember during the union lands in ONE org
# ---------------------------------------------------------------------------


def _count(db, org_id: str) -> int:
    return db.execute(
        text("SELECT COUNT(*) FROM session_memories WHERE org_id = :o"),
        {"o": org_id},
    ).scalar()


@pytest.mark.asyncio
async def test_a_write_during_the_union_lands_only_in_the_active_org(db):
    """The union is a READ-ONLY widening. The write path takes one org_id and
    has no extra_org_id at all; this pins the row counts so a future fan-out
    cannot land silently.
    """
    active = _org(db, "org_active")
    personal = _org(db, "org_personal")

    with patch(
        "shared.memory.embed_text",
        new_callable=AsyncMock,
        return_value=FIXED_EMBEDDING,
    ):
        result = await write_scoped_memory(
            db,
            org_id=active,
            agent_id="agent-1",
            content="a fact written while the personal union is active",
            memory_type="fact",
            session_id=f"sess-{uuid.uuid4().hex[:6]}",
            dedup=False,
            user_id=ME,
        )
    db.commit()

    assert result.get("stored") is True
    assert _count(db, active) == 1
    assert _count(db, personal) == 0
