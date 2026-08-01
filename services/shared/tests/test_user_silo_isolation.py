"""Adversarial cross-user tests for the user silo. Real Postgres.

Every test in this file is an ATTACK. Each one seeds a memory belonging to
Priya, then does its best to reach it as Dev, and asserts the result is empty
or refused. They are the acceptance criterion for phase 2: a recall bug here is
a privacy breach, not a bad result, so the boundary is tested the way a tenancy
boundary is tested rather than the way a feature is.

Why real Postgres rather than a mocked session: the boundary IS the SQL. A
fake DB that records the emitted string can prove the predicate was written;
only a real engine proves it was *effective*, that the ``project_members``
subquery joins the way it reads, and that migration 031's CHECK constraints
fire. ``embed_text`` is still mocked so every row shares one vector and cosine
ranking cannot mask an admission bug — what is under test is which rows the
WHERE clause admits, never their order.

Each test uses a unique org so a failure cannot be a leak from another test.
"""

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

from shared import projects
from shared.memory import retract_memory, retrieve_memories_scoped
from shared.promotion import promote_memory

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",
)


def _db_reachable() -> bool:
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_reachable(),
    reason="Postgres not reachable; the user-silo boundary tests require it",
)

EMBED_DIM = 1024
FIXED_EMBEDDING = [0.1] * EMBED_DIM
FIXED_EMBEDDING_LITERAL = "[" + ",".join(str(x) for x in FIXED_EMBEDDING) + "]"

# Two people in one org. PRIYA owns things; DEV tries to read them.
PRIYA = "11111111-2222-3333-4444-555555555555"
DEV = "99999999-8888-7777-6666-555555555555"


def _unique_org() -> str:
    return f"org_{uuid.uuid4().hex[:8]}"


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
                    text("DELETE FROM project_grant_audit WHERE org_id = :o"),
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
    org_id = _unique_org()
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
    memory_type: str = "fact",
    provenance: str = "EXTRACTED",
) -> str:
    memory_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    db.execute(
        text("""
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, execution_id,
                memory_type, content, embedding, scope, status,
                user_id, provenance, project_id, metadata,
                created_at, updated_at
            ) VALUES (
                :id, :session_id, :org_id, :agent_id, NULL,
                :memory_type, :content, :embedding, :scope, 'active',
                :user_id, :provenance, :project_id, '{}'::jsonb,
                :created_at, :updated_at
            )
        """),
        {
            "id": memory_id,
            "session_id": f"sess-{uuid.uuid4().hex[:6]}",
            "org_id": org_id,
            "agent_id": agent_id,
            "memory_type": memory_type,
            "content": content,
            "embedding": FIXED_EMBEDDING_LITERAL,
            "scope": scope,
            "user_id": user_id,
            "provenance": provenance,
            "project_id": project_id,
            "created_at": now,
            "updated_at": now,
        },
    )
    db.commit()
    return memory_id


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


def _mock_embed():
    return patch(
        "shared.memory.embed_text",
        new_callable=AsyncMock,
        return_value=FIXED_EMBEDDING,
    )


# ---------------------------------------------------------------------------
# ATTACK 1 — read another user's private memory by asking for it directly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dev_cannot_recall_priyas_private_memory(db):
    """The base case. Dev recalls, in Priya's org, with the same query text.

    Proves: ``scope='private' AND user_id = :me`` admits nothing for a
    different :me, at BOTH the default scope and the narrowed one.
    """
    org = _org(db)
    _seed(
        db,
        org_id=org,
        content="priya is interviewing at a competitor",
        scope="private",
        user_id=PRIYA,
    )

    with _mock_embed():
        default_scope = await retrieve_memories_scoped(
            db, org, "priya is interviewing at a competitor", user_id=DEV, limit=50
        )
        own_scope = await retrieve_memories_scoped(
            db,
            org,
            "priya is interviewing at a competitor",
            scope="private",
            user_id=DEV,
            limit=50,
        )

    assert default_scope == []
    assert own_scope == []


@pytest.mark.asyncio
async def test_priya_can_recall_her_own_private_memory(db):
    """The control. Without this, every other test in this file could pass
    against a recall that returns nothing to anybody."""
    org = _org(db)
    _seed(
        db,
        org_id=org,
        content="priya is interviewing at a competitor",
        scope="private",
        user_id=PRIYA,
    )

    with _mock_embed():
        results = await retrieve_memories_scoped(
            db, org, "priya is interviewing at a competitor", user_id=PRIYA, limit=50
        )

    assert len(results) == 1
    assert results[0].user_id == PRIYA
    assert results[0].scope == "private"


# ---------------------------------------------------------------------------
# ATTACK 2 — reach a private memory through some OTHER scope argument
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_scope_argument_exposes_another_users_private_row(db):
    """Sweep every scope the API accepts, as Dev, and as an anonymous caller.

    A boundary that holds for the scope you thought of and not the one you
    forgot is not a boundary. ``scope='agent'`` matters most here: it is the
    deprecated arm that still ORs in extra rows, and the agent id is
    client-declared, so it is the natural place for an omitted user predicate
    to hide.
    """
    org = _org(db)
    _seed(
        db,
        org_id=org,
        content="a secret",
        scope="private",
        user_id=PRIYA,
        agent_id="priya-laptop",
    )

    with _mock_embed():
        for scope in ("org", "agent"):
            as_dev = await retrieve_memories_scoped(
                db,
                org,
                "a secret",
                scope=scope,
                agent_id="priya-laptop",  # Dev DECLARES Priya's agent id
                user_id=DEV,
                limit=50,
            )
            assert as_dev == [], f"scope={scope} leaked to another user"

            anonymous = await retrieve_memories_scoped(
                db,
                org,
                "a secret",
                scope=scope,
                agent_id="priya-laptop",
                user_id=None,  # unattributable caller
                limit=50,
            )
            assert anonymous == [], f"scope={scope} leaked to an anonymous caller"


@pytest.mark.asyncio
async def test_private_scope_without_a_user_is_refused_not_widened(db):
    """An unattributable caller asking for 'mine' must not receive everyone's.

    Raising is the point: returning the org brain instead would answer a
    narrower question with a broader one, silently.
    """
    org = _org(db)
    with _mock_embed():
        with pytest.raises(ValueError):
            await retrieve_memories_scoped(db, org, "q", scope="private", user_id=None)


# ---------------------------------------------------------------------------
# ATTACK 3 — read a project's memories without being a member
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_member_cannot_recall_project_scoped_rows(db):
    """Dev is in the org and knows the project exists. He is not a member."""
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
    _seed(
        db,
        org_id=org,
        content="the apollo launch date slipped",
        scope="project",
        user_id=PRIYA,
        project_id=project_id,
    )

    with _mock_embed():
        as_dev = await retrieve_memories_scoped(
            db, org, "the apollo launch date slipped", user_id=DEV, limit=50
        )
        # ...and again while explicitly naming the project, in case naming it
        # is what unlocks it.
        as_dev_named = await retrieve_memories_scoped(
            db,
            org,
            "the apollo launch date slipped",
            user_id=DEV,
            project_id=project_id,
            limit=50,
        )
        as_priya = await retrieve_memories_scoped(
            db, org, "the apollo launch date slipped", user_id=PRIYA, limit=50
        )

    assert as_dev == []
    assert as_dev_named == []
    assert len(as_priya) == 1  # the control


@pytest.mark.asyncio
async def test_membership_of_one_project_does_not_open_another(db):
    """Dev administers project A. That must buy him nothing in project B.

    The membership subquery is keyed on user_id alone, so a bug that dropped
    the project_id correlation would surface exactly here and nowhere else.
    """
    org = _org(db)
    project_a = _project(db, org_id=org, name="alpha")
    project_b = _project(db, org_id=org, name="beta")
    projects.add_member(
        db,
        org_id=org,
        project_id=project_a,
        user_id=DEV,
        role=projects.ROLE_ADMIN,
        granted_by=DEV,
    )
    _seed(
        db,
        org_id=org,
        content="beta credentials rotated",
        scope="project",
        user_id=PRIYA,
        project_id=project_b,
    )

    with _mock_embed():
        results = await retrieve_memories_scoped(
            db, org, "beta credentials rotated", user_id=DEV, limit=50
        )

    assert results == []


@pytest.mark.asyncio
async def test_a_membership_row_from_another_org_grants_nothing(db):
    """Dev is a member of a project with the SAME id shape in another tenant.

    project_members has no org_id column of its own, so every read of it has to
    be joined through projects.org_id. This is the test that fails if one is
    not.
    """
    org_a = _org(db)
    org_b = _org(db)
    project_b = _project(db, org_id=org_b, name="theirs")
    # Dev genuinely administers a project — just not one in org_a.
    projects.add_member(
        db,
        org_id=org_b,
        project_id=project_b,
        user_id=DEV,
        role=projects.ROLE_ADMIN,
        granted_by=DEV,
    )
    _seed(
        db,
        org_id=org_a,
        content="cross tenant probe",
        scope="project",
        user_id=PRIYA,
        project_id=project_b,  # a project id from the OTHER org
    )

    with _mock_embed():
        results = await retrieve_memories_scoped(
            db, org_a, "cross tenant probe", user_id=DEV, limit=50
        )

    assert results == []
    # And the ACL helpers refuse to answer for a project outside the org at all.
    assert not projects.is_member(db, org_id=org_a, project_id=project_b, user_id=DEV)
    assert not projects.is_admin(db, org_id=org_a, project_id=project_b, user_id=DEV)
    assert projects.list_members(db, org_id=org_a, project_id=project_b) is None


# ---------------------------------------------------------------------------
# ATTACK 4 — the 32,568-row trap: project_id must not become a boundary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_org_rows_filed_under_a_project_stay_visible_to_everyone(db):
    """The inverse failure, and the more likely one.

    32,568 existing rows are scope='org' WITH a project_id, where project_id is
    a relevance filter and explicitly not a boundary. An implementer who reads
    "project_id IS NOT NULL" as "restricted" hides all of them from the org.
    Dev is not a member of this project and must still see the row.
    """
    org = _org(db)
    project_id = _project(db, org_id=org)
    _seed(
        db,
        org_id=org,
        content="we deploy on fridays",
        scope="org",
        user_id=PRIYA,  # authorship on a shared row grants Priya nothing extra
        project_id=project_id,
    )

    with _mock_embed():
        as_dev = await retrieve_memories_scoped(
            db, org, "we deploy on fridays", user_id=DEV, limit=50
        )
        anonymous = await retrieve_memories_scoped(
            db, org, "we deploy on fridays", user_id=None, limit=50
        )

    assert len(as_dev) == 1
    assert len(anonymous) == 1
    assert as_dev[0].scope == "org"


# ---------------------------------------------------------------------------
# ATTACK 5 — destroy or move a memory you cannot read
# ---------------------------------------------------------------------------


def test_dev_cannot_forget_priyas_private_memory(db):
    """Deleting what you cannot read is the same breach with a worse outcome."""
    org = _org(db)
    memory_id = _seed(
        db, org_id=org, content="priya's note", scope="private", user_id=PRIYA
    )

    as_dev = retract_memory(
        db,
        org_id=org,
        agent_id="dev-laptop",
        memory_id=memory_id,
        via="mcp",
        user_id=DEV,
    )
    anonymous = retract_memory(
        db,
        org_id=org,
        agent_id="dev-laptop",
        memory_id=memory_id,
        via="mcp",
        user_id=None,
    )

    # not_found, not forbidden: forbidden would confirm the row exists.
    assert as_dev == {"forgotten": False, "error": "not_found"}
    assert anonymous == {"forgotten": False, "error": "not_found"}

    still_active = db.execute(
        text("SELECT status FROM session_memories WHERE id = :id"), {"id": memory_id}
    ).fetchone()
    assert still_active.status == "active"

    # Control: Priya can.
    assert (
        retract_memory(
            db, org_id=org, agent_id="x", memory_id=memory_id, via="mcp", user_id=PRIYA
        )["forgotten"]
        is True
    )


def test_dev_cannot_share_priyas_private_memory(db):
    """Publishing someone else's private memory to the org is the worst case:
    it is a breach that cannot be undone by revoking access afterwards."""
    org = _org(db)
    memory_id = _seed(
        db, org_id=org, content="priya's note", scope="private", user_id=PRIYA
    )

    assert promote_memory(
        db, org_id=org, memory_id=memory_id, user_id=DEV, to="org"
    ) == {"shared": False, "error": "not_found"}
    assert promote_memory(
        db, org_id=org, memory_id=memory_id, user_id=None, to="org"
    ) == {"shared": False, "error": "not_found"}

    row = db.execute(
        text("SELECT scope, user_id FROM session_memories WHERE id = :id"),
        {"id": memory_id},
    ).fetchone()
    assert row.scope == "private"
    assert row.user_id == PRIYA


def test_promotion_keeps_the_inferred_label(db):
    """A curator-derived fact stays INFERRED after Priya shares it.

    The moment an LLM's deduction becomes team truth is exactly when the label
    matters; a promotion that laundered it into EXTRACTED would erase the
    distinction at the only point anyone was going to check.
    """
    org = _org(db)
    memory_id = _seed(
        db,
        org_id=org,
        content="the team prefers trunk-based development",
        scope="private",
        user_id=PRIYA,
        provenance="INFERRED",
    )

    result = promote_memory(
        db, org_id=org, memory_id=memory_id, user_id=PRIYA, to="org"
    )
    assert result["shared"] is True
    assert result["provenance"] == "INFERRED"

    row = db.execute(
        text("SELECT scope, user_id, provenance FROM session_memories WHERE id = :id"),
        {"id": memory_id},
    ).fetchone()
    assert row.scope == "org"
    assert row.provenance == "INFERRED"
    # user_id is RETAINED through the move — it is authorship now, not access.
    assert row.user_id == PRIYA


def test_an_already_shared_memory_cannot_be_re_promoted(db):
    """to='org' on a project-scoped row would WIDEN it. Refuse."""
    org = _org(db)
    project_id = _project(db, org_id=org)
    memory_id = _seed(
        db,
        org_id=org,
        content="scoped to apollo",
        scope="project",
        user_id=PRIYA,
        project_id=project_id,
    )

    result = promote_memory(
        db, org_id=org, memory_id=memory_id, user_id=PRIYA, to="org"
    )
    assert result["shared"] is False
    assert result["error"] == "already_shared"

    row = db.execute(
        text("SELECT scope FROM session_memories WHERE id = :id"), {"id": memory_id}
    ).fetchone()
    assert row.scope == "project"


# ---------------------------------------------------------------------------
# ATTACK 6 — the database itself must reject an owner-less private row
# ---------------------------------------------------------------------------


def test_the_schema_rejects_a_private_row_with_no_owner(db):
    """Belt to the code's braces.

    Such a row is unreachable by every caller including its author, so it is a
    bug by construction. If a future write path forgets the owner, this fires
    at the storage layer rather than silently destroying the data.
    """
    org = _org(db)
    with pytest.raises(Exception):
        _seed(db, org_id=org, content="orphan", scope="private", user_id=None)
    db.rollback()
