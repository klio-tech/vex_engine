"""Branch-aware reads: ``retrieve_memories_scoped(branch_id=...)``. Real
Postgres.

The load-bearing default (Task C2, global constraint): a caller who does not
name a branch must NEVER receive branch rows — the failure mode is a
half-formed branch conclusion surfacing as project truth. That default is one
SQL clause (``branch_id IS NULL``) shared by construction between the vector
and lexical arms (both build their WHERE from the same ``base_clauses`` list),
and mirrored in the graph-expansion hop
(:func:`shared.graph_store.neighbors`), which is the one other path a row can
ride in on.

Naming a branch widens the read to ``(branch_id IS NULL OR branch_id =
:branch_id)`` — branch and main both live, never OTHER branches — and only on
the ACTIVE org's arm: the personal-node union's extra org always stays on
main, because branches live under org projects and a personal-node branch row
is a corruption, not a feature.

Real Postgres for the same reason as ``test_recall_personal_union.py``: the
boundary IS the SQL. ``embed_text`` is mocked so every row shares one vector
and ranking cannot mask an admission bug — under test is which rows the WHERE
clause admits, never their order.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

from shared import graph_store
from shared.memory import retrieve_memories_scoped
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the branch-aware read tests")

EMBED_DIM = 1024
FIXED_EMBEDDING = [0.1] * EMBED_DIM
FIXED_EMBEDDING_LITERAL = "[" + ",".join(str(x) for x in FIXED_EMBEDDING) + "]"

ME = "aaaaaaaa-1111-2222-3333-444444444444"
TEAMMATE = "bbbbbbbb-1111-2222-3333-444444444444"
OUTSIDER = "cccccccc-1111-2222-3333-444444444444"


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
                    text("DELETE FROM entities WHERE org_id = :o"), {"o": org_id}
                )
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


def _org(db, prefix: str = "org_br") -> str:
    org_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
    db.created_org_ids.add(org_id)
    return org_id


def _project(db, *, org_id: str, members: list[str]) -> str:
    project_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO projects (id, org_id, git_remote, display_name)
            VALUES (:id, :org_id, :remote, 'branch-read-test')
            """
        ),
        {
            "id": project_id,
            "org_id": org_id,
            "remote": f"git@example.com:{uuid.uuid4().hex[:8]}.git",
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
    scope: str = "project",
    user_id: str | None = None,
    project_id: str | None = None,
    branch_id: str | None = None,
) -> str:
    memory_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    db.execute(
        text(
            """
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, execution_id,
                memory_type, content, embedding, scope, status,
                user_id, provenance, project_id, branch_id, metadata,
                created_at, updated_at
            ) VALUES (
                :id, :session_id, :org_id, :agent_id, NULL,
                'fact', :content, :embedding, :scope, 'active',
                :user_id, 'EXTRACTED', :project_id, :branch_id, '{}'::jsonb,
                :created_at, :updated_at
            )
            """
        ),
        {
            "id": memory_id,
            "session_id": f"sess-{uuid.uuid4().hex[:6]}",
            "org_id": org_id,
            "agent_id": "agent-1",
            "content": content,
            "embedding": FIXED_EMBEDDING_LITERAL,
            "scope": scope,
            "user_id": user_id,
            "project_id": project_id,
            "branch_id": branch_id,
            "created_at": now,
            "updated_at": now,
        },
    )
    db.commit()
    return memory_id


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
# THE load-bearing default: no branch named ⇒ branch rows never surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_branch_recall_never_surfaces_branch_rows(db):
    """Every no-branch recall shape, one seeded branch row each: the branch
    row must not appear in ANY of them. This is the test the mutation check
    targets — drop the ``branch_id IS NULL`` default clause and it fails.

    Both retrieval arms are covered by construction: the vector and lexical
    queries build their WHERE from the same ``base_clauses`` list, so a
    clause present on one is present on the other.
    """
    org = _org(db)
    project = _project(db, org_id=org, members=[ME])
    branch = _branch(db, org_id=org, project_id=project)
    on_main = _seed(
        db, org_id=org, content="main truth", project_id=project, user_id=ME
    )
    on_branch = _seed(
        db,
        org_id=org,
        content="half-formed branch conclusion",
        project_id=project,
        branch_id=branch,
        user_id=ME,
    )

    # Shape 1: default org recall, attributable member (the project arm is
    # live — without the branch clause this WOULD admit the branch row).
    records = await _recall(db, org, user_id=ME)
    got = {str(r.id) for r in records}
    assert on_main in got
    assert on_branch not in got

    # Shape 2: project-filtered recall (tier-1 admits project rows).
    records = await _recall(db, org, user_id=ME, project_id=project)
    got = {str(r.id) for r in records}
    assert on_main in got
    assert on_branch not in got

    # Shape 3: unattributable caller (org tier only) — nothing to see, but
    # run it so the clause is proven present on this arm layout too.
    records = await _recall(db, org)
    assert on_branch not in {str(r.id) for r in records}

    # Shape 4: agent scope with a user (widest ladder this org offers).
    records = await _recall(db, org, scope="agent", agent_id="agent-1", user_id=ME)
    got = {str(r.id) for r in records}
    assert on_main in got
    assert on_branch not in got


@pytest.mark.asyncio
async def test_no_branch_recall_excludes_a_corrupt_private_branch_row(db):
    """A ``scope='private'`` row carrying a branch_id is structurally
    impossible through the write path (C1 refuses it); seed one directly and
    assert the default read still excludes it. Corruption tolerance, not a
    feature."""
    org = _org(db)
    project = _project(db, org_id=org, members=[ME])
    branch = _branch(db, org_id=org, project_id=project)
    corrupt = _seed(
        db,
        org_id=org,
        content="corrupt private branch row",
        scope="private",
        user_id=ME,
        project_id=project,
        branch_id=branch,
    )

    records = await _recall(db, org, scope="private", user_id=ME)
    assert corrupt not in {str(r.id) for r in records}


@pytest.mark.asyncio
async def test_no_branch_graph_expansion_never_pulls_a_branch_row(db):
    """The graph hop is the one other path a row can ride in on: an edge from
    a main entry row to a branch row must not drag the branch row into a
    no-branch recall. :func:`shared.graph_store.neighbors` carries the same
    default ``branch_id IS NULL`` filter."""
    org = _org(db)
    project = _project(db, org_id=org, members=[ME])
    branch = _branch(db, org_id=org, project_id=project)
    entry = _seed(db, org_id=org, content="main entry", project_id=project, user_id=ME)
    on_branch = _seed(
        db,
        org_id=org,
        content="connected branch row",
        project_id=project,
        branch_id=branch,
        user_id=ME,
    )
    graph_store.add_edge(
        db,
        org_id=org,
        src_id=entry,
        dst_kind="memory",
        dst_id=on_branch,
        relation="relates_to",
    )
    db.commit()

    records = await _recall(db, org, user_id=ME, project_id=project)
    got = {str(r.id) for r in records}
    assert entry in got
    assert on_branch not in got


# ---------------------------------------------------------------------------
# Naming a branch: branch + main both live, tagged, other branches stay out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_named_branch_surfaces_branch_and_main_distinguishably(db):
    org = _org(db)
    project = _project(db, org_id=org, members=[ME])
    branch = _branch(db, org_id=org, project_id=project)
    on_main = _seed(
        db, org_id=org, content="main truth", project_id=project, user_id=ME
    )
    on_branch = _seed(
        db,
        org_id=org,
        content="branch draft",
        project_id=project,
        branch_id=branch,
        user_id=ME,
    )

    records = await _recall(db, org, user_id=ME, project_id=project, branch_id=branch)
    by_id = {str(r.id): r for r in records}
    assert on_main in by_id
    assert on_branch in by_id
    # Distinguishable in the result: the row's branch_id rides on the record.
    assert by_id[on_branch].branch_id == branch
    assert by_id[on_main].branch_id is None


@pytest.mark.asyncio
async def test_named_branch_excludes_other_branches(db):
    """Opting into one branch is not opting into all of them."""
    org = _org(db)
    project = _project(db, org_id=org, members=[ME])
    mine = _branch(db, org_id=org, project_id=project, name="feat/mine")
    other = _branch(db, org_id=org, project_id=project, name="feat/other")
    on_mine = _seed(
        db,
        org_id=org,
        content="my branch draft",
        project_id=project,
        branch_id=mine,
        user_id=ME,
    )
    on_other = _seed(
        db,
        org_id=org,
        content="someone else's draft",
        project_id=project,
        branch_id=other,
        user_id=ME,
    )

    records = await _recall(db, org, user_id=ME, project_id=project, branch_id=mine)
    got = {str(r.id) for r in records}
    assert on_mine in got
    assert on_other not in got


# ---------------------------------------------------------------------------
# Pushed-branch visibility rides the existing ladder — never widens it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_member_reads_another_members_branch(db):
    """Any project member may read a branch by naming it, whoever created it."""
    org = _org(db)
    project = _project(db, org_id=org, members=[ME, TEAMMATE])
    branch = _branch(db, org_id=org, project_id=project, name="feat/teammates")
    theirs = _seed(
        db,
        org_id=org,
        content="teammate's branch draft",
        project_id=project,
        branch_id=branch,
        user_id=TEAMMATE,
    )

    records = await _recall(db, org, user_id=ME, project_id=project, branch_id=branch)
    assert theirs in {str(r.id) for r in records}


@pytest.mark.asyncio
async def test_non_member_gets_nothing_even_naming_the_branch(db):
    """The branch filter is a row filter, never a new access path: the
    project-membership arm still binds with it present."""
    org = _org(db)
    project = _project(db, org_id=org, members=[ME])
    branch = _branch(db, org_id=org, project_id=project)
    _seed(
        db,
        org_id=org,
        content="members-only branch draft",
        project_id=project,
        branch_id=branch,
        user_id=ME,
    )

    records = await _recall(
        db, org, user_id=OUTSIDER, project_id=project, branch_id=branch
    )
    assert records == []


# ---------------------------------------------------------------------------
# The B3 personal union composes: the extra org's arm always stays on main
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_personal_union_arm_stays_on_main_even_with_a_named_branch(db):
    """Branches live under org projects; a personal-node branch row is a
    corruption and must not leak through the union — even when the SAME
    branch id is (adversarially) stamped on a row in the personal org."""
    active = _org(db, "org_active")
    personal = _org(db, "org_personal")
    project = _project(db, org_id=active, members=[ME])
    branch = _branch(db, org_id=active, project_id=project)
    on_branch = _seed(
        db,
        org_id=active,
        content="active-org branch draft",
        project_id=project,
        branch_id=branch,
        user_id=ME,
    )
    personal_main = _seed(
        db,
        org_id=personal,
        content="my personal main note",
        scope="private",
        user_id=ME,
    )
    personal_branch_row = _seed(
        db,
        org_id=personal,
        content="corrupt personal branch row",
        scope="private",
        user_id=ME,
        branch_id=branch,
    )

    records = await _recall(
        db,
        active,
        user_id=ME,
        project_id=project,
        branch_id=branch,
        extra_org_id=personal,
    )
    got = {str(r.id) for r in records}
    assert on_branch in got
    assert personal_main in got
    assert personal_branch_row not in got


@pytest.mark.asyncio
async def test_personal_union_default_read_excludes_branch_rows_in_both_orgs(db):
    """The no-branch default under the union: main only, in BOTH orgs."""
    active = _org(db, "org_active")
    personal = _org(db, "org_personal")
    project = _project(db, org_id=active, members=[ME])
    branch = _branch(db, org_id=active, project_id=project)
    on_branch = _seed(
        db,
        org_id=active,
        content="active-org branch draft",
        project_id=project,
        branch_id=branch,
        user_id=ME,
    )
    personal_main = _seed(
        db,
        org_id=personal,
        content="my personal main note",
        scope="private",
        user_id=ME,
    )

    records = await _recall(db, active, user_id=ME, extra_org_id=personal)
    got = {str(r.id) for r in records}
    assert personal_main in got
    assert on_branch not in got


# ---------------------------------------------------------------------------
# The explore path: entity_brief keeps the same default fence (Task C3)
# ---------------------------------------------------------------------------


def test_entity_brief_never_returns_branch_rows(db):
    """``explore`` has no branch parameter, so its brief is a MAIN-only view
    by definition: an edge from an entity into a branch row must not drag the
    branch draft into the evidence slice, its decisions, or its counts."""
    org = _org(db)
    project = _project(db, org_id=org, members=[ME])
    branch = _branch(db, org_id=org, project_id=project)
    on_main = _seed(
        db, org_id=org, content="main evidence", project_id=project, user_id=ME
    )
    on_branch = _seed(
        db,
        org_id=org,
        content="half-formed branch evidence",
        project_id=project,
        branch_id=branch,
        user_id=ME,
    )
    entity = graph_store.add_entity(
        db, org_id=org, name="auth service", kind="system"
    ).id
    for memory_id in (on_main, on_branch):
        graph_store.add_edge(
            db,
            org_id=org,
            src_id=memory_id,
            dst_kind="entity",
            dst_id=entity,
            relation="about",
        )
    db.commit()

    brief = graph_store.entity_brief(
        db, org_id=org, viewer_user_id=ME, entity="auth service"
    )
    assert brief is not None
    got = {str(m.id) for m in brief["memories"]}
    assert on_main in got
    assert on_branch not in got


def test_entity_brief_supersession_section_never_surfaces_branch_rows(db):
    """The brief's ``superseded`` section returns the RETIRED side's content;
    a retired row still parked on a branch must not surface through it."""
    org = _org(db)
    project = _project(db, org_id=org, members=[ME])
    branch = _branch(db, org_id=org, project_id=project)
    evidence = _seed(
        db, org_id=org, content="current belief", project_id=project, user_id=ME
    )
    retired_on_branch = _seed(
        db,
        org_id=org,
        content="retired branch belief",
        project_id=project,
        branch_id=branch,
        user_id=ME,
    )
    db.execute(
        text("UPDATE session_memories SET status = 'superseded' WHERE id = :id"),
        {"id": retired_on_branch},
    )
    entity = graph_store.add_entity(
        db, org_id=org, name="auth service", kind="system"
    ).id
    graph_store.add_edge(
        db,
        org_id=org,
        src_id=evidence,
        dst_kind="entity",
        dst_id=entity,
        relation="about",
    )
    graph_store.add_edge(
        db,
        org_id=org,
        src_id=evidence,
        dst_kind="memory",
        dst_id=retired_on_branch,
        relation="supersedes",
    )
    db.commit()

    brief = graph_store.entity_brief(
        db, org_id=org, viewer_user_id=ME, entity="auth service"
    )
    assert brief is not None
    assert brief["superseded"] == []
