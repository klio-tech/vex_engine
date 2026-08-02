"""Adversarial traversal tests for the knowledge graph. Real Postgres.

Edges are the new tenancy leak class: a private memory linked to an org
entity must never surface through a walk to someone who could not read it
directly. Every test in this file is an ATTACK in the style of
``test_user_silo_isolation.py`` — seed something belonging to Priya, try to
reach it as Dev *through the graph*, assert refusal — and every negative has
a positive control, because a store that returns nothing to anyone passes
every negative test.

Real Postgres for the same reason as the silo: the boundary IS the SQL.
The ladder predicate and the destination org join in ``graph_store`` are
what is under test; a mocked pool would pass against a query with no WHERE
clause at all.

Each test uses a unique org so a failure cannot be a leak from another test.
"""

import os
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text

from shared import graph_store, projects
from shared.memory import retract_memory

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
    reason="Postgres not reachable; the graph boundary tests require it",
)

EMBED_DIM = 1024
FIXED_EMBEDDING_LITERAL = "[" + ",".join(["0.1"] * EMBED_DIM) + "]"

# Two people in one org. PRIYA owns things; DEV tries to reach them.
PRIYA = "11111111-2222-3333-4444-555555555555"
DEV = "99999999-8888-7777-6666-555555555555"
ADMIN = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


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
    status: str = "active",
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
                :memory_type, :content, :embedding, :scope, :status,
                :user_id, :provenance, :project_id, '{}'::jsonb,
                :created_at, :updated_at
            )
            """
        ),
        {
            "id": memory_id,
            "session_id": f"sess-{uuid.uuid4().hex[:6]}",
            "org_id": org_id,
            "agent_id": agent_id,
            "memory_type": memory_type,
            "content": content,
            "embedding": FIXED_EMBEDDING_LITERAL,
            "scope": scope,
            "status": status,
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


def _edge(db, *, org_id, src_id, dst_id, dst_kind="memory", relation="relates_to"):
    return graph_store.add_edge(
        db,
        org_id=org_id,
        src_id=src_id,
        dst_kind=dst_kind,
        dst_id=dst_id,
        relation=relation,
    )


def _neighbor_ids(db, *, org_id, viewer, entry, scope="org", agent_id=None) -> set:
    result = graph_store.neighbors(
        db,
        org_id=org_id,
        viewer_user_id=viewer,
        memory_ids=[entry] if isinstance(entry, str) else list(entry),
        scope=scope,
        agent_id=agent_id,
    )
    return {str(r.id) for r in result["records"]}


# ---------------------------------------------------------------------------
# ATTACK 1 — walk from a shared memory into Priya's private one
# ---------------------------------------------------------------------------


def test_dev_cannot_reach_priyas_private_memory_via_an_edge(db):
    """The base case for the new leak class: org row --edge--> private row."""
    org = _org(db)
    shared = _seed(db, org_id=org, content="the pricing page is being redesigned")
    secret = _seed(
        db,
        org_id=org,
        content="priya thinks the pricing model is wrong",
        scope="private",
        user_id=PRIYA,
    )
    _edge(db, org_id=org, src_id=shared, dst_id=secret)

    assert secret not in _neighbor_ids(db, org_id=org, viewer=DEV, entry=shared)
    assert secret not in _neighbor_ids(db, org_id=org, viewer=None, entry=shared)
    # Control: the SAME walk works for Priya.
    assert secret in _neighbor_ids(db, org_id=org, viewer=PRIYA, entry=shared)


def test_reverse_direction_is_filtered_too(db):
    """Edges traverse both ways, so the ladder must hold both ways: here the
    PRIVATE row is the edge's src and the shared row is the dst."""
    org = _org(db)
    shared = _seed(db, org_id=org, content="release notes drafted")
    secret = _seed(
        db,
        org_id=org,
        content="priya's private draft feedback",
        scope="private",
        user_id=PRIYA,
    )
    _edge(db, org_id=org, src_id=secret, dst_id=shared)

    assert secret not in _neighbor_ids(db, org_id=org, viewer=DEV, entry=shared)
    assert secret in _neighbor_ids(db, org_id=org, viewer=PRIYA, entry=shared)


def test_walk_through_a_shared_entity_is_filtered(db):
    """Two memories about one entity: the org one is the entry, Priya's
    private one is the sibling. The entity is org-level by design; the
    sibling's visibility must still be the MEMORY's, not the entity's."""
    org = _org(db)
    entity = graph_store.add_entity(db, org_id=org, name="auth service", kind="system")
    entry = _seed(db, org_id=org, content="auth service uses jwt")
    secret = _seed(
        db,
        org_id=org,
        content="priya found an auth service vulnerability",
        scope="private",
        user_id=PRIYA,
    )
    _edge(
        db, org_id=org, src_id=entry, dst_id=entity, dst_kind="entity", relation="about"
    )
    _edge(
        db,
        org_id=org,
        src_id=secret,
        dst_id=entity,
        dst_kind="entity",
        relation="about",
    )

    assert secret not in _neighbor_ids(db, org_id=org, viewer=DEV, entry=entry)
    assert secret in _neighbor_ids(db, org_id=org, viewer=PRIYA, entry=entry)


def test_no_scope_argument_reaches_the_forbidden_row(db):
    """Requirement: the connected-but-forbidden row is absent at EVERY scope
    argument the API accepts, not just the one the implementer thought of."""
    org = _org(db)
    shared = _seed(db, org_id=org, content="entry point")
    secret = _seed(
        db,
        org_id=org,
        content="not for dev",
        scope="private",
        user_id=PRIYA,
        agent_id="priya-laptop",
    )
    _edge(db, org_id=org, src_id=shared, dst_id=secret)

    for scope, agent in (("org", None), ("agent", "priya-laptop"), ("private", None)):
        found = _neighbor_ids(
            db, org_id=org, viewer=DEV, entry=shared, scope=scope, agent_id=agent
        )
        assert secret not in found, f"scope={scope} leaked through the edge"
    # Control so the sweep isn't vacuous.
    assert secret in _neighbor_ids(db, org_id=org, viewer=PRIYA, entry=shared)


def test_private_scope_narrows_expansion_to_mine_alone(db):
    """scope='private' means MINE: expansion must not pull in even org rows."""
    org = _org(db)
    mine = _seed(db, org_id=org, content="my note", scope="private", user_id=PRIYA)
    org_row = _seed(db, org_id=org, content="team fact")
    mine_too = _seed(
        db, org_id=org, content="my other note", scope="private", user_id=PRIYA
    )
    _edge(db, org_id=org, src_id=mine, dst_id=org_row)
    _edge(db, org_id=org, src_id=mine, dst_id=mine_too)

    found = _neighbor_ids(db, org_id=org, viewer=PRIYA, entry=mine, scope="private")
    assert mine_too in found
    assert org_row not in found


# ---------------------------------------------------------------------------
# ATTACK 2 — project boundary through the graph
# ---------------------------------------------------------------------------


def test_edge_into_a_project_row_respects_membership(db):
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
    shared = _seed(db, org_id=org, content="entry")
    fenced = _seed(
        db,
        org_id=org,
        content="apollo launch slipped",
        scope="project",
        user_id=PRIYA,
        project_id=project_id,
    )
    _edge(db, org_id=org, src_id=shared, dst_id=fenced)

    assert fenced not in _neighbor_ids(db, org_id=org, viewer=DEV, entry=shared)
    assert fenced in _neighbor_ids(db, org_id=org, viewer=PRIYA, entry=shared)


def test_cross_org_membership_grants_nothing_through_the_graph(db):
    """Same shape as the project_members hole the silo tests caught: Dev's
    genuine membership lives in ANOTHER tenant."""
    org_a = _org(db)
    org_b = _org(db)
    project_b = _project(db, org_id=org_b, name="theirs")
    projects.add_member(
        db,
        org_id=org_b,
        project_id=project_b,
        user_id=DEV,
        role=projects.ROLE_ADMIN,
        granted_by=DEV,
    )
    shared = _seed(db, org_id=org_a, content="entry")
    fenced = _seed(
        db,
        org_id=org_a,
        content="cross tenant probe",
        scope="project",
        user_id=PRIYA,
        project_id=project_b,
    )
    _edge(db, org_id=org_a, src_id=shared, dst_id=fenced)

    assert fenced not in _neighbor_ids(db, org_id=org_a, viewer=DEV, entry=shared)


# ---------------------------------------------------------------------------
# ATTACK 3 — a tampered cross-org edge
# ---------------------------------------------------------------------------


def _tampered_edge(db, *, org_id, src_id, dst_id, dst_kind="memory"):
    """Seed an edge DIRECTLY in SQL, bypassing the store's integrity checks —
    simulating a bug or tamper. The read path must still refuse it."""
    db.execute(
        text(
            "INSERT INTO memory_edges (id, org_id, src_id, dst_kind, dst_id, relation) "
            "VALUES (:i, :o, :s, :k, :d, 'relates_to')"
        ),
        {
            "i": str(uuid.uuid4()),
            "o": org_id,
            "s": src_id,
            "k": dst_kind,
            "d": dst_id,
        },
    )
    db.commit()


def test_a_cross_org_edge_cannot_grant_a_read(db):
    """An edge in org A pointing at org B's memory surfaces nothing: the org
    join on the DESTINATION row wins, not the edge's own org_id."""
    org_a = _org(db)
    org_b = _org(db)
    entry = _seed(db, org_id=org_a, content="org A entry")
    foreign = _seed(db, org_id=org_b, content="org B secret")
    _tampered_edge(db, org_id=org_a, src_id=entry, dst_id=foreign)

    for viewer in (DEV, PRIYA, None):
        assert foreign not in _neighbor_ids(
            db, org_id=org_a, viewer=viewer, entry=entry
        )


def test_a_cross_org_edge_in_the_victims_org_grants_nothing_either(db):
    """The reverse tamper: the edge row claims org B (the victim's org), with
    src pointing at B's memory. A caller in B walking from their own rows must
    not be led out to A's row, and a caller in A gains nothing from B's edge."""
    org_a = _org(db)
    org_b = _org(db)
    a_row = _seed(db, org_id=org_a, content="attacker row")
    b_row = _seed(db, org_id=org_b, content="victim row")
    _tampered_edge(db, org_id=org_b, src_id=b_row, dst_id=a_row)

    # In org B, walking from b_row: a_row is in another org — refused.
    assert a_row not in _neighbor_ids(db, org_id=org_b, viewer=PRIYA, entry=b_row)
    # In org A, the edge is not even in this org's edge set.
    assert b_row not in _neighbor_ids(db, org_id=org_a, viewer=DEV, entry=a_row)


# ---------------------------------------------------------------------------
# ATTACK 4 — entity existence must not leak
# ---------------------------------------------------------------------------


def test_entity_evidenced_only_privately_is_not_found_for_everyone_else(db):
    """Not an empty brief — an empty brief confirms the name exists."""
    org = _org(db)
    other_org = _org(db)
    entity = graph_store.add_entity(
        db, org_id=org, name="project-mercury", kind="concept"
    )
    secret = _seed(
        db,
        org_id=org,
        content="priya's private notes on project-mercury",
        scope="private",
        user_id=PRIYA,
        memory_type="decision",
    )
    _edge(
        db,
        org_id=org,
        src_id=secret,
        dst_id=entity,
        dst_kind="entity",
        relation="about",
    )

    # Dev, an org admin, an anonymous caller: not_found (None), all of them.
    for viewer in (DEV, ADMIN, None):
        assert (
            graph_store.entity_brief(
                db, org_id=org, viewer_user_id=viewer, entity="project-mercury"
            )
            is None
        )
    # Another org entirely: also None, even for "Priya" as a user id there.
    assert (
        graph_store.entity_brief(
            db, org_id=other_org, viewer_user_id=PRIYA, entity="project-mercury"
        )
        is None
    )
    # ...and INDISTINGUISHABLE from a name that has no entity row at all.
    assert (
        graph_store.entity_brief(
            db, org_id=org, viewer_user_id=DEV, entity="never-heard-of-it"
        )
        is None
    )

    # Control: the owner gets a real brief.
    brief = graph_store.entity_brief(
        db, org_id=org, viewer_user_id=PRIYA, entity="project-mercury"
    )
    assert brief is not None
    assert brief["entity"] == "project-mercury"
    assert brief["evidence_count"] == 1


def test_two_teammates_see_different_briefs_of_the_same_entity(db):
    """Priya's private memory about an ORG entity: absent from Dev's brief,
    present in Priya's. Different briefs is the feature, not a bug."""
    org = _org(db)
    entity = graph_store.add_entity(db, org_id=org, name="billing", kind="system")
    shared = _seed(
        db, org_id=org, content="billing moved to stripe", memory_type="decision"
    )
    secret = _seed(
        db,
        org_id=org,
        content="priya suspects billing double-charges",
        scope="private",
        user_id=PRIYA,
    )
    _edge(
        db,
        org_id=org,
        src_id=shared,
        dst_id=entity,
        dst_kind="entity",
        relation="about",
    )
    _edge(
        db,
        org_id=org,
        src_id=secret,
        dst_id=entity,
        dst_kind="entity",
        relation="about",
    )

    dev_brief = graph_store.entity_brief(
        db, org_id=org, viewer_user_id=DEV, entity="billing"
    )
    priya_brief = graph_store.entity_brief(
        db, org_id=org, viewer_user_id=PRIYA, entity="billing"
    )

    assert dev_brief is not None  # the org evidence makes it visible to Dev
    dev_contents = " ".join(m.content for m in dev_brief["memories"])
    assert "double-charges" not in dev_contents
    assert dev_brief["evidence_count"] == 1

    assert priya_brief is not None
    priya_contents = " ".join(m.content for m in priya_brief["memories"])
    assert "double-charges" in priya_contents
    assert priya_brief["evidence_count"] == 2


def test_brief_contributors_are_counts_never_ids(db):
    """The brief may say HOW MANY people evidenced an entity, never WHO
    (beyond the caller's own visible slice's contents)."""
    org = _org(db)
    entity = graph_store.add_entity(db, org_id=org, name="onboarding", kind="concept")
    m1 = _seed(db, org_id=org, content="onboarding flow shipped", user_id=PRIYA)
    m2 = _seed(db, org_id=org, content="onboarding drop-off is 40%", user_id=DEV)
    for m in (m1, m2):
        _edge(
            db, org_id=org, src_id=m, dst_id=entity, dst_kind="entity", relation="about"
        )

    brief = graph_store.entity_brief(
        db, org_id=org, viewer_user_id=ADMIN, entity="onboarding"
    )
    assert brief is not None
    assert brief["contributors"] == 2
    assert not isinstance(brief["contributors"], (list, dict)), (
        "contributors must be a count, not an enumeration of people"
    )


def test_cross_org_caller_never_resolves_another_orgs_entity(db):
    org_a = _org(db)
    org_b = _org(db)
    entity = graph_store.add_entity(db, org_id=org_a, name="skunkworks", kind="concept")
    evidence = _seed(db, org_id=org_a, content="skunkworks is greenlit")
    _edge(
        db,
        org_id=org_a,
        src_id=evidence,
        dst_id=entity,
        dst_kind="entity",
        relation="about",
    )

    assert (
        graph_store.entity_brief(
            db, org_id=org_b, viewer_user_id=DEV, entity="skunkworks"
        )
        is None
    )
    # Control in the right org.
    assert (
        graph_store.entity_brief(
            db, org_id=org_a, viewer_user_id=DEV, entity="skunkworks"
        )
        is not None
    )


# ---------------------------------------------------------------------------
# ATTACK 5 — forgotten memories stay forgotten, graph or no graph
# ---------------------------------------------------------------------------


def test_edge_to_a_forgotten_memory_is_silently_absent_then_swept(db):
    org = _org(db)
    entry = _seed(db, org_id=org, content="entry")
    doomed = _seed(db, org_id=org, content="soon forgotten")
    _edge(db, org_id=org, src_id=entry, dst_id=doomed)

    # Visible while alive (sanity), gone after the forget, no error either way.
    assert doomed in _neighbor_ids(db, org_id=org, viewer=DEV, entry=entry)
    retract_memory(db, org_id=org, agent_id="agent-1", memory_id=doomed, via="mcp")
    assert doomed not in _neighbor_ids(db, org_id=org, viewer=DEV, entry=entry)

    # And the sweep removes the orphan edge outright.
    assert graph_store.sweep_orphan_edges(db, org_id=org) == 1
    remaining = db.execute(
        text("SELECT count(*) FROM memory_edges WHERE org_id = :o"), {"o": org}
    ).scalar()
    assert remaining == 0
