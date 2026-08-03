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

import json
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text

from shared import graph_store, projects
from shared.memory import retract_memory
from shared.testing import database_url, requires_db

DB_URL = database_url()


pytestmark = requires_db("the graph boundary tests")

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
    recall_hidden: bool = False,
    superseded_by: str | None = None,
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
                recall_hidden, superseded_by,
                created_at, updated_at
            ) VALUES (
                :id, :session_id, :org_id, :agent_id, NULL,
                :memory_type, :content, :embedding, :scope, :status,
                :user_id, :provenance, :project_id, '{}'::jsonb,
                :recall_hidden, :superseded_by,
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
            "recall_hidden": recall_hidden,
            "superseded_by": superseded_by,
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


# ---------------------------------------------------------------------------
# ATTACK 6 — the destination org filter
#
# `neighbors` pins the destination row's org TWICE: once in the JOIN
# (`ON m.id = scored.other_id AND m.org_id = :org_id`) and once in the WHERE
# clause list (`"m.org_id = :org_id"`). Defence in depth against a tampered
# or buggy cross-org edge.
#
# WHAT THESE TESTS COVER, exactly — an earlier version of this comment
# claimed they "pin BOTH sites", and re-running the mutation showed that is
# false:
#
#   drop the WHERE-clause guard alone → 579 passed (nothing fails)
#   drop the JOIN guard alone         → 579 passed (nothing fails)
#   drop both                         → 3 failed
#
# That is not a gap in the tests; it is a property of the SQL. On an INNER
# JOIN, `ON ... AND m.org_id = :org_id` and `WHERE ... m.org_id = :org_id`
# are the SAME predicate on the SAME column of the SAME row. No query result
# can ever distinguish them, so NO behavioural test can pin one while the
# other stands. Anyone who writes one has written a test that passes for a
# different reason.
#
# So the behavioural tests below pin the destination filter as a whole (the
# boundary holds, by content and not merely by id), and the redundancy itself
# — the thing a refactor could quietly halve — is pinned structurally by
# `test_neighbors_keeps_both_destination_org_guards` in test_graph_store.py.
# ---------------------------------------------------------------------------


def test_entity_walk_pins_the_destination_org_filter(db):
    """Sibling-walk through a shared entity name across two orgs.

    Each org has its own entity row (entities are org-scoped), so a caller
    in org A resolving "billing" must see only A's evidence.

    Scope note: this covers ``entity_brief``'s own org scoping, which is a
    DIFFERENT query from ``neighbors``. Mutating either of ``neighbors``'
    destination org guards leaves this test passing — it is not, and cannot
    be, a pin on them.
    """
    org_a = _org(db)
    org_b = _org(db)

    a_mem = _seed(db, org_id=org_a, content="org A billing note")
    b_mem = _seed(db, org_id=org_b, content="org B billing secret")

    a_ent = graph_store.add_entity(db, org_id=org_a, name="billing", kind="concept")
    b_ent = graph_store.add_entity(db, org_id=org_b, name="billing", kind="concept")
    assert a_ent != b_ent, "entities must not be shared across tenants"

    graph_store.add_edge(
        db,
        org_id=org_a,
        src_id=a_mem,
        dst_kind="entity",
        dst_id=a_ent,
        relation="about",
    )
    graph_store.add_edge(
        db,
        org_id=org_b,
        src_id=b_mem,
        dst_kind="entity",
        dst_id=b_ent,
        relation="about",
    )
    db.commit()

    brief = graph_store.entity_brief(
        db, org_id=org_a, viewer_user_id=DEV, entity="billing"
    )
    assert brief is not None, "control: org A must resolve its own entity"
    blob = json.dumps(brief, default=str)
    assert "org B billing secret" not in blob
    assert brief["evidence_count"] == 1


def test_neighbors_destination_filter_holds_without_the_join_guard(db):
    """A tampered edge in org A pointing at org B, walked from an org A entry.

    Asserted on CONTENT rather than only ids, so the test fails if a foreign
    row is returned under any shape.
    """
    org_a = _org(db)
    org_b = _org(db)
    entry = _seed(db, org_id=org_a, content="org A entry")
    foreign = _seed(db, org_id=org_b, content="ORG-B-SECRET-MARKER")
    _tampered_edge(db, org_id=org_a, src_id=entry, dst_id=foreign)

    result = graph_store.neighbors(
        db, org_id=org_a, viewer_user_id=DEV, memory_ids=[entry], scope="org"
    )
    blob = json.dumps(result, default=str)
    assert "ORG-B-SECRET-MARKER" not in blob


# ---------------------------------------------------------------------------
# ATTACK 7 — the right to be forgotten, applied to the SUPERSEDED half
#
# The curator auto-supersedes a memory when a newer one contradicts it. The
# old row stays in the table as history, and ``entity_brief`` deliberately
# reports it as ``superseded[].old_content`` so a caller can see what changed.
# That is the feature. What is NOT the feature: the owner asking to forget
# that row, being told ``not_found``, and the graph serving its content back
# to them forever.
# ---------------------------------------------------------------------------


def _superseded_pair(db, *, org, entity_name, old_content, new_content, **seed_kw):
    """Seed old <-supersedes- new, both evidencing one entity. Returns ids."""
    entity = graph_store.add_entity(db, org_id=org, name=entity_name, kind="concept")
    new_id = _seed(db, org_id=org, content=new_content, **seed_kw)
    old_id = _seed(
        db,
        org_id=org,
        content=old_content,
        status="superseded",
        superseded_by=new_id,
        **seed_kw,
    )
    for src in (new_id, old_id):
        _edge(
            db,
            org_id=org,
            src_id=src,
            dst_id=entity,
            dst_kind="entity",
            relation="about",
        )
    _edge(db, org_id=org, src_id=new_id, dst_id=old_id, relation="supersedes")
    return old_id, new_id


def _brief_old_contents(db, *, org, viewer, entity, scope="org", agent_id=None):
    brief = graph_store.entity_brief(
        db,
        org_id=org,
        viewer_user_id=viewer,
        entity=entity,
        scope=scope,
        agent_id=agent_id,
    )
    if brief is None:
        return None
    return [s["old_content"] for s in brief["superseded"]]


def test_a_superseded_memory_is_forgettable(db):
    """H1, half one: ``retract_memory`` used to require status='active', so a
    row the curator had already superseded answered the indistinguishable
    ``not_found`` and could never be retracted. The owner cannot tell "you may
    not" from "it isn't there" — so they believe it is gone."""
    org = _org(db)
    old_id, _new_id = _superseded_pair(
        db,
        org=org,
        entity_name="api key rotation",
        old_content="My personal API key is sk-live-ABC123",
        new_content="I rotated my key; it lives in 1Password now",
        scope="private",
        user_id=PRIYA,
    )

    result = retract_memory(
        db, org_id=org, agent_id="agent-1", memory_id=old_id, via="mcp", user_id=PRIYA
    )
    assert result == {"forgotten": True, "id": old_id}, (
        "the owner of a superseded memory must be able to forget it"
    )
    status = db.execute(
        text("SELECT status FROM session_memories WHERE id = :i"), {"i": old_id}
    ).scalar()
    assert status == "retracted"


def test_forgetting_a_superseded_memory_removes_it_from_the_brief(db):
    """H1, the full sequence, production functions only: forget then explore."""
    org = _org(db)
    old_id, _new_id = _superseded_pair(
        db,
        org=org,
        entity_name="api key rotation",
        old_content="My personal API key is sk-live-ABC123",
        new_content="I rotated my key; it lives in 1Password now",
        scope="private",
        user_id=PRIYA,
    )

    # Control: before the forget, the supersession history IS served — the
    # fixture is not vacuous and the feature still works.
    before = _brief_old_contents(db, org=org, viewer=PRIYA, entity="api key rotation")
    assert before == ["My personal API key is sk-live-ABC123"]

    assert (
        retract_memory(
            db,
            org_id=org,
            agent_id="agent-1",
            memory_id=old_id,
            via="mcp",
            user_id=PRIYA,
        )["forgotten"]
        is True
    )

    brief = graph_store.entity_brief(
        db, org_id=org, viewer_user_id=PRIYA, entity="api key rotation"
    )
    assert brief is not None, "control: the surviving memory still evidences it"
    blob = json.dumps(brief, default=str)
    assert "sk-live-ABC123" not in blob, "forgotten content came back through explore"
    assert brief["superseded"] == []


def test_a_retracted_memory_is_never_served_as_old_content(db):
    """H1, half two, at the store layer: the superseded query must carry a
    status guard. Attacks the SQL directly, with no forget call in sight."""
    org = _org(db)
    entity = graph_store.add_entity(
        db, org_id=org, name="deploy target", kind="concept"
    )
    new_id = _seed(db, org_id=org, content="we deploy to fly.io now")
    dead_id = _seed(
        db, org_id=org, content="RETRACTED-MARKER heroku creds", status="retracted"
    )
    live_id = _seed(
        db,
        org_id=org,
        content="LIVE-MARKER we used to deploy to heroku",
        status="superseded",
        superseded_by=new_id,
    )
    _edge(
        db,
        org_id=org,
        src_id=new_id,
        dst_id=entity,
        dst_kind="entity",
        relation="about",
    )
    _edge(db, org_id=org, src_id=new_id, dst_id=dead_id, relation="supersedes")
    _edge(db, org_id=org, src_id=new_id, dst_id=live_id, relation="supersedes")

    olds = _brief_old_contents(db, org=org, viewer=DEV, entity="deploy target")
    assert "LIVE-MARKER we used to deploy to heroku" in olds, (
        "control: a genuinely superseded row is still history worth showing"
    )
    assert not any("RETRACTED-MARKER" in o for o in olds)


def test_a_hidden_raw_capture_is_never_served_as_old_content(db):
    """H1, half two continued: ``recall_hidden`` rows are curator fuel that no
    read path may return. The superseded query omitted the predicate."""
    org = _org(db)
    entity = graph_store.add_entity(db, org_id=org, name="oncall rota", kind="concept")
    new_id = _seed(db, org_id=org, content="priya is on call this week")
    hidden_id = _seed(
        db,
        org_id=org,
        content="HIDDEN-MARKER raw transcript",
        recall_hidden=True,
        status="superseded",
        superseded_by=new_id,
    )
    shown_id = _seed(
        db,
        org_id=org,
        content="SHOWN-MARKER dev was on call",
        status="superseded",
        superseded_by=new_id,
    )
    _edge(
        db,
        org_id=org,
        src_id=new_id,
        dst_id=entity,
        dst_kind="entity",
        relation="about",
    )
    _edge(db, org_id=org, src_id=new_id, dst_id=hidden_id, relation="supersedes")
    _edge(db, org_id=org, src_id=new_id, dst_id=shown_id, relation="supersedes")

    olds = _brief_old_contents(db, org=org, viewer=DEV, entity="oncall rota")
    assert "SHOWN-MARKER dev was on call" in olds, "control: the query returns rows"
    assert not any("HIDDEN-MARKER" in o for o in olds)


def test_retract_still_answers_not_found_for_someone_elses_superseded_private_row(db):
    """Dropping the status filter must NOT turn ``forget`` into an existence
    oracle: a superseded private row belonging to Priya answers ``not_found``
    for Dev, exactly as an id that was never issued does."""
    org = _org(db)
    old_id, _new = _superseded_pair(
        db,
        org=org,
        entity_name="priyas topic",
        old_content="priya's old private note",
        new_content="priya's new private note",
        scope="private",
        user_id=PRIYA,
    )
    never_issued = str(uuid.uuid4())

    mine = retract_memory(
        db, org_id=org, agent_id="agent-1", memory_id=old_id, via="mcp", user_id=DEV
    )
    ghost = retract_memory(
        db,
        org_id=org,
        agent_id="agent-1",
        memory_id=never_issued,
        via="mcp",
        user_id=DEV,
    )
    assert mine == ghost == {"forgotten": False, "error": "not_found"}

    # ...and cross-org is likewise indistinguishable.
    other_org = _org(db)
    assert retract_memory(
        db,
        org_id=other_org,
        agent_id="agent-1",
        memory_id=old_id,
        via="mcp",
        user_id=PRIYA,
    ) == {"forgotten": False, "error": "not_found"}

    # Control: the owner still gets through.
    assert (
        retract_memory(
            db,
            org_id=org,
            agent_id="agent-1",
            memory_id=old_id,
            via="mcp",
            user_id=PRIYA,
        )["forgotten"]
        is True
    )


def test_retract_still_forbids_another_agents_superseded_row(db):
    """The agent arm of the ladder survives the status change too."""
    org = _org(db)
    old_id, _new = _superseded_pair(
        db,
        org=org,
        entity_name="agent topic",
        old_content="old agent-scoped note",
        new_content="new agent-scoped note",
        scope="agent",
        agent_id="priya-laptop",
    )
    assert retract_memory(
        db, org_id=org, agent_id="dev-laptop", memory_id=old_id, via="mcp"
    ) == {"forgotten": False, "error": "forbidden"}
    # Control: the owning agent may.
    assert (
        retract_memory(
            db, org_id=org, agent_id="priya-laptop", memory_id=old_id, via="mcp"
        )["forgotten"]
        is True
    )


# ---------------------------------------------------------------------------
# ATTACK 8 — the unguarded predicates (H2)
#
# Each predicate below survived mutation: deleting it produced zero test
# failures, because no test anywhere seeded a row that made it evaluate.
# One test per predicate, each with a positive control.
# ---------------------------------------------------------------------------


def test_neighbors_never_returns_a_hidden_raw_capture(db):
    """Pins ``m.recall_hidden = FALSE`` in ``neighbors``."""
    org = _org(db)
    entry = _seed(db, org_id=org, content="entry")
    hidden = _seed(db, org_id=org, content="raw transcript fuel", recall_hidden=True)
    shown = _seed(db, org_id=org, content="a normal neighbour")
    _edge(db, org_id=org, src_id=entry, dst_id=hidden)
    _edge(db, org_id=org, src_id=entry, dst_id=shown)

    found = _neighbor_ids(db, org_id=org, viewer=DEV, entry=entry)
    assert shown in found, "control: the walk reaches ordinary neighbours"
    assert hidden not in found


def test_entity_brief_never_admits_a_hidden_raw_capture(db):
    """Pins ``m.recall_hidden = FALSE`` in ``entity_brief``'s evidence query."""
    org = _org(db)
    entity = graph_store.add_entity(db, org_id=org, name="checkout", kind="system")
    hidden = _seed(
        db, org_id=org, content="HIDDEN-EVIDENCE raw checkout log", recall_hidden=True
    )
    shown = _seed(db, org_id=org, content="checkout uses stripe")
    for m in (hidden, shown):
        _edge(
            db, org_id=org, src_id=m, dst_id=entity, dst_kind="entity", relation="about"
        )

    brief = graph_store.entity_brief(
        db, org_id=org, viewer_user_id=DEV, entity="checkout"
    )
    assert brief is not None, "control: the visible evidence still resolves"
    assert brief["evidence_count"] == 1
    assert "HIDDEN-EVIDENCE" not in json.dumps(brief, default=str)


def test_entity_brief_is_not_found_when_only_hidden_evidence_exists(db):
    """The stronger form: hidden evidence must not even prove the name."""
    org = _org(db)
    entity = graph_store.add_entity(db, org_id=org, name="shadow-topic", kind="concept")
    hidden = _seed(db, org_id=org, content="raw capture only", recall_hidden=True)
    _edge(
        db,
        org_id=org,
        src_id=hidden,
        dst_id=entity,
        dst_kind="entity",
        relation="about",
    )

    assert (
        graph_store.entity_brief(
            db, org_id=org, viewer_user_id=DEV, entity="shadow-topic"
        )
        is None
    )


def test_neighbors_never_returns_a_row_marked_superseded_by(db):
    """Pins ``m.superseded_by IS NULL`` in ``neighbors``, ON ITS OWN.

    ``status`` and ``superseded_by`` are two columns that are supposed to move
    together, and the sibling ``m.status = 'active'`` predicate masks this one
    for any row where they agree. So the row seeded here has them DISAGREE —
    superseded_by set, status still 'active' — which is precisely the drift the
    second predicate exists to catch.
    """
    org = _org(db)
    entry = _seed(db, org_id=org, content="entry")
    replacement = _seed(db, org_id=org, content="the current version")
    retired = _seed(
        db,
        org_id=org,
        content="the retired version",
        status="active",
        superseded_by=replacement,
    )
    _edge(db, org_id=org, src_id=entry, dst_id=retired)
    _edge(db, org_id=org, src_id=entry, dst_id=replacement)

    found = _neighbor_ids(db, org_id=org, viewer=DEV, entry=entry)
    assert replacement in found, "control: the un-retired neighbour is returned"
    assert retired not in found


def test_agent_scoped_rows_expand_only_for_their_own_agent(db):
    """Pins the agent arm of ``_ladder_clause`` from BOTH sides.

    Deleting the arm narrows the read rather than widening it, so a negative
    test alone cannot catch it — the positive half is what dies. The existing
    sweep at ``test_no_scope_argument_reaches_the_forbidden_row`` iterates
    scope='agent' but its victim row is 'private', so the arm never evaluates.
    """
    org = _org(db)
    entry = _seed(db, org_id=org, content="entry")
    mine = _seed(
        db,
        org_id=org,
        content="my agent's working note",
        scope="agent",
        agent_id="dev-laptop",
    )
    theirs = _seed(
        db,
        org_id=org,
        content="another agent's working note",
        scope="agent",
        agent_id="priya-laptop",
    )
    _edge(db, org_id=org, src_id=entry, dst_id=mine)
    _edge(db, org_id=org, src_id=entry, dst_id=theirs)

    found = _neighbor_ids(
        db, org_id=org, viewer=DEV, entry=entry, scope="agent", agent_id="dev-laptop"
    )
    assert mine in found, "control: scope='agent' must reach my own agent's rows"
    assert theirs not in found

    # And at scope='org' the agent arm is not built at all: neither surfaces.
    org_found = _neighbor_ids(db, org_id=org, viewer=DEV, entry=entry, scope="org")
    assert mine not in org_found
    assert theirs not in org_found


def test_entity_brief_agent_arm_admits_only_the_callers_agent(db):
    """The same arm, through the other read: entity_brief shares _ladder_clause."""
    org = _org(db)
    entity = graph_store.add_entity(
        db, org_id=org, name="migration plan", kind="concept"
    )
    mine = _seed(
        db,
        org_id=org,
        content="MINE-MARKER migration plan draft",
        scope="agent",
        agent_id="dev-laptop",
    )
    theirs = _seed(
        db,
        org_id=org,
        content="THEIRS-MARKER migration plan draft",
        scope="agent",
        agent_id="priya-laptop",
    )
    for m in (mine, theirs):
        _edge(
            db, org_id=org, src_id=m, dst_id=entity, dst_kind="entity", relation="about"
        )

    brief = graph_store.entity_brief(
        db,
        org_id=org,
        viewer_user_id=DEV,
        entity="migration plan",
        scope="agent",
        agent_id="dev-laptop",
    )
    assert brief is not None, "control: my own agent's evidence resolves the entity"
    blob = json.dumps(brief, default=str)
    assert "MINE-MARKER" in blob
    assert "THEIRS-MARKER" not in blob
    assert brief["evidence_count"] == 1


# ---------------------------------------------------------------------------
# ATTACK 9 (M5) — retention must not leave the graph behind
# ---------------------------------------------------------------------------


def test_retention_leaves_no_orphan_edges_once_the_sweep_runs(db):
    """``enforce_memory_retention`` deletes from ``session_memories`` only;
    ``memory_edges`` has no FKs by design, so without the sweep the edges
    outlive their endpoints forever."""
    org = _org(db)
    keep = _seed(db, org_id=org, content="recent enough to survive")
    old_a = _seed(db, org_id=org, content="old one")
    old_b = _seed(db, org_id=org, content="old two")
    _edge(db, org_id=org, src_id=keep, dst_id=old_a)
    _edge(db, org_id=org, src_id=old_a, dst_id=old_b)
    _edge(db, org_id=org, src_id=old_b, dst_id=keep)

    db.execute(
        text(
            "UPDATE session_memories SET created_at = NOW() - INTERVAL '400 days' "
            "WHERE id = ANY(CAST(:ids AS uuid[]))"
        ),
        {"ids": [old_a, old_b]},
    )
    db.commit()

    db.execute(
        text("SELECT enforce_memory_retention(:org_id, :days)"),
        {"org_id": org, "days": 30},
    )
    db.commit()

    survivors = db.execute(
        text("SELECT count(*) FROM session_memories WHERE org_id = :o"), {"o": org}
    ).scalar()
    assert survivors == 1, "control: retention actually deleted the aged rows"

    swept = graph_store.sweep_orphan_edges(db, org_id=org)
    assert swept == 3
    orphans = db.execute(
        text(
            """
            SELECT count(*) FROM memory_edges e
            WHERE e.org_id = :o
              AND NOT EXISTS (
                SELECT 1 FROM session_memories s WHERE s.id = e.src_id)
            """
        ),
        {"o": org},
    ).scalar()
    assert orphans == 0
    remaining = db.execute(
        text("SELECT count(*) FROM memory_edges WHERE org_id = :o"), {"o": org}
    ).scalar()
    assert remaining == 0
