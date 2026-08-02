"""The curator as graph builder. Real Postgres for everything that lands rows.

Phase 2 of the knowledge graph: the per-user curator tick emits entities and
edges alongside its facts (one extended LLM pass — the LLM seams are mocked
here, the storage is not), and reconciliation records the supersession
judgment it already makes as a ``supersedes`` edge.

The tenancy stakes are the same as the silo's: the curator reads ONE user's
private memories, so everything it draws must stay invisible to everyone
else until promoted. The last test here is the acceptance shape — a tick for
user A creates nothing readable by user B — with A's own read as the control.

Storage rows go through a real Postgres connection (reachability-gated,
mirroring test_user_silo_isolation.py): a mocked pool would pass against
graph writes that ignore the GraphStore seam entirely.
"""

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

from shared import graph_store

from app.curator import curator_tick, extract_graph, reconcile_fact

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
    reason="Postgres not reachable; the curator graph tests require it",
)

EMBED_DIM = 1024
FIXED_EMBEDDING = [0.1] * EMBED_DIM

PRIYA = "11111111-2222-3333-4444-555555555555"
DEV = "99999999-8888-7777-6666-555555555555"


def _unique_org() -> str:
    return f"org_{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def db():
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
                    text("DELETE FROM brain_curator_state WHERE org_id = :o"),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM session_memories WHERE org_id = :o"),
                    {"o": org_id},
                )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _org(db) -> str:
    org_id = _unique_org()
    db.created_org_ids.add(org_id)
    return org_id


def _seed_raw(db, *, org_id: str, user_id: str, content: str) -> str:
    """A deliberate private write — curator fuel."""
    memory_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    db.execute(
        text(
            """
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, memory_type, content,
                embedding, scope, status, user_id, metadata,
                created_at, updated_at, recall_hidden
            ) VALUES (
                :id, :session_id, :org_id, 'agent-1', 'note', :content,
                :embedding, 'private', 'active', :user_id, '{}'::jsonb,
                :now, :now, FALSE
            )
            """
        ),
        {
            "id": memory_id,
            "session_id": f"sess-{uuid.uuid4().hex[:6]}",
            "org_id": org_id,
            "content": content,
            "embedding": "[" + ",".join(["0.1"] * EMBED_DIM) + "]",
            "user_id": user_id,
            "now": now,
        },
    )
    db.commit()
    return memory_id


def _graph_result(facts=None, entities=None, edges=None):
    return {
        "facts": facts if facts is not None else [],
        "entities": entities if entities is not None else [],
        "edges": edges if edges is not None else [],
    }


def _patches(graph):
    """Mock the LLM seams; leave every storage path real."""
    return (
        patch(
            "app.curator.generate_summary", new_callable=AsyncMock, return_value=None
        ),
        patch("app.curator.extract_graph", new_callable=AsyncMock, return_value=graph),
        # write_scoped_memory (real) embeds via shared.memory.embed_text;
        # reconcile embeds via app.curator.embed_text. Pin both to one vector
        # so ranking can never mask an admission bug.
        patch(
            "shared.memory.embed_text",
            new_callable=AsyncMock,
            return_value=FIXED_EMBEDDING,
        ),
        patch(
            "app.curator.embed_text",
            new_callable=AsyncMock,
            return_value=FIXED_EMBEDDING,
        ),
        # Every row shares the fixed vector, so the REAL dedup would collapse
        # the new fact into the seeded raw notes; that path has its own tests.
        patch(
            "shared.brain.find_similar_memory",
            new_callable=AsyncMock,
            return_value=None,
        ),
        # No LLM judge in these tests unless a test overrides it.
        patch("app.curator.call_llm", new_callable=AsyncMock, return_value=None),
    )


async def _run_tick(db, org, user, graph):
    patches = _patches(graph)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        return await curator_tick(db, org, user, min_inputs=3)


# ---------------------------------------------------------------------------
# The extended extraction pass
# ---------------------------------------------------------------------------


async def test_extract_graph_parses_the_spec_edge_shape():
    """The spec draws edges as {"src": <fact idx>, "<relation>": "<entity>"};
    the parser must take that shape (and refuse unknown relations)."""
    canned = {
        "facts": ["the auth service uses jwt", "rate limiter added"],
        "entities": [
            {"name": "auth service", "kind": "system"},
            {"name": "rate limiter", "kind": "system"},
        ],
        "edges": [
            {"src": 0, "about": "auth service"},
            {"src": 1, "about": "rate limiter"},
            {"src": 0, "relates_to": "rate limiter"},
            {"src": 0, "supersedes": "auth service"},  # not an LLM's to draw
            {"src": "nonsense", "about": "auth service"},  # bad index
        ],
    }
    with patch(
        "app.curator.call_llm", new_callable=AsyncMock, return_value=canned
    ) as llm:
        graph = await extract_graph("combined text")

    assert llm.await_count == 1, "facts + graph must be ONE pass, not two"
    assert graph["facts"] == canned["facts"]
    assert graph["entities"] == canned["entities"]
    # Only the well-formed about/relates_to edges survive parsing.
    assert graph["edges"] == [
        {"src": 0, "relation": "about", "entity": "auth service"},
        {"src": 1, "relation": "about", "entity": "rate limiter"},
        {"src": 0, "relation": "relates_to", "entity": "rate limiter"},
    ]


async def test_extract_graph_llm_failure_yields_empty_graph():
    with patch("app.curator.call_llm", new_callable=AsyncMock, return_value=None):
        graph = await extract_graph("combined text")
    assert graph == {"facts": [], "entities": [], "edges": []}


# ---------------------------------------------------------------------------
# The tick writes the graph — INFERRED, via the seam, capped, blocklisted
# ---------------------------------------------------------------------------


async def test_tick_stores_entities_and_inferred_edges(db):
    org = _org(db)
    for i in range(3):
        _seed_raw(db, org_id=org, user_id=PRIYA, content=f"note {i} about auth")

    graph = _graph_result(
        facts=["the auth service verifies jwts"],
        entities=[{"name": "Auth Service", "kind": "system"}],
        edges=[{"src": 0, "relation": "about", "entity": "Auth Service"}],
    )
    result = await _run_tick(db, org, PRIYA, graph)
    assert result["synthesized"] == 1  # the fact

    entity = db.execute(
        text(
            "SELECT id, kind FROM entities "
            "WHERE org_id = :o AND norm_name = 'auth service'"
        ),
        {"o": org},
    ).fetchone()
    assert entity is not None
    assert entity.kind == "system"

    edges = db.execute(
        text(
            "SELECT e.relation, e.provenance, m.content, m.scope, m.user_id "
            "FROM memory_edges e JOIN session_memories m ON m.id = e.src_id "
            "WHERE e.org_id = :o AND e.dst_kind = 'entity' AND e.dst_id = :d"
        ),
        {"o": org, "d": str(entity.id)},
    ).fetchall()
    assert len(edges) == 1
    assert edges[0].relation == "about"
    assert edges[0].provenance == "INFERRED", "everything the curator draws is INFERRED"
    assert edges[0].content == "the auth service verifies jwts"
    # The evidencing fact stays in the author's private silo.
    assert edges[0].scope == "private"
    assert edges[0].user_id == PRIYA


async def test_blocklisted_entities_are_refused_not_stored_and_tick_survives(
    db, caplog
):
    org = _org(db)
    for i in range(3):
        _seed_raw(db, org_id=org, user_id=PRIYA, content=f"note {i}")

    graph = _graph_result(
        facts=["someone wrote some code on monday"],
        entities=[
            {"name": "the code", "kind": "concept"},
            {"name": "Monday", "kind": "concept"},
            {"name": "deploy pipeline", "kind": "system"},
            {"name": "nameless", "kind": "not_a_kind"},  # bad kind: skipped too
        ],
        edges=[
            {"src": 0, "relation": "about", "entity": "the code"},
            {"src": 0, "relation": "about", "entity": "deploy pipeline"},
        ],
    )
    result = await _run_tick(db, org, PRIYA, graph)
    assert result["synthesized"] == 1  # the tick survived the refusals

    names = {
        r.norm_name
        for r in db.execute(
            text("SELECT norm_name FROM entities WHERE org_id = :o"), {"o": org}
        ).fetchall()
    }
    assert names == {"deploy pipeline"}

    # Only the edge to the surviving entity landed.
    n_edges = db.execute(
        text("SELECT count(*) FROM memory_edges WHERE org_id = :o"), {"o": org}
    ).scalar()
    assert n_edges == 1


async def test_per_tick_entity_cap_is_enforced(db, caplog):
    org = _org(db)
    for i in range(3):
        _seed_raw(db, org_id=org, user_id=PRIYA, content=f"note {i}")

    entities = [{"name": f"service-{i}", "kind": "system"} for i in range(20)]
    graph = _graph_result(facts=["many services exist"], entities=entities)

    with patch.dict(os.environ, {"CURATOR_ENTITY_CAP": "12"}):
        import logging

        with caplog.at_level(logging.WARNING, logger="agentguard.mcp.curator"):
            await _run_tick(db, org, PRIYA, graph)

    stored = db.execute(
        text("SELECT count(*) FROM entities WHERE org_id = :o"), {"o": org}
    ).scalar()
    assert stored == 12, "the per-tick cap must hold"
    assert any("entity cap" in r.message for r in caplog.records), (
        "overflow must be logged, not silent"
    )


async def test_edge_to_a_deduped_fact_is_skipped_not_misattached(db):
    """A fact that dedups away has no new row; its edges must simply not be
    drawn — never attached to some other row."""
    org = _org(db)
    for i in range(3):
        _seed_raw(db, org_id=org, user_id=PRIYA, content=f"note {i}")

    graph = _graph_result(
        facts=["fact that will dedup away"],
        entities=[{"name": "payments", "kind": "system"}],
        edges=[{"src": 0, "relation": "about", "entity": "payments"}],
    )
    patches = _patches(graph)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch(
            "app.curator.write_scoped_memory",
            new_callable=AsyncMock,
            return_value={"stored": False, "deduplicated": True, "id": "existing"},
        ):
            await curator_tick(db, org, PRIYA, min_inputs=3)

    n_edges = db.execute(
        text("SELECT count(*) FROM memory_edges WHERE org_id = :o"), {"o": org}
    ).scalar()
    assert n_edges == 0


# ---------------------------------------------------------------------------
# Reconciliation records its judgment as an edge
# ---------------------------------------------------------------------------


async def test_reconcile_writes_a_supersedes_edge_when_it_flips_status(db):
    org = _org(db)
    old = _seed_raw(db, org_id=org, user_id=PRIYA, content="deploys are on fridays")
    # Make the old row a FACT so the candidate query finds it.
    db.execute(
        text("UPDATE session_memories SET memory_type = 'fact' WHERE id = :i"),
        {"i": old},
    )
    db.commit()
    new = _seed_raw(db, org_id=org, user_id=PRIYA, content="deploys are on tuesdays")
    db.execute(
        text("UPDATE session_memories SET memory_type = 'fact' WHERE id = :i"),
        {"i": new},
    )
    db.commit()

    with patch(
        "app.curator.call_llm",
        new_callable=AsyncMock,
        return_value={"contradicts": True},
    ):
        superseded, judged = await reconcile_fact(
            db,
            org_id=org,
            user_id=PRIYA,
            new_fact_id=new,
            new_fact_content="deploys are on tuesdays",
            embedding=FIXED_EMBEDDING,
            max_judgments=5,
        )
    assert superseded == 1

    edge = db.execute(
        text(
            "SELECT src_id, dst_id, relation, provenance FROM memory_edges "
            "WHERE org_id = :o"
        ),
        {"o": org},
    ).fetchone()
    assert edge is not None
    assert str(edge.src_id) == new  # successor --supersedes--> retired row
    assert str(edge.dst_id) == old
    assert edge.relation == "supersedes"
    assert edge.provenance == "INFERRED"

    row = db.execute(
        text("SELECT status, superseded_by FROM session_memories WHERE id = :i"),
        {"i": old},
    ).fetchone()
    assert row.status == "superseded"
    assert str(row.superseded_by) == new


# ---------------------------------------------------------------------------
# The acceptance shape: a tick for A creates nothing readable by B
# ---------------------------------------------------------------------------


async def test_a_ticks_graph_is_invisible_to_user_b(db):
    org = _org(db)
    for i in range(3):
        _seed_raw(
            db, org_id=org, user_id=PRIYA, content=f"priya's secret project note {i}"
        )

    graph = _graph_result(
        facts=["project skunkworks is behind schedule"],
        entities=[{"name": "skunkworks", "kind": "concept"}],
        edges=[{"src": 0, "relation": "about", "entity": "skunkworks"}],
    )
    await _run_tick(db, org, PRIYA, graph)

    # Dev cannot even learn the entity exists...
    assert (
        graph_store.entity_brief(
            db, org_id=org, viewer_user_id=DEV, entity="skunkworks"
        )
        is None
    )
    # ...and an anonymous caller cannot either.
    assert (
        graph_store.entity_brief(
            db, org_id=org, viewer_user_id=None, entity="skunkworks"
        )
        is None
    )
    # Control: Priya can, and the brief carries her INFERRED fact.
    brief = graph_store.entity_brief(
        db, org_id=org, viewer_user_id=PRIYA, entity="skunkworks"
    )
    assert brief is not None
    assert brief["provenance"] == {"INFERRED": 1}
