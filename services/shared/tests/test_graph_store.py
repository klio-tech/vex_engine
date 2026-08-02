"""GraphStore contract tests. Real Postgres.

The seam's *functional* contract: race-safe entity upserts, closed-set
validation, store-layer referential integrity (the schema deliberately has no
FKs), the god-entity guard, forget-independence, and the orphan sweep. The
adversarial visibility suite lives in ``test_graph_isolation.py``; this file
is about the store behaving correctly for a single legitimate caller.

Real Postgres for the same reason as the silo tests: the upsert arbiter, the
CHECK constraints, and the sweep's anti-joins are SQL, and a mocked session
proves nothing about any of them.
"""

import os
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text

from shared import graph_store
from shared.graph_store import EntityRefused
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
    reason="Postgres not reachable; the GraphStore contract tests require it",
)

EMBED_DIM = 1024
FIXED_EMBEDDING_LITERAL = "[" + ",".join(["0.1"] * EMBED_DIM) + "]"

PRIYA = "11111111-2222-3333-4444-555555555555"


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


def _seed(
    db,
    *,
    org_id: str,
    content: str,
    scope: str = "org",
    user_id: str | None = None,
    memory_type: str = "fact",
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
                user_id, metadata, created_at, updated_at
            ) VALUES (
                :id, :session_id, :org_id, 'agent-1', NULL,
                :memory_type, :content, :embedding, :scope, :status,
                :user_id, '{}'::jsonb, :now, :now
            )
            """
        ),
        {
            "id": memory_id,
            "session_id": f"sess-{uuid.uuid4().hex[:6]}",
            "org_id": org_id,
            "memory_type": memory_type,
            "content": content,
            "embedding": FIXED_EMBEDDING_LITERAL,
            "scope": scope,
            "status": status,
            "user_id": user_id,
            "now": now,
        },
    )
    db.commit()
    return memory_id


# ---------------------------------------------------------------------------
# add_entity — upsert, validation, god-entity guard
# ---------------------------------------------------------------------------


def test_add_entity_upserts_on_normalized_name(db):
    org = _org(db)
    first = graph_store.add_entity(db, org_id=org, name="Auth Service", kind="system")
    second = graph_store.add_entity(
        db, org_id=org, name="  auth   SERVICE ", kind="system"
    )
    assert first == second, "case/whitespace variants must converge on one row"

    row = db.execute(
        text("SELECT name, norm_name, kind FROM entities WHERE id = :i"),
        {"i": first},
    ).fetchone()
    assert row.norm_name == "auth service"
    assert row.kind == "system"
    # The first spelling wins; a repeat upsert must not thrash the display name.
    assert row.name == "Auth Service"


def test_add_entity_same_name_different_kind_is_a_different_entity(db):
    org = _org(db)
    a = graph_store.add_entity(db, org_id=org, name="mercury", kind="system")
    b = graph_store.add_entity(db, org_id=org, name="mercury", kind="person")
    assert a != b


def test_add_entity_rejects_unknown_kind(db):
    org = _org(db)
    with pytest.raises(ValueError):
        graph_store.add_entity(db, org_id=org, name="x-service", kind="service")


def test_add_entity_refuses_blocklisted_trivial_names(db):
    """The god-entity guard, part 1: graphify's hardest-won lesson."""
    org = _org(db)
    for name in ("the code", "The User", "the system", "monday", "python", "it"):
        with pytest.raises(EntityRefused):
            graph_store.add_entity(db, org_id=org, name=name, kind="concept")
    count = db.execute(
        text("SELECT count(*) FROM entities WHERE org_id = :o"), {"o": org}
    ).scalar()
    assert count == 0


def test_add_entity_refuses_blank_and_oversized_names(db):
    org = _org(db)
    with pytest.raises(EntityRefused):
        graph_store.add_entity(db, org_id=org, name="   ", kind="concept")
    with pytest.raises(EntityRefused):
        graph_store.add_entity(db, org_id=org, name="x" * 201, kind="concept")


# ---------------------------------------------------------------------------
# add_edge — closed sets, store-layer integrity, dedup
# ---------------------------------------------------------------------------


def test_add_edge_defaults_to_inferred_and_dedups(db):
    org = _org(db)
    src = _seed(db, org_id=org, content="we chose postgres")
    entity = graph_store.add_entity(db, org_id=org, name="postgres", kind="tool")

    first = graph_store.add_edge(
        db, org_id=org, src_id=src, dst_kind="entity", dst_id=entity, relation="about"
    )
    second = graph_store.add_edge(
        db, org_id=org, src_id=src, dst_kind="entity", dst_id=entity, relation="about"
    )
    assert first == second, "the unique key makes a repeat draw idempotent"

    row = db.execute(
        text("SELECT provenance, created_by FROM memory_edges WHERE id = :i"),
        {"i": first},
    ).fetchone()
    assert row.provenance == "INFERRED"
    assert row.created_by is None


def test_add_edge_rejects_unknown_relation_and_dst_kind(db):
    org = _org(db)
    src = _seed(db, org_id=org, content="m")
    dst = _seed(db, org_id=org, content="n")
    with pytest.raises(ValueError):
        graph_store.add_edge(
            db, org_id=org, src_id=src, dst_kind="memory", dst_id=dst, relation="links"
        )
    with pytest.raises(ValueError):
        graph_store.add_edge(
            db, org_id=org, src_id=src, dst_kind="row", dst_id=dst, relation="about"
        )
    with pytest.raises(ValueError):
        graph_store.add_edge(
            db,
            org_id=org,
            src_id=src,
            dst_kind="memory",
            dst_id=dst,
            relation="relates_to",
            provenance="GUESSED",
        )


def test_add_edge_enforces_endpoint_existence_in_the_org(db):
    """The schema has no FKs on purpose; the store IS the integrity layer."""
    org = _org(db)
    other_org = _org(db)
    src = _seed(db, org_id=org, content="real row")
    foreign = _seed(db, org_id=other_org, content="other tenant's row")

    # Nonexistent src.
    with pytest.raises(LookupError):
        graph_store.add_edge(
            db,
            org_id=org,
            src_id=str(uuid.uuid4()),
            dst_kind="memory",
            dst_id=src,
            relation="relates_to",
        )
    # Nonexistent dst memory.
    with pytest.raises(LookupError):
        graph_store.add_edge(
            db,
            org_id=org,
            src_id=src,
            dst_kind="memory",
            dst_id=str(uuid.uuid4()),
            relation="relates_to",
        )
    # Nonexistent dst entity.
    with pytest.raises(LookupError):
        graph_store.add_edge(
            db,
            org_id=org,
            src_id=src,
            dst_kind="entity",
            dst_id=str(uuid.uuid4()),
            relation="about",
        )
    # A row that exists — in ANOTHER org. Same refusal: an endpoint outside
    # the tenant does not exist as far as this org is concerned.
    with pytest.raises(LookupError):
        graph_store.add_edge(
            db,
            org_id=org,
            src_id=src,
            dst_kind="memory",
            dst_id=foreign,
            relation="relates_to",
        )
    count = db.execute(
        text("SELECT count(*) FROM memory_edges WHERE org_id = :o"), {"o": org}
    ).scalar()
    assert count == 0


# ---------------------------------------------------------------------------
# forget-independence and the orphan sweep
# ---------------------------------------------------------------------------


def test_forget_succeeds_with_edges_attached(db):
    """An edge must never be the reason a forget fails."""
    org = _org(db)
    a = _seed(db, org_id=org, content="a", scope="private", user_id=PRIYA)
    b = _seed(db, org_id=org, content="b", scope="private", user_id=PRIYA)
    graph_store.add_edge(
        db, org_id=org, src_id=a, dst_kind="memory", dst_id=b, relation="relates_to"
    )

    result = retract_memory(
        db, org_id=org, agent_id="agent-1", memory_id=b, via="mcp", user_id=PRIYA
    )
    assert result["forgotten"] is True


def test_sweep_removes_edges_to_forgotten_memories_only(db):
    org = _org(db)
    a = _seed(db, org_id=org, content="a")
    b = _seed(db, org_id=org, content="b")
    keep = _seed(db, org_id=org, content="keep")
    old = _seed(db, org_id=org, content="old truth", status="superseded")

    doomed = graph_store.add_edge(
        db, org_id=org, src_id=a, dst_kind="memory", dst_id=b, relation="relates_to"
    )
    kept = graph_store.add_edge(
        db, org_id=org, src_id=a, dst_kind="memory", dst_id=keep, relation="relates_to"
    )
    # Supersession history: an edge INTO a superseded row is the point of the
    # feature and must survive the sweep.
    history = graph_store.add_edge(
        db, org_id=org, src_id=a, dst_kind="memory", dst_id=old, relation="supersedes"
    )

    retract_memory(db, org_id=org, agent_id="agent-1", memory_id=b, via="mcp")
    removed = graph_store.sweep_orphan_edges(db, org_id=org)
    assert removed == 1

    remaining = {
        str(r.id)
        for r in db.execute(
            text("SELECT id FROM memory_edges WHERE org_id = :o"), {"o": org}
        ).fetchall()
    }
    assert doomed not in remaining
    assert kept in remaining
    assert history in remaining


def test_sweep_removes_edges_whose_entity_vanished(db):
    org = _org(db)
    src = _seed(db, org_id=org, content="m")
    entity = graph_store.add_entity(db, org_id=org, name="ghost-svc", kind="system")
    graph_store.add_edge(
        db, org_id=org, src_id=src, dst_kind="entity", dst_id=entity, relation="about"
    )
    db.execute(text("DELETE FROM entities WHERE id = :i"), {"i": entity})
    db.commit()

    assert graph_store.sweep_orphan_edges(db, org_id=org) == 1


# ---------------------------------------------------------------------------
# neighbors — the depth cap
# ---------------------------------------------------------------------------


def test_neighbors_depth_is_hard_capped_at_one(db):
    org = _org(db)
    with pytest.raises(ValueError):
        graph_store.neighbors(
            db,
            org_id=org,
            viewer_user_id=PRIYA,
            memory_ids=[str(uuid.uuid4())],
            depth=2,
        )


def test_neighbors_empty_entry_set_is_empty(db):
    org = _org(db)
    result = graph_store.neighbors(db, org_id=org, viewer_user_id=PRIYA, memory_ids=[])
    assert result["records"] == []
    assert result["connections"] == {}


# ---------------------------------------------------------------------------
# The engine-swap boundary: no SQL against the graph tables anywhere else
# ---------------------------------------------------------------------------


def test_no_module_but_graph_store_touches_the_graph_tables():
    """The seam is only a seam if nothing goes around it.

    Scans every executable module under services/ (tests, migrations, and
    graph_store.py itself excluded) for SQL against ``entities`` /
    ``memory_edges``. Mirrors the silo's loader-invariant style: named files
    would be caught by review; this catches the one nobody reviews.
    """
    import pathlib
    import re

    services_dir = pathlib.Path(__file__).resolve().parents[2]
    assert services_dir.name == "services"

    edge_pattern = re.compile(r"\bmemory_edges\b")
    entity_sql_pattern = re.compile(
        r"(?i)\b(from|join|into|update|delete\s+from)\s+entities\b"
    )

    offenders = []
    for path in services_dir.rglob("*.py"):
        rel = path.relative_to(services_dir)
        parts = set(rel.parts)
        if (
            path.name == "graph_store.py"
            or "tests" in parts
            or "migrations" in rel.parts[0]
            or ".venv" in parts
            or "node_modules" in parts
        ):
            continue
        try:
            source = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if edge_pattern.search(source) or entity_sql_pattern.search(source):
            offenders.append(str(rel))

    assert offenders == [], (
        f"SQL against the graph tables outside graph_store.py: {offenders}. "
        "The GraphStore seam is the engine-swap boundary; route through it."
    )
