"""Migration 035: the knowledge graph — entities and typed memory edges.

Mirrors test_migration_031.py's conventions — real Postgres via DATABASE_URL,
skips ONLY when the DB is unreachable (a skip is not a pass), CHECK
constraints exercised with real INSERTs, and the downgrade genuinely run.

The backfill-honesty tests are the heart of this file: migration 035 creates
``supersedes`` edges from ``status='superseded'`` rows ONLY where the
successor id is genuinely recoverable from ``superseded_by``. Where it is not
(NULL, or pointing at a row that no longer exists), no edge may be invented.
Those tests seed all three shapes, re-run the ``035 → 034 → 035`` cycle with
the real alembic machinery, and count what the backfill produced.
"""

import os
import subprocess
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[1]

ENTITY_KINDS = ("system", "person", "concept", "decision_subject", "tool", "other")
EDGE_RELATIONS = ("about", "supersedes", "derived_from", "relates_to")


@pytest.fixture(scope="module")
def conn():
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception:
        pytest.skip(
            "Postgres not reachable; migration 035 tests require the live test DB"
        )
    with create_engine(DB_URL).connect() as c:
        yield c


def _columns(conn, table):
    rows = conn.execute(
        text(
            "SELECT column_name, is_nullable, data_type "
            "FROM information_schema.columns WHERE table_name = :t"
        ),
        {"t": table},
    ).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def _has_constraint(conn, name):
    return (
        conn.execute(
            text("SELECT count(*) FROM pg_constraint WHERE conname = :n"),
            {"n": name},
        ).scalar()
        > 0
    )


# ---------------------------------------------------------------------------
# Schema shape — entities
# ---------------------------------------------------------------------------


def test_entities_table_exists_with_tenancy_column(conn):
    cols = _columns(conn, "entities")
    assert {"id", "org_id", "name", "norm_name", "kind", "embedding", "created_at"} <= (
        set(cols)
    )
    assert cols["org_id"][0] == "NO", "org_id is the tenancy boundary; never null"
    assert cols["name"][0] == "NO"
    assert cols["norm_name"][0] == "NO"
    assert cols["kind"][0] == "NO"
    assert cols["embedding"][0] == "YES", "embedding arrives lazily; nullable"


def test_entities_dedup_unique_on_org_norm_name_kind(conn):
    """The upsert arbiter: one entity row per (org, norm_name, kind)."""
    org = f"t035_{uuid.uuid4().hex[:8]}"
    tx = conn.begin_nested()
    conn.execute(
        text(
            "INSERT INTO entities (id, org_id, name, norm_name, kind) "
            "VALUES (:a, :o, 'Auth Service', 'auth service', 'system')"
        ),
        {"a": str(uuid.uuid4()), "o": org},
    )
    with pytest.raises(IntegrityError):
        conn.execute(
            text(
                "INSERT INTO entities (id, org_id, name, norm_name, kind) "
                "VALUES (:a, :o, 'auth SERVICE', 'auth service', 'system')"
            ),
            {"a": str(uuid.uuid4()), "o": org},
        )
    tx.rollback()


def test_same_norm_name_in_another_org_is_a_separate_row(conn):
    """The unique key includes org_id: tenants never share an entity row."""
    tx = conn.begin_nested()
    for org in (f"t035a_{uuid.uuid4().hex[:6]}", f"t035b_{uuid.uuid4().hex[:6]}"):
        conn.execute(
            text(
                "INSERT INTO entities (id, org_id, name, norm_name, kind) "
                "VALUES (:a, :o, 'pricing', 'pricing', 'concept')"
            ),
            {"a": str(uuid.uuid4()), "o": org},
        )
    tx.rollback()


def test_entity_kind_check_rejects_a_typo(conn):
    tx = conn.begin_nested()
    with pytest.raises(IntegrityError):
        conn.execute(
            text(
                "INSERT INTO entities (id, org_id, name, norm_name, kind) "
                "VALUES (:a, 't035', 'x', 'x', 'systm')"
            ),
            {"a": str(uuid.uuid4())},
        )
    tx.rollback()


def test_entity_kind_check_admits_the_closed_set(conn):
    assert _has_constraint(conn, "ck_entities_kind")
    tx = conn.begin_nested()
    for kind in ENTITY_KINDS:
        conn.execute(
            text(
                "INSERT INTO entities (id, org_id, name, norm_name, kind) "
                "VALUES (:a, 't035', :n, :n, :k)"
            ),
            {"a": str(uuid.uuid4()), "n": f"e-{kind}", "k": kind},
        )
    tx.rollback()


# ---------------------------------------------------------------------------
# Schema shape — memory_edges
# ---------------------------------------------------------------------------


def test_memory_edges_table_exists_with_tenancy_column(conn):
    cols = _columns(conn, "memory_edges")
    assert {
        "id",
        "org_id",
        "src_id",
        "dst_kind",
        "dst_id",
        "relation",
        "provenance",
        "created_by",
        "created_at",
    } <= set(cols)
    assert cols["org_id"][0] == "NO"
    assert cols["src_id"][0] == "NO"
    assert cols["dst_id"][0] == "NO"
    assert cols["created_by"][0] == "YES", "human-drawn edges are v2; nullable now"


def _insert_edge(conn, **overrides):
    params = {
        "id": str(uuid.uuid4()),
        "org_id": "t035",
        "src_id": str(uuid.uuid4()),
        "dst_kind": "memory",
        "dst_id": str(uuid.uuid4()),
        "relation": "relates_to",
    }
    params.update(overrides)
    conn.execute(
        text(
            "INSERT INTO memory_edges (id, org_id, src_id, dst_kind, dst_id, relation) "
            "VALUES (:id, :org_id, :src_id, :dst_kind, :dst_id, :relation)"
        ),
        params,
    )


def test_edge_relation_check_rejects_a_typo(conn):
    tx = conn.begin_nested()
    with pytest.raises(IntegrityError):
        _insert_edge(conn, relation="superseds")
    tx.rollback()


def test_edge_relation_check_admits_the_closed_set(conn):
    assert _has_constraint(conn, "ck_memory_edges_relation")
    tx = conn.begin_nested()
    for relation in EDGE_RELATIONS:
        _insert_edge(conn, relation=relation)
    tx.rollback()


def test_edge_dst_kind_check_is_memory_or_entity(conn):
    assert _has_constraint(conn, "ck_memory_edges_dst_kind")
    tx = conn.begin_nested()
    for dst_kind in ("memory", "entity"):
        _insert_edge(conn, dst_kind=dst_kind)
    tx.rollback()
    tx = conn.begin_nested()
    with pytest.raises(IntegrityError):
        _insert_edge(conn, dst_kind="row")
    tx.rollback()


def test_edge_provenance_defaults_to_inferred_and_rejects_unknown(conn):
    cols = _columns(conn, "memory_edges")
    assert cols["provenance"][0] == "NO"
    assert _has_constraint(conn, "ck_memory_edges_provenance")

    tx = conn.begin_nested()
    edge_id = str(uuid.uuid4())
    _insert_edge(conn, id=edge_id)
    got = conn.execute(
        text("SELECT provenance FROM memory_edges WHERE id = :i"), {"i": edge_id}
    ).scalar()
    assert got == "INFERRED"
    tx.rollback()

    tx = conn.begin_nested()
    with pytest.raises(IntegrityError):
        conn.execute(
            text(
                "INSERT INTO memory_edges "
                "(id, org_id, src_id, dst_kind, dst_id, relation, provenance) "
                "VALUES (:i, 't035', :s, 'memory', :d, 'about', 'GUESSED')"
            ),
            {
                "i": str(uuid.uuid4()),
                "s": str(uuid.uuid4()),
                "d": str(uuid.uuid4()),
            },
        )
    tx.rollback()


def test_edge_uniqueness_over_org_src_dst_relation(conn):
    src, dst = str(uuid.uuid4()), str(uuid.uuid4())
    tx = conn.begin_nested()
    _insert_edge(conn, src_id=src, dst_id=dst, relation="about", dst_kind="entity")
    with pytest.raises(IntegrityError):
        _insert_edge(conn, src_id=src, dst_id=dst, relation="about", dst_kind="entity")
    tx.rollback()


def test_traversal_indexes_exist_both_directions(conn):
    idx = conn.execute(
        text("SELECT indexdef FROM pg_indexes WHERE tablename = 'memory_edges'")
    ).fetchall()
    defs = [r[0] for r in idx]
    assert any("org_id" in d and "src_id" in d for d in defs), (
        "forward traversal needs (org_id, src_id)"
    )
    assert any("org_id" in d and "dst_kind" in d and "dst_id" in d for d in defs), (
        "reverse traversal needs (org_id, dst_kind, dst_id)"
    )


def test_no_foreign_key_from_edges_to_session_memories(conn):
    """dst_id is polymorphic and a forget must never fail on an edge, so
    integrity is the store layer's job — the schema must NOT cascade."""
    fks = conn.execute(
        text(
            "SELECT count(*) FROM pg_constraint "
            "WHERE conrelid = 'memory_edges'::regclass AND contype = 'f'"
        )
    ).scalar()
    assert fks == 0, "memory_edges must carry no foreign keys"


# ---------------------------------------------------------------------------
# Backfill honesty — the only backfill 035 performs
# ---------------------------------------------------------------------------


def _seed_memory(conn, *, org_id, content, status="active", superseded_by=None):
    memory_id = str(uuid.uuid4())
    conn.execute(
        text(
            "INSERT INTO session_memories "
            "(id, session_id, org_id, agent_id, memory_type, content, scope, "
            " status, superseded_by, metadata, created_at, updated_at) "
            "VALUES (:id, 's035', :org, 'a', 'fact', :content, 'org', "
            "        :status, :superseded_by, '{}'::jsonb, NOW(), NOW())"
        ),
        {
            "id": memory_id,
            "org": org_id,
            "content": content,
            "status": status,
            "superseded_by": superseded_by,
        },
    )
    return memory_id


def _alembic(*args):
    env = dict(os.environ, DATABASE_URL=DB_URL)
    result = subprocess.run(
        ["alembic", *args],
        cwd=str(MIGRATIONS_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"alembic {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}"
    )


def test_backfill_creates_edges_only_where_the_successor_is_recoverable(conn):
    """Seed all three supersession shapes, replay 035, count what it made.

    * recoverable   — superseded_by names a row that exists → ONE edge
    * amnesiac      — superseded_by IS NULL → no edge (no history invented)
    * dangling      — superseded_by names a vanished row → no edge

    Also exercises the downgrade for real: the cycle is 035 → 034 → 035.
    """
    org = f"t035bf_{uuid.uuid4().hex[:8]}"

    successor = _seed_memory(conn, org_id=org, content="new truth")
    recoverable = _seed_memory(
        conn,
        org_id=org,
        content="old truth",
        status="superseded",
        superseded_by=successor,
    )
    amnesiac = _seed_memory(
        conn, org_id=org, content="old, successor unknown", status="superseded"
    )
    dangling = _seed_memory(
        conn,
        org_id=org,
        content="old, successor vanished",
        status="superseded",
        superseded_by=str(uuid.uuid4()),
    )
    conn.commit()

    try:
        _alembic("downgrade", "034")
        _alembic("upgrade", "head")

        edges = conn.execute(
            text(
                "SELECT src_id, dst_id, dst_kind, relation, provenance "
                "FROM memory_edges WHERE org_id = :o"
            ),
            {"o": org},
        ).fetchall()

        assert len(edges) == 1, (
            f"exactly the one recoverable supersession may become an edge; "
            f"got {len(edges)}"
        )
        edge = edges[0]
        # Direction: the successor supersedes the old row.
        assert str(edge.src_id) == successor
        assert str(edge.dst_id) == recoverable
        assert edge.dst_kind == "memory"
        assert edge.relation == "supersedes"
        assert edge.provenance == "INFERRED", (
            "the supersession judgment was an LLM's; the edge must say so"
        )

        # The amnesiac and dangling rows produced NOTHING, anywhere.
        for orphan in (amnesiac, dangling):
            n = conn.execute(
                text(
                    "SELECT count(*) FROM memory_edges WHERE dst_id = :d OR src_id = :d"
                ),
                {"d": orphan},
            ).scalar()
            assert n == 0, "an edge was invented for an unrecoverable supersession"
    finally:
        conn.execute(text("DELETE FROM memory_edges WHERE org_id = :o"), {"o": org})
        conn.execute(text("DELETE FROM session_memories WHERE org_id = :o"), {"o": org})
        conn.commit()


def test_backfilled_edges_never_exceed_superseded_rows(conn):
    """Global honesty bound: you cannot have recovered more history than
    exists. Every backfilled supersedes edge's endpoints must both exist."""
    superseded = conn.execute(
        text("SELECT count(*) FROM session_memories WHERE status = 'superseded'")
    ).scalar()
    supersedes_edges = conn.execute(
        text(
            "SELECT count(*) FROM memory_edges "
            "WHERE relation = 'supersedes' AND dst_kind = 'memory'"
        )
    ).scalar()
    assert supersedes_edges <= superseded

    broken = conn.execute(
        text(
            """
            SELECT count(*) FROM memory_edges e
            WHERE e.relation = 'supersedes' AND e.dst_kind = 'memory'
              AND (
                NOT EXISTS (SELECT 1 FROM session_memories s
                            WHERE s.id = e.src_id AND s.org_id = e.org_id)
                OR NOT EXISTS (SELECT 1 FROM session_memories s
                               WHERE s.id = e.dst_id AND s.org_id = e.org_id)
              )
            """
        )
    ).scalar()
    assert broken == 0, "a supersedes edge references a nonexistent endpoint"
