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

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text

from shared import graph_store
from shared.graph_store import EntityRefused
from shared.memory import retract_memory
from shared.testing import database_url, requires_db

DB_URL = database_url()


pytestmark = requires_db("the GraphStore contract tests")

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
    first = graph_store.add_entity(
        db, org_id=org, name="Auth Service", kind="system"
    ).id
    second = graph_store.add_entity(
        db, org_id=org, name="  auth   SERVICE ", kind="system"
    ).id
    assert first == second, "case/whitespace variants must converge on one row"

    row = db.execute(
        text("SELECT name, norm_name, kind FROM entities WHERE id = :i"),
        {"i": first},
    ).fetchone()
    assert row.norm_name == "auth service"
    assert row.kind == "system"
    # The first spelling wins; a repeat upsert must not thrash the display name.
    assert row.name == "Auth Service"


def test_add_entity_reports_whether_it_inserted_or_hit(db):
    """``inserted`` must come from the row, not from a constant.

    This is the flag the curator's per-tick entity cap is spent on. If it were
    hardcoded ``True`` the cap would still be charged for every reference —
    the exact defect — and if it were hardcoded ``False`` the god-entity guard
    would never fire at all. Only real Postgres can tell the difference:
    ``(xmax = 0)`` is a property of the row the upsert produced.
    """
    org = _org(db)
    first = graph_store.add_entity(db, org_id=org, name="Auth Service", kind="system")
    assert first.inserted is True, "the first upsert of a name creates the row"

    second = graph_store.add_entity(
        db, org_id=org, name="  auth   SERVICE ", kind="system"
    )
    assert second.id == first.id, "control: the second call did resolve the same row"
    assert second.inserted is False, (
        "the second upsert took the CONFLICT arm and created nothing; "
        "reporting it as an insert is what emptied the curator's entity budget "
        "on references"
    )

    # A different kind is a different entity, so that one IS an insert again —
    # proving the flag tracks the row rather than the call ordinal.
    other_kind = graph_store.add_entity(
        db, org_id=org, name="Auth Service", kind="concept"
    )
    assert other_kind.inserted is True


def test_add_entity_same_name_different_kind_is_a_different_entity(db):
    org = _org(db)
    a = graph_store.add_entity(db, org_id=org, name="mercury", kind="system").id
    b = graph_store.add_entity(db, org_id=org, name="mercury", kind="person").id
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
    entity = graph_store.add_entity(db, org_id=org, name="postgres", kind="tool").id

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
    entity = graph_store.add_entity(db, org_id=org, name="ghost-svc", kind="system").id
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
# The destination org guard, pinned where behaviour cannot reach it
# ---------------------------------------------------------------------------


def test_neighbors_keeps_both_destination_org_guards():
    """``neighbors`` must pin the destination row's org in BOTH places.

    The JOIN (``ON m.id = scored.other_id AND m.org_id = :org_id``) and the
    WHERE clause list (``"m.org_id = :org_id"``) are deliberate defence in
    depth: a tampered cross-org edge must grant nothing even if one guard is
    later refactored away by accident.

    This assertion is STRUCTURAL, and that is not laziness. On an INNER JOIN
    the two are the same predicate on the same column of the same row, so
    every query returns identical results with either one removed — verified
    by mutation, each single-site deletion leaves the whole 579-test shared
    suite green. There is no behavioural test that can distinguish them.
    Redundancy that no result can observe can only be pinned at the source,
    or it is not pinned at all.
    """
    import inspect
    import re

    source = inspect.getsource(graph_store.neighbors)
    flat = re.sub(r"\s+", " ", source)

    assert '"m.org_id = :org_id",' in source, (
        "the WHERE-clause destination org guard is gone from neighbors(); "
        "restore it — the JOIN guard alone leaves the tenancy boundary "
        "resting on a single unasserted line"
    )
    assert "ON m.id = scored.other_id AND m.org_id = :org_id" in flat, (
        "the JOIN destination org guard is gone from neighbors(); restore it "
        "— the WHERE guard alone leaves the tenancy boundary resting on a "
        "single unasserted line"
    )


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


# ---------------------------------------------------------------------------
# sweep_orphan_entities — the entity GC
#
# Retention deletes from ``session_memories``; ``sweep_orphan_edges`` then
# collects the edges whose endpoints vanished. Nothing collected the ENTITIES
# those edges used to evidence, so an entity outlived every memory that ever
# mentioned it, forever.
#
# The delete is user data, so the tests below assert both halves: exactly the
# unevidenced entities go, and nothing else does.
# ---------------------------------------------------------------------------


def _entity_ids(db, *, org_id: str) -> set[str]:
    rows = db.execute(
        text("SELECT id FROM entities WHERE org_id = :o"), {"o": org_id}
    ).fetchall()
    return {str(r[0]) for r in rows}


def _age_entity(db, entity_id: str, *, days: int = 30) -> None:
    """Backdate an entity past any grace window, so the test is about
    evidence and not about the anti-race guard."""
    db.execute(
        text(
            "UPDATE entities SET created_at = NOW() - make_interval(days => :d) "
            "WHERE id = :id"
        ),
        {"d": days, "id": entity_id},
    )
    db.commit()


def test_sweep_collects_an_entity_whose_every_evidencing_memory_is_gone(db):
    org = _org(db)
    mem = _seed(db, org_id=org, content="the billing service is flaky")
    ent = graph_store.add_entity(db, org_id=org, name="billing", kind="system").id
    graph_store.add_edge(
        db, org_id=org, src_id=mem, dst_kind="entity", dst_id=ent, relation="about"
    )
    _age_entity(db, ent)

    # Positive control: while the evidence stands, the sweep must take nothing.
    assert graph_store.sweep_orphan_entities(db, org_id=org) == 0
    assert ent in _entity_ids(db, org_id=org)

    # Retention removes the memory; the edge sweep then removes the edge.
    db.execute(text("DELETE FROM session_memories WHERE id = :id"), {"id": mem})
    db.commit()
    graph_store.sweep_orphan_edges(db, org_id=org)

    assert graph_store.sweep_orphan_entities(db, org_id=org) == 1
    assert ent not in _entity_ids(db, org_id=org)


def test_sweep_spares_an_entity_evidenced_by_a_memory_the_caller_cannot_see(db):
    """Evidence is evidence, whoever can see it.

    The visibility ladder decides what a CALLER is shown. A retention sweep is
    not a caller: an entity evidenced only by somebody's private memory is
    still evidenced, and collecting it would delete one user's data because a
    different user could not see it.
    """
    org = _org(db)
    private_mem = _seed(
        db,
        org_id=org,
        content="priya's private note about billing",
        scope="private",
        user_id=PRIYA,
    )
    ent = graph_store.add_entity(db, org_id=org, name="billing", kind="system").id
    graph_store.add_edge(
        db,
        org_id=org,
        src_id=private_mem,
        dst_kind="entity",
        dst_id=ent,
        relation="about",
    )
    _age_entity(db, ent)

    assert graph_store.sweep_orphan_entities(db, org_id=org) == 0
    assert ent in _entity_ids(db, org_id=org)


def test_sweep_never_reaches_into_another_org(db):
    org_a = _org(db)
    org_b = _org(db)
    orphan_a = graph_store.add_entity(
        db, org_id=org_a, name="billing", kind="system"
    ).id
    orphan_b = graph_store.add_entity(
        db, org_id=org_b, name="billing", kind="system"
    ).id
    _age_entity(db, orphan_a)
    _age_entity(db, orphan_b)

    assert graph_store.sweep_orphan_entities(db, org_id=org_a) == 1
    assert orphan_a not in _entity_ids(db, org_id=org_a)
    assert orphan_b in _entity_ids(db, org_id=org_b), (
        "one org's retention run deleted another org's entity"
    )


def test_sweep_spares_an_entity_that_is_merely_new(db):
    """add_entity and add_edge are two statements.

    An entity created microseconds before the sweep has no edge yet. Deleting
    it is a race that loses data the caller is in the middle of writing, so
    the sweep only collects entities older than its grace window.
    """
    org = _org(db)
    fresh = graph_store.add_entity(db, org_id=org, name="brand-new", kind="concept").id

    assert graph_store.sweep_orphan_entities(db, org_id=org) == 0
    assert fresh in _entity_ids(db, org_id=org)

    # Positive control: the fixture is not vacuous — age it and it goes.
    _age_entity(db, fresh)
    assert graph_store.sweep_orphan_entities(db, org_id=org) == 1


def test_sweep_leaves_evidenced_siblings_untouched(db):
    """Exactness: the sweep removes the orphan and ONLY the orphan."""
    org = _org(db)
    mem = _seed(db, org_id=org, content="auth service owns login")
    kept = graph_store.add_entity(db, org_id=org, name="auth", kind="system").id
    orphan = graph_store.add_entity(db, org_id=org, name="ghost", kind="concept").id
    graph_store.add_edge(
        db, org_id=org, src_id=mem, dst_kind="entity", dst_id=kept, relation="about"
    )
    _age_entity(db, kept)
    _age_entity(db, orphan)

    assert graph_store.sweep_orphan_entities(db, org_id=org) == 1
    assert _entity_ids(db, org_id=org) == {kept}
