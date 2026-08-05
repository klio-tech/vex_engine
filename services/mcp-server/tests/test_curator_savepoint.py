"""One failed graph item must not discard the batch's other work. Real Postgres.

The seven best-effort handlers fixed in ``shared`` and elsewhere take a bare
``db.rollback()``: each is a pure read or already-committed work, so there is
nothing in flight for the rollback to throw away. The curator's three are
different in kind. They sit inside a loop — facts, then entities, then edges —
and the loop's earlier iterations may still be uncommitted when a later one
fails. A bare rollback there clears the aborted transaction (necessary) and
destroys the partition's other writes on the way (not).

``ROLLBACK TO SAVEPOINT`` undoes exactly the failed item. These tests pin that
difference directly, because nothing else can: with the write helpers
committing eagerly as they do today the two behave identically most of the
time, and "identical today" is not the contract — the moment a helper stops
committing per row, a bare rollback silently starts eating batches.

So each test puts a genuinely uncommitted statement on the session first, then
fails one item with a REAL aborting SQL error, then asks whether the
uncommitted statement survived. That is the whole difference between the two
fixes, isolated.

Why the failures are injected rather than provoked: ``add_entity`` and
``add_edge`` validate hard in Python before they touch SQL — an over-long name
raises ``EntityRefused``, an unknown endpoint raises ``LookupError``, and
neither aborts a transaction, so neither can distinguish the two fixes. The
class of failure the handlers exist for is the one that reaches Postgres and is
refused there. That is what is injected: a statement Postgres rejects, on the
real session, from inside the call.
"""

import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text

from app.curator import EntityBudget, _store_graph
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the curator savepoint tests")

_EMBED_LITERAL = "[" + ",".join(["0.1"] * 1024) + "]"
PRIYA = "11111111-2222-3333-4444-555555555555"


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
                for table in ("memory_edges", "entities", "session_memories"):
                    conn.execute(
                        text(f"DELETE FROM {table} WHERE org_id = :o"), {"o": org_id}
                    )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _org(db) -> str:
    org_id = f"org_{uuid.uuid4().hex[:8]}"
    db.created_org_ids.add(org_id)
    return org_id


def _in_flight_marker(db, org_id: str) -> str:
    """An UNCOMMITTED insert: the 'good work' a bare rollback would discard."""
    entity_id = str(uuid.uuid4())
    db.execute(
        text(
            "INSERT INTO entities (id, org_id, name, norm_name, kind) "
            "VALUES (:id, :o, 'in flight', :norm, 'system')"
        ),
        {"id": entity_id, "o": org_id, "norm": f"in flight {uuid.uuid4().hex[:6]}"},
    )
    return entity_id


def _marker_survives(db, entity_id: str) -> bool:
    return (
        db.execute(
            text("SELECT count(*) FROM entities WHERE id = :id"), {"id": entity_id}
        ).scalar()
        == 1
    )


def _aborting(*_args, **kwargs):
    """Stand in for a store call that reaches Postgres and is refused there."""
    db = kwargs.get("db") or _args[0]
    db.execute(text("SELECT no_such_column_at_all"))
    raise AssertionError("unreachable: the statement above raises")


def test_a_failed_entity_does_not_discard_the_batchs_in_flight_work(db):
    org_id = _org(db)
    marker = _in_flight_marker(db, org_id)

    with patch("app.curator.graph_store.add_entity", side_effect=_aborting):
        _store_graph(
            db,
            org_id=org_id,
            entities=[{"name": "auth service", "kind": "system"}],
            edges=[],
            fact_row_ids={},
            entity_budget=EntityBudget(12),
        )

    assert _marker_survives(db, marker), (
        "the failed entity took the batch's uncommitted work with it; a bare "
        "rollback undoes the whole transaction, a savepoint undoes one item"
    )
    assert db.execute(text("SELECT 1")).scalar() == 1, (
        "and the session must still be usable — the rest of the tick runs on it"
    )


def test_a_failed_edge_does_not_discard_the_batchs_in_flight_work(db):
    org_id = _org(db)
    marker = _in_flight_marker(db, org_id)

    # ``add_entity`` commits internally today, which would make the marker
    # durable before the edge ever runs and hide the difference this test
    # exists to show. So the entity step is stubbed to resolve a name without
    # touching the database: the marker is still UNCOMMITTED when the edge
    # fails, which is the state a bare rollback destroys.
    entity_id = _in_flight_marker(db, org_id)
    with (
        patch("app.curator.graph_store.add_entity", return_value=entity_id),
        patch("app.curator.graph_store.add_edge", side_effect=_aborting),
    ):
        _store_graph(
            db,
            org_id=org_id,
            entities=[{"name": "auth service", "kind": "system"}],
            edges=[{"src": 0, "relation": "about", "entity": "auth service"}],
            fact_row_ids={0: str(uuid.uuid4())},
            entity_budget=EntityBudget(12),
        )

    assert _marker_survives(db, marker), (
        "the failed edge rolled the whole transaction back, discarding work "
        "that had nothing to do with it"
    )
    assert db.execute(text("SELECT 1")).scalar() == 1


def test_a_healthy_batch_still_stores_its_entities_and_edges(db):
    """Positive control: the savepoints do not break the ordinary path.

    Every assertion above is that something SURVIVED. A ``_store_graph`` that
    quietly stored nothing at all would pass both of them.
    """
    org_id = _org(db)
    memory_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, memory_type, content,
                embedding, scope, status, user_id, metadata,
                created_at, updated_at
            ) VALUES (
                :id, 'sess-x', :o, 'agent-1', 'fact', 'a fact',
                :embedding, 'private', 'active', :user_id, '{}'::jsonb,
                now(), now()
            )
            """
        ),
        {
            "id": memory_id,
            "o": org_id,
            "embedding": _EMBED_LITERAL,
            "user_id": PRIYA,
        },
    )
    db.commit()

    drawn = _store_graph(
        db,
        org_id=org_id,
        entities=[{"name": "auth service", "kind": "system"}],
        edges=[{"src": 0, "relation": "about", "entity": "auth service"}],
        fact_row_ids={0: memory_id},
        entity_budget=EntityBudget(12),
    )

    assert drawn == 1
    assert (
        db.execute(
            text("SELECT count(*) FROM entities WHERE org_id = :o"), {"o": org_id}
        ).scalar()
        == 1
    )
