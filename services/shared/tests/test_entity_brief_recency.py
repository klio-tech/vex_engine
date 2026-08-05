"""``entity_brief`` must show the caller the NEWEST evidence, and say when it capped.

The evidence query was ``ORDER BY m.created_at ASC LIMIT 50``. Past the
fiftieth memory about an entity, that is a query for the oldest fifty — so
``explore`` froze in the past. The failure is quiet and it gets worse with use:
the more an org talks about something, the more stale its brief, and every
derived field is computed off that stale slice. ``decisions``,
``contributors``, ``provenance`` and ``superseded`` are all reported as if they
described the entity, while describing only its first fifty mentions. A
decision superseded twice last week does not appear at all.

Nothing told the caller either. The brief carried no truncation flag, so a
50-memory slice of a 500-memory entity was indistinguishable from the whole
thing.

Two changes, tested here: select the newest bounded set (then reorder oldest-
first for presentation, so the brief still reads as a chronology), and report
``truncated``.

Real Postgres: ``ORDER BY`` direction and ``LIMIT`` are the defect.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text

from shared import graph_store
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the entity_brief recency tests")

EMBED_DIM = 1024
FIXED_EMBEDDING_LITERAL = "[" + ",".join(["0.1"] * EMBED_DIM) + "]"

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


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


def _seed(db, *, org_id: str, content: str, created_at, memory_type="fact") -> str:
    memory_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, memory_type, content,
                embedding, scope, status, metadata, created_at, updated_at
            ) VALUES (
                :id, :session_id, :org_id, 'agent-1', :memory_type, :content,
                :embedding, 'org', 'active', '{}'::jsonb, :now, :now
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
            "now": created_at,
        },
    )
    db.commit()
    return memory_id


def _entity_with_evidence(db, org_id: str, count: int, *, newest_type="fact"):
    """One entity with ``count`` memories about it, oldest first.

    The LAST memory gets ``newest_type`` so a test can ask whether the newest
    evidence made it into the brief at all.
    """
    entity = graph_store.add_entity(
        db, org_id=org_id, name="auth service", kind="system"
    ).id
    ids = []
    for i in range(count):
        memory_type = newest_type if i == count - 1 else "fact"
        memory_id = _seed(
            db,
            org_id=org_id,
            content=f"note {i}",
            created_at=_BASE + timedelta(minutes=i),
            memory_type=memory_type,
        )
        graph_store.add_edge(
            db,
            org_id=org_id,
            src_id=memory_id,
            dst_kind="entity",
            dst_id=entity,
            relation="about",
        )
        ids.append(memory_id)
    return entity, ids


def _brief(db, org_id: str):
    return graph_store.entity_brief(
        db, org_id=org_id, viewer_user_id=None, entity="auth service"
    )


# ---------------------------------------------------------------------------
# Positive control first: an unbounded entity is reported whole and untruncated.
# ---------------------------------------------------------------------------


def test_a_small_entity_is_complete_and_not_truncated(db):
    org_id = _org(db)
    _entity_with_evidence(db, org_id, 5)

    brief = _brief(db, org_id)

    assert brief["evidence_count"] == 5
    assert brief["truncated"] is False, (
        "an entity inside the bound must not be reported as capped, or the "
        "flag says nothing"
    )
    contents = [m.content for m in brief["memories"]]
    assert contents == [f"note {i}" for i in range(5)], (
        "and the brief still reads oldest-first — it is a chronology"
    )


# ---------------------------------------------------------------------------
# The defect.
# ---------------------------------------------------------------------------


def test_the_brief_shows_the_newest_evidence_not_the_oldest(db):
    org_id = _org(db)
    total = graph_store._BRIEF_EVIDENCE_LIMIT + 10
    _entity_with_evidence(db, org_id, total)

    brief = _brief(db, org_id)

    contents = [m.content for m in brief["memories"]]
    assert f"note {total - 1}" in contents, (
        "the most recent memory about the entity is missing from its brief; "
        "past the limit the query returned the OLDEST slice, so explore "
        "reports the entity as it was 50 memories ago"
    )
    assert "note 0" not in contents, (
        "and the very first memory must have fallen out — a bounded window "
        "that still starts at the beginning has not moved"
    )


def test_a_capped_brief_says_it_was_capped(db):
    org_id = _org(db)
    total = graph_store._BRIEF_EVIDENCE_LIMIT + 10
    _entity_with_evidence(db, org_id, total)

    brief = _brief(db, org_id)

    assert brief["evidence_count"] == graph_store._BRIEF_EVIDENCE_LIMIT
    assert brief["truncated"] is True, (
        "the caller has no way to know the derived counts describe a slice "
        "rather than the entity; evidence_count alone cannot say, because it "
        "is exactly what a 50-memory entity reports"
    )


def test_a_capped_brief_is_still_presented_oldest_first(db):
    """Newest SELECTED, oldest PRESENTED — the two are separate concerns."""
    org_id = _org(db)
    total = graph_store._BRIEF_EVIDENCE_LIMIT + 10
    _entity_with_evidence(db, org_id, total)

    brief = _brief(db, org_id)

    created = [m.created_at for m in brief["memories"]]
    assert created == sorted(created), (
        "the brief is read as a chronology and the LLM summary is built from "
        "it in order; selecting newest-first must not invert the narrative"
    )


def test_the_newest_decision_survives_the_cap(db):
    """The consequence that matters: derived fields follow the window.

    ``decisions`` is computed from the selected slice. With the old ordering a
    decision recorded after the fiftieth memory could never appear, which is
    the opposite of what a caller asking about an entity wants.
    """
    org_id = _org(db)
    total = graph_store._BRIEF_EVIDENCE_LIMIT + 10
    _entity_with_evidence(db, org_id, total, newest_type="decision")

    brief = _brief(db, org_id)

    assert [d["content"] for d in brief["decisions"]] == [f"note {total - 1}"]
