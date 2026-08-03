"""Hidden RAW captures are the knowledge graph's fuel.

``recall_hidden = TRUE`` exists so raw hook noise never surfaces in *recall*.
It was never meant to starve the curator. In production every hook capture
lands as an ``observation`` with ``recall_hidden = TRUE`` (migration 026), so
the curator's original ``recall_hidden = FALSE`` input predicate matched
nothing at all and the graph could never populate.

These tests pin the corrected input contract, and — just as importantly —
every guarantee that must survive the relaxation:

* hidden RAW rows ARE eligible fuel;
* curator OUTPUT types (``fact``/``summary``) are never eligible input, so a
  tick cannot re-digest its own distillate;
* the user silo holds — a tick for A is never fed by B, whether B's rows are
  hidden or visible;
* ``status`` still filters — retracted and superseded rows are not fuel;
* the tick stays idempotent, and the cursor neither stalls nor skips now that
  a hidden backlog is finally visible to it.

Real Postgres throughout: a mocked pool would happily pass a predicate that
selects nothing.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

from app.curator import CursorState, _read_new_raw_rows, curator_tick

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
    reason="Postgres not reachable; the curator fuel tests require it",
)

EMBED_DIM = 1024
FIXED_EMBEDDING = [0.1] * EMBED_DIM
_EMBED_LITERAL = "[" + ",".join(["0.1"] * EMBED_DIM) + "]"

PRIYA = "11111111-2222-3333-4444-555555555555"
DEV = "99999999-8888-7777-6666-555555555555"


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
                for table in (
                    "memory_edges",
                    "entities",
                    "brain_curator_state",
                    "session_memories",
                ):
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


def _seed(
    db,
    *,
    org_id: str,
    user_id: str,
    content: str,
    memory_type: str = "observation",
    recall_hidden: bool = True,
    status: str = "active",
    created_at=None,
) -> str:
    """Insert one private memory row with full control of every predicate."""
    memory_id = str(uuid.uuid4())
    now = created_at or datetime.now(timezone.utc)
    db.execute(
        text(
            """
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, memory_type, content,
                embedding, scope, status, user_id, metadata,
                created_at, updated_at, recall_hidden
            ) VALUES (
                :id, :session_id, :org_id, 'agent-1', :memory_type, :content,
                :embedding, 'private', :status, :user_id, '{}'::jsonb,
                :now, :now, :recall_hidden
            )
            """
        ),
        {
            "id": memory_id,
            "session_id": f"sess-{uuid.uuid4().hex[:6]}",
            "org_id": org_id,
            "memory_type": memory_type,
            "content": content,
            "embedding": _EMBED_LITERAL,
            "status": status,
            "user_id": user_id,
            "now": now,
            "recall_hidden": recall_hidden,
        },
    )
    db.commit()
    return memory_id


def _seed_production_shape(db, *, org_id: str, user_id: str, hook_captures: int = 6):
    """The row shape production actually holds for a private user silo.

    Hidden ``observation`` hook captures (the RAW fuel), plus the curator's own
    visible ``fact`` and ``summary`` output. Nothing else.
    """
    base = datetime.now(timezone.utc)
    for i in range(hook_captures):
        _seed(
            db,
            org_id=org_id,
            user_id=user_id,
            content=f"hook capture {i}: touched the auth service",
            memory_type="observation",
            recall_hidden=True,
            created_at=base + timedelta(seconds=i),
        )
    for i in range(4):
        _seed(
            db,
            org_id=org_id,
            user_id=user_id,
            content=f"prior distilled fact {i}",
            memory_type="fact",
            recall_hidden=False,
            created_at=base + timedelta(seconds=100 + i),
        )
    _seed(
        db,
        org_id=org_id,
        user_id=user_id,
        content="prior distilled summary",
        memory_type="summary",
        recall_hidden=False,
        created_at=base + timedelta(seconds=200),
    )


def _graph_result(facts=None, entities=None, edges=None):
    return {
        "facts": facts if facts is not None else [],
        "entities": entities if entities is not None else [],
        "edges": edges if edges is not None else [],
    }


def _patches(graph):
    """Mock only the LLM seams; every storage path stays real."""
    return (
        patch(
            "app.curator.generate_summary", new_callable=AsyncMock, return_value=None
        ),
        patch("app.curator.extract_graph", new_callable=AsyncMock, return_value=graph),
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
        # Every row shares one vector, so the REAL dedup would collapse the new
        # fact into the seeded rows; dedup has its own tests.
        patch(
            "shared.brain.find_similar_memory",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch("app.curator.call_llm", new_callable=AsyncMock, return_value=None),
    )


async def _run_tick(db, org, user, graph, **kwargs):
    p = _patches(graph)
    with p[0], p[1], p[2], p[3], p[4], p[5]:
        return await curator_tick(db, org, user, min_inputs=3, **kwargs)


_AUTH_GRAPH = _graph_result(
    facts=["the auth service verifies jwts"],
    entities=[{"name": "Auth Service", "kind": "system"}],
    edges=[{"src": 0, "relation": "about", "entity": "Auth Service"}],
)


def _entities(db, org):
    return [
        r.norm_name
        for r in db.execute(
            text("SELECT norm_name FROM entities WHERE org_id = :o ORDER BY norm_name"),
            {"o": org},
        ).fetchall()
    ]


def _about_edges(db, org):
    return db.execute(
        text(
            "SELECT count(*) FROM memory_edges "
            "WHERE org_id = :o AND relation = 'about' AND dst_kind = 'entity'"
        ),
        {"o": org},
    ).scalar()


# ---------------------------------------------------------------------------
# The bug: production's row shape yields an empty batch
# ---------------------------------------------------------------------------


def test_hidden_raw_captures_are_eligible_fuel(db):
    """The production silo: hidden observations + the curator's own output.

    The batch must be exactly the hidden RAW rows.
    """
    org = _org(db)
    _seed_production_shape(db, org_id=org, user_id=PRIYA, hook_captures=6)

    rows = _read_new_raw_rows(db, org, PRIYA, CursorState(None, None), 50)

    assert len(rows) == 6, (
        "hidden hook captures are the graph's fuel; a filter that excludes them "
        "leaves the curator with nothing to extract"
    )
    assert all("hook capture" in r.content for r in rows)


async def test_tick_over_the_production_shape_writes_entities_and_about_edges(db):
    """End to end: the tick selects a non-empty batch and populates the graph."""
    org = _org(db)
    _seed_production_shape(db, org_id=org, user_id=PRIYA, hook_captures=6)

    result = await _run_tick(db, org, PRIYA, _AUTH_GRAPH)

    assert result["processed"] == 6
    assert result["synthesized"] == 1  # the one fact
    assert _entities(db, org) == ["auth service"]

    edges = db.execute(
        text(
            "SELECT e.relation, e.provenance, m.scope, m.user_id, m.memory_type "
            "FROM memory_edges e JOIN session_memories m ON m.id = e.src_id "
            "WHERE e.org_id = :o AND e.dst_kind = 'entity'"
        ),
        {"o": org},
    ).fetchall()
    assert len(edges) == 1
    assert edges[0].relation == "about"
    assert edges[0].provenance == "INFERRED"
    assert edges[0].memory_type == "fact"
    # The evidencing fact stays in the author's private silo.
    assert edges[0].scope == "private"
    assert edges[0].user_id == PRIYA


# ---------------------------------------------------------------------------
# What the relaxation must NOT relax
# ---------------------------------------------------------------------------


def test_curator_output_types_never_become_curator_input(db):
    """No feedback loop: facts and summaries are output, never fuel.

    Visible or hidden, they must never enter a batch.
    """
    org = _org(db)
    base = datetime.now(timezone.utc)
    for i in range(4):
        _seed(
            db,
            org_id=org,
            user_id=PRIYA,
            content=f"distilled fact {i}",
            memory_type="fact",
            recall_hidden=i % 2 == 0,
            created_at=base + timedelta(seconds=i),
        )
        _seed(
            db,
            org_id=org,
            user_id=PRIYA,
            content=f"distilled summary {i}",
            memory_type="summary",
            recall_hidden=i % 2 == 0,
            created_at=base + timedelta(seconds=10 + i),
        )

    assert _read_new_raw_rows(db, org, PRIYA, CursorState(None, None), 50) == []

    # Positive control: the same silo with RAW rows added does yield a batch,
    # and yields ONLY the RAW rows.
    for i in range(3):
        _seed(
            db,
            org_id=org,
            user_id=PRIYA,
            content=f"hook capture {i}",
            memory_type="observation",
            recall_hidden=True,
            created_at=base + timedelta(seconds=20 + i),
        )
    rows = _read_new_raw_rows(db, org, PRIYA, CursorState(None, None), 50)
    assert len(rows) == 3
    assert all("hook capture" in r.content for r in rows)


def test_raw_type_list_excludes_the_curators_own_outputs():
    """The constant itself, not just its effect in SQL."""
    from app.curator import RAW_MEMORY_TYPES

    assert "fact" not in RAW_MEMORY_TYPES
    assert "summary" not in RAW_MEMORY_TYPES
    assert "observation" in RAW_MEMORY_TYPES  # positive control


def test_hidden_rows_of_another_user_are_not_fuel(db):
    """The silo holds. Hidden rows are eligible, but only the owner's."""
    org = _org(db)
    base = datetime.now(timezone.utc)
    for i in range(5):
        _seed(
            db,
            org_id=org,
            user_id=DEV,
            content=f"dev's hidden capture {i}",
            memory_type="observation",
            recall_hidden=True,
            created_at=base + timedelta(seconds=i),
        )

    assert _read_new_raw_rows(db, org, PRIYA, CursorState(None, None), 50) == []

    # Positive control: the very same rows ARE fuel for their own owner.
    dev_rows = _read_new_raw_rows(db, org, DEV, CursorState(None, None), 50)
    assert len(dev_rows) == 5


async def test_a_tick_for_one_user_creates_nothing_readable_by_another(db):
    """Acceptance shape: A's tick over hidden fuel leaves B's silo empty."""
    org = _org(db)
    _seed_production_shape(db, org_id=org, user_id=PRIYA, hook_captures=4)

    await _run_tick(db, org, PRIYA, _AUTH_GRAPH)

    dev_visible = db.execute(
        text(
            "SELECT count(*) FROM session_memories "
            "WHERE org_id = :o AND user_id = :u AND memory_type IN ('fact','summary')"
        ),
        {"o": org, "u": DEV},
    ).scalar()
    assert dev_visible == 0

    # Positive control: the author's own silo did gain the distilled fact.
    priya_new = db.execute(
        text(
            "SELECT count(*) FROM session_memories "
            "WHERE org_id = :o AND user_id = :u AND memory_type = 'fact' "
            "AND content = 'the auth service verifies jwts'"
        ),
        {"o": org, "u": PRIYA},
    ).scalar()
    assert priya_new == 1


def test_retracted_and_superseded_hidden_rows_are_not_fuel(db):
    """Relaxing recall_hidden must not relax status."""
    org = _org(db)
    base = datetime.now(timezone.utc)
    for i, status in enumerate(("retracted", "superseded", "retracted")):
        _seed(
            db,
            org_id=org,
            user_id=PRIYA,
            content=f"dead hidden capture {i}",
            memory_type="observation",
            recall_hidden=True,
            status=status,
            created_at=base + timedelta(seconds=i),
        )

    assert _read_new_raw_rows(db, org, PRIYA, CursorState(None, None), 50) == []

    # Positive control: an identical row that is merely `active` IS fuel.
    _seed(
        db,
        org_id=org,
        user_id=PRIYA,
        content="live hidden capture",
        memory_type="observation",
        recall_hidden=True,
        status="active",
        created_at=base + timedelta(seconds=10),
    )
    rows = _read_new_raw_rows(db, org, PRIYA, CursorState(None, None), 50)
    assert [r.content for r in rows] == ["live hidden capture"]


# ---------------------------------------------------------------------------
# Idempotency and cursor behaviour over a newly-visible hidden backlog
# ---------------------------------------------------------------------------


async def test_two_ticks_do_not_double_write_entities(db):
    org = _org(db)
    _seed_production_shape(db, org_id=org, user_id=PRIYA, hook_captures=5)

    first = await _run_tick(db, org, PRIYA, _AUTH_GRAPH)
    entities_after_first = _entities(db, org)
    edges_after_first = _about_edges(db, org)

    second = await _run_tick(db, org, PRIYA, _AUTH_GRAPH)

    assert first["processed"] == 5
    assert second["processed"] == 0, "the cursor consumed the backlog"
    assert second["reason"] == "below_threshold"
    assert entities_after_first == ["auth service"]
    assert _entities(db, org) == entities_after_first, "entities must not double-write"
    assert _about_edges(db, org) == edges_after_first == 1


async def test_the_cursor_walks_a_hidden_backlog_without_stalling_or_skipping(db):
    """A backlog larger than one batch must drain across ticks, losing nothing.

    Ten hidden captures, batch_size=4: the ticks must consume 4, 4, 2 — no row
    seen twice, none skipped, and the cursor never parked on rows it already
    processed.
    """
    org = _org(db)
    base = datetime.now(timezone.utc)
    for i in range(10):
        _seed(
            db,
            org_id=org,
            user_id=PRIYA,
            content=f"hidden capture {i:02d}",
            memory_type="observation",
            recall_hidden=True,
            created_at=base + timedelta(seconds=i),
        )

    seen: list[str] = []
    processed: list[int] = []
    cursor = CursorState(None, None)
    for _ in range(4):
        rows = _read_new_raw_rows(db, org, PRIYA, cursor, 4)
        processed.append(len(rows))
        if not rows:
            break
        seen.extend(r.content for r in rows)
        cursor = CursorState(rows[-1].created_at, rows[-1].id)

    assert processed == [4, 4, 2, 0]
    assert seen == [f"hidden capture {i:02d}" for i in range(10)]

    # And a real tick advances the stored cursor over hidden rows the same way.
    result = await _run_tick(db, org, PRIYA, _AUTH_GRAPH, batch_size=4)
    assert result["processed"] == 4
    stored = db.execute(
        text(
            "SELECT last_processed_at, last_processed_id FROM brain_curator_state "
            "WHERE org_id = :o AND user_id = :u"
        ),
        {"o": org, "u": PRIYA},
    ).fetchone()
    assert stored.last_processed_at is not None
    assert stored.last_processed_id is not None
    # The next read starts exactly after the 4th hidden row — not at row 0, and
    # not past the remaining 6.
    remaining = _read_new_raw_rows(
        db,
        org,
        PRIYA,
        CursorState(stored.last_processed_at, stored.last_processed_id),
        50,
    )
    assert [r.content for r in remaining] == [
        f"hidden capture {i:02d}" for i in range(4, 10)
    ]


async def test_ticks_drain_a_hidden_backlog_and_park_on_the_sub_threshold_tail(db):
    """Repeated ticks drain the hidden backlog, then hold — they do not loop.

    Nine hidden captures, ``batch_size=4``, ``min_inputs=3``: two ticks consume
    4 each, and the 1-row tail falls below ``min_inputs``. That tick takes the
    documented no-op path — cursor untouched, so the row is *retained* for when
    more accumulates, not silently dropped. Every subsequent tick reports the
    same single pending row rather than advancing past it or re-synthesising.
    """
    org = _org(db)
    base = datetime.now(timezone.utc)
    for i in range(9):
        _seed(
            db,
            org_id=org,
            user_id=PRIYA,
            content=f"hidden capture {i:02d}",
            memory_type="observation",
            recall_hidden=True,
            created_at=base + timedelta(seconds=i),
        )

    results = [
        await _run_tick(db, org, PRIYA, _AUTH_GRAPH, batch_size=4) for _ in range(4)
    ]

    assert [r["processed"] for r in results] == [4, 4, 1, 1]
    assert [r.get("reason") for r in results] == [
        None,
        None,
        "below_threshold",
        "below_threshold",
    ]
    # The tail is pending, not consumed and not lost.
    cursor_row = db.execute(
        text(
            "SELECT last_processed_at, last_processed_id FROM brain_curator_state "
            "WHERE org_id = :o AND user_id = :u"
        ),
        {"o": org, "u": PRIYA},
    ).fetchone()
    pending = _read_new_raw_rows(
        db,
        org,
        PRIYA,
        CursorState(cursor_row.last_processed_at, cursor_row.last_processed_id),
        50,
    )
    assert [r.content for r in pending] == ["hidden capture 08"]

    # Positive control: two more hidden rows push the tail over min_inputs and
    # the stalled remainder is finally consumed — the parking is a wait, not a
    # permanent stall.
    for i in (9, 10):
        _seed(
            db,
            org_id=org,
            user_id=PRIYA,
            content=f"hidden capture {i:02d}",
            memory_type="observation",
            recall_hidden=True,
            created_at=base + timedelta(seconds=i),
        )
    final = await _run_tick(db, org, PRIYA, _AUTH_GRAPH, batch_size=4)
    assert final["processed"] == 3
    assert "reason" not in final
