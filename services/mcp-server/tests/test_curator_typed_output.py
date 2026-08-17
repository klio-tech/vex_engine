"""The curator emits the product taxonomy, and does not eat what it emits.

Two coupled properties. The first is the point of the change: a distilled
``decision`` must land in the table AS ``decision``, because the Context
workspace groups by ``memory_type`` and a decision flattened into ``fact``
appears under Constraints and never under Decisions.

The second is what the first breaks if left alone. ``decision`` / ``plan`` /
``note`` are RAW input types — the curator's own read predicate names them. So
the moment the curator can write them, the type list alone no longer separates
input from output, and a freshly written row's ``created_at`` is by
construction past the keyset cursor. The curator would re-digest its own
deductions on the very next tick, forever, and each generation would be a
model's reading of a model's reading. ``provenance <> 'INFERRED'`` is the
separator that holds: this job distils what was captured, never what was
concluded.

Real Postgres throughout: the guard IS the SQL predicate, and a mocked session
would pass a query with no predicate at all.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

from app.curator import CursorState, _read_new_raw_rows, curator_tick
from shared.taxonomy import ExtractedMemory

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
    reason="Postgres not reachable; the curator taxonomy tests require it",
)

EMBED_DIM = 1024
FIXED_EMBEDDING = [0.1] * EMBED_DIM
_EMBED_LITERAL = "[" + ",".join(["0.1"] * EMBED_DIM) + "]"

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
    provenance: str = "EXTRACTED",
    created_at=None,
) -> str:
    memory_id = str(uuid.uuid4())
    now = created_at or datetime.now(timezone.utc)
    db.execute(
        text(
            """
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, memory_type, content,
                embedding, scope, status, user_id, metadata,
                created_at, updated_at, recall_hidden, provenance
            ) VALUES (
                :id, :session_id, :org_id, 'agent-1', :memory_type, :content,
                :embedding, 'private', 'active', :user_id, '{}'::jsonb,
                :now, :now, TRUE, :provenance
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
            "user_id": user_id,
            "now": now,
            "provenance": provenance,
        },
    )
    db.commit()
    return memory_id


def _patches(graph):
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
        patch(
            "shared.brain.find_similar_memory",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch("app.curator.call_llm", new_callable=AsyncMock, return_value=None),
    )


async def _run_tick(db, org, user, graph):
    p = _patches(graph)
    with p[0], p[1], p[2], p[3], p[4], p[5]:
        return await curator_tick(db, org, user, min_inputs=3)


def _types(db, org, provenance):
    rows = db.execute(
        text(
            "SELECT memory_type, content FROM session_memories "
            "WHERE org_id = :o AND provenance = :p ORDER BY memory_type"
        ),
        {"o": org, "p": provenance},
    ).fetchall()
    return [(r.memory_type, r.content) for r in rows]


TYPED_GRAPH = {
    "memories": [
        ExtractedMemory("decision", "Deploys run from GitHub source, not Docker Hub."),
        ExtractedMemory("plan", "Next the proposals UI is wired to approve."),
        ExtractedMemory("fact", "The pooled endpoint refuses libpq startup params."),
        ExtractedMemory("note", "Abhishek prefers Linear's design language here."),
    ],
    "entities": [],
    "edges": [],
}


@pytest.mark.asyncio
async def test_a_distilled_decision_is_stored_as_a_decision(db):
    """The whole point: the kind survives the write path.

    Before this change every distilled item was written ``memory_type='fact'``
    regardless of what it was, which is why the reference org held 0 decisions
    and 0 plans across 5,197 rows.
    """
    org = _org(db)
    base = datetime.now(timezone.utc)
    for i in range(4):
        _seed(
            db,
            org_id=org,
            user_id=PRIYA,
            content=f"hook capture {i}: touched the deploy pipeline",
            created_at=base + timedelta(seconds=i),
        )

    await _run_tick(db, org, PRIYA, TYPED_GRAPH)

    written = dict(_types(db, org, "INFERRED"))
    assert set(written) == {"decision", "plan", "fact", "note"}


@pytest.mark.asyncio
async def test_the_curators_own_output_is_not_fuel_for_the_next_tick(db):
    """``provenance <> 'INFERRED'`` is what stops the feedback loop.

    ``decision`` / ``plan`` / ``note`` are all RAW input types, so without the
    provenance predicate the rows written by the tick above are eligible input
    the moment they exist — their ``created_at`` is always past the cursor.
    """
    org = _org(db)
    base = datetime.now(timezone.utc)
    _seed(
        db,
        org_id=org,
        user_id=PRIYA,
        content="a genuine hook capture about the deploy pipeline",
        created_at=base,
    )
    for kind in ("decision", "plan", "note", "observation"):
        _seed(
            db,
            org_id=org,
            user_id=PRIYA,
            content=f"a curator-authored {kind} nobody captured",
            memory_type=kind,
            provenance="INFERRED",
            created_at=base + timedelta(seconds=1),
        )

    rows = _read_new_raw_rows(db, org, PRIYA, CursorState(None, None), batch_size=50)

    contents = [r.content for r in rows]
    assert contents == ["a genuine hook capture about the deploy pipeline"], (
        "an INFERRED row is a model's conclusion; re-reading it is the "
        "curator digesting its own distillate"
    )


@pytest.mark.asyncio
async def test_an_extracted_raw_row_of_every_kind_is_still_fuel(db):
    """Positive control: the predicate narrows on provenance, not on kind.

    A ``decision`` an agent actually wrote through the ``decide`` tool, or a
    hook captured, is EXTRACTED and must stay eligible input.
    """
    org = _org(db)
    base = datetime.now(timezone.utc)
    for i, kind in enumerate(("observation", "note", "memory", "plan", "decision")):
        _seed(
            db,
            org_id=org,
            user_id=PRIYA,
            content=f"an agent-authored {kind} about the deploy pipeline",
            memory_type=kind,
            created_at=base + timedelta(seconds=i),
        )

    rows = _read_new_raw_rows(db, org, PRIYA, CursorState(None, None), batch_size=50)
    assert len(rows) == 5
