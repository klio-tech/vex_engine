"""The cursor must never move past a row the tick did not distil. Real Postgres.

**Why this file and the rollback fix are one change.**

``_store_one`` swallows every exception from ``write_scoped_memory``, so a tick
whose writes all fail still reaches the end of ``curator_tick`` and calls
``_advance_cursor``. Until now that call *raised*: the swallowed failure was a
SQL error, Postgres had aborted the transaction, and nothing had rolled it
back — so the cursor upsert could not execute. The tick died, the cursor stayed
where it was, and the next tick reconsidered the same rows. A self-limiting
retry loop, held together by a bug.

Add the missing rollback on its own and that stops being true. The session is
healthy, ``_advance_cursor`` succeeds, and the cursor moves past rows nothing
was ever synthesised from. Those rows are not merely skipped: the curator never
reads behind its cursor, and ``scripts/run_retention.py`` deletes RAW
observations on a 7-day TTL. Abandoned means gone.

So the rollback fix converts a loud, self-correcting failure into silent
permanent data loss unless the cursor fix lands with it. That is the whole
reason shapes 2 and 3 are one work package.

**What this test tolerates, and why.**

``_run_tick_tolerating_failure`` lets ``curator_tick`` raise. That is not
laxness — it is the invariant stated precisely. Under a failed write the tick
may abort loudly or it may complete quietly; what it must never do, in either
case, is leave the cursor past rows it did not distil. Asserting only "does not
raise" would test the rollback; asserting only the cursor while requiring a
clean return would make the test unable to run against the rollback-reverted
tree. The invariant is about the cursor, so the assertion is about the cursor.

Real Postgres, because the coupling *is* the transaction state: a mocked
session has none, so the failed write would never poison anything and the
before/after would look identical.
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

from app.curator import curator_tick
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the curator cursor-abandonment tests")

EMBED_DIM = 1024
_EMBED_LITERAL = "[" + ",".join(["0.1"] * EMBED_DIM) + "]"

PRIYA = "11111111-2222-3333-4444-555555555555"

_BASE = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


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
                    "projects",
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


def _seed(db, *, org_id: str, content: str, created_at, project_id=None) -> str:
    memory_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, memory_type, content,
                embedding, scope, status, user_id, project_id, metadata,
                created_at, updated_at, recall_hidden
            ) VALUES (
                :id, :session_id, :org_id, 'agent-1', 'observation', :content,
                :embedding, 'private', 'active', :user_id, :project_id,
                '{}'::jsonb, :now, :now, TRUE
            )
            """
        ),
        {
            "id": memory_id,
            "session_id": f"sess-{uuid.uuid4().hex[:6]}",
            "org_id": org_id,
            "content": content,
            "embedding": _EMBED_LITERAL,
            "user_id": PRIYA,
            "project_id": project_id,
            "now": created_at,
        },
    )
    db.commit()
    return memory_id


def _cursor(db, org_id: str):
    """The persisted ``(last_processed_at, last_processed_id)``, or ``None``."""
    row = db.execute(
        text(
            "SELECT last_processed_at, last_processed_id FROM brain_curator_state "
            "WHERE org_id = :o AND user_id = :u"
        ),
        {"o": org_id, "u": PRIYA},
    ).fetchone()
    if row is None:
        return None
    return row.last_processed_at, row.last_processed_id


def _patched(*, summary="a summary", facts=("f1",), write=None):
    """Patch the three LLM/write seams the tick reaches through."""
    return (
        patch("app.curator.generate_summary", AsyncMock(return_value=summary)),
        patch(
            "app.curator.extract_graph",
            AsyncMock(return_value={"facts": list(facts), "entities": [], "edges": []}),
        ),
        patch("app.curator.write_scoped_memory", write),
        # No live embedder in this suite: reconciliation is not what is under
        # test, and a real embed call would make the tick depend on a network.
        patch("app.curator.embed_text", AsyncMock(return_value=None)),
    )


async def _run_tick_tolerating_failure(db, org_id, **kwargs):
    """Run the tick and swallow its exception. See the module docstring.

    Returns the exception if one escaped, else ``None`` — so a test can still
    say which of the two acceptable outcomes it saw.
    """
    try:
        await curator_tick(db, org_id, PRIYA, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the point of the helper
        return exc
    return None


# ---------------------------------------------------------------------------
# The positive control comes first. Every assertion below is that the cursor
# did NOT move; a tick that never advanced the cursor under any circumstance
# would pass all of them.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_successful_tick_does_advance_the_cursor(db):
    org_id = _org(db)
    ids = [
        _seed(
            db, org_id=org_id, content=f"c{i}", created_at=_BASE + timedelta(minutes=i)
        )
        for i in range(5)
    ]
    write = AsyncMock(return_value={"stored": True, "id": str(uuid.uuid4())})
    gen, ext, wsm, emb = _patched(write=write)

    with gen, ext, wsm, emb:
        result = await curator_tick(db, org_id, PRIYA, min_inputs=3)

    assert result["synthesized"] > 0
    assert _cursor(db, org_id) == (_BASE + timedelta(minutes=4), uuid.UUID(ids[4])), (
        "control: when the tick DOES distil the batch, the cursor must advance "
        "to the batch's composite max — otherwise every test below is vacuous"
    )


# ---------------------------------------------------------------------------
# Abandonment: every write fails with a real SQL error.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_cursor_is_not_advanced_when_every_write_failed(db):
    org_id = _org(db)
    for i in range(5):
        _seed(
            db, org_id=org_id, content=f"c{i}", created_at=_BASE + timedelta(minutes=i)
        )

    async def failing_write(db_session, **kwargs):
        # A REAL failure, not a raise: this is what a constraint violation or a
        # dropped column does — the statement fails and Postgres aborts the
        # transaction. Anything less and there is nothing for the rollback to
        # clear, and the coupling this file exists to pin disappears.
        db_session.execute(text("SELECT no_such_column_at_all"))
        raise AssertionError("unreachable: the statement above raises")

    gen, ext, wsm, emb = _patched(write=AsyncMock(side_effect=failing_write))

    with gen, ext, wsm, emb:
        await _run_tick_tolerating_failure(db, org_id, min_inputs=3)

    db.rollback()
    assert _cursor(db, org_id) in (None, (None, None)), (
        "the tick synthesised nothing and advanced the cursor anyway. The "
        "curator never reads behind its cursor and run_retention.py deletes "
        "RAW observations after 7 days, so those five rows are now "
        "unrecoverable — this is the failure the missing rollback used to mask"
    )


# ---------------------------------------------------------------------------
# Per-partition abandonment: the batch qualifies, one partition does not.
#
# This needs no write failure at all. It is the ordinary path, and the rows it
# abandons are abandoned on a healthy session with everything working.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_cursor_stops_before_a_skipped_partitions_rows(db):
    org_id = _org(db)
    # Ordered by created_at: three for project A (distilled), then two for
    # project B (below the per-partition threshold, skipped).
    a_ids = [
        _seed(
            db,
            org_id=org_id,
            content=f"a{i}",
            created_at=_BASE + timedelta(minutes=i),
            project_id=None,
        )
        for i in range(3)
    ]
    project_id = _make_project(db, org_id)
    for i in range(3, 5):
        _seed(
            db,
            org_id=org_id,
            content=f"b{i}",
            created_at=_BASE + timedelta(minutes=i),
            project_id=project_id,
        )

    write = AsyncMock(return_value={"stored": True, "id": str(uuid.uuid4())})
    gen, ext, wsm, emb = _patched(write=write)

    with gen, ext, wsm, emb:
        result = await curator_tick(db, org_id, PRIYA, min_inputs=3)

    assert result["synthesized"] > 0, "project A's partition did qualify"
    assert _cursor(db, org_id) == (_BASE + timedelta(minutes=2), uuid.UUID(a_ids[2])), (
        "the cursor advanced past project B's two rows, which nothing was ever "
        "synthesised from. They can never be read again, and the 7-day TTL "
        "deletes them"
    )


@pytest.mark.asyncio
async def test_the_held_rows_are_still_eligible_on_the_next_tick(db):
    """The point of holding the cursor: the rows come back when they qualify.

    Positive control for the test above — it asserts the cursor stayed put;
    this asserts that staying put is worth something.
    """
    org_id = _org(db)
    project_id = _make_project(db, org_id)
    for i in range(3):
        _seed(
            db, org_id=org_id, content=f"a{i}", created_at=_BASE + timedelta(minutes=i)
        )
    for i in range(3, 5):
        _seed(
            db,
            org_id=org_id,
            content=f"b{i}",
            created_at=_BASE + timedelta(minutes=i),
            project_id=project_id,
        )

    write = AsyncMock(return_value={"stored": True, "id": str(uuid.uuid4())})
    gen, ext, wsm, emb = _patched(write=write)
    with gen, ext, wsm, emb:
        await curator_tick(db, org_id, PRIYA, min_inputs=3)

    # A third row arrives for project B, taking it over the threshold.
    _seed(
        db,
        org_id=org_id,
        content="b5",
        created_at=_BASE + timedelta(minutes=5),
        project_id=project_id,
    )

    write2 = AsyncMock(return_value={"stored": True, "id": str(uuid.uuid4())})
    gen, ext, wsm, emb = _patched(write=write2)
    with gen, ext, wsm, emb:
        result = await curator_tick(db, org_id, PRIYA, min_inputs=3)

    assert result["synthesized"] > 0, (
        "project B's three rows must be re-read and distilled on this tick; if "
        "the first tick had advanced the cursor over them only one would remain"
    )
    input_counts = {
        str(c.kwargs["project_id"]): c.kwargs["metadata"]["input_count"]
        for c in write2.await_args_list
    }
    assert input_counts.get(project_id) == 3, (
        "project B must be distilled from all THREE of its rows. A count of 1 "
        f"means the first tick advanced over b3 and b4 (got {input_counts})"
    )


def _make_project(db, org_id: str) -> str:
    project_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO projects (
                id, org_id, display_name, git_remote, created_at, last_seen_at
            )
            VALUES (:id, :o, 'fixture', :remote, now(), now())
            """
        ),
        {"id": project_id, "o": org_id, "remote": f"https://example.test/{project_id}"},
    )
    db.commit()
    return project_id
