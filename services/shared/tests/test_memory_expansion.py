"""Recall stage B: one-hop graph expansion inside the visibility ladder.

Real Postgres, ``embed_text`` mocked to one fixed vector (the silo-test
convention): what is under test is WHICH rows the combined vector+graph read
admits, never their cosine order. Every negative here carries a positive
control — an expansion that adds nothing to anybody passes every negative.

The security property (spec invariant: "expansion never widens visibility"):
a row connected to an entry point by an edge, but forbidden by the caller's
ladder, is absent at EVERY scope argument. The mechanism (the ladder inside
``graph_store.neighbors``) is attacked directly in test_graph_isolation.py;
this file proves ``retrieve_memories_scoped`` composes with it rather than
around it.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

from shared import graph_store
from shared.memory import GRAPH_EXPANSION_CAP, retrieve_memories_scoped

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
    reason="Postgres not reachable; the expansion tests require it",
)

EMBED_DIM = 1024
FIXED_EMBEDDING = [0.1] * EMBED_DIM
FIXED_EMBEDDING_LITERAL = "[" + ",".join(str(x) for x in FIXED_EMBEDDING) + "]"

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
    age_days: float = 0.0,
) -> str:
    memory_id = str(uuid.uuid4())
    created = datetime.now(timezone.utc) - timedelta(days=age_days)
    db.execute(
        text(
            """
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, memory_type, content,
                embedding, scope, status, user_id, project_id, metadata,
                created_at, updated_at
            ) VALUES (
                :id, :session_id, :org_id, 'agent-1', 'fact', :content,
                :embedding, :scope, 'active', :user_id, :project_id,
                '{}'::jsonb, :created, :created
            )
            """
        ),
        {
            "id": memory_id,
            "session_id": f"sess-{uuid.uuid4().hex[:6]}",
            "org_id": org_id,
            "content": content,
            "embedding": FIXED_EMBEDDING_LITERAL,
            "scope": scope,
            "user_id": user_id,
            "project_id": project_id,
            "created": created,
        },
    )
    db.commit()
    return memory_id


def _edge(db, *, org_id, src_id, dst_id, relation="relates_to"):
    graph_store.add_edge(
        db,
        org_id=org_id,
        src_id=src_id,
        dst_kind="memory",
        dst_id=dst_id,
        relation=relation,
    )


def _mock_embed():
    return patch(
        "shared.memory.embed_text",
        new_callable=AsyncMock,
        return_value=FIXED_EMBEDDING,
    )


async def _recall_ids(
    db, org, *, user_id=None, scope="org", agent_id=None, limit=1, project_id=None
):
    with _mock_embed():
        records = await retrieve_memories_scoped(
            db,
            org,
            "query",
            scope=scope,
            agent_id=agent_id,
            user_id=user_id,
            project_id=project_id,
            limit=limit,
        )
    return records


# ---------------------------------------------------------------------------
# The feature: connected memories surface beyond the vector stage
# ---------------------------------------------------------------------------


async def test_expansion_surfaces_a_connected_row_beyond_the_limit(db):
    """limit=1 admits only the newest row on the vector stage; the connected
    older row rides in on the edge — the 'why' behind the decision."""
    org = _org(db)
    entry = _seed(db, org_id=org, content="we chose flat pricing", age_days=0)
    connected = _seed(
        db, org_id=org, content="usage pricing confused customers", age_days=20
    )
    _edge(db, org_id=org, src_id=entry, dst_id=connected)
    # An unconnected old row does NOT ride in.
    bystander = _seed(db, org_id=org, content="unrelated old note", age_days=20)

    records = await _recall_ids(db, org, user_id=DEV, limit=1)
    ids = {str(r.id) for r in records}
    assert entry in ids
    assert connected in ids, "the edge should have pulled the connected row in"
    assert bystander not in ids

    # Entry rows carry their similarity; expanded rows carry connections
    # instead of a fake similarity score.
    by_id = {str(r.id): r for r in records}
    assert by_id[entry].relevance is not None
    assert by_id[connected].relevance is None


async def test_the_entry_row_still_ranks_first(db):
    """0.70·sim keeps a genuinely similar row ahead of a merely-connected one."""
    org = _org(db)
    entry = _seed(db, org_id=org, content="entry", age_days=0)
    connected = _seed(db, org_id=org, content="context", age_days=20)
    _edge(db, org_id=org, src_id=entry, dst_id=connected)

    records = await _recall_ids(db, org, user_id=DEV, limit=1)
    assert str(records[0].id) == entry


async def test_expansion_is_capped_at_five_rows(db):
    org = _org(db)
    entry = _seed(db, org_id=org, content="hub", age_days=0)
    for i in range(GRAPH_EXPANSION_CAP + 3):
        other = _seed(db, org_id=org, content=f"spoke {i}", age_days=10 + i)
        _edge(db, org_id=org, src_id=entry, dst_id=other)

    records = await _recall_ids(db, org, user_id=DEV, limit=1)
    assert len(records) <= 1 + GRAPH_EXPANSION_CAP


# ---------------------------------------------------------------------------
# The invariant: expansion never widens visibility
# ---------------------------------------------------------------------------


async def test_connected_but_forbidden_row_is_absent_at_every_scope(db):
    org = _org(db)
    entry = _seed(db, org_id=org, content="entry", age_days=0)
    secret = _seed(
        db,
        org_id=org,
        content="priya's connected secret",
        scope="private",
        user_id=PRIYA,
        age_days=5,
    )
    _edge(db, org_id=org, src_id=entry, dst_id=secret)

    # Dev, at every scope argument the API accepts.
    for scope, agent in (("org", None), ("agent", "priya-laptop")):
        records = await _recall_ids(
            db, org, user_id=DEV, scope=scope, agent_id=agent, limit=10
        )
        assert secret not in {str(r.id) for r in records}, f"scope={scope} leaked"
    private_records = await _recall_ids(db, org, user_id=DEV, scope="private", limit=10)
    assert secret not in {str(r.id) for r in private_records}
    # An anonymous caller.
    anon = await _recall_ids(db, org, user_id=None, limit=10)
    assert secret not in {str(r.id) for r in anon}

    # Control: the same walk delivers the row to its owner (limit=1 keeps the
    # vector stage to the entry row, so it can only have arrived via the edge).
    priya = await _recall_ids(db, org, user_id=PRIYA, limit=1)
    assert secret in {str(r.id) for r in priya}


async def test_project_filtered_recall_is_not_widened_by_expansion(db):
    """With a project filter, rows filed under OTHER projects never surface —
    the expansion obeys the same rule as the vector stage."""
    org = _org(db)
    project_a, project_b = str(uuid.uuid4()), str(uuid.uuid4())
    for pid, name in ((project_a, "alpha"), (project_b, "beta")):
        db.execute(
            text(
                "INSERT INTO projects (id, org_id, git_remote, display_name) "
                "VALUES (:id, :org, :remote, :name)"
            ),
            {
                "id": pid,
                "org": org,
                "remote": f"git@example.com:{uuid.uuid4().hex[:8]}.git",
                "name": name,
            },
        )
    db.commit()
    entry = _seed(db, org_id=org, content="entry", project_id=project_a, age_days=0)
    other_project_row = _seed(
        db, org_id=org, content="filed elsewhere", project_id=project_b, age_days=5
    )
    unfiled = _seed(db, org_id=org, content="unfiled context", age_days=5)
    _edge(db, org_id=org, src_id=entry, dst_id=other_project_row)
    _edge(db, org_id=org, src_id=entry, dst_id=unfiled)

    records = await _recall_ids(db, org, user_id=DEV, limit=1, project_id=project_a)
    ids = {str(r.id) for r in records}
    assert other_project_row not in ids
    assert unfiled in ids  # the control: expansion itself is alive


# ---------------------------------------------------------------------------
# Failure containment
# ---------------------------------------------------------------------------


async def test_expansion_failure_never_breaks_recall(db, monkeypatch):
    org = _org(db)
    entry = _seed(db, org_id=org, content="entry")

    def _boom(*args, **kwargs):
        raise RuntimeError("graph engine down")

    monkeypatch.setattr(graph_store, "neighbors", _boom)
    records = await _recall_ids(db, org, user_id=DEV, limit=10)
    assert {str(r.id) for r in records} == {entry}


# ---------------------------------------------------------------------------
# M2 — why a row is here must survive the wire, not just the process
# ---------------------------------------------------------------------------


async def test_a_graph_expanded_row_is_labelled_as_such(db):
    """``relevance=None`` was the ONLY thing marking an expanded row.

    Structural compression drops null keys per row, so on the wire the
    expanded row simply had no ``relevance`` key and NOTHING distinguished a
    semantic match from a graph walk. A caller weighing "the model found this
    similar to my question" against "this was one hop away from something
    similar" could not tell them apart. The marker has to be positive.
    """
    org = _org(db)
    entry = _seed(db, org_id=org, content="we chose flat pricing", age_days=0)
    connected = _seed(
        db, org_id=org, content="usage pricing confused customers", age_days=20
    )
    _edge(db, org_id=org, src_id=entry, dst_id=connected)

    records = await _recall_ids(db, org, user_id=DEV, limit=1)
    by_id = {str(r.id): r for r in records}
    assert set(by_id) == {entry, connected}, "control: the expansion happened"

    assert by_id[connected].via == "graph"
    assert by_id[connected].relevance is None
    # The vector row is NOT relabelled — it earned its similarity score.
    assert by_id[entry].via != "graph"
    assert by_id[entry].relevance is not None


async def test_the_graph_marker_survives_structural_compression(db):
    """The marker is only worth having if it reaches the caller. Compression
    strips null-valued keys row by row, which is exactly how the old marker
    was erased; a non-null value is never stripped."""
    from shared.compression.structural import compress_payload

    org = _org(db)
    entry = _seed(db, org_id=org, content="we chose flat pricing", age_days=0)
    connected = _seed(
        db, org_id=org, content="usage pricing confused customers", age_days=20
    )
    _edge(db, org_id=org, src_id=entry, dst_id=connected)

    records = await _recall_ids(db, org, user_id=DEV, limit=1)
    payload = {
        "memories": [
            {
                "id": str(r.id),
                "content": r.content,
                "relevance": r.relevance,
                "via": r.via,
            }
            for r in records
        ]
    }
    compressed, _stats = compress_payload(payload)

    rows = {row["id"]: row for row in compressed["memories"]}
    hoisted = compressed.get("common", {})
    assert rows[connected].get("via", hoisted.get("via")) == "graph", (
        "the graph marker did not survive compression"
    )
    # And the semantic row is still distinguishable after the strip.
    assert rows[entry].get("via", hoisted.get("via")) != "graph"


# ---------------------------------------------------------------------------
# M4 — a failed expansion must not poison the session it borrowed
# ---------------------------------------------------------------------------


async def test_a_real_database_error_in_expansion_leaves_the_session_usable(db):
    """The existing containment test raises a pure ``RuntimeError``, which
    never touches the DB, so it cannot see this: a genuine SQL error aborts
    the transaction and leaves psycopg2 in ``InFailedSqlTransaction``. Recall
    survives (the handler catches), but EVERY later statement on that session
    fails — including the recall audit/billing event, whose own handler
    swallows the failure. Result: zero audit rows, silently.

    Assert on the session, not on the log: after a failed expansion the
    caller must be able to write.
    """
    org = _org(db)
    entry = _seed(db, org_id=org, content="entry")

    def _bad_sql(db_session, **kwargs):
        db_session.execute(text("SELECT * FROM a_table_that_does_not_exist"))
        raise AssertionError("unreachable: the statement above must raise")

    with patch.object(graph_store, "neighbors", _bad_sql):
        records = await _recall_ids(db, org, user_id=DEV, limit=10)
    assert {str(r.id) for r in records} == {entry}, "recall still returns its rows"

    # THE assertion: the session is not left in a failed transaction, so the
    # audit write that follows a recall actually lands.
    probe = db.execute(text("SELECT 1")).scalar()
    assert probe == 1, "expansion left the session in a failed transaction"
