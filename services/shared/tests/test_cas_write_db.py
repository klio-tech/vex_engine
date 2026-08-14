"""``supersede_with_revision`` against real Postgres.

The mocked-DB tests in ``test_cas_write.py`` prove the guard's SQL shape and
the conflict/success branching in isolation. They cannot prove the one thing
the whole feature exists for: that the guard is actually race-safe when two
writers really contend for the same row. That needs a real database, real
transactions, and a real row lock — a mock's ``rowcount`` is whatever the test
tells it to be.

Two shapes, both DB-gated:

* A genuine two-thread race (``threading.Barrier``) — two independent
  connections both read the same starting ``revision``, then both call
  ``supersede_with_revision`` with it at (as close as this process can make
  it) the same instant. Postgres's row lock on the guard ``UPDATE`` is what
  decides the winner, not anything in this test.
* A sequential same-assertion check (read once, write twice with the same
  stale token) — deterministic, and kept alongside the threaded version
  rather than replacing it, because a thread-timing flake in CI must never be
  the reason this boundary goes untested for a run.
"""

import threading
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text

from shared import memory as memory_mod
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the CAS write-conflict race tests")

EMBED_DIM = 1024
AGENT = "agent-cas"
ORG_PREFIX = "org_cas_"


def _unique_org() -> str:
    return f"{ORG_PREFIX}{uuid.uuid4().hex[:8]}"


def _fixed_vector(seed: str) -> list[float]:
    # Deterministic and cheap — _embed_text_sync is monkeypatched to this in
    # every test below, so no network call and no litellm dependency here.
    base = (hash(seed) % 1000) / 1000.0
    return [base] * EMBED_DIM


@pytest.fixture(autouse=True)
def _patch_sync_embedding(monkeypatch):
    monkeypatch.setattr(
        memory_mod, "_embed_text_sync", lambda content: _fixed_vector(content)
    )


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
            for org_id in orgs:
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


def _seed(db, *, org_id: str, content: str = "the on-call rotation is Priya") -> str:
    memory_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    db.execute(
        text("""
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, execution_id,
                memory_type, content, embedding, scope, status,
                user_id, provenance, metadata,
                created_at, updated_at
            ) VALUES (
                :id, :session_id, :org_id, :agent_id, NULL,
                'fact', :content, :embedding, 'org', 'active',
                NULL, 'EXTRACTED', '{}'::jsonb,
                :created_at, :updated_at
            )
        """),
        {
            "id": memory_id,
            "session_id": f"sess-{uuid.uuid4().hex[:6]}",
            "org_id": org_id,
            "agent_id": AGENT,
            "content": content,
            "embedding": "[" + ",".join(["0.05"] * EMBED_DIM) + "]",
            "created_at": now,
            "updated_at": now,
        },
    )
    db.commit()
    return memory_id


def _read_revision(db, *, org_id: str, memory_id: str) -> str:
    row = db.execute(
        text(
            "SELECT revision FROM session_memories WHERE id = :id AND org_id = :o"
        ),
        {"id": memory_id, "o": org_id},
    ).fetchone()
    return str(row[0])


def test_sequential_stale_writer_loses(db):
    """Read once, write twice with the same base_revision: second is a conflict.

    Deterministic control for the threaded race below — no timing involved.
    """
    org_id = _org(db)
    memory_id = _seed(db, org_id=org_id)
    base_revision = _read_revision(db, org_id=org_id, memory_id=memory_id)

    first = memory_mod.supersede_with_revision(
        db,
        org_id=org_id,
        memory_id=memory_id,
        base_revision=base_revision,
        new_content="Priya handed the rotation to Sam",
        agent_id=AGENT,
        memory_type="fact",
        session_id="sess-cas-1",
    )
    assert first["conflict"] is False
    assert first["new_id"] is not None

    second = memory_mod.supersede_with_revision(
        db,
        org_id=org_id,
        memory_id=memory_id,
        base_revision=base_revision,  # same, now-stale token
        new_content="Priya is still on-call",
        agent_id=AGENT,
        memory_type="fact",
        session_id="sess-cas-2",
    )
    assert second["conflict"] is True
    assert second["current"]["id"] == memory_id
    assert second["current"]["revision"] != base_revision

    # The old row was actually superseded — supersede_memory's semantics are
    # untouched by the CAS wrapper.
    status_row = db.execute(
        text("SELECT status, superseded_by FROM session_memories WHERE id = :id"),
        {"id": memory_id},
    ).fetchone()
    assert status_row.status == "superseded"
    assert str(status_row.superseded_by) == first["new_id"]


def test_two_concurrent_writers_exactly_one_wins(db):
    """Two independent connections race the SAME stale base_revision.

    Postgres's row lock on the guard UPDATE serialises the two: whichever
    commits first wins outright (conflict=False); the other's predicate no
    longer matches once it re-evaluates against the now-changed revision, so
    it gets conflict=True. Never both winners, never both losers.
    """
    org_id = _org(db)
    memory_id = _seed(db, org_id=org_id)
    base_revision = _read_revision(db, org_id=org_id, memory_id=memory_id)

    barrier = threading.Barrier(2)
    results: dict[str, dict] = {}
    errors: list[BaseException] = []

    def _attempt(tag: str, content: str) -> None:
        engine = create_engine(DB_URL)
        conn = engine.connect()
        try:
            barrier.wait(timeout=5)
            results[tag] = memory_mod.supersede_with_revision(
                conn,
                org_id=org_id,
                memory_id=memory_id,
                base_revision=base_revision,
                new_content=content,
                agent_id=AGENT,
                memory_type="fact",
                session_id=f"sess-cas-race-{tag}",
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced via `errors`
            errors.append(exc)
        finally:
            conn.close()
            engine.dispose()

    t1 = threading.Thread(target=_attempt, args=("a", "writer A's correction"))
    t2 = threading.Thread(target=_attempt, args=("b", "writer B's correction"))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"race threads raised: {errors}"
    assert set(results) == {"a", "b"}

    conflicts = [r["conflict"] for r in results.values()]
    assert conflicts.count(False) == 1, (
        f"expected exactly one winner, got {results}"
    )
    assert conflicts.count(True) == 1, (
        f"expected exactly one conflict, got {results}"
    )

    winner = next(r for r in results.values() if r["conflict"] is False)
    loser = next(r for r in results.values() if r["conflict"] is True)
    assert loser["current"]["id"] == memory_id
    assert loser["current"]["revision"] != base_revision

    status_row = db.execute(
        text("SELECT status, superseded_by FROM session_memories WHERE id = :id"),
        {"id": memory_id},
    ).fetchone()
    assert status_row.status == "superseded"
    assert str(status_row.superseded_by) == winner["new_id"]
