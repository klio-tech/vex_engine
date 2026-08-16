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

from shared import memory as memory_mod, projects
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the CAS write-conflict race tests")

EMBED_DIM = 1024
AGENT = "agent-cas"
ORG_PREFIX = "org_cas_"
PRIYA = "11111111-2222-3333-4444-555555555555"
DEV = "99999999-8888-7777-6666-555555555555"


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
        memory_mod,
        "_embed_text_sync",
        lambda content, **_kwargs: _fixed_vector(content),
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
                    text(
                        """
                        DELETE FROM project_members WHERE project_id IN (
                            SELECT id FROM projects WHERE org_id = :o)
                        """
                    ),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM project_grant_audit WHERE org_id = :o"),
                    {"o": org_id},
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
    content: str = "the on-call rotation is Priya",
    scope: str = "org",
    user_id: str | None = None,
    agent_id: str = AGENT,
    project_id: str | None = None,
) -> str:
    memory_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    db.execute(
        text("""
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, execution_id,
                memory_type, content, embedding, scope, status,
                user_id, provenance, project_id, metadata,
                created_at, updated_at
            ) VALUES (
                :id, :session_id, :org_id, :agent_id, NULL,
                'fact', :content, :embedding, :scope, 'active',
                :user_id, 'EXTRACTED', :project_id, '{}'::jsonb,
                :created_at, :updated_at
            )
        """),
        {
            "id": memory_id,
            "session_id": f"sess-{uuid.uuid4().hex[:6]}",
            "org_id": org_id,
            "agent_id": agent_id,
            "content": content,
            "embedding": "[" + ",".join(["0.05"] * EMBED_DIM) + "]",
            "scope": scope,
            "user_id": user_id,
            "project_id": project_id,
            "created_at": now,
            "updated_at": now,
        },
    )
    db.commit()
    return memory_id


def _project(db, *, org_id: str, name: str = "apollo") -> str:
    project_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO projects (id, org_id, git_remote, display_name)
            VALUES (:id, :org_id, :remote, :name)
            """
        ),
        {
            "id": project_id,
            "org_id": org_id,
            "remote": f"git@example.com:{uuid.uuid4().hex[:8]}.git",
            "name": name,
        },
    )
    db.commit()
    return project_id


def _status(db, memory_id: str) -> str | None:
    row = db.execute(
        text("SELECT status FROM session_memories WHERE id = :id"),
        {"id": memory_id},
    ).fetchone()
    return None if row is None else row.status


def _read_revision(db, *, org_id: str, memory_id: str) -> str:
    row = db.execute(
        text("SELECT revision FROM session_memories WHERE id = :id AND org_id = :o"),
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
    # H2: `current` follows the supersede chain to the LIVE belief — the
    # winner's new row, not the now-dead memory_id the caller named.
    assert second["current"]["id"] == first["new_id"]
    assert second["current"]["revision"] == first["new_revision"]
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
    assert conflicts.count(False) == 1, f"expected exactly one winner, got {results}"
    assert conflicts.count(True) == 1, f"expected exactly one conflict, got {results}"

    winner = next(r for r in results.values() if r["conflict"] is False)
    loser = next(r for r in results.values() if r["conflict"] is True)
    # H2: `current` follows the chain to the winner's new row, not the dead
    # memory_id both callers started from.
    assert loser["current"]["id"] == winner["new_id"]
    assert loser["current"]["revision"] == winner["new_revision"]
    assert loser["current"]["revision"] != base_revision

    status_row = db.execute(
        text("SELECT status, superseded_by FROM session_memories WHERE id = :id"),
        {"id": memory_id},
    ).fetchone()
    assert status_row.status == "superseded"
    assert str(status_row.superseded_by) == winner["new_id"]


# ---------------------------------------------------------------------------
# H1: the authorization ladder. Same shape as test_forget_isolation.py's
# attack suite for retract_memory — both now share _authorize_scoped_action,
# so this is also a regression guard against the two ever drifting apart.
# Every negative is paired with a positive control performing the identical
# supersede, so a ladder that refused everyone would not pass this file by
# accident.
# ---------------------------------------------------------------------------


def test_private_row_owner_can_supersede_it(db):
    org_id = _org(db)
    memory_id = _seed(db, org_id=org_id, scope="private", user_id=PRIYA)
    base_revision = _read_revision(db, org_id=org_id, memory_id=memory_id)

    result = memory_mod.supersede_with_revision(
        db,
        org_id=org_id,
        memory_id=memory_id,
        base_revision=base_revision,
        new_content="updated private note",
        user_id=PRIYA,
        agent_id=AGENT,
        memory_type="fact",
        session_id="sess-priv-owner",
        scope="private",
    )

    assert result["conflict"] is False
    assert result["stored"] is True


def test_private_row_non_owner_gets_not_found_and_leaks_nothing(db):
    """H1: this is the exact attack the review flagged — CAS-probing a
    private row for its content+revision, then using that to destroy it.
    Both halves must fail: no leak, no write."""
    org_id = _org(db)
    memory_id = _seed(
        db,
        org_id=org_id,
        scope="private",
        user_id=PRIYA,
        content="Priya's salary negotiation notes",
    )
    base_revision = _read_revision(db, org_id=org_id, memory_id=memory_id)

    result = memory_mod.supersede_with_revision(
        db,
        org_id=org_id,
        memory_id=memory_id,
        base_revision=base_revision,
        new_content="attacker's overwrite",
        user_id=DEV,
        agent_id=AGENT,
        memory_type="fact",
        session_id="sess-priv-attack",
        scope="private",
    )

    assert result == {"conflict": False, "stored": False, "error": "not_found"}
    # Never destroyed, and no `current` payload carrying its content ever
    # left the function — the refusal is a bare error code, nothing else.
    assert set(result) == {"conflict", "stored", "error"}
    assert _status(db, memory_id) == "active"
    assert _read_revision(db, org_id=org_id, memory_id=memory_id) == base_revision


def test_an_unattributed_caller_cannot_supersede_a_private_row(db):
    """user_id=None is not a member of anything, and not a wildcard."""
    org_id = _org(db)
    memory_id = _seed(db, org_id=org_id, scope="private", user_id=PRIYA)
    base_revision = _read_revision(db, org_id=org_id, memory_id=memory_id)

    result = memory_mod.supersede_with_revision(
        db,
        org_id=org_id,
        memory_id=memory_id,
        base_revision=base_revision,
        new_content="attacker's overwrite",
        user_id=None,
        agent_id=AGENT,
        memory_type="fact",
        session_id="sess-priv-anon",
        scope="private",
    )

    assert result == {"conflict": False, "stored": False, "error": "not_found"}
    assert _status(db, memory_id) == "active"


def test_the_private_refusal_is_identical_to_an_unknown_id(db):
    """No existence oracle: a real private row a non-owner cannot see and an
    id that was never issued must be the SAME answer."""
    org_id = _org(db)
    memory_id = _seed(db, org_id=org_id, scope="private", user_id=PRIYA)
    base_revision = _read_revision(db, org_id=org_id, memory_id=memory_id)

    refused = memory_mod.supersede_with_revision(
        db,
        org_id=org_id,
        memory_id=memory_id,
        base_revision=base_revision,
        new_content="x",
        user_id=DEV,
        agent_id=AGENT,
        memory_type="fact",
        session_id="s1",
    )
    unknown = memory_mod.supersede_with_revision(
        db,
        org_id=org_id,
        memory_id=str(uuid.uuid4()),
        base_revision=base_revision,
        new_content="x",
        user_id=DEV,
        agent_id=AGENT,
        memory_type="fact",
        session_id="s2",
    )

    assert (
        refused
        == unknown
        == {
            "conflict": False,
            "stored": False,
            "error": "not_found",
        }
    )


def test_agent_scope_owning_agent_can_supersede(db):
    org_id = _org(db)
    memory_id = _seed(db, org_id=org_id, scope="agent", agent_id=AGENT)
    base_revision = _read_revision(db, org_id=org_id, memory_id=memory_id)

    result = memory_mod.supersede_with_revision(
        db,
        org_id=org_id,
        memory_id=memory_id,
        base_revision=base_revision,
        new_content="updated agent note",
        agent_id=AGENT,
        memory_type="fact",
        session_id="sess-agent-owner",
        scope="agent",
    )

    assert result["conflict"] is False
    assert result["stored"] is True


def test_agent_scope_non_owning_agent_gets_forbidden(db):
    """Unlike the private/project arms, this answers forbidden — an agent
    row's existence was never the secret, only the caller's standing."""
    org_id = _org(db)
    memory_id = _seed(db, org_id=org_id, scope="agent", agent_id=AGENT)
    base_revision = _read_revision(db, org_id=org_id, memory_id=memory_id)

    result = memory_mod.supersede_with_revision(
        db,
        org_id=org_id,
        memory_id=memory_id,
        base_revision=base_revision,
        new_content="attacker's overwrite",
        agent_id="other-agent",
        memory_type="fact",
        session_id="sess-agent-attack",
        scope="agent",
    )

    assert result == {"conflict": False, "stored": False, "error": "forbidden"}
    assert _status(db, memory_id) == "active"


def test_project_scope_member_can_supersede(db):
    org_id = _org(db)
    project_id = _project(db, org_id=org_id)
    assert projects.add_member(
        db, org_id=org_id, project_id=project_id, user_id=PRIYA, granted_by=None
    )
    db.commit()
    memory_id = _seed(db, org_id=org_id, scope="project", project_id=project_id)
    base_revision = _read_revision(db, org_id=org_id, memory_id=memory_id)

    result = memory_mod.supersede_with_revision(
        db,
        org_id=org_id,
        memory_id=memory_id,
        base_revision=base_revision,
        new_content="updated project note",
        user_id=PRIYA,
        agent_id=AGENT,
        memory_type="fact",
        session_id="sess-proj-member",
        scope="project",
        project_id=project_id,
    )

    assert result["conflict"] is False
    assert result["stored"] is True


def test_project_scope_non_member_gets_not_found(db):
    org_id = _org(db)
    project_id = _project(db, org_id=org_id)
    assert projects.add_member(
        db, org_id=org_id, project_id=project_id, user_id=PRIYA, granted_by=None
    )
    db.commit()
    memory_id = _seed(db, org_id=org_id, scope="project", project_id=project_id)
    base_revision = _read_revision(db, org_id=org_id, memory_id=memory_id)

    result = memory_mod.supersede_with_revision(
        db,
        org_id=org_id,
        memory_id=memory_id,
        base_revision=base_revision,
        new_content="attacker's overwrite",
        user_id=DEV,
        agent_id=AGENT,
        memory_type="fact",
        session_id="sess-proj-attack",
        scope="project",
        project_id=project_id,
    )

    assert result == {"conflict": False, "stored": False, "error": "not_found"}
    assert _status(db, memory_id) == "active"


# ---------------------------------------------------------------------------
# H2: a superseded row's own (frozen) revision must not be a live CAS token.
# ---------------------------------------------------------------------------


def test_a_superseded_rows_frozen_revision_cannot_be_reused_as_a_cas_token(db):
    """The guard's revision-only predicate would have MATCHED here: a
    superseded row's own ``revision`` never changes again after the flip, so
    a caller who reads it back post-supersession holds an exact-match token
    for a dead row. ``status = 'active' AND superseded_by IS NULL`` is what
    refuses it — without those two predicates this would silently succeed at
    re-superseding an already-dead row."""
    org_id = _org(db)
    memory_id = _seed(db, org_id=org_id)
    base_revision = _read_revision(db, org_id=org_id, memory_id=memory_id)

    first = memory_mod.supersede_with_revision(
        db,
        org_id=org_id,
        memory_id=memory_id,
        base_revision=base_revision,
        new_content="first correction",
        agent_id=AGENT,
        memory_type="fact",
        session_id="sess-h2-1",
    )
    assert first["conflict"] is False

    # The dead row's revision, read AFTER it was superseded — frozen at
    # whatever the guard bumped it to during `first`.
    frozen_revision = _read_revision(db, org_id=org_id, memory_id=memory_id)
    assert frozen_revision != base_revision  # sanity: it DID change once

    replay = memory_mod.supersede_with_revision(
        db,
        org_id=org_id,
        memory_id=memory_id,
        base_revision=frozen_revision,
        new_content="replay attempt",
        agent_id=AGENT,
        memory_type="fact",
        session_id="sess-h2-2",
    )

    assert replay["conflict"] is True
    assert replay["current"]["id"] == first["new_id"]
    assert replay["current"]["revision"] == first["new_revision"]
    # The dead row was never touched a second time.
    assert _status(db, memory_id) == "superseded"


# ---------------------------------------------------------------------------
# N1: the chain-follow authorizes the TERMINAL row, not just the one named
# by memory_id. The chain can cross scope — the caller picks the new row's
# scope on every CAS write, and nothing forces it to match what it replaces
# — so a caller authorized on the ORIGINAL id must not see a narrower-scoped
# successor's content through the conflict payload just because they were
# allowed to name the id that led there.
# ---------------------------------------------------------------------------


def test_chain_follow_hides_a_project_scoped_successor_from_a_non_member(db):
    org_id = _org(db)
    project_id = _project(db, org_id=org_id)
    assert projects.add_member(
        db, org_id=org_id, project_id=project_id, user_id=PRIYA, granted_by=None
    )
    db.commit()

    memory_id = _seed(db, org_id=org_id, scope="org", content="org-wide fact")
    base_revision = _read_revision(db, org_id=org_id, memory_id=memory_id)

    # A project member supersedes the ORG row with a PROJECT-scoped
    # successor. Legitimate: any caller may choose the scope of what they
    # write, and nothing ties a new row's scope to the one it replaces.
    first = memory_mod.supersede_with_revision(
        db,
        org_id=org_id,
        memory_id=memory_id,
        base_revision=base_revision,
        new_content="project-private follow-up",
        user_id=PRIYA,
        agent_id=AGENT,
        memory_type="fact",
        session_id="sess-n1-1",
        scope="project",
        project_id=project_id,
    )
    assert first["conflict"] is False

    # A non-member, holding the now-stale base_revision of the ORIGINAL
    # (org-scoped) row, is authorized on memory_id itself (scope='org' has
    # no owner/membership check) but must NOT see the project row's content
    # once the guard misses and the chain-follow lands on it.
    non_member = memory_mod.supersede_with_revision(
        db,
        org_id=org_id,
        memory_id=memory_id,
        base_revision=base_revision,
        new_content="attempt",
        user_id=DEV,
        agent_id=AGENT,
        memory_type="fact",
        session_id="sess-n1-2",
    )
    assert non_member["conflict"] is True
    assert non_member["current"] is None

    # Positive control: the project member DOES see the terminal row — a
    # ladder that refused everyone would pass the negative above by accident.
    member = memory_mod.supersede_with_revision(
        db,
        org_id=org_id,
        memory_id=memory_id,
        base_revision=base_revision,
        new_content="attempt",
        user_id=PRIYA,
        agent_id=AGENT,
        memory_type="fact",
        session_id="sess-n1-3",
    )
    assert member["conflict"] is True
    assert member["current"]["id"] == first["new_id"]
    assert member["current"]["content"] == "project-private follow-up"


def test_chain_follow_hides_an_agent_scoped_successor_from_another_agent(db):
    org_id = _org(db)
    memory_id = _seed(db, org_id=org_id, scope="org", content="org-wide fact")
    base_revision = _read_revision(db, org_id=org_id, memory_id=memory_id)

    first = memory_mod.supersede_with_revision(
        db,
        org_id=org_id,
        memory_id=memory_id,
        base_revision=base_revision,
        new_content="agent-private follow-up",
        agent_id=AGENT,
        memory_type="fact",
        session_id="sess-n1-agent-1",
        scope="agent",
    )
    assert first["conflict"] is False

    other_agent = memory_mod.supersede_with_revision(
        db,
        org_id=org_id,
        memory_id=memory_id,
        base_revision=base_revision,
        new_content="attempt",
        agent_id="other-agent",
        memory_type="fact",
        session_id="sess-n1-agent-2",
    )
    assert other_agent["conflict"] is True
    assert other_agent["current"] is None

    owning_agent = memory_mod.supersede_with_revision(
        db,
        org_id=org_id,
        memory_id=memory_id,
        base_revision=base_revision,
        new_content="attempt",
        agent_id=AGENT,
        memory_type="fact",
        session_id="sess-n1-agent-3",
    )
    assert owning_agent["conflict"] is True
    assert owning_agent["current"]["id"] == first["new_id"]
    assert owning_agent["current"]["content"] == "agent-private follow-up"
