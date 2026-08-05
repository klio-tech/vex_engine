"""Can a concurrent promotion land a row wider than it was reported? Real Postgres.

``promote_memory`` reads a row, checks ``scope == 'private'`` in Python, and
then issues an UPDATE keyed only on ``id``/``org_id``/``user_id``. Between
those two statements the row can change. Under READ COMMITTED — the default
here — the UPDATE re-reads the latest committed version, so the check and the
write disagree about what they are looking at:

    T1  SELECT ... -> scope='private'
    T2  promote to 'project', COMMIT
    T1  UPDATE ... SET scope='org'      <- no guard, so this wins

The row ends at ``org``. The caller who asked for ``project`` is told
``{"shared": True, "scope": "project"}``. Nothing anywhere reports that a
memory the owner scoped to a project is now readable by the whole org, and the
``already_shared`` refusal that exists specifically to prevent the
project→org downgrade was evaluated against a row that no longer existed in
that form.

The interleaving here is deterministic, not a race the test hopes to hit: a
proxy session runs the interfering promotion on a SECOND connection at exactly
the moment between the SELECT and the UPDATE. Two connections and a real
server are both load-bearing — one connection cannot exhibit a lost update,
and a fake session cannot evaluate a WHERE clause.
"""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text

from shared.promotion import promote_memory
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the promotion lost-update tests")

EMBED_DIM = 1024
FIXED_EMBEDDING_LITERAL = "[" + ",".join(["0.1"] * EMBED_DIM) + "]"

PRIYA = "11111111-2222-3333-4444-555555555555"


@pytest.fixture()
def engine():
    eng = create_engine(DB_URL)
    orgs: set[str] = set()
    eng.created_org_ids = orgs  # type: ignore[attr-defined]
    try:
        yield eng
    finally:
        with eng.connect() as conn:
            for org_id in orgs:
                conn.execute(
                    text("DELETE FROM session_memories WHERE org_id = :o"),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM projects WHERE org_id = :o"), {"o": org_id}
                )
            conn.commit()
        eng.dispose()


def _org(engine) -> str:
    org_id = f"org_{uuid.uuid4().hex[:8]}"
    engine.created_org_ids.add(org_id)
    return org_id


def _project(conn, *, org_id: str) -> str:
    project_id = str(uuid.uuid4())
    conn.execute(
        text(
            """
            INSERT INTO projects (id, org_id, git_remote, display_name)
            VALUES (:id, :org_id, :remote, 'apollo')
            """
        ),
        {
            "id": project_id,
            "org_id": org_id,
            "remote": f"git@example.com:{uuid.uuid4().hex[:8]}.git",
        },
    )
    conn.commit()
    return project_id


def _seed_private(conn, *, org_id: str, project_id: str | None) -> str:
    memory_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    conn.execute(
        text("""
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, execution_id,
                memory_type, content, embedding, scope, status,
                user_id, provenance, project_id, metadata,
                created_at, updated_at
            ) VALUES (
                :id, :session_id, :org_id, 'agent-1', NULL,
                'fact', 'my salary review is on friday', :embedding,
                'private', 'active', :user_id, 'EXTRACTED', :project_id,
                '{}'::jsonb, :now, :now
            )
        """),
        {
            "id": memory_id,
            "session_id": f"sess-{uuid.uuid4().hex[:6]}",
            "org_id": org_id,
            "embedding": FIXED_EMBEDDING_LITERAL,
            "user_id": PRIYA,
            "project_id": project_id,
            "now": now,
        },
    )
    conn.commit()
    return memory_id


def _scope(conn, memory_id: str) -> str:
    return conn.execute(
        text("SELECT scope FROM session_memories WHERE id = :i"), {"i": memory_id}
    ).scalar()


class _InterleavingSession:
    """A real connection that runs ``interfere`` once, after the first SELECT.

    This is the whole experiment: it puts a committed write from another
    session precisely into the window between the check and the write.
    """

    def __init__(self, conn, interfere):
        self._conn = conn
        self._interfere = interfere
        self.fired = False

    def execute(self, statement, params=None):
        result = self._conn.execute(statement, params)
        if not self.fired and "SELECT" in str(statement).upper():
            self.fired = True
            self._interfere()
        return result

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_a_concurrent_promotion_cannot_be_widened_to_org(engine):
    """The defect, in its damaging direction: project → org behind the caller."""
    org = _org(engine)
    with engine.connect() as setup:
        project = _project(setup, org_id=org)
        memory_id = _seed_private(setup, org_id=org, project_id=project)

    def _interfere():
        with engine.connect() as other:
            other.execute(
                text(
                    "UPDATE session_memories SET scope = 'project' "
                    "WHERE id = :i AND org_id = :o"
                ),
                {"i": memory_id, "o": org},
            )
            other.commit()

    with engine.connect() as conn:
        session = _InterleavingSession(conn, _interfere)
        result = promote_memory(
            session, org_id=org, memory_id=memory_id, user_id=PRIYA, to="org"
        )

    assert session.fired, "the interleaving never happened; the test proves nothing"

    with engine.connect() as check:
        final = _scope(check, memory_id)

    assert final == "project", (
        f"a concurrent promotion was overwritten: the row ended at {final!r}, "
        "wider than the scope its owner last chose"
    )
    assert result["shared"] is False, (
        f"the caller was told the promotion succeeded; got {result}"
    )
    assert result["error"] == "already_shared"


def test_the_uncontended_promotion_still_works(engine):
    """Positive control: with nothing interfering, promotion is unchanged."""
    org = _org(engine)
    with engine.connect() as setup:
        memory_id = _seed_private(setup, org_id=org, project_id=None)

    with engine.connect() as conn:
        result = promote_memory(
            conn, org_id=org, memory_id=memory_id, user_id=PRIYA, to="org"
        )

    assert result["shared"] is True, result
    assert result["scope"] == "org"
    with engine.connect() as check:
        assert _scope(check, memory_id) == "org"


def test_a_concurrent_retraction_is_not_silently_promoted(engine):
    """``status='active'`` belongs in the guard for the same reason.

    The SELECT filters on it; without it in the UPDATE, a memory the owner
    retracted in the meantime is quietly resurrected into a shared scope.
    """
    org = _org(engine)
    with engine.connect() as setup:
        memory_id = _seed_private(setup, org_id=org, project_id=None)

    def _interfere():
        with engine.connect() as other:
            other.execute(
                text(
                    "UPDATE session_memories SET status = 'retracted' "
                    "WHERE id = :i AND org_id = :o"
                ),
                {"i": memory_id, "o": org},
            )
            other.commit()

    with engine.connect() as conn:
        session = _InterleavingSession(conn, _interfere)
        result = promote_memory(
            session, org_id=org, memory_id=memory_id, user_id=PRIYA, to="org"
        )

    assert session.fired
    with engine.connect() as check:
        assert _scope(check, memory_id) == "private", (
            "a retracted memory was promoted to a shared scope"
        )
    assert result["shared"] is False, result
