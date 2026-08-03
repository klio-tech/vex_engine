"""Cross-org attacks on ``forget``. Real Postgres.

``retract_memory`` is the only write in the memory layer that a caller can aim
at an arbitrary row id. Its whole defence is ``org_id = :org_id`` in the SELECT
that resolves the id — and until this file existed, deleting that predicate
left the entire suite green. Any org could have retracted any other org's
memory by id and nothing would have noticed.

Why real Postgres and not the ``FakeDB`` in
``services/mcp-server/tests/test_forget_tool.py``: a fake cannot catch this,
even a fake that inspects params. The parameter dict is built by
``retract_memory`` from its own arguments and is passed identically whether or
not the SQL references ``:org_id``, so a params-inspecting fake sees no
difference between the guarded query and an unguarded one. Only a server that
actually evaluates the WHERE clause can tell them apart. The fake keeps its
job — the pure-Python authorization ladder (malformed id, agent-scope refusal)
— and the tenancy boundary is tested here.

Every test is an ATTACK: seed a memory in one org, try to retract it as
another, assert the refusal AND that the row is still active. Refusing while
retracting anyway would pass a return-value-only assertion. Each negative has
a positive control, because a ``retract_memory`` that refused everyone would
otherwise pass this whole file.

Each test uses two fresh orgs so a failure cannot be residue from another test.
"""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text

from shared.memory import retract_memory
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the cross-org forget tests")

EMBED_DIM = 1024
FIXED_EMBEDDING_LITERAL = "[" + ",".join(["0.1"] * EMBED_DIM) + "]"

PRIYA = "11111111-2222-3333-4444-555555555555"
AGENT = "agent-1"


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


def _seed(
    db,
    *,
    org_id: str,
    content: str = "the deploy key rotates on the first of the month",
    scope: str = "org",
    user_id: str | None = None,
    agent_id: str = AGENT,
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
                :user_id, 'EXTRACTED', NULL, '{}'::jsonb,
                :created_at, :updated_at
            )
        """),
        {
            "id": memory_id,
            "session_id": f"sess-{uuid.uuid4().hex[:6]}",
            "org_id": org_id,
            "agent_id": agent_id,
            "content": content,
            "embedding": FIXED_EMBEDDING_LITERAL,
            "scope": scope,
            "user_id": user_id,
            "created_at": now,
            "updated_at": now,
        },
    )
    db.commit()
    return memory_id


def _status(db, memory_id: str) -> str | None:
    row = db.execute(
        text("SELECT status FROM session_memories WHERE id = :id"),
        {"id": memory_id},
    ).fetchone()
    return None if row is None else row.status


def _metadata(db, memory_id: str) -> dict:
    row = db.execute(
        text("SELECT metadata FROM session_memories WHERE id = :id"),
        {"id": memory_id},
    ).fetchone()
    return {} if row is None else (row.metadata or {})


# ── the attack ────────────────────────────────────────────────────────


def test_a_foreign_org_cannot_retract_an_org_memory(db):
    """The core cross-tenant attack: acme's id, initech's credentials."""
    acme, initech = _org(db), _org(db)
    memory_id = _seed(db, org_id=acme)

    result = retract_memory(
        db, org_id=initech, agent_id=AGENT, memory_id=memory_id, via="mcp"
    )

    assert result == {"forgotten": False, "error": "not_found"}
    assert _status(db, memory_id) == "active"


def test_the_owning_org_can_retract_it(db):
    """Positive control. Without it, a retract_memory that refused every caller
    would pass every other test in this file."""
    acme = _org(db)
    memory_id = _seed(db, org_id=acme)

    result = retract_memory(
        db, org_id=acme, agent_id=AGENT, memory_id=memory_id, via="mcp"
    )

    assert result == {"forgotten": True, "id": memory_id}
    assert _status(db, memory_id) == "retracted"


def test_a_foreign_org_cannot_retract_a_private_memory(db):
    """Same user id in both orgs, so ownership cannot be what refuses this —
    only the org predicate can."""
    acme, initech = _org(db), _org(db)
    memory_id = _seed(db, org_id=acme, scope="private", user_id=PRIYA)

    result = retract_memory(
        db,
        org_id=initech,
        agent_id=AGENT,
        memory_id=memory_id,
        via="mcp",
        user_id=PRIYA,
    )

    assert result == {"forgotten": False, "error": "not_found"}
    assert _status(db, memory_id) == "active"


def test_the_owner_can_retract_their_own_private_memory(db):
    """Positive control for the private path."""
    acme = _org(db)
    memory_id = _seed(db, org_id=acme, scope="private", user_id=PRIYA)

    result = retract_memory(
        db, org_id=acme, agent_id=AGENT, memory_id=memory_id, via="mcp", user_id=PRIYA
    )

    assert result == {"forgotten": True, "id": memory_id}
    assert _status(db, memory_id) == "retracted"


def test_a_foreign_org_cannot_retract_an_agent_memory(db):
    """Same agent id in both orgs. The answer must be ``not_found``, never
    ``forbidden``: ``forbidden`` is the agent-ladder's answer and it confirms
    the row exists, which would tell an outsider that this id names a real
    memory in some org."""
    acme, initech = _org(db), _org(db)
    memory_id = _seed(db, org_id=acme, scope="agent", agent_id=AGENT)

    result = retract_memory(
        db, org_id=initech, agent_id=AGENT, memory_id=memory_id, via="mcp"
    )

    assert result == {"forgotten": False, "error": "not_found"}
    assert _status(db, memory_id) == "active"


def test_the_cross_org_refusal_is_identical_to_an_unknown_id(db):
    """A distinguishable refusal is an existence oracle: an attacker holding a
    list of ids could sort the real ones from the invented ones without ever
    reading a memory."""
    acme, initech = _org(db), _org(db)
    memory_id = _seed(db, org_id=acme)

    foreign = retract_memory(
        db, org_id=initech, agent_id=AGENT, memory_id=memory_id, via="mcp"
    )
    invented = retract_memory(
        db, org_id=initech, agent_id=AGENT, memory_id=str(uuid.uuid4()), via="mcp"
    )

    assert foreign == invented


def test_a_refused_cross_org_retraction_writes_no_audit_metadata(db):
    """The status assertions above would still pass if the UPDATE ran and set
    only the metadata. It must not run at all."""
    acme, initech = _org(db), _org(db)
    memory_id = _seed(db, org_id=acme)

    retract_memory(db, org_id=initech, agent_id=AGENT, memory_id=memory_id, via="mcp")

    assert _metadata(db, memory_id) == {}


def test_a_foreign_org_cannot_retract_a_memory_it_can_name_exactly(db):
    """The attacker knows the id, the agent id, the user id and the scope —
    everything except the org. That is the threat model: ids leak (logs,
    screenshots, a shared transcript) and must not be bearer tokens."""
    acme, initech = _org(db), _org(db)
    memory_id = _seed(db, org_id=acme, scope="private", user_id=PRIYA, agent_id=AGENT)

    result = retract_memory(
        db,
        org_id=initech,
        agent_id=AGENT,
        memory_id=memory_id,
        via="mcp",
        user_id=PRIYA,
    )

    assert result["forgotten"] is False
    assert _status(db, memory_id) == "active"
