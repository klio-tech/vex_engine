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

from shared import projects
from shared.memory import retract_memory, supersede_memory
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
    content: str = "the deploy key rotates on the first of the month",
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
            "embedding": FIXED_EMBEDDING_LITERAL,
            "scope": scope,
            "user_id": user_id,
            "project_id": project_id,
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


# ── the project arm ───────────────────────────────────────────────────
#
# ``retract_memory``'s ladder had a private arm and an agent arm and nothing
# for ``scope='project'``. A project row was therefore retractable by ANY
# caller in the org who held its id — including someone the recall SQL would
# never have shown it to, because the id is not the secret; membership is.
#
# The refusal is ``not_found``, matching the private arm rather than the agent
# arm's ``forbidden``: a project row's existence is exactly what membership
# hides, so confirming it would be the disclosure. Each attack below is paired
# with a member performing the identical retraction.

DEV = "99999999-8888-7777-6666-555555555555"


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


def test_a_non_member_cannot_retract_a_project_memory(db):
    org = _org(db)
    project = _project(db, org_id=org)
    assert projects.add_member(
        db, org_id=org, project_id=project, user_id=PRIYA, granted_by=None
    )
    db.commit()
    memory_id = _seed(db, org_id=org, scope="project", project_id=project)

    refused = retract_memory(
        db,
        org_id=org,
        agent_id=AGENT,
        memory_id=memory_id,
        via="mcp",
        user_id=DEV,
    )

    assert refused == {"forgotten": False, "error": "not_found"}, refused
    assert _status(db, memory_id) == "active", "refused, then retracted anyway"

    # Positive control: the member retracts the identical row.
    allowed = retract_memory(
        db,
        org_id=org,
        agent_id=AGENT,
        memory_id=memory_id,
        via="mcp",
        user_id=PRIYA,
    )
    assert allowed["forgotten"] is True, allowed
    assert _status(db, memory_id) == "retracted"


def test_an_unattributed_caller_cannot_retract_a_project_memory(db):
    """``user_id=None`` is not a member of anything, and not a wildcard."""
    org = _org(db)
    project = _project(db, org_id=org)
    assert projects.add_member(
        db, org_id=org, project_id=project, user_id=PRIYA, granted_by=None
    )
    db.commit()
    memory_id = _seed(db, org_id=org, scope="project", project_id=project)

    refused = retract_memory(
        db, org_id=org, agent_id=AGENT, memory_id=memory_id, via="mcp", user_id=None
    )

    assert refused == {"forgotten": False, "error": "not_found"}, refused
    assert _status(db, memory_id) == "active"

    allowed = retract_memory(
        db, org_id=org, agent_id=AGENT, memory_id=memory_id, via="mcp", user_id=PRIYA
    )
    assert allowed["forgotten"] is True, "positive control failed"


def test_the_project_refusal_is_identical_to_an_unknown_id(db):
    """No existence oracle: the two answers are the same object."""
    org = _org(db)
    project = _project(db, org_id=org)
    memory_id = _seed(db, org_id=org, scope="project", project_id=project)

    real = retract_memory(
        db, org_id=org, agent_id=AGENT, memory_id=memory_id, via="mcp", user_id=DEV
    )
    invented = retract_memory(
        db,
        org_id=org,
        agent_id=AGENT,
        memory_id=str(uuid.uuid4()),
        via="mcp",
        user_id=DEV,
    )
    assert real == invented == {"forgotten": False, "error": "not_found"}


def test_an_org_row_that_merely_names_a_project_is_not_fenced_by_it(db):
    """``project_id`` on an ``org``-scoped row RANKS; it does not FENCE.

    ``_explicit_write_scope`` deliberately files non-members' writes under a
    project at ``scope='org'``. Reading membership off ``project_id`` alone
    would make those rows unforgettable by the very people who wrote them.
    """
    org = _org(db)
    project = _project(db, org_id=org)
    memory_id = _seed(db, org_id=org, scope="org", project_id=project)

    result = retract_memory(
        db, org_id=org, agent_id=AGENT, memory_id=memory_id, via="mcp", user_id=DEV
    )

    assert result["forgotten"] is True, result
    assert _status(db, memory_id) == "retracted"


def test_a_project_row_with_no_project_id_is_not_retractable_by_anyone(db):
    """Fail closed. A fenced scope with no fence is not an open gate.

    No such row should exist, but ``scope`` and ``project_id`` are separate
    columns with no constraint tying them together, so the code has to answer
    for the combination. ``is_member`` cannot vouch for a NULL project, so
    neither can this.
    """
    org = _org(db)
    memory_id = _seed(db, org_id=org, scope="project", project_id=None)

    refused = retract_memory(
        db, org_id=org, agent_id=AGENT, memory_id=memory_id, via="mcp", user_id=PRIYA
    )

    assert refused == {"forgotten": False, "error": "not_found"}, refused
    assert _status(db, memory_id) == "active"


# ── supersede_memory's tenancy argument ───────────────────────────────
#
# ``supersede_memory`` had no tenancy argument at all: its UPDATE was keyed on
# ``id`` alone. Both current callers derive their ids from org-scoped queries,
# so nothing was reachable through it — but it is the only write in the memory
# layer with no ``org_id`` in its WHERE, and "the caller happens to be careful"
# is not a boundary. One new caller passing an id from anywhere else silently
# retires another tenant's memory.


def test_supersede_requires_the_owning_org(db):
    org = _org(db)
    other = _org(db)
    old_id = _seed(db, org_id=org, content="the old fact")
    new_id = _seed(db, org_id=org, content="the new fact")

    # The attacker's org names a memory that is not theirs.
    supersede_memory(db, uuid.UUID(old_id), uuid.UUID(new_id), org_id=other)

    assert _status(db, old_id) == "active", (
        "a foreign org retired a memory it does not own"
    )

    # Positive control: the owning org supersedes the identical pair.
    supersede_memory(db, uuid.UUID(old_id), uuid.UUID(new_id), org_id=org)
    assert _status(db, old_id) == "superseded"
    row = db.execute(
        text("SELECT superseded_by FROM session_memories WHERE id = :i"),
        {"i": old_id},
    ).fetchone()
    assert str(row.superseded_by) == new_id


def test_supersede_will_not_run_without_an_org():
    """The argument is mandatory: there is no 'apply to every tenant' call."""
    with pytest.raises(TypeError):
        supersede_memory(object(), uuid.uuid4(), uuid.uuid4())
