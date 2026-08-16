"""Project scoping on the READ path, against a real Postgres.

The reported symptom: a session entirely about the Klio proxy opened with two
memories from a Vedic astrology app and two from a hiring app. All four were
real rows, correctly filed under their own projects, and correctly recent. The
recall that surfaced them resolved no project at all.

Two defects stacked, and both are asserted here.

**Reads refused the identifier writes accepted.** ``_find_project_id`` returned
``None`` the instant ``git_remote`` was falsy, while ``_ensure_project_id``
passed both ``git_remote`` and ``repo_root`` and ``brain.find_project_id`` has
always resolved ``repo_root_path``. In production 79 of 118 projects have no
remote — including ``Astrology``, ``hireloop`` and ``Agent Memory``, the exact
projects that leaked — so writes filed by path and reads could not.

**An unresolved project meant "everything" rather than "unfiled".** The recency
predicate was ``(:project_id IS NULL OR project_id = :project_id OR project_id
IS NULL)``: two escape hatches in one line, on a query whose only ordering is
``created_at DESC``. That is "the last N things anyone wrote anywhere",
injected at the start of every session.

Real Postgres, because both defects are WHERE clauses and a fake session would
pass against a query with no predicate at all.
"""

import uuid

import pytest
from sqlalchemy import create_engine, text

from app.capture import _find_project_id, _recent_memories
from shared.testing import database_url, requires_db

pytestmark = requires_db("the capture project-fence tests")

DB_URL = database_url()
USER = "11111111-2222-3333-4444-555555555555"


@pytest.fixture()
def db():
    engine = create_engine(DB_URL)
    conn = engine.connect()
    org_id = f"org_fence_{uuid.uuid4().hex[:8]}"
    conn.org_id = org_id  # type: ignore[attr-defined]
    try:
        yield conn
    finally:
        try:
            conn.rollback()
            conn.execute(
                text("DELETE FROM session_memories WHERE org_id = :o"), {"o": org_id}
            )
            conn.execute(text("DELETE FROM projects WHERE org_id = :o"), {"o": org_id})
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _project(db, *, name: str, remote=None, path=None) -> str:
    project_id = str(uuid.uuid4())
    db.execute(
        text(
            "INSERT INTO projects (id, org_id, display_name, git_remote, "
            "repo_root_path) VALUES (:id, :org, :name, :remote, :path)"
        ),
        {
            "id": project_id,
            "org": db.org_id,
            "name": name,
            "remote": remote,
            "path": path,
        },
    )
    db.commit()
    return project_id


def _memory(db, *, content: str, project_id=None, scope="private") -> str:
    memory_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, memory_type, content,
                scope, status, created_at, updated_at, recall_hidden,
                user_id, project_id
            ) VALUES (
                :id, 's-fence', :org, 'agent-1', 'fact', :content,
                :scope, 'active', NOW(), NOW(), FALSE, :user, :project
            )
            """
        ),
        {
            "id": memory_id,
            "org": db.org_id,
            "content": content,
            "scope": scope,
            "user": USER if scope == "private" else None,
            "project": project_id,
        },
    )
    db.commit()
    return memory_id


# ── resolving a project without a git remote ──────────────────────────────


def test_a_project_with_no_remote_resolves_by_repo_root(db):
    """The single highest-leverage line in the fix.

    Two thirds of production projects are path-only. Refusing to resolve them
    on read is why 94.8 % of hook recalls ran with no project filter at all,
    against a corpus where 95.2 % of rows carry one.
    """
    project_id = _project(db, name="Astrology", path="/Users/x/Code/Astrology")
    resolved = _find_project_id(
        db, org_id=db.org_id, git_remote=None, repo_root="/Users/x/Code/Astrology"
    )
    assert resolved == project_id


def test_the_git_remote_is_tried_before_the_path(db):
    """The remote is stable across machines and checkouts; the path is not.

    A repo cloned to a different directory must still resolve to the one
    project, so the more specific identifier has to win. Here the path points
    at a DIFFERENT project, and the remote must decide.
    """
    _project(db, name="decoy", path="/Users/x/Code/klio")
    wanted = _project(
        db, name="Klio", remote="git@github.com:klio-tech/klio.git", path="/elsewhere"
    )
    resolved = _find_project_id(
        db,
        org_id=db.org_id,
        git_remote="git@github.com:klio-tech/klio.git",
        repo_root="/Users/x/Code/klio",
    )
    assert resolved == wanted


def test_neither_identifier_resolving_is_unfiled_not_org_wide(db):
    """``None`` still means "no project" — what CHANGED is what the callers do
    with it. See the recency tests below."""
    assert (
        _find_project_id(
            db, org_id=db.org_id, git_remote=None, repo_root="/nowhere/at/all"
        )
        is None
    )


def test_no_identifiers_at_all_resolves_to_nothing(db):
    assert (
        _find_project_id(db, org_id=db.org_id, git_remote=None, repo_root=None) is None
    )


# ── the SessionStart recency fence ────────────────────────────────────────


def test_session_start_never_returns_another_projects_memories(db):
    """The reported bug, stated directly.

    Both decoys are newer than the wanted row, so under ``ORDER BY created_at
    DESC`` they came first — which is exactly how the astrology and hiring-app
    facts reached a session about the proxy.
    """
    klio = _project(db, name="Klio", path="/Users/x/Code/klio")
    astrology = _project(db, name="Astrology", path="/Users/x/Code/Astrology")
    hirly = _project(db, name="hirly", path="/Users/x/Code/hirly")

    wanted = _memory(db, content="the proxy injects at SessionStart", project_id=klio)
    _memory(db, content="D30 uses unequal Trimsamsa portions", project_id=astrology)
    _memory(db, content="supabase_db_hirly runs on port 54422", project_id=hirly)

    rows = _recent_memories(
        db, org_id=db.org_id, project_id=klio, limit=12, user_id=USER
    )
    assert [r["id"] for r in rows] == [wanted]


def test_a_resolved_project_does_not_admit_unfiled_rows(db):
    """The second escape hatch.

    ``OR project_id IS NULL`` is defensible on the VECTOR path, which can rank
    what it lets in. On a path ordered by nothing but the clock, every row it
    admits beyond the fence is chosen by its timestamp — that is not extra
    context, it is a coin flip.
    """
    klio = _project(db, name="Klio", path="/Users/x/Code/klio")
    wanted = _memory(db, content="the proxy injects at SessionStart", project_id=klio)
    _memory(db, content="an unfiled note written just now", project_id=None)

    rows = _recent_memories(
        db, org_id=db.org_id, project_id=klio, limit=12, user_id=USER
    )
    assert [r["id"] for r in rows] == [wanted]


def test_an_unresolved_project_reads_exactly_the_unfiled_bucket(db):
    """Symmetry with the write path, which is the whole design.

    A capture from a directory that resolves to no project is written unfiled.
    A recall from that same directory now reads that bucket back — and nothing
    else. Previously it read every project in the org.
    """
    astrology = _project(db, name="Astrology", path="/Users/x/Code/Astrology")
    _memory(db, content="D30 uses unequal Trimsamsa portions", project_id=astrology)
    unfiled = _memory(db, content="an unfiled note", project_id=None)

    rows = _recent_memories(
        db, org_id=db.org_id, project_id=None, limit=12, user_id=USER
    )
    assert [r["id"] for r in rows] == [unfiled]


def test_an_empty_result_is_a_legitimate_answer(db):
    """A project with nothing in it gets nothing, rather than somebody else's.

    Every caller has to handle this — the Node proxy already reports
    ``reason: "empty"``.
    """
    empty = _project(db, name="brand-new", path="/Users/x/Code/brand-new")
    other = _project(db, name="Astrology", path="/Users/x/Code/Astrology")
    _memory(db, content="D30 uses unequal Trimsamsa portions", project_id=other)

    rows = _recent_memories(
        db, org_id=db.org_id, project_id=empty, limit=12, user_id=USER
    )
    assert rows == []


def test_the_fence_does_not_widen_the_scope_ladder(db):
    """The fence narrows by project; it must not accidentally admit a row the
    visibility ladder excludes. A teammate's private row stays invisible even
    when it is filed under the project being read."""
    klio = _project(db, name="Klio", path="/Users/x/Code/klio")
    teammate = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, memory_type, content,
                scope, status, created_at, updated_at, recall_hidden,
                user_id, project_id
            ) VALUES (
                :id, 's-fence', :org, 'agent-1', 'fact', 'a teammate note',
                'private', 'active', NOW(), NOW(), FALSE, :user, :project
            )
            """
        ),
        {
            "id": str(uuid.uuid4()),
            "org": db.org_id,
            "user": teammate,
            "project": klio,
        },
    )
    db.commit()

    rows = _recent_memories(
        db, org_id=db.org_id, project_id=klio, limit=12, user_id=USER
    )
    assert rows == []
