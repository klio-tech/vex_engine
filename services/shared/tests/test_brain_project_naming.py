"""Resolving a project by the name a human typed. Real Postgres.

``find_project_id`` matched ``display_name`` with ``=``, so a project stored as
``klio`` resolved for ``klio`` and not for ``Klio`` — and "Klio" is what a
person types, including the person who just created it and was told that was
its name. These tests are about which rows the lookup ADMITS, so they run
against a real engine: ``lower(a) = lower(b)`` is a claim about Postgres, not
about Python.

The determinism tests matter as much as the matching ones. Case-insensitivity
widens the candidate set — ``web``, ``Web`` and ``WEB`` become one bucket — so
it can only be safe if the documented tie-break (``last_seen_at DESC, id ASC``)
still picks the same winner every time.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text

from shared.brain import find_project_id
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the project name-lookup tests")


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
            for org_id in orgs:
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


def _project(
    db,
    *,
    org_id: str,
    display_name: str,
    git_remote: str | None = None,
    repo_root_path: str | None = None,
    last_seen_at: datetime | None = None,
    project_id: str | None = None,
) -> str:
    project_id = project_id or str(uuid.uuid4())
    if git_remote is None and repo_root_path is None:
        git_remote = f"git@example.com:{uuid.uuid4().hex[:8]}.git"
    db.execute(
        text(
            """
            INSERT INTO projects (
                id, org_id, git_remote, repo_root_path, display_name, last_seen_at
            ) VALUES (
                :id, :org_id, :remote, :path, :name,
                COALESCE(:seen, NOW())
            )
            """
        ),
        {
            "id": project_id,
            "org_id": org_id,
            "remote": git_remote,
            "path": repo_root_path,
            "name": display_name,
            "seen": last_seen_at,
        },
    )
    db.commit()
    return project_id


# ---------------------------------------------------------------------------
# (b) name lookup is case-insensitive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("typed", ["klio", "Klio", "KLIO", "kLiO"])
def test_display_name_resolves_regardless_of_case(db, typed):
    """The live probe: `klio` resolved, `Klio` returned unknown_project."""
    org = _org(db)
    project_id = _project(db, org_id=org, display_name="klio")

    assert find_project_id(db, org_id=org, identifier=typed) == project_id


def test_a_name_that_matches_nothing_still_resolves_to_none(db):
    """The control. Case-insensitivity must widen case, not widen matching.

    Without this, `lower(display_name) LIKE ...` or any other loosening would
    pass the test above while quietly resolving every near-miss to a project
    the caller never named.
    """
    org = _org(db)
    _project(db, org_id=org, display_name="klio")

    assert find_project_id(db, org_id=org, identifier="kli") is None
    assert find_project_id(db, org_id=org, identifier="klio2") is None
    assert find_project_id(db, org_id=org, identifier="not-a-project") is None


def test_case_insensitivity_stops_at_the_org_boundary(db):
    """A name is only resolvable inside the org that owns it."""
    mine = _org(db)
    theirs = _org(db)
    _project(db, org_id=theirs, display_name="klio")

    assert find_project_id(db, org_id=mine, identifier="Klio") is None


# ---------------------------------------------------------------------------
# precedence and the tie-break, under the widened candidate set
# ---------------------------------------------------------------------------


def test_an_exact_git_remote_still_outranks_a_display_name_match(db):
    """Precedence is unchanged: id -> git_remote -> repo_root_path -> name."""
    org = _org(db)
    remote = "git@example.com:acme/web.git"
    by_remote = _project(
        db, org_id=org, display_name="something-else", git_remote=remote
    )
    _project(db, org_id=org, display_name=remote)  # a name that IS the remote string

    assert find_project_id(db, org_id=org, identifier=remote) == by_remote


def test_a_repo_path_still_outranks_a_display_name_match(db):
    org = _org(db)
    path = "/Users/dev/Me/Agent Memory"
    by_path = _project(
        db, org_id=org, display_name="unrelated", git_remote=None, repo_root_path=path
    )
    _project(db, org_id=org, display_name=path)

    assert find_project_id(db, org_id=org, identifier=path) == by_path


def test_names_differing_only_in_case_break_the_tie_on_last_seen_then_id(db):
    """Case-insensitivity merges `web` and `Web` into ONE candidate bucket.

    That is exactly where it could introduce a non-deterministic winner, so
    the documented tie-break has to survive it: most-recently-seen first, id
    ascending after that.
    """
    org = _org(db)
    now = datetime.now(timezone.utc)
    _project(db, org_id=org, display_name="web", last_seen_at=now - timedelta(days=5))
    recent = _project(db, org_id=org, display_name="WEB", last_seen_at=now)
    _project(db, org_id=org, display_name="Web", last_seen_at=now - timedelta(days=1))

    # Same answer every time, and it is the most-recently-seen row.
    assert {find_project_id(db, org_id=org, identifier="web") for _ in range(5)} == {
        recent
    }
    assert find_project_id(db, org_id=org, identifier="WeB") == recent


def test_id_breaks_a_last_seen_at_tie_across_cases(db):
    """Two rows can share a last_seen_at; id ASC is the final total order."""
    org = _org(db)
    same_instant = datetime.now(timezone.utc)
    low = "00000000-0000-0000-0000-00000000000a"
    high = "ffffffff-0000-0000-0000-00000000000f"
    _project(
        db,
        org_id=org,
        display_name="APP",
        last_seen_at=same_instant,
        project_id=high,
    )
    _project(
        db,
        org_id=org,
        display_name="app",
        last_seen_at=same_instant,
        project_id=low,
    )

    assert {find_project_id(db, org_id=org, identifier="App") for _ in range(5)} == {
        low
    }
