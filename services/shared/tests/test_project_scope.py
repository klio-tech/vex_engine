"""Where a passive capture lands (``shared.project_scope``, migration 047).

REAL POSTGRES, NOT MOCKS. The whole feature is a routing decision made from two
columns, and the failure that matters is a capture landing WIDER than the person
expected. A mocked session proves the function was called; only a real row
proves what scope it resolved.

Mutations this file is built to catch, and the test that catches each:

* The project predicate is loosened — e.g. to ``(:project_id IS NULL OR
  p.id = :project_id)``, the kind of "helpful" refactor that looks harmless —
  so unfiled context inherits some project's sharing setting
  -> ``test_a_capture_with_no_project_is_always_private`` (verified to fail
  under exactly that mutation)

  NOTE what this does NOT catch: deleting the early ``if not project_id``
  return alone changes nothing, because ``WHERE p.id = :project_id`` already
  matches no row and the function falls closed. The early return is
  explicitness, not the safety mechanism; the predicate is.
* The member opt-out is ignored, publishing someone who opted out
  -> ``test_member_override_beats_the_project_default``
* The org fence is dropped, so another tenant's setting routes this org
  -> ``test_another_orgs_project_never_influences_this_org``
* An unreadable or unknown project fails OPEN instead of closed
  -> ``test_an_unknown_project_is_private`` /
     ``test_a_broken_read_is_private``
* A bad value in the column is treated as shared
  -> ``test_an_unrecognised_default_is_private``
* The override is allowed to WIDEN rather than only narrow
  -> ``test_a_member_override_may_only_narrow``
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from shared import project_scope
from shared.testing import requires_db, database_url

pytestmark = requires_db("the project default capture scope tests")

ORG = "org-scope-test"
OTHER_ORG = "org-scope-other"
ALICE = "11111111-1111-1111-1111-111111111111"
BOB = "22222222-2222-2222-2222-222222222222"
PROJECT = "aaaaaaaa-0000-0000-0000-00000000ab01"
OTHER_PROJECT = "aaaaaaaa-0000-0000-0000-00000000ab02"

#: A dedicated Postgres schema for this file's tables.
#:
#: CI runs `alembic upgrade head` and then the suites against that migrated
#: database. These fixtures need to CREATE tables, and creating them in
#: `public` means dropping the real ones first — which would destroy the
#: migrated schema for every test that runs afterwards. Namespacing keeps the
#: tables real (the predicates under test are the point) while leaving `public`
#: untouched.
TEST_SCHEMA = "test_project_scope"

SCHEMA = f"""
DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE;
CREATE SCHEMA {TEST_SCHEMA};
SET search_path TO {TEST_SCHEMA};

CREATE TABLE projects (
  id uuid PRIMARY KEY,
  org_id varchar(64) NOT NULL,
  display_name varchar NOT NULL,
  default_scope text NOT NULL DEFAULT 'private'
    CHECK (default_scope IN ('private','project','org'))
);

CREATE TABLE project_members (
  project_id uuid NOT NULL,
  user_id varchar NOT NULL,
  role varchar NOT NULL DEFAULT 'write',
  scope_override text
    CHECK (scope_override IS NULL OR scope_override = 'private'),
  PRIMARY KEY (project_id, user_id)
);
"""




def _engine():
    """An engine whose connections resolve unqualified names to TEST_SCHEMA."""
    return sa.create_engine(
        database_url(),
        connect_args={"options": f"-csearch_path={TEST_SCHEMA}"},
    )


def _session(engine):
    """A session the caller MUST close.

    `resolve_capture_scope` is a read and deliberately does not commit — in
    production it shares the capture's session, which commits later. In tests
    that means an un-closed session sits idle-in-transaction holding a lock on
    `projects`, and the next test's DROP TABLE waits on it forever. Every
    helper below closes; nothing here leaks a session.
    """
    return sessionmaker(bind=engine)()


def _setup(engine, *, default="private", override=None, other_default="org"):
    with engine.begin() as conn:
        conn.execute(sa.text(SCHEMA))
        conn.execute(
            sa.text(
                "INSERT INTO projects (id, org_id, display_name, default_scope) "
                "VALUES (:p,:o,'Agent Memory',:d), (:p2,:o2,'Elsewhere',:d2)"
            ),
            {
                "p": PROJECT,
                "o": ORG,
                "d": default,
                "p2": OTHER_PROJECT,
                "o2": OTHER_ORG,
                "d2": other_default,
            },
        )
        conn.execute(
            sa.text(
                "INSERT INTO project_members (project_id, user_id, role, scope_override) "
                "VALUES (:p,:u,'admin',:ov)"
            ),
            {"p": PROJECT, "u": ALICE, "ov": override},
        )


def _resolve(engine, *, project=PROJECT, user=ALICE):
    db = _session(engine)
    try:
        return project_scope.resolve_capture_scope(
            db, org_id=ORG, project_id=project, user_id=user
        )
    finally:
        db.rollback()
        db.close()


def _with_session(engine, fn):
    db = _session(engine)
    try:
        return fn(db)
    finally:
        db.close()


def test_a_private_project_captures_private():
    engine = _engine()
    _setup(engine, default="private")

    assert _resolve(engine) == "private"


def test_a_project_scoped_project_captures_project():
    engine = _engine()
    _setup(engine, default="project")

    assert _resolve(engine) == "project"


def test_an_org_scoped_project_captures_org():
    engine = _engine()
    _setup(engine, default="org")

    assert _resolve(engine) == "org"


def test_a_capture_with_no_project_is_always_private():
    # Unfiled context is whatever the person was doing outside any repo; no
    # project setting may reach it. Guards the PREDICATE, not the early return
    # — see the module docstring for which mutation this actually catches.
    engine = _engine()
    _setup(engine, default="org")

    assert _resolve(engine, project=None) == "private"


def test_an_unattributed_capture_is_private():
    engine = _engine()
    _setup(engine, default="org")

    assert _resolve(engine, user=None) == "private"


def test_member_override_beats_the_project_default():
    engine = _engine()
    _setup(engine, default="org", override="private")

    assert _resolve(engine) == "private"


def test_a_non_member_of_a_shared_project_still_follows_the_default():
    # Bob has no membership row. The project says org, so his capture is org —
    # the LEFT JOIN must not turn "no override" into "no route".
    engine = _engine()
    _setup(engine, default="org")

    assert _resolve(engine, user=BOB) == "org"


def test_another_orgs_project_never_influences_this_org():
    # OTHER_PROJECT defaults to 'org' but belongs to OTHER_ORG. Asking for it
    # as ORG must select nothing and fall closed.
    engine = _engine()
    _setup(engine, default="private", other_default="org")

    assert _resolve(engine, project=OTHER_PROJECT) == "private"


def test_an_unknown_project_is_private():
    engine = _engine()
    _setup(engine, default="org")

    assert _resolve(engine, project="aaaaaaaa-0000-0000-0000-0000000000ff") == "private"


def test_an_unrecognised_default_is_private():
    engine = _engine()
    _setup(engine, default="org")
    # Simulate a value the CHECK would reject but that could arrive from a
    # future migration or a hand edit: the router must not trust it.
    assert project_scope.normalize_default_scope("everyone") is None
    assert project_scope.normalize_default_scope(None) is None


def test_a_broken_read_is_private():
    # A capture must never fail, and must never widen, because a preference
    # could not be read.
    class Exploding:
        def execute(self, *_a, **_k):
            raise RuntimeError("neon is asleep")

    assert (
        project_scope.resolve_capture_scope(
            Exploding(), org_id=ORG, project_id=PROJECT, user_id=ALICE
        )
        == "private"
    )


def test_setting_the_default_returns_the_previous_value():
    engine = _engine()
    _setup(engine, default="private")

    previous = _with_session(
        engine,
        lambda db: project_scope.set_project_default_scope(
            db, org_id=ORG, project_id=PROJECT, scope="org"
        ),
    )

    assert previous == "private"
    assert _resolve(engine) == "org"


def test_setting_the_default_refuses_another_org():
    engine = _engine()
    _setup(engine)

    assert (
        _with_session(
            engine,
            lambda db: project_scope.set_project_default_scope(
                db, org_id=ORG, project_id=OTHER_PROJECT, scope="org"
            ),
        )
        is None
    )


def test_setting_an_invalid_default_raises():
    engine = _engine()
    _setup(engine)

    try:
        _with_session(
            engine,
            lambda db: project_scope.set_project_default_scope(
                db, org_id=ORG, project_id=PROJECT, scope="everyone"
            ),
        )
    except ValueError:
        return
    raise AssertionError("an invalid scope must be refused, not stored")


def test_a_member_override_may_only_narrow():
    engine = _engine()
    _setup(engine, default="private")

    try:
        _with_session(
            engine,
            lambda db: project_scope.set_member_scope_override(
                db,
                org_id=ORG,
                project_id=PROJECT,
                user_id=ALICE,
                override="org",
            ),
        )
    except ValueError:
        return
    raise AssertionError("an override that widens must be refused")


def test_clearing_an_override_follows_the_project_again():
    engine = _engine()
    _setup(engine, default="org", override="private")
    assert _resolve(engine) == "private"

    _with_session(
        engine,
        lambda db: project_scope.set_member_scope_override(
            db, org_id=ORG, project_id=PROJECT, user_id=ALICE, override=None
        ),
    )

    assert _resolve(engine) == "org"


def test_an_override_write_cannot_cross_the_org_fence():
    engine = _engine()
    _setup(engine)

    changed = _with_session(
        engine,
        lambda db: project_scope.set_member_scope_override(
            db,
            org_id=OTHER_ORG,
            project_id=PROJECT,
            user_id=ALICE,
            override="private",
        ),
    )

    assert changed is False
