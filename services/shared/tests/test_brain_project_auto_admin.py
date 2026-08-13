"""``ensure_project`` seats the creator as the project's first admin.

Ruling 2026-08-12 ("projects are visible only to their members"): a members-
only ``project_list`` is only safe to ship because every project that gets
auto-created from here on out actually HAS a member — the person whose write
caused it to exist. This is that seating, tested against real Postgres
(mirroring ``test_brain.py``'s convention for this module).

Mutations this file is built to catch:

* The ``xmax = 0`` insert-detection is dropped (membership seeded on every
  call, including the ON CONFLICT hit path)
  -> ``test_a_second_writer_in_the_same_repo_does_not_become_a_member``
* ``created_by`` is written on every call instead of INSERT-only
  -> ``test_a_second_writer_never_overwrites_created_by``
* The admin INSERT is unconditional instead of gated on ``created_by`` being
  truthy
  -> ``test_an_unattributed_write_creates_no_membership``
* The membership INSERT lacks ``ON CONFLICT DO NOTHING``
  -> ``test_calling_ensure_project_twice_for_a_brand_new_project_id_is_safe``
"""

import uuid

import pytest
from sqlalchemy import create_engine, text

from shared.brain import ensure_project
from shared.projects import ROLE_ADMIN, is_member
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the ensure_project auto-admin tests")


def _unique_org() -> str:
    return f"org_{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def db_session():
    engine = create_engine(DB_URL)
    conn = engine.connect()
    created_org_ids: set[str] = set()
    conn.created_org_ids = created_org_ids  # type: ignore[attr-defined]
    try:
        yield conn
    finally:
        try:
            for org_id in created_org_ids:
                conn.execute(
                    text(
                        "DELETE FROM project_members WHERE project_id IN "
                        "(SELECT id FROM projects WHERE org_id = :o)"
                    ),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM projects WHERE org_id = :o"), {"o": org_id}
                )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _track(db_session, org_id: str) -> str:
    db_session.created_org_ids.add(org_id)
    return org_id


def _member_role(db_session, *, project_id: str, user_id: str) -> str | None:
    return db_session.execute(
        text("SELECT role FROM project_members WHERE project_id = :p AND user_id = :u"),
        {"p": project_id, "u": user_id},
    ).scalar()


def _created_by(db_session, project_id: str) -> str | None:
    return db_session.execute(
        text("SELECT created_by FROM projects WHERE id = :p"), {"p": project_id}
    ).scalar()


PRIYA = "11111111-2222-3333-4444-555555555555"
DEV = "99999999-8888-7777-6666-555555555555"


def test_a_fresh_project_seats_its_creator_as_admin(db_session):
    """The whole point: an auto-created project must not start memberless."""
    org_id = _track(db_session, _unique_org())
    remote = f"git@example.com:acme/{uuid.uuid4().hex[:8]}.git"

    project_id = ensure_project(
        db_session,
        org_id=org_id,
        display_name="fresh",
        git_remote=remote,
        created_by=PRIYA,
    )

    assert _created_by(db_session, project_id) == PRIYA
    assert _member_role(db_session, project_id=project_id, user_id=PRIYA) == ROLE_ADMIN
    assert is_member(db_session, org_id=org_id, project_id=project_id, user_id=PRIYA)


def test_an_unattributed_write_creates_no_membership(db_session):
    """No user id -> no admin, no invented owner. created_by stays NULL."""
    org_id = _track(db_session, _unique_org())
    remote = f"git@example.com:acme/{uuid.uuid4().hex[:8]}.git"

    project_id = ensure_project(
        db_session,
        org_id=org_id,
        display_name="unattributed",
        git_remote=remote,
        created_by=None,
    )

    assert _created_by(db_session, project_id) is None
    count = db_session.execute(
        text("SELECT COUNT(*) FROM project_members WHERE project_id = :p"),
        {"p": project_id},
    ).scalar()
    assert count == 0


def test_a_second_writer_in_the_same_repo_does_not_become_a_member(db_session):
    """The ON CONFLICT (hit) path must never seat a member — only the INSERT
    (miss) path does. A second person writing into an already-auto-created
    repo does not silently gain admin access to it."""
    org_id = _track(db_session, _unique_org())
    remote = f"git@example.com:acme/{uuid.uuid4().hex[:8]}.git"

    first_id = ensure_project(
        db_session,
        org_id=org_id,
        display_name="repo",
        git_remote=remote,
        created_by=PRIYA,
    )
    second_id = ensure_project(
        db_session,
        org_id=org_id,
        display_name="repo",
        git_remote=remote,
        created_by=DEV,
    )

    assert first_id == second_id
    assert _member_role(db_session, project_id=first_id, user_id=DEV) is None
    assert not is_member(db_session, org_id=org_id, project_id=first_id, user_id=DEV)
    # PRIYA, the original creator, is unaffected by the second call.
    assert _member_role(db_session, project_id=first_id, user_id=PRIYA) == ROLE_ADMIN


def test_a_second_writer_never_overwrites_created_by(db_session):
    """First writer wins: created_by is set on INSERT only."""
    org_id = _track(db_session, _unique_org())
    remote = f"git@example.com:acme/{uuid.uuid4().hex[:8]}.git"

    ensure_project(
        db_session,
        org_id=org_id,
        display_name="repo",
        git_remote=remote,
        created_by=PRIYA,
    )
    project_id = ensure_project(
        db_session,
        org_id=org_id,
        display_name="repo",
        git_remote=remote,
        created_by=DEV,
    )

    assert _created_by(db_session, project_id) == PRIYA


def test_calling_ensure_project_twice_for_a_brand_new_project_id_is_safe(db_session):
    """The same creator hitting the same repo twice must not error or
    duplicate the admin row — ON CONFLICT DO NOTHING on the membership
    insert, exercised on the very first (insert) call followed immediately
    by a hit-path call from the identical user."""
    org_id = _track(db_session, _unique_org())
    remote = f"git@example.com:acme/{uuid.uuid4().hex[:8]}.git"

    ensure_project(
        db_session,
        org_id=org_id,
        display_name="repo",
        git_remote=remote,
        created_by=PRIYA,
    )
    project_id = ensure_project(
        db_session,
        org_id=org_id,
        display_name="repo",
        git_remote=remote,
        created_by=PRIYA,
    )

    count = db_session.execute(
        text(
            "SELECT COUNT(*) FROM project_members WHERE project_id = :p AND user_id = :u"
        ),
        {"p": project_id, "u": PRIYA},
    ).scalar()
    assert count == 1


def test_path_only_projects_get_the_same_treatment_as_remote_backed_ones(db_session):
    """The auto-admin seat does not depend on git_remote vs repo_root_path —
    the ruling rejected treating those differently (§4 of the design spec)."""
    org_id = _track(db_session, _unique_org())
    path = f"/Users/dev/scratch/{uuid.uuid4().hex[:8]}"

    project_id = ensure_project(
        db_session,
        org_id=org_id,
        display_name="local",
        repo_root_path=path,
        created_by=PRIYA,
    )

    assert _created_by(db_session, project_id) == PRIYA
    assert _member_role(db_session, project_id=project_id, user_id=PRIYA) == ROLE_ADMIN
