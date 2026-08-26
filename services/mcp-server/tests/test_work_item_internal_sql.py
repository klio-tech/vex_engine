"""The work-item internal endpoint's SQL, executed against a real database.

A unit test with a mocked session proves the Python around these queries and
NOTHING about the queries themselves. The first version of
``_WORK_ITEM_MEMORIES_SQL`` compared ``user_id`` (a ``character varying``
column) against ``CAST(:user_id AS uuid)`` and raised
``operator does not exist: character varying = uuid`` the first time it ran
in production — invisible to every mock, obvious to Postgres.

So these tests do the one thing that catches that class of bug: they hand
the statements to a real server and check it accepts them. Both the
``user_id IS NULL`` and the ``user_id`` present shapes run, because the
NULL branch is the one a mock is most likely to be the only exercise of.
"""

import uuid

import pytest
from sqlalchemy import create_engine, text

from app.internal_api import (
    _WORK_ITEM_COUNTS_SQL,
    _WORK_ITEM_MEMORIES_SQL,
)
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the work-item internal SQL tests")


@pytest.fixture()
def db():
    engine = create_engine(DB_URL)
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()
        engine.dispose()


ORG = "org_wi_sql_probe"


def test_the_memories_query_runs_with_no_user(db):
    """The anonymous read: no private tier, and no type error reaching it."""
    rows = db.execute(
        _WORK_ITEM_MEMORIES_SQL,
        {
            "org_id": ORG,
            "work_item_id": str(uuid.uuid4()),
            "user_id": None,
            "limit": 5,
        },
    ).fetchall()
    assert rows == []


def test_the_memories_query_runs_with_a_user(db):
    """The attributed read: the private tier's comparison must type-check."""
    rows = db.execute(
        _WORK_ITEM_MEMORIES_SQL,
        {
            "org_id": ORG,
            "work_item_id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "limit": 5,
        },
    ).fetchall()
    assert rows == []


def test_the_counts_query_runs(db):
    rows = db.execute(
        _WORK_ITEM_COUNTS_SQL,
        {"org_id": ORG, "work_item_id": str(uuid.uuid4())},
    ).fetchall()
    assert rows == []


def test_a_private_row_reaches_only_its_owner(db):
    """The predicate is a boundary, so prove it both ways, not just that it runs."""
    work_item_id = str(uuid.uuid4())
    owner = str(uuid.uuid4())
    stranger = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO session_memories
                (id, session_id, org_id, agent_id, memory_type, content,
                 scope, status, user_id, work_item_id)
            VALUES (:id, 'probe', :org, 'agent', 'note', 'a private note',
                    'private', 'active', :owner, CAST(:wi AS uuid))
            """
        ),
        {"id": str(uuid.uuid4()), "org": ORG, "owner": owner, "wi": work_item_id},
    )

    def _read(user_id):
        return db.execute(
            _WORK_ITEM_MEMORIES_SQL,
            {
                "org_id": ORG,
                "work_item_id": work_item_id,
                "user_id": user_id,
                "limit": 5,
            },
        ).fetchall()

    assert len(_read(owner)) == 1, "the owner must see their own private row"
    assert _read(stranger) == [], "nobody else may"
    assert _read(None) == [], "and an unattributed read sees none of it"
    # the fixture rolls back, so nothing is left behind
