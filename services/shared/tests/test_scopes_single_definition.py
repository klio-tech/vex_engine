"""The scope value set has one definition, and the database agrees with it.

Before this module, the set was written out independently in six places:
migration 031's VALID_SCOPES, the CHECK constraint it installs, and bare
tuples in memory.py (x2), brain.py, graph_store.py and promotion.py. Nothing
tied them together, so a new scope could be accepted by one and dropped by
another.
"""

import re

import pytest
from sqlalchemy import create_engine, text

from shared.scopes import (
    GRAPH_SCOPES,
    MEMORY_SCOPES,
    PROMOTABLE_SCOPES,
    RECALL_MODES,
    RECALL_SCOPES,
    WRITABLE_SCOPES,
)
from shared.testing import database_url, requires_db

DB_URL = database_url()


@pytest.fixture()
def db_session():
    """A read-only connection to the dev database.

    ``services/shared/tests`` has no shared session fixture -- each DB test
    declares its own (see ``test_brain.py``) -- so this follows that pattern.
    Nothing here writes, so there is nothing to clean up.
    """
    engine = create_engine(DB_URL)
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()
        engine.dispose()


@requires_db("the scope CHECK-constraint test")
def test_memory_scopes_match_the_database_check_constraint(db_session):
    row = db_session.execute(
        text(
            """
            -- aliased ``def_`` (not ``def``) so the row attribute is reachable:
            -- ``def`` is a Python keyword and SQLAlchemy exposes the
            -- column under its literal name.
            SELECT pg_get_constraintdef(oid) AS def_
            FROM pg_constraint
            WHERE conname = 'ck_session_memories_scope'
            """
        )
    ).first()
    assert row is not None, "ck_session_memories_scope is missing"
    in_db = set(re.findall(r"'([a-z]+)'::character varying", row.def_)) or set(
        re.findall(r"'([a-z]+)'", row.def_)
    )
    assert in_db == set(MEMORY_SCOPES)


def test_the_derived_sets_are_subsets_of_the_whole():
    for name, subset in (
        ("WRITABLE_SCOPES", WRITABLE_SCOPES),
        ("RECALL_SCOPES", RECALL_SCOPES),
        ("PROMOTABLE_SCOPES", PROMOTABLE_SCOPES),
        ("GRAPH_SCOPES", GRAPH_SCOPES),
    ):
        assert subset <= MEMORY_SCOPES, f"{name} has values the column rejects"


def test_session_is_not_writable_through_the_scoped_path():
    # write_memory hardcodes scope='session' on INSERT and write_scoped_memory
    # then UPDATEs it to the real scope. 'session' is therefore never a value a
    # caller may request.
    assert "session" not in WRITABLE_SCOPES


def test_private_is_not_promotable():
    # promote_memory moves a row OUT of private; private is the source, never a
    # destination.
    assert "private" not in PROMOTABLE_SCOPES


def test_recall_modes_extends_recall_scopes_without_widening_the_column():
    """`agent_only` is a READ MODE, not a row value.

    The set above must keep satisfying "these are values the column accepts";
    a read mode that leaked into it would be accepted on write and invisible
    on read, which is the whole failure this module prevents. So the modes
    live in their own set, and every mode that is NOT a column value must be
    absent from every column-valued set.
    """
    assert RECALL_SCOPES <= RECALL_MODES
    read_only_modes = RECALL_MODES - RECALL_SCOPES
    assert read_only_modes == {"agent_only"}
    for mode in read_only_modes:
        assert mode not in MEMORY_SCOPES, f"{mode} must not be a column value"
        assert mode not in WRITABLE_SCOPES, f"{mode} must not be writable"
        assert mode not in PROMOTABLE_SCOPES
        assert mode not in GRAPH_SCOPES
