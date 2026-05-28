"""Unit tests for the read-only brain space-listing helpers.

:func:`shared.brain.list_spaces` and :func:`shared.brain.get_space` enumerate
and look up spaces on the *management* (read) path. Like the other brain read
helpers (:func:`find_space_id` / :func:`find_project_id`) they MUST NEVER create
a row — they only ``SELECT`` and project rows to dicts. These tests use a
``MagicMock`` db_session so no real Postgres is touched, asserting both the
returned shape and that the executed SQL is tenant-scoped (filtered by
``org_id``) and never issues an INSERT/commit.

The dict shape mirrors the helper contract: ``id`` (str), ``name``, ``slug``,
``embedding_model`` (str|None), ``created_at`` (kept as a ``datetime``; the MCP
tool layer serializes it). ``id`` is stringified because ``spaces.id`` is a
UUID column.
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from shared.brain import get_space, list_spaces


def _row(*, id, name, slug, embedding_model, created_at):
    """Build a fake DB row exposing attribute access (like a SQLAlchemy Row)."""
    return SimpleNamespace(
        id=id,
        name=name,
        slug=slug,
        embedding_model=embedding_model,
        created_at=created_at,
    )


def _result_with_rows(rows):
    """Build a fake execute() result whose fetchall() yields *rows*."""
    result = MagicMock()
    result.fetchall.return_value = rows
    return result


def _result_with_one(row):
    """Build a fake execute() result whose fetchone() yields *row* (or None)."""
    result = MagicMock()
    result.fetchone.return_value = row
    return result


def _executed_sql(db_session) -> str:
    """Return the SQL text of the single execute() call as a string."""
    assert db_session.execute.call_count == 1
    return str(db_session.execute.call_args.args[0])


def _executed_params(db_session) -> dict:
    """Return the bound params dict of the single execute() call."""
    return db_session.execute.call_args.args[1]


# ---------------------------------------------------------------------------
# list_spaces
# ---------------------------------------------------------------------------


def test_list_spaces_maps_rows_to_dicts_with_str_id():
    """Each org-owned row is projected to the dict shape; id is stringified and
    created_at stays a datetime."""
    id1 = uuid.uuid4()
    id2 = uuid.uuid4()
    created1 = datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc)
    created2 = datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc)
    rows = [
        _row(
            id=id1,
            name="Research",
            slug="research",
            embedding_model="text-embedding-3-small",
            created_at=created1,
        ),
        _row(
            id=id2,
            name="Personal",
            slug="personal",
            embedding_model=None,
            created_at=created2,
        ),
    ]
    db_session = MagicMock()
    db_session.execute.return_value = _result_with_rows(rows)

    result = list_spaces(db_session, org_id="org-1")

    assert result == [
        {
            "id": str(id1),
            "name": "Research",
            "slug": "research",
            "embedding_model": "text-embedding-3-small",
            "created_at": created1,
        },
        {
            "id": str(id2),
            "name": "Personal",
            "slug": "personal",
            "embedding_model": None,
            "created_at": created2,
        },
    ]
    # id is a string; created_at is left as a datetime for the tool to serialize.
    assert isinstance(result[0]["id"], str)
    assert isinstance(result[0]["created_at"], datetime)


def test_list_spaces_is_tenant_scoped_and_newest_first():
    """The query filters by :org_id and orders by created_at DESC."""
    db_session = MagicMock()
    db_session.execute.return_value = _result_with_rows([])

    list_spaces(db_session, org_id="org-secure")

    sql = _executed_sql(db_session)
    assert "FROM spaces" in sql
    assert "org_id = :org_id" in sql
    # Newest first.
    assert "ORDER BY created_at DESC" in sql
    params = _executed_params(db_session)
    assert params == {"org_id": "org-secure"}
    # Read-only: never writes.
    db_session.commit.assert_not_called()


def test_list_spaces_empty_returns_empty_list():
    """No rows yields an empty list (not None)."""
    db_session = MagicMock()
    db_session.execute.return_value = _result_with_rows([])

    result = list_spaces(db_session, org_id="org-1")

    assert result == []
    db_session.commit.assert_not_called()


# ---------------------------------------------------------------------------
# get_space
# ---------------------------------------------------------------------------


def test_get_space_returns_dict_with_str_id_when_found():
    """A matching (org_id, slug) row is projected to the dict shape with a
    stringified id."""
    space_id = uuid.uuid4()
    created = datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc)
    row = _row(
        id=space_id,
        name="Research",
        slug="research",
        embedding_model="text-embedding-3-small",
        created_at=created,
    )
    db_session = MagicMock()
    db_session.execute.return_value = _result_with_one(row)

    result = get_space(db_session, org_id="org-1", slug="research")

    assert result == {
        "id": str(space_id),
        "name": "Research",
        "slug": "research",
        "embedding_model": "text-embedding-3-small",
        "created_at": created,
    }
    assert isinstance(result["id"], str)
    sql = _executed_sql(db_session)
    assert "FROM spaces" in sql
    # Tenant-scoped AND slug-filtered.
    assert "org_id = :org_id" in sql
    assert "slug = :slug" in sql
    params = _executed_params(db_session)
    assert params == {"org_id": "org-1", "slug": "research"}
    # Read-only: never writes.
    db_session.commit.assert_not_called()


def test_get_space_returns_none_when_no_row():
    """No matching row yields None (management read never creates a space)."""
    db_session = MagicMock()
    db_session.execute.return_value = _result_with_one(None)

    result = get_space(db_session, org_id="org-1", slug="missing")

    assert result is None
    sql = _executed_sql(db_session)
    assert "org_id = :org_id" in sql
    assert "slug = :slug" in sql
    db_session.commit.assert_not_called()
