"""Unit tests for the read-only brain lookup helpers.

:func:`shared.brain.find_space_id` and :func:`shared.brain.find_project_id`
resolve user-facing names/identifiers to ids on the *recall* (read) path.
Unlike the ``ensure_*`` helpers they MUST NEVER create a row — they only
``SELECT`` and return an id (or ``None``). These tests use a ``MagicMock``
db_session so no real Postgres is touched, asserting both the returned value
and that the executed SQL is tenant-scoped (filtered by ``org_id``) and never
issues an INSERT/commit.
"""

import uuid
from unittest.mock import MagicMock

from shared.brain import find_project_id, find_space_id


def _result_with_row(row):
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
# find_space_id
# ---------------------------------------------------------------------------


def test_find_space_id_returns_id_when_row_exists():
    """A matching (org_id, slug) row yields its id as a string."""
    space_id = uuid.uuid4()
    db_session = MagicMock()
    db_session.execute.return_value = _result_with_row((space_id,))

    result = find_space_id(db_session, org_id="org-1", slug="research")

    assert result == str(space_id)
    sql = _executed_sql(db_session)
    assert "FROM spaces" in sql
    # Tenant-scoped AND deduped on slug.
    assert "org_id = :org_id" in sql
    assert "slug = :slug" in sql
    params = _executed_params(db_session)
    assert params == {"org_id": "org-1", "slug": "research"}
    # Read-only: never writes.
    db_session.commit.assert_not_called()


def test_find_space_id_returns_none_when_no_row():
    """No matching row yields None (recall never creates a space)."""
    db_session = MagicMock()
    db_session.execute.return_value = _result_with_row(None)

    result = find_space_id(db_session, org_id="org-1", slug="missing")

    assert result is None
    db_session.commit.assert_not_called()


# ---------------------------------------------------------------------------
# find_project_id
# ---------------------------------------------------------------------------


def test_find_project_id_uuid_identifier_hits_id_query():
    """A UUID identifier resolves via the id-equality query."""
    proj_id = uuid.uuid4()
    db_session = MagicMock()
    db_session.execute.return_value = _result_with_row((proj_id,))

    result = find_project_id(db_session, org_id="org-1", identifier=str(proj_id))

    assert result == str(proj_id)
    sql = _executed_sql(db_session)
    assert "FROM projects" in sql
    assert "org_id = :org_id" in sql
    assert "id = :id" in sql
    # The id query never references git_remote/display_name.
    assert "git_remote" not in sql
    params = _executed_params(db_session)
    assert params == {"org_id": "org-1", "id": str(proj_id)}
    db_session.commit.assert_not_called()


def test_find_project_id_uuid_miss_falls_back_to_name_query():
    """A UUID that matches no id row falls back to the git_remote/name query."""
    valid_uuid = str(uuid.uuid4())
    proj_id = uuid.uuid4()
    db_session = MagicMock()
    # First call (id query) misses, second call (name query) hits.
    db_session.execute.side_effect = [
        _result_with_row(None),
        _result_with_row((proj_id,)),
    ]

    result = find_project_id(db_session, org_id="org-1", identifier=valid_uuid)

    assert result == str(proj_id)
    assert db_session.execute.call_count == 2
    fallback_sql = str(db_session.execute.call_args_list[1].args[0])
    assert "git_remote = :ident" in fallback_sql
    assert "display_name = :ident" in fallback_sql
    fallback_params = db_session.execute.call_args_list[1].args[1]
    assert fallback_params == {"org_id": "org-1", "ident": valid_uuid}
    db_session.commit.assert_not_called()


def test_find_project_id_non_uuid_hits_name_query():
    """A non-UUID identifier goes straight to the git_remote/display_name query."""
    proj_id = uuid.uuid4()
    db_session = MagicMock()
    db_session.execute.return_value = _result_with_row((proj_id,))

    result = find_project_id(
        db_session, org_id="org-1", identifier="git@github.com:acme/app.git"
    )

    assert result == str(proj_id)
    sql = _executed_sql(db_session)
    assert "FROM projects" in sql
    assert "org_id = :org_id" in sql
    assert "git_remote = :ident" in sql
    assert "display_name = :ident" in sql
    params = _executed_params(db_session)
    assert params == {"org_id": "org-1", "ident": "git@github.com:acme/app.git"}
    db_session.commit.assert_not_called()


def test_find_project_id_returns_none_when_nothing_matches():
    """A non-UUID identifier with no matching row yields None."""
    db_session = MagicMock()
    db_session.execute.return_value = _result_with_row(None)

    result = find_project_id(db_session, org_id="org-1", identifier="nope")

    assert result is None
    db_session.commit.assert_not_called()


def test_find_project_id_uuid_miss_then_name_miss_returns_none():
    """A UUID matching neither id nor name yields None after both queries."""
    valid_uuid = str(uuid.uuid4())
    db_session = MagicMock()
    db_session.execute.side_effect = [
        _result_with_row(None),
        _result_with_row(None),
    ]

    result = find_project_id(db_session, org_id="org-1", identifier=valid_uuid)

    assert result is None
    assert db_session.execute.call_count == 2
    db_session.commit.assert_not_called()
