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
    # Names are matched case-insensitively; the exact keys are not.
    assert "lower(display_name) = lower(:ident)" in fallback_sql
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
    # A human types "Klio" for a project stored as "klio"; case is not a
    # distinction any caller means to draw on a display name.
    assert "lower(display_name) = lower(:ident)" in sql
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


# ---------------------------------------------------------------------------
# find_project_id — path resolution and deterministic ranking
#
# ensure_project dedups on git_remote when present and repo_root_path
# otherwise, so a project created outside a git repo is keyed by its
# directory. The read path used to match only git_remote/display_name, which
# made those projects unresolvable — the caller silently fell back to org-wide
# recall. In production two thirds of projects are path-only.
#
# display_name is the last path segment, so one org holds several projects
# called "web" or "app". The old query ended in a bare LIMIT 1 with no
# ORDER BY, so the winner was arbitrary and could change between calls —
# resolving to the WRONG project, which is worse than not resolving.
# ---------------------------------------------------------------------------


def test_find_project_id_matches_repo_root_path():
    """A non-git project must be resolvable by the directory it was keyed on."""
    proj_id = uuid.uuid4()
    db_session = MagicMock()
    db_session.execute.return_value = _result_with_row((proj_id,))

    result = find_project_id(
        db_session, org_id="org-1", identifier="/Users/dev/Me/Agent Memory"
    )

    assert result == str(proj_id)
    sql = _executed_sql(db_session)
    assert "repo_root_path = :ident" in sql
    params = _executed_params(db_session)
    assert params == {"org_id": "org-1", "ident": "/Users/dev/Me/Agent Memory"}
    db_session.commit.assert_not_called()


def test_find_project_id_ranks_exact_keys_above_display_name():
    """git_remote and repo_root_path are unique per org; display_name is not.

    A display_name collision must never outrank an exact key match, so the
    ordering is by specificity first.
    """
    db_session = MagicMock()
    db_session.execute.return_value = _result_with_row((uuid.uuid4(),))

    find_project_id(db_session, org_id="org-1", identifier="web")

    sql = _executed_sql(db_session)
    remote_rank = sql.index("WHEN git_remote = :ident THEN 0")
    path_rank = sql.index("WHEN repo_root_path = :ident THEN 1")
    assert remote_rank < path_rank, "git_remote must rank above repo_root_path"
    assert "ELSE 2" in sql, "display_name must rank last"


def test_find_project_id_is_deterministic_under_display_name_collision():
    """Ties must break on a stable, total ordering — never planner order.

    Without this the same identifier can resolve to a different project
    between two calls, so memories file and recall against whichever row the
    planner happened to return.
    """
    db_session = MagicMock()
    db_session.execute.return_value = _result_with_row((uuid.uuid4(),))

    find_project_id(db_session, org_id="org-1", identifier="web")

    sql = _executed_sql(db_session)
    assert "ORDER BY" in sql, "a bare LIMIT 1 is non-deterministic"
    order_clause = sql[sql.index("ORDER BY") :]
    assert "last_seen_at DESC" in order_clause
    # id is the final tie-break: two projects can share a last_seen_at.
    assert "id ASC" in order_clause
    assert order_clause.index("last_seen_at DESC") < order_clause.index("id ASC")


def test_find_project_id_still_returns_none_for_unknown_identifier():
    """Fail-open behaviour is unchanged: unknown resolves to None, not a guess."""
    db_session = MagicMock()
    db_session.execute.return_value = _result_with_row(None)

    assert (
        find_project_id(db_session, org_id="org-1", identifier="/no/such/dir") is None
    )
    db_session.commit.assert_not_called()
