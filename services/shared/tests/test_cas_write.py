"""Mocked-DB unit tests for ``supersede_with_revision``'s control flow.

These prove the WIRING — which branch runs, what gets returned, what the
guard SQL says — with the expensive/impure pieces (redaction, embedding, the
full ``_apply_supersession_for_cas`` insert-and-flip, the chain-follow query)
patched out as seams. The seams themselves get their own tests:

* the authorization ladder and the superseded-row guard predicate (H1/H2)
  are exercised against REAL Postgres in ``test_cas_write_db.py`` — a mock
  cannot catch a missing ``WHERE`` clause, only a database that actually
  evaluates one can;
* the two-thread race is also there, for the same reason.
"""

from unittest.mock import MagicMock, patch

from shared import memory as memory_mod

_MEMORY_ID = "11111111-1111-1111-1111-111111111111"
# revision is a UUID column; base_revision is now UUID-validated (N3) before
# any DB round trip, so every fixture value below has to actually parse.
_BASE_REVISION = "22222222-2222-2222-2222-222222222222"
_CURRENT_REVISION = "33333333-3333-3333-3333-333333333333"


def _org_scoped_row():
    """A row that clears the authorization ladder trivially (scope='org')."""
    row = MagicMock()
    row.scope = "org"
    row.agent_id = "agent-x"
    row.user_id = None
    row.project_id = None
    return row


def _fetchone_result(value):
    result = MagicMock()
    result.fetchone.return_value = value
    return result


def test_stale_base_revision_returns_conflict_payload():
    db = MagicMock()
    guard = MagicMock(rowcount=0)  # CAS guard missed
    db.execute.side_effect = [_fetchone_result(_org_scoped_row()), guard]

    with (
        patch.object(
            memory_mod,
            "_prepare_cas_write",
            return_value=("mine", {}, [0.1] * 4, False, []),
        ),
        patch.object(
            memory_mod,
            "_current_belief_in_chain",
            return_value={
                "id": _MEMORY_ID,
                "content": "newer text",
                "revision": _CURRENT_REVISION,
                "user_id": "u2",
            },
        ) as chain_mock,
    ):
        result = memory_mod.supersede_with_revision(
            db,
            org_id="o",
            memory_id=_MEMORY_ID,
            base_revision=_BASE_REVISION,
            new_content="mine",
            agent_id="agent-x",
            memory_type="fact",
            session_id="s1",
        )

    assert result["conflict"] is True
    assert result["stored"] is False
    assert result["current"]["revision"] == _CURRENT_REVISION
    # N1: the chain-follow's own authorization runs against the caller's
    # principals too, so user_id/agent_id must reach it.
    chain_mock.assert_called_once_with(
        db,
        org_id="o",
        memory_id=_MEMORY_ID,
        user_id=None,
        agent_id="agent-x",
    )

    guard_sql = " ".join(str(db.execute.call_args_list[1].args[0]).split())
    assert "revision = :base_revision" in guard_sql
    assert "org_id = :org_id" in guard_sql
    # H2: a superseded row's own (frozen) revision must not be a live token.
    assert "status = 'active'" in guard_sql
    assert "superseded_by IS NULL" in guard_sql


def test_current_base_revision_proceeds_to_supersede():
    db = MagicMock()
    guard = MagicMock(rowcount=1)
    db.execute.side_effect = [_fetchone_result(_org_scoped_row()), guard]

    with (
        patch.object(
            memory_mod,
            "_prepare_cas_write",
            return_value=("mine", {}, [0.1] * 4, False, []),
        ),
        patch.object(
            memory_mod,
            "_apply_supersession_for_cas",
            return_value={"new_id": "m2", "new_revision": _CURRENT_REVISION},
        ) as ap,
    ):
        result = memory_mod.supersede_with_revision(
            db,
            org_id="o",
            memory_id=_MEMORY_ID,
            base_revision=_BASE_REVISION,
            new_content="mine",
            agent_id="agent-x",
            memory_type="fact",
            session_id="s1",
        )

    assert result["conflict"] is False
    assert result["stored"] is True
    assert result["new_id"] == "m2"
    assert result["new_revision"] == _CURRENT_REVISION
    assert ap.called
    # The redacted/embedded content from phase 3 is what gets forwarded, not
    # the raw new_content string.
    apkw = ap.call_args.kwargs
    assert apkw["content"] == "mine"
    assert apkw["embedding"] == [0.1] * 4


def test_unknown_memory_id_returns_not_found_without_touching_the_guard():
    """No row at all: the SAME refusal an unauthorized caller gets (H1) — the
    guard UPDATE is never reached, so no lock is taken on nothing."""
    db = MagicMock()
    db.execute.side_effect = [_fetchone_result(None)]

    result = memory_mod.supersede_with_revision(
        db,
        org_id="o",
        memory_id=_MEMORY_ID,
        base_revision=_BASE_REVISION,
        new_content="mine",
        agent_id="agent-x",
        memory_type="fact",
        session_id="s1",
    )

    assert result == {"conflict": False, "stored": False, "error": "not_found"}
    assert db.execute.call_count == 1


def test_invalid_memory_id_is_refused_before_any_query():
    """M5: a non-UUID id is refused before the DB is touched at all."""
    db = MagicMock()

    result = memory_mod.supersede_with_revision(
        db,
        org_id="o",
        memory_id="not-a-uuid",
        base_revision=_BASE_REVISION,
        new_content="mine",
        agent_id="agent-x",
        memory_type="fact",
        session_id="s1",
    )

    assert result == {"conflict": False, "stored": False, "error": "invalid_id"}
    db.execute.assert_not_called()


def test_invalid_base_revision_is_refused_before_any_query():
    """N3: a non-UUID base_revision is refused before any DB round trip or
    embedding spend, same error code as an invalid memory_id."""
    db = MagicMock()

    result = memory_mod.supersede_with_revision(
        db,
        org_id="o",
        memory_id=_MEMORY_ID,
        base_revision="not-a-uuid-either",
        new_content="mine",
        agent_id="agent-x",
        memory_type="fact",
        session_id="s1",
    )

    assert result == {"conflict": False, "stored": False, "error": "invalid_id"}
    db.execute.assert_not_called()


def test_private_row_non_owner_gets_not_found_before_the_guard():
    """H1: a caller who is not the private row's owner never reaches the
    guard — same indistinguishable not_found retract_memory gives."""
    db = MagicMock()
    row = MagicMock(
        scope="private", agent_id="agent-x", user_id="owner-1", project_id=None
    )
    db.execute.side_effect = [_fetchone_result(row)]

    result = memory_mod.supersede_with_revision(
        db,
        org_id="o",
        memory_id=_MEMORY_ID,
        base_revision=_BASE_REVISION,
        new_content="mine",
        user_id="someone-else",
        agent_id="agent-x",
        memory_type="fact",
        session_id="s1",
    )

    assert result == {"conflict": False, "stored": False, "error": "not_found"}
    assert db.execute.call_count == 1


def test_agent_scope_wrong_agent_gets_forbidden_before_the_guard():
    """H1: the agent-ownership mismatch answers forbidden, not not_found —
    existence was never the secret for an agent-scoped row."""
    db = MagicMock()
    row = MagicMock(
        scope="agent", agent_id="owning-agent", user_id=None, project_id=None
    )
    db.execute.side_effect = [_fetchone_result(row)]

    result = memory_mod.supersede_with_revision(
        db,
        org_id="o",
        memory_id=_MEMORY_ID,
        base_revision=_BASE_REVISION,
        new_content="mine",
        agent_id="other-agent",
        memory_type="fact",
        session_id="s1",
    )

    assert result == {"conflict": False, "stored": False, "error": "forbidden"}
    assert db.execute.call_count == 1


def test_authorized_read_is_rolled_back_before_the_embed():
    """N2: the authz SELECT's transaction is closed (rolled back) before
    phase 3's network call, so the pooled connection is not pinned across it
    under a transaction-mode pooler."""
    db = MagicMock()
    guard = MagicMock(rowcount=1)
    db.execute.side_effect = [_fetchone_result(_org_scoped_row()), guard]

    call_order: list[str] = []
    db.rollback.side_effect = lambda: call_order.append("rollback")

    def _prepare(*args, **kwargs):
        call_order.append("prepare")
        return ("mine", {}, [0.1] * 4, False, [])

    with (
        patch.object(memory_mod, "_prepare_cas_write", side_effect=_prepare),
        patch.object(
            memory_mod,
            "_apply_supersession_for_cas",
            return_value={"new_id": "m2", "new_revision": _CURRENT_REVISION},
        ),
    ):
        memory_mod.supersede_with_revision(
            db,
            org_id="o",
            memory_id=_MEMORY_ID,
            base_revision=_BASE_REVISION,
            new_content="mine",
            agent_id="agent-x",
            memory_type="fact",
            session_id="s1",
        )

    assert call_order == ["rollback", "prepare"]


def test_conflict_increments_the_write_conflicts_counter():
    """L8: the stale-revision path increments klio_write_conflicts_total."""
    from shared import metrics

    metrics.reset_for_test()
    db = MagicMock()
    guard = MagicMock(rowcount=0)
    db.execute.side_effect = [_fetchone_result(_org_scoped_row()), guard]

    with (
        patch.object(
            memory_mod,
            "_prepare_cas_write",
            return_value=("mine", {}, [0.1] * 4, False, []),
        ),
        patch.object(memory_mod, "_current_belief_in_chain", return_value=None),
    ):
        memory_mod.supersede_with_revision(
            db,
            org_id="o",
            memory_id=_MEMORY_ID,
            base_revision=_BASE_REVISION,
            new_content="mine",
            agent_id="agent-x",
            memory_type="fact",
            session_id="s1",
        )

    assert metrics.WRITE_CONFLICTS.value() == 1.0
    metrics.reset_for_test()
