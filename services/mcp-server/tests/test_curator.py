"""Unit tests for the org-scoped Klio brain curator (``app.curator``).

Mock-based, mirroring ``shared/tests/test_brain_write.py`` and
``mcp-server/tests/test_recall_tool.py``: no real Postgres, no real LLM. The
LLM helpers (``generate_summary`` / ``extract_facts``) and the write primitive
(``write_scoped_memory``) are patched at the ``app.curator`` module boundary so
the synthesis flow can be asserted in isolation.

The DB seam is a ``MagicMock`` whose ``.execute(...)`` is fed a ``side_effect``
list, one entry per SQL statement the tick issues in order:

    1. cursor SELECT  -> a result whose ``.fetchone()`` yields a row exposing
       ``last_processed_at`` + ``last_processed_id`` (the composite keyset
       cursor), or ``None`` for the first-ever tick
    2. new-rows SELECT -> a result whose ``.fetchall()`` yields the raw rows
    3. cursor upsert   -> an opaque result (return value unused)

Each fake row is a ``SimpleNamespace`` exposing ``id`` / ``content`` /
``created_at`` (attribute access, like a SQLAlchemy ``Row``).
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.curator import curator_tick, run_curator_once


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


def _row(
    idx: int,
    content: str,
    *,
    created_at: datetime | None = None,
    id: str | None = None,
    space_id: str | None = None,
    project_id: str | None = None,
):
    """A raw session_memories row as the new-rows SELECT returns it."""
    return SimpleNamespace(
        id=id or f"mem-{idx}",
        content=content,
        created_at=created_at or (_BASE + timedelta(minutes=idx)),
        space_id=space_id,
        project_id=project_id,
    )


def _cursor_result(last_processed_at, last_processed_id=None):
    """Fake result for the cursor SELECT.

    The cursor read is now keyset (composite ``(created_at, id)``), so the
    result must surface BOTH columns: ``.fetchone()`` -> a row exposing
    ``last_processed_at`` / ``last_processed_id`` (or ``None`` for the no-row,
    first-ever-tick case).
    """
    res = MagicMock()
    if last_processed_at is None and last_processed_id is None:
        res.fetchone.return_value = None
    else:
        res.fetchone.return_value = SimpleNamespace(
            last_processed_at=last_processed_at,
            last_processed_id=last_processed_id,
        )
    return res


def _rows_result(rows):
    """Fake result for the new-rows SELECT: ``.fetchall()`` -> rows."""
    res = MagicMock()
    res.fetchall.return_value = rows
    return res


def _make_session(*, cursor=None, cursor_id=None, rows=None):
    """A MagicMock db_session wired with an execute side_effect sequence.

    Sequence: cursor SELECT, new-rows SELECT, then any number of subsequent
    statements (the upsert) return a generic MagicMock. ``cursor`` /
    ``cursor_id`` form the composite keyset cursor the SELECT surfaces.
    """
    session = MagicMock()
    results = [
        _cursor_result(cursor, cursor_id),
        _rows_result(rows or []),
    ]
    session.execute.side_effect = results + [MagicMock()] * 8
    return session


def _patch_curator(*, summary="a synthesized summary", facts=None):
    """Context-manager bundle patching the LLM + write seams at app.curator."""
    facts = facts if facts is not None else ["fact one", "fact two"]
    return (
        patch(
            "app.curator.generate_summary",
            new_callable=AsyncMock,
            return_value=summary,
        ),
        patch(
            "app.curator.extract_facts",
            new_callable=AsyncMock,
            return_value=facts,
        ),
        patch(
            "app.curator.write_scoped_memory",
            new_callable=AsyncMock,
            return_value={"stored": True, "deduplicated": False, "id": "new"},
        ),
    )


def _select_sql_calls(session) -> list[str]:
    """Return the stringified SQL of every db_session.execute call."""
    return [str(c.args[0]) for c in session.execute.call_args_list]


# ---------------------------------------------------------------------------
# Synthesis happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesis_happy_path_writes_summary_and_facts():
    """cursor=None + 5 raw rows -> summary written once, a fact per extracted
    fact, synthesized>=1, and the cursor upsert binds the batch max created_at."""
    rows = [_row(i, f"content {i}") for i in range(1, 6)]  # created_at minute 1..5
    session = _make_session(cursor=None, rows=rows)
    gen, ext, wsm = _patch_curator(summary="weekly digest", facts=["f1", "f2", "f3"])

    with gen as mock_gen, ext as mock_ext, wsm as mock_write:
        result = await curator_tick(session, "org-1", min_inputs=3)

    # generate_summary saw the combined (newline-joined) content of the batch.
    mock_gen.assert_awaited_once()
    combined = mock_gen.await_args.args[0]
    for i in range(1, 6):
        assert f"content {i}" in combined
    mock_ext.assert_awaited_once()

    # One summary write + one write per fact = 1 + 3 = 4 writes.
    assert mock_write.await_count == 4
    types_written = [c.kwargs["memory_type"] for c in mock_write.await_args_list]
    assert types_written.count("summary") == 1
    assert types_written.count("fact") == 3
    # The summary write carries curator provenance + the input count.
    summary_call = next(
        c for c in mock_write.await_args_list if c.kwargs["memory_type"] == "summary"
    )
    assert summary_call.kwargs["org_id"] == "org-1"
    assert summary_call.kwargs["metadata"]["source"] == "curator"
    assert summary_call.kwargs["metadata"]["input_count"] == 5

    assert result["synthesized"] >= 1
    assert result["synthesized"] == 4
    assert result["processed"] == 5

    # The cursor upsert (the statement AFTER the two SELECTs) advanced the
    # COMPOSITE cursor to the batch's max (created_at, id) — the last row, since
    # rows are ordered created_at ASC, id ASC.
    upsert_call = session.execute.call_args_list[2]
    upsert_sql = str(upsert_call.args[0])
    assert "brain_curator_state" in upsert_sql
    assert "ON CONFLICT" in upsert_sql
    assert "last_processed_id" in upsert_sql
    params = upsert_call.args[1]
    assert params["last_processed_at"] == rows[-1].created_at
    assert params["last_processed_at"] == _BASE + timedelta(minutes=5)
    assert params["last_processed_id"] == rows[-1].id
    assert params["org_id"] == "org-1"
    session.commit.assert_called()


@pytest.mark.asyncio
async def test_synthesis_returns_iso_cursor():
    """The result cursor is the ISO string of the advanced last_processed_at."""
    rows = [_row(i, f"c{i}") for i in range(1, 5)]
    session = _make_session(cursor=None, rows=rows)
    gen, ext, wsm = _patch_curator()

    with gen, ext, wsm:
        result = await curator_tick(session, "org-1", min_inputs=3)

    assert result["cursor"] == rows[-1].created_at.isoformat()


# ---------------------------------------------------------------------------
# Same-timestamp straddle regression (keyset (created_at, id) pagination)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_timestamp_straddle_does_not_skip_twin():
    """Two RAW rows A and B share the EXACT same created_at (T) with id_A < id_B.

    With batch_size=1 the LIMIT boundary splits them: tick 1 reads [A] and
    advances the cursor to the COMPOSITE (T, id_A) — NOT just T. Tick 2 must
    then bind that composite cursor and the keyset WHERE clause must keep B
    (created_at=T, id_B>id_A) eligible — i.e. the twin is NOT silently skipped.
    """
    T = _BASE
    id_a, id_b = (
        "00000000-0000-0000-0000-00000000000a",
        ("00000000-0000-0000-0000-00000000000b"),
    )
    # min_inputs=1 so a single row triggers the advance path.
    # --- tick 1: cursor empty -> read returns [A] -------------------------
    row_a = _row(1, "content A", created_at=T, id=id_a)
    session1 = _make_session(cursor=None, cursor_id=None, rows=[row_a])
    gen, ext, wsm = _patch_curator(summary="s", facts=["f1"])
    with gen, ext, wsm:
        await curator_tick(session1, "org-1", batch_size=1, min_inputs=1)

    # tick 1 advance binds BOTH composite-cursor columns to A's values.
    upsert1 = session1.execute.call_args_list[2]
    upsert1_sql = str(upsert1.args[0])
    assert "last_processed_at" in upsert1_sql
    assert "last_processed_id" in upsert1_sql
    params1 = upsert1.args[1]
    assert params1["last_processed_at"] == T
    assert params1["last_processed_id"] == id_a

    # --- tick 2: cursor now (T, id_A) -> read must keep B eligible ---------
    row_b = _row(2, "content B", created_at=T, id=id_b)
    session2 = _make_session(cursor=T, cursor_id=id_a, rows=[row_b])
    gen2, ext2, wsm2 = _patch_curator(summary="s", facts=["f1"])
    with gen2, ext2, wsm2:
        await curator_tick(session2, "org-1", batch_size=1, min_inputs=1)

    # The new-rows SELECT (2nd execute) binds the composite cursor...
    read_call = session2.execute.call_args_list[1]
    read_sql = str(read_call.args[0])
    read_params = read_call.args[1]
    assert read_params["cursor_ts"] == T
    assert read_params["cursor_id"] == id_a
    # ...and the keyset clause keeps B (same ts, greater id) returnable rather
    # than excluding it via a bare `created_at > cursor`.
    assert "created_at = :cursor_ts AND id > :cursor_id" in read_sql
    assert "created_at > :cursor_ts" in read_sql
    assert ":cursor_ts IS NULL" in read_sql
    assert "ORDER BY created_at ASC, id ASC" in read_sql


# ---------------------------------------------------------------------------
# Idempotent / no new rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_new_rows_is_idempotent_no_synthesis():
    """Empty new-rows SELECT -> no LLM, no writes, synthesized=0, and the
    cursor (last_processed_at) is NOT advanced (only last_run_at touched)."""
    session = _make_session(cursor=_BASE, rows=[])
    gen, ext, wsm = _patch_curator()

    with gen as mock_gen, ext as mock_ext, wsm as mock_write:
        result = await curator_tick(session, "org-1", min_inputs=3)

    mock_gen.assert_not_called()
    mock_ext.assert_not_called()
    mock_write.assert_not_called()
    assert result["synthesized"] == 0
    assert result["processed"] == 0
    assert result["reason"] == "below_threshold"

    # The state upsert touched only last_run_at, leaving BOTH composite-cursor
    # columns absent from the statement so neither half of the cursor moves.
    upsert_call = session.execute.call_args_list[2]
    upsert_sql = str(upsert_call.args[0])
    assert "brain_curator_state" in upsert_sql
    assert "last_run_at" in upsert_sql
    assert "last_processed_at" not in upsert_sql
    assert "last_processed_id" not in upsert_sql


# ---------------------------------------------------------------------------
# Below threshold
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_below_threshold_skips_synthesis():
    """Fewer than min_inputs rows -> no synthesis, reason=below_threshold."""
    rows = [_row(1, "only one"), _row(2, "and two")]  # 2 < min_inputs=3
    session = _make_session(cursor=None, rows=rows)
    gen, ext, wsm = _patch_curator()

    with gen as mock_gen, ext as mock_ext, wsm as mock_write:
        result = await curator_tick(session, "org-1", min_inputs=3)

    mock_gen.assert_not_called()
    mock_ext.assert_not_called()
    mock_write.assert_not_called()
    assert result["synthesized"] == 0
    assert result["processed"] == 2
    assert result["reason"] == "below_threshold"


# ---------------------------------------------------------------------------
# LLM failure resilience
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_none_skips_summary_but_still_extracts_and_advances():
    """generate_summary -> None: no summary write, facts still attempted, and
    the cursor still advances (the tick does not crash)."""
    rows = [_row(i, f"c{i}") for i in range(1, 6)]
    session = _make_session(cursor=None, rows=rows)
    gen, ext, wsm = _patch_curator(summary=None, facts=["f1", "f2"])

    with gen as mock_gen, ext as mock_ext, wsm as mock_write:
        result = await curator_tick(session, "org-1", min_inputs=3)

    mock_gen.assert_awaited_once()
    mock_ext.assert_awaited_once()  # facts still attempted
    # Only fact writes (no summary write).
    types_written = [c.kwargs["memory_type"] for c in mock_write.await_args_list]
    assert "summary" not in types_written
    assert types_written.count("fact") == 2

    # Cursor still advanced to the batch's composite (created_at, id) max.
    upsert_call = session.execute.call_args_list[2]
    assert upsert_call.args[1]["last_processed_at"] == rows[-1].created_at
    assert upsert_call.args[1]["last_processed_id"] == rows[-1].id
    assert result["synthesized"] == 2


@pytest.mark.asyncio
async def test_extract_facts_empty_still_advances():
    """generate_summary ok but extract_facts returns [] -> only the summary is
    written, and the cursor still advances."""
    rows = [_row(i, f"c{i}") for i in range(1, 6)]
    session = _make_session(cursor=None, rows=rows)
    gen, ext, wsm = _patch_curator(summary="just a summary", facts=[])

    with gen, ext, wsm as mock_write:
        result = await curator_tick(session, "org-1", min_inputs=3)

    types_written = [c.kwargs["memory_type"] for c in mock_write.await_args_list]
    assert types_written == ["summary"]
    assert result["synthesized"] == 1
    advance_params = session.execute.call_args_list[2].args[1]
    assert advance_params["last_processed_at"] == rows[-1].created_at
    assert advance_params["last_processed_id"] == rows[-1].id


# ---------------------------------------------------------------------------
# Dedup accounting: a deduped synthesized write is NOT counted as synthesized.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deduped_synthesized_write_not_counted():
    """write_scoped_memory reporting stored=False (dedup) does not increment
    the synthesized count; only newly-stored memories count."""
    rows = [_row(i, f"c{i}") for i in range(1, 6)]
    session = _make_session(cursor=None, rows=rows)

    with (
        patch(
            "app.curator.generate_summary",
            new_callable=AsyncMock,
            return_value="dup summary",
        ),
        patch(
            "app.curator.extract_facts",
            new_callable=AsyncMock,
            return_value=["f1", "f2"],
        ),
        patch(
            "app.curator.write_scoped_memory",
            new_callable=AsyncMock,
            # summary stored, first fact deduped, second fact stored.
            side_effect=[
                {"stored": True, "deduplicated": False, "id": "s"},
                {"stored": False, "deduplicated": True, "id": "dup"},
                {"stored": True, "deduplicated": False, "id": "f2"},
            ],
        ),
    ):
        result = await curator_tick(session, "org-1", min_inputs=3)

    # 3 write attempts, but only 2 newly stored.
    assert result["synthesized"] == 2
    assert result["processed"] == 5


# ---------------------------------------------------------------------------
# Isolation: the new-rows SELECT excludes verification + the curator's own output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_rows_select_is_org_scoped_and_excludes_self_output():
    """The new-rows SELECT must filter scope='org' and the memory_type IN list
    must include RAW types but EXCLUDE 'summary'/'fact' (so the curator cannot
    reprocess its own synthesized output, nor verification's session rows)."""
    rows = [_row(i, f"c{i}") for i in range(1, 6)]
    session = _make_session(cursor=None, rows=rows)
    gen, ext, wsm = _patch_curator()

    with gen, ext, wsm:
        await curator_tick(session, "org-1", min_inputs=3)

    # The new-rows SELECT is the 2nd execute call.
    new_rows_sql = str(session.execute.call_args_list[1].args[0])
    assert "scope = 'org'" in new_rows_sql
    assert "session_memories" in new_rows_sql
    # RAW types present.
    for raw in ("observation", "note", "memory", "plan", "decision"):
        assert raw in new_rows_sql
    # Self-output types absent from the filter.
    assert "summary" not in new_rows_sql
    assert "fact" not in new_rows_sql


@pytest.mark.asyncio
async def test_new_rows_select_filters_status_active():
    """The new-rows SELECT only reads active rows (not superseded/archived)."""
    rows = [_row(i, f"c{i}") for i in range(1, 6)]
    session = _make_session(cursor=None, rows=rows)
    gen, ext, wsm = _patch_curator()

    with gen, ext, wsm:
        await curator_tick(session, "org-1", min_inputs=3)

    new_rows_sql = str(session.execute.call_args_list[1].args[0])
    assert "status = 'active'" in new_rows_sql


@pytest.mark.asyncio
async def test_new_rows_select_includes_scope_columns():
    """The new-rows SELECT must also read space_id + project_id so the tick can
    partition the batch by project/space."""
    rows = [_row(i, f"c{i}") for i in range(1, 6)]
    session = _make_session(cursor=None, rows=rows)
    gen, ext, wsm = _patch_curator()

    with gen, ext, wsm:
        await curator_tick(session, "org-1", min_inputs=3)

    new_rows_sql = str(session.execute.call_args_list[1].args[0])
    assert "project_id" in new_rows_sql
    assert "space_id" in new_rows_sql


@pytest.mark.asyncio
async def test_writes_target_org_scope_via_curator_agent():
    """Synthesized memories are written through write_scoped_memory for the same
    org under the klio-curator agent id and a curator session id."""
    rows = [_row(i, f"c{i}") for i in range(1, 6)]
    session = _make_session(cursor=None, rows=rows)
    gen, ext, wsm = _patch_curator(summary="s", facts=["f1"])

    with gen, ext, wsm as mock_write:
        await curator_tick(session, "org-1", min_inputs=3, agent_id="klio-curator")

    for c in mock_write.await_args_list:
        assert c.kwargs["org_id"] == "org-1"
        assert c.kwargs["agent_id"] == "klio-curator"
        assert c.kwargs["session_id"] == "klio-curator:org-1"


# ---------------------------------------------------------------------------
# run_curator_once orchestration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_curator_once_iterates_orgs_and_aggregates():
    """The DISTINCT-org query yields 2 orgs; curator_tick runs once per org via
    a fresh session from the factory (each closed), and counts aggregate."""
    org_rows = [SimpleNamespace(org_id="org-a"), SimpleNamespace(org_id="org-b")]
    distinct_result = MagicMock()
    distinct_result.fetchall.return_value = org_rows

    sessions = [MagicMock(), MagicMock()]
    # First session is the discovery session; then one per org.
    discovery_session = MagicMock()
    discovery_session.execute.return_value = distinct_result
    made = [discovery_session, sessions[0], sessions[1]]
    session_factory = MagicMock(side_effect=made)

    tick_mock = AsyncMock(
        side_effect=[
            {"synthesized": 3, "processed": 5, "cursor": "x"},
            {"synthesized": 2, "processed": 4, "cursor": "y"},
        ]
    )

    with patch("app.curator.curator_tick", tick_mock):
        result = await run_curator_once(session_factory, min_inputs=3)

    assert result == {"orgs": 2, "synthesized_total": 5}
    assert tick_mock.await_count == 2
    ticked_orgs = {c.args[1] for c in tick_mock.await_args_list}
    assert ticked_orgs == {"org-a", "org-b"}

    # Every opened session was closed (discovery + per-org).
    discovery_session.close.assert_called_once()
    for s in sessions:
        s.close.assert_called_once()


@pytest.mark.asyncio
async def test_run_curator_once_no_orgs_returns_zeroes():
    """No orgs with raw org memories -> aggregate counts are zero."""
    distinct_result = MagicMock()
    distinct_result.fetchall.return_value = []
    discovery_session = MagicMock()
    discovery_session.execute.return_value = distinct_result
    session_factory = MagicMock(return_value=discovery_session)

    tick_mock = AsyncMock()
    with patch("app.curator.curator_tick", tick_mock):
        result = await run_curator_once(session_factory)

    assert result == {"orgs": 0, "synthesized_total": 0}
    tick_mock.assert_not_called()
    discovery_session.close.assert_called_once()


@pytest.mark.asyncio
async def test_run_curator_once_distinct_query_is_org_scope_and_raw_only():
    """The org-discovery query reads scope='org' and only RAW types (so it does
    not surface orgs that only have verification/session rows or curator output)."""
    distinct_result = MagicMock()
    distinct_result.fetchall.return_value = []
    discovery_session = MagicMock()
    discovery_session.execute.return_value = distinct_result
    session_factory = MagicMock(return_value=discovery_session)

    with patch("app.curator.curator_tick", AsyncMock()):
        await run_curator_once(session_factory)

    sql = str(discovery_session.execute.call_args_list[0].args[0])
    assert "DISTINCT org_id" in sql
    assert "scope = 'org'" in sql
    for raw in ("observation", "note", "memory", "plan", "decision"):
        assert raw in sql
    assert "summary" not in sql
    assert "fact" not in sql


@pytest.mark.asyncio
async def test_run_curator_once_closes_session_even_if_tick_raises():
    """A per-org tick failure still closes that org's session (finally), and
    does not abort the whole run for remaining orgs."""
    org_rows = [SimpleNamespace(org_id="org-a"), SimpleNamespace(org_id="org-b")]
    distinct_result = MagicMock()
    distinct_result.fetchall.return_value = org_rows
    discovery_session = MagicMock()
    discovery_session.execute.return_value = distinct_result

    org_a_session = MagicMock()
    org_b_session = MagicMock()
    session_factory = MagicMock(
        side_effect=[discovery_session, org_a_session, org_b_session]
    )

    tick_mock = AsyncMock(
        side_effect=[RuntimeError("boom"), {"synthesized": 4, "processed": 6}]
    )

    with patch("app.curator.curator_tick", tick_mock):
        result = await run_curator_once(session_factory)

    # org-a raised, org-b succeeded: both sessions closed, run continued.
    org_a_session.close.assert_called_once()
    org_b_session.close.assert_called_once()
    assert result["orgs"] == 2
    assert result["synthesized_total"] == 4


# ---------------------------------------------------------------------------
# Project-awareness: distill per (space_id, project_id), tag the output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_distills_each_project_separately_and_tags_scope():
    """A batch spanning two projects (each >= min_inputs) distills per project,
    tagging every write with that partition's project_id/space_id."""
    rows = [
        _row(i, f"a{i}", project_id="proj-A", space_id="sp-A") for i in range(1, 4)
    ] + [_row(i, f"b{i}", project_id="proj-B", space_id="sp-B") for i in range(4, 7)]
    session = _make_session(cursor=None, rows=rows)
    gen, ext, wsm = _patch_curator(summary="s", facts=["f1"])

    with gen as mock_gen, ext as mock_ext, wsm as mock_write:
        result = await curator_tick(session, "org-1", min_inputs=3)

    # Two partitions => synthesis ran once per partition.
    assert mock_gen.await_count == 2
    assert mock_ext.await_count == 2

    # Every write is tagged with a real project_id/space_id (never None here),
    # and each project's writes carry only that project's scope.
    by_project = {}
    for c in mock_write.await_args_list:
        by_project.setdefault(c.kwargs["project_id"], set()).add(c.kwargs["space_id"])
    assert by_project == {"proj-A": {"sp-A"}, "proj-B": {"sp-B"}}
    # processed counts the whole batch.
    assert result["processed"] == 6


@pytest.mark.asyncio
async def test_null_project_partition_stays_org_wide():
    """Rows with no project_id distill with project_id=None (genuinely org-wide,
    the now-correct meaning of NULL)."""
    rows = [_row(i, f"c{i}") for i in range(1, 4)]  # project_id defaults to None
    session = _make_session(cursor=None, rows=rows)
    gen, ext, wsm = _patch_curator(summary="s", facts=["f1"])

    with gen, ext, wsm as mock_write:
        await curator_tick(session, "org-1", min_inputs=3)

    for c in mock_write.await_args_list:
        assert c.kwargs["project_id"] is None
        assert c.kwargs["space_id"] is None


@pytest.mark.asyncio
async def test_subthreshold_partition_is_not_distilled():
    """When the batch total >= min_inputs but a single project's partition is
    below it, that partition is skipped (no writes) while a qualifying one runs.
    The cursor still advances over the whole batch."""
    rows = (
        [_row(i, f"a{i}", project_id="proj-A") for i in range(1, 4)]  # 3 -> distilled
        + [_row(i, f"b{i}", project_id="proj-B") for i in range(4, 6)]  # 2 -> skipped
    )
    session = _make_session(cursor=None, rows=rows)
    gen, ext, wsm = _patch_curator(summary="s", facts=["f1"])

    with gen, ext, wsm as mock_write:
        result = await curator_tick(session, "org-1", min_inputs=3)

    written_projects = {c.kwargs["project_id"] for c in mock_write.await_args_list}
    assert written_projects == {"proj-A"}  # proj-B skipped
    assert result["processed"] == 5  # whole batch counted
    # Cursor advanced to the whole batch's max (created_at, id) = last row.
    advance_params = session.execute.call_args_list[2].args[1]
    assert advance_params["last_processed_at"] == rows[-1].created_at
    assert advance_params["last_processed_id"] == rows[-1].id


@pytest.mark.asyncio
async def test_input_count_is_per_partition():
    """metadata.input_count reflects the partition's row count, not the batch."""
    rows = (
        [_row(i, f"a{i}", project_id="proj-A") for i in range(1, 4)]  # 3
        + [_row(i, f"b{i}", project_id="proj-B") for i in range(4, 8)]  # 4
    )
    session = _make_session(cursor=None, rows=rows)
    gen, ext, wsm = _patch_curator(
        summary="s", facts=[]
    )  # summary-only keeps it simple

    with gen, ext, wsm as mock_write:
        await curator_tick(session, "org-1", min_inputs=3)

    counts = {
        c.kwargs["project_id"]: c.kwargs["metadata"]["input_count"]
        for c in mock_write.await_args_list
    }
    assert counts == {"proj-A": 3, "proj-B": 4}


# ---------------------------------------------------------------------------
# reconcile_fact: LLM-judge supersession
# ---------------------------------------------------------------------------


async def test_reconcile_supersedes_direct_contradiction(monkeypatch):
    import app.curator as curator_mod

    superseded = []

    def _fake_supersede(db, old_id, new_id):
        superseded.append((old_id, new_id))

    async def _fake_judge(prompt, system=None):
        return {"contradicts": True}

    monkeypatch.setattr(curator_mod, "supersede_memory", _fake_supersede)
    monkeypatch.setattr(curator_mod, "call_llm", _fake_judge)

    import uuid

    old_id, new_id = uuid.uuid4(), uuid.uuid4()

    class _DB:
        def execute(self, sql, params=None):
            class R:
                def fetchall(self_inner):
                    class Row:
                        id = old_id
                        content = "Stripe MCP is connected to LoopReel sandbox"

                    return [Row()]

            return R()

    sup_count, judged = await curator_mod.reconcile_fact(
        _DB(),
        org_id="org-1",
        new_fact_id=new_id,
        new_fact_content="Stripe MCP is connected to the MoonForge account",
        embedding=[0.1, 0.2],
        max_judgments=5,
    )
    assert sup_count == 1 and judged == 1
    assert superseded == [(old_id, new_id)]


async def test_reconcile_related_but_not_contradicting_is_untouched(monkeypatch):
    import app.curator as curator_mod

    superseded = []
    monkeypatch.setattr(
        curator_mod, "supersede_memory", lambda db, o, n: superseded.append((o, n))
    )

    async def _fake_judge(prompt, system=None):
        return {"contradicts": False}

    monkeypatch.setattr(curator_mod, "call_llm", _fake_judge)

    import uuid

    class _DB:
        def execute(self, sql, params=None):
            class R:
                def fetchall(self_inner):
                    class Row:
                        id = uuid.uuid4()
                        content = "The team uses Stripe for billing"

                    return [Row()]

            return R()

    sup_count, judged = await curator_mod.reconcile_fact(
        _DB(),
        org_id="org-1",
        new_fact_id=uuid.uuid4(),
        new_fact_content="Stripe MCP is connected to the MoonForge account",
        embedding=[0.1, 0.2],
        max_judgments=5,
    )
    assert sup_count == 0 and judged == 1
    assert superseded == []


async def test_reconcile_judge_error_skips_pair(monkeypatch):
    import app.curator as curator_mod

    async def _boom(prompt, system=None):
        return None  # LLM failure path of call_llm

    monkeypatch.setattr(curator_mod, "call_llm", _boom)

    import uuid

    class _DB:
        def execute(self, sql, params=None):
            class R:
                def fetchall(self_inner):
                    class Row:
                        id = uuid.uuid4()
                        content = "old fact"

                    return [Row()]

            return R()

    sup_count, judged = await curator_mod.reconcile_fact(
        _DB(),
        org_id="org-1",
        new_fact_id=uuid.uuid4(),
        new_fact_content="new fact",
        embedding=[0.1],
        max_judgments=5,
    )
    assert (
        sup_count == 0 and judged == 1
    )  # call attempted, skipped on failure, no raise


# ---------------------------------------------------------------------------
# _store_one: returns id string on new store, None on dedup
# ---------------------------------------------------------------------------


async def test_store_one_returns_id_on_success(monkeypatch):
    """_store_one returns the new row id string when write_scoped_memory stored=True."""
    import app.curator as curator_mod

    monkeypatch.setattr(
        curator_mod,
        "write_scoped_memory",
        AsyncMock(
            return_value={"stored": True, "deduplicated": False, "id": "abc-123"}
        ),
    )
    result = await curator_mod._store_one(
        object(),
        org_id="org-1",
        agent_id="agent-1",
        content="some fact",
        memory_type="fact",
        session_id="agent-1:org-1",
        metadata={"source": "curator"},
    )
    assert result == "abc-123"


async def test_store_one_returns_none_on_dedup(monkeypatch):
    """_store_one returns None when write_scoped_memory reports stored=False (dedup)."""
    import app.curator as curator_mod

    monkeypatch.setattr(
        curator_mod,
        "write_scoped_memory",
        AsyncMock(
            return_value={"stored": False, "deduplicated": True, "id": "existing-id"}
        ),
    )
    result = await curator_mod._store_one(
        object(),
        org_id="org-1",
        agent_id="agent-1",
        content="duplicate fact",
        memory_type="fact",
        session_id="agent-1:org-1",
        metadata={"source": "curator"},
    )
    assert result is None


# ---------------------------------------------------------------------------
# _synthesize_and_store: reconcile_fact called for facts only, not summaries
# ---------------------------------------------------------------------------


async def test_synthesize_and_store_reconciles_facts_not_summaries(monkeypatch):
    """reconcile_fact is called once per newly stored fact, never for summaries."""
    import app.curator as curator_mod

    monkeypatch.setattr(
        curator_mod,
        "generate_summary",
        AsyncMock(return_value="a summary"),
    )
    monkeypatch.setattr(
        curator_mod,
        "extract_facts",
        AsyncMock(return_value=["fact one", "fact two"]),
    )
    monkeypatch.setattr(
        curator_mod,
        "write_scoped_memory",
        AsyncMock(
            side_effect=[
                # summary stored
                {"stored": True, "deduplicated": False, "id": "sum-1"},
                # fact one stored
                {"stored": True, "deduplicated": False, "id": "fact-1"},
                # fact two stored
                {"stored": True, "deduplicated": False, "id": "fact-2"},
            ]
        ),
    )
    fake_embedding = [0.1, 0.2, 0.3]
    monkeypatch.setattr(
        curator_mod, "embed_text", AsyncMock(return_value=fake_embedding)
    )
    reconcile_calls = []

    async def _fake_reconcile(
        db_session,
        *,
        org_id,
        new_fact_id,
        new_fact_content,
        embedding,
        max_judgments,
    ):
        reconcile_calls.append(
            {
                "org_id": org_id,
                "new_fact_id": new_fact_id,
                "new_fact_content": new_fact_content,
                "embedding": embedding,
                "max_judgments": max_judgments,
            }
        )
        # Return (superseded, judged) tuple as the real implementation does.
        return 0, 1

    monkeypatch.setattr(curator_mod, "reconcile_fact", _fake_reconcile)

    stored = await curator_mod._synthesize_and_store(
        object(),
        org_id="org-1",
        agent_id="agent-1",
        combined="content one\ncontent two",
        input_count=2,
    )

    assert stored == 3  # summary + 2 facts
    # reconcile_fact called once per stored fact (2 facts), never for summary.
    assert len(reconcile_calls) == 2
    fact_contents = {c["new_fact_content"] for c in reconcile_calls}
    assert fact_contents == {"fact one", "fact two"}
    # org_id forwarded correctly
    for c in reconcile_calls:
        assert c["org_id"] == "org-1"
        assert c["embedding"] == fake_embedding
    # max_judgments kwarg must be forwarded (budget enforcement).
    for c in reconcile_calls:
        assert isinstance(c["max_judgments"], int) and c["max_judgments"] > 0
