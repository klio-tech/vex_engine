"""SQL-shape regression tests for recall queries — fake DB, no Postgres required.

These tests verify the *structure* of the SQL emitted by recall helpers by
capturing the rendered SQL string through a lightweight fake DB session.  They
require no live Postgres instance and must run in every CI environment without
skipping.
"""

import pytest

import shared.memory as memory_mod


@pytest.mark.asyncio
async def test_recall_sql_excludes_hidden_and_inactive(monkeypatch):
    """The generated recall SQL must filter recall_hidden and status.

    This is a unit-level SQL-shape test: no live DB is required. A fake DB
    session captures the rendered SQL string, which must contain both
    ``recall_hidden = FALSE`` and ``status = 'active'``.  The test is
    intentionally isolated from Postgres so it runs in every CI environment.
    """
    captured = {}

    async def _embed(_):
        return [0.1, 0.2]

    monkeypatch.setattr(memory_mod, "embed_text", _embed)

    class _DB:
        def execute(self, sql, params=None):
            captured["sql"] = str(sql)

            class R:
                def fetchall(self_inner):
                    return []

            return R()

    await memory_mod.retrieve_memories_scoped(_DB(), "org-1", "query")
    assert "recall_hidden = FALSE" in captured["sql"]
    assert "status = 'active'" in captured["sql"]


# ---------------------------------------------------------------------------
# Task 10: blended score, two-tier project recall, relevance field
# ---------------------------------------------------------------------------


async def _run_recall_sql_capture(monkeypatch, **kwargs):
    """Helper: run retrieve_memories_scoped with a fake DB that captures SQL."""
    captured = {"sqls": []}

    async def _embed(_):
        return [0.1, 0.2]

    monkeypatch.setattr(memory_mod, "embed_text", _embed)

    class _DB:
        def execute(self, sql, params=None):
            captured["sqls"].append((str(sql), params))

            class R:
                def fetchall(self_inner):
                    return []

            return R()

    await memory_mod.retrieve_memories_scoped(_DB(), "org-1", "q", **kwargs)
    return captured


@pytest.mark.asyncio
async def test_blended_score_in_order_by(monkeypatch):
    """recall SQL must use blended_score (0.85 * similarity + 0.15 * recency)
    for ordering instead of raw cosine distance."""
    captured = await _run_recall_sql_capture(monkeypatch)
    sql = captured["sqls"][0][0]
    assert "blended_score" in sql
    assert "0.85" in sql and "0.15" in sql


@pytest.mark.asyncio
async def test_project_recall_is_two_tier(monkeypatch):
    """With project_id set, two separate queries are issued:
    tier-1 (project_id = :project_id), tier-2 (project_id IS NULL for remainder).
    Rows from OTHER projects must never appear."""
    captured = await _run_recall_sql_capture(monkeypatch, project_id="proj-1")
    tier1_sql = captured["sqls"][0][0]
    assert "project_id = :project_id" in tier1_sql
    assert "project_id IS NULL" not in tier1_sql
    tier2_sql = captured["sqls"][1][0]
    assert "project_id IS NULL" in tier2_sql


@pytest.mark.asyncio
async def test_relevance_populated_on_records(monkeypatch):
    """retrieve_memories_scoped must populate MemoryRecord.relevance from the
    ``(1 - (embedding <=> :embedding))`` column in the SQL result."""
    import uuid
    from datetime import datetime, timezone

    async def _embed(_):
        return [0.1, 0.2]

    monkeypatch.setattr(memory_mod, "embed_text", _embed)
    now = datetime.now(timezone.utc)
    fake_row = (
        uuid.uuid4(), "s", "org-1", "a", None, "fact", "content", None, None,
        None, "org", "active", None, {}, now, now, None, None,
        0.91, 0.85,
    )

    class _DB:
        def execute(self, sql, params=None):
            class R:
                def fetchall(self_inner):
                    return [fake_row]

            return R()

    records = await memory_mod.retrieve_memories_scoped(_DB(), "org-1", "q")
    assert records[0].relevance == 0.91
