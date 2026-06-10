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
    """Helper: run retrieve_memories_scoped with a fake DB that captures SQL
    and the bound params for each execute call."""
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
    """recall SQL must use the ANN-inner / re-rank-outer shape:
    - inner ORDER BY embedding <=> :embedding (HNSW-friendly)
    - inner LIMIT :candidate_pool (bounded candidate pool)
    - outer ORDER BY with 0.85/0.15 blended score
    The old ``blended_score`` alias must NOT appear as a selected column."""
    captured = await _run_recall_sql_capture(monkeypatch)
    sql = captured["sqls"][0][0]
    # Inner ANN-accelerated pass
    assert "ORDER BY embedding <=> :embedding" in sql
    assert "LIMIT :candidate_pool" in sql
    # Outer blended re-rank (not a column alias — lives only in ORDER BY)
    assert "0.85" in sql and "0.15" in sql
    # blended_score must NOT be a selected column alias
    assert "blended_score" not in sql
    # params must include candidate_pool
    params = captured["sqls"][0][1]
    assert "candidate_pool" in params


@pytest.mark.asyncio
async def test_project_recall_is_two_tier(monkeypatch):
    """With project_id set, two separate queries are issued:
    tier-1 (project_id = :project_id), tier-2 (project_id IS NULL for remainder).
    Rows from OTHER projects must never appear.
    Tier-2 params must override both limit and candidate_pool."""
    captured = await _run_recall_sql_capture(monkeypatch, project_id="proj-1")
    tier1_sql, tier1_params = captured["sqls"][0]
    assert "project_id = :project_id" in tier1_sql
    assert "project_id IS NULL" not in tier1_sql
    tier2_sql, tier2_params = captured["sqls"][1]
    assert "project_id IS NULL" in tier2_sql
    # tier-2 must supply its own limit and candidate_pool for the fill count
    assert "limit" in tier2_params
    assert "candidate_pool" in tier2_params


@pytest.mark.asyncio
async def test_relevance_populated_on_records(monkeypatch):
    """retrieve_memories_scoped must populate MemoryRecord.relevance from the
    ``(1 - (embedding <=> :embedding))`` column at index 18.

    The outer SELECT * preserves 18 record columns + relevance at index 18.
    blended_score is NOT a selected column (it drives ORDER BY only), so
    the row has exactly 19 elements (indices 0–18).
    """
    import uuid
    from datetime import datetime, timezone

    async def _embed(_):
        return [0.1, 0.2]

    monkeypatch.setattr(memory_mod, "embed_text", _embed)
    now = datetime.now(timezone.utc)
    # 19 elements: 18 record columns (indices 0-17) + relevance at index 18.
    # No blended_score column — it is only used in the outer ORDER BY.
    fake_row = (
        uuid.uuid4(), "s", "org-1", "a", None, "fact", "content", None, None,
        None, "org", "active", None, {}, now, now, None, None,
        0.91,
    )

    class _DB:
        def execute(self, sql, params=None):
            class R:
                def fetchall(self_inner):
                    return [fake_row]

            return R()

    records = await memory_mod.retrieve_memories_scoped(_DB(), "org-1", "q")
    assert records[0].relevance == 0.91


def test_no_bind_param_immediately_followed_by_cast():
    """Ban `:param::type` in raw SQL — SQLAlchemy's text() does not recognize
    a bind param whose name is immediately followed by `::` (the tokenizer
    requires a non-word, non-colon character after the name), so the raw
    `:param` reaches Postgres and fails with a syntax error at runtime.
    Use `CAST(:param AS type)` instead. This broke the production `forget`
    tool and would have broken the redaction backfill's --apply mode.
    """
    import os
    import re

    here = os.path.dirname(__file__)
    sources = [
        os.path.join(here, "..", "shared", "memory.py"),
        os.path.join(here, "..", "shared", "brain.py"),
        os.path.join(here, "..", "..", "scripts", "run_redaction_backfill.py"),
        os.path.join(here, "..", "..", "scripts", "run_retention.py"),
        os.path.join(here, "..", "..", "mcp-server", "app", "capture.py"),
        os.path.join(here, "..", "..", "mcp-server", "app", "curator.py"),
        os.path.join(here, "..", "..", "mcp-server", "app", "mcp_app.py"),
        os.path.join(here, "..", "..", "mcp-server", "app", "recall_log.py"),
    ]
    pattern = re.compile(r":[A-Za-z_]\w*::")
    offenders = {}
    for path in sources:
        with open(path) as f:
            hits = [
                f"line {i}: {line.strip()}"
                for i, line in enumerate(f, 1)
                if pattern.search(line)
            ]
        if hits:
            offenders[os.path.basename(path)] = hits
    assert not offenders, f"bind-then-cast SQL found (use CAST()): {offenders}"
