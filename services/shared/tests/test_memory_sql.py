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
