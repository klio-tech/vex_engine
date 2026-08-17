"""``CURATOR_RECONCILE_MAX`` is a per-TICK cap, and must actually be one.

The docstring on ``_reconcile_budget`` says what it means: the maximum number
of ``call_llm`` judge attempts "across all facts in one curator tick". The
implementation read the environment variable fresh inside
``_synthesize_and_store``, which runs once per ``(space_id, project_id)``
partition — so each partition got its own full allowance and the documented
per-tick cap of 50 was a per-partition cap of 50. A 50-row batch can partition
16 ways, so the tick's real ceiling was 800 LLM calls: 16× the number an
operator set, on a knob whose entire purpose is bounding spend.

``EntityBudget`` already solved exactly this problem for the entity cap — one
mutable budget created in ``curator_tick`` and threaded through every
partition. This mirrors it.

Mock-based: the budget is arithmetic, and the seam that spends it
(``reconcile_fact``) is patched so the accounting is what is measured.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.curator import curator_tick
from shared.taxonomy import ExtractedMemory

_BASE = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
USER = "11111111-2222-3333-4444-555555555555"


def _row(idx: int, *, project_id=None):
    return SimpleNamespace(
        id=f"mem-{idx}",
        content=f"c{idx}",
        created_at=_BASE + timedelta(minutes=idx),
        space_id=None,
        project_id=project_id,
    )


def _session(rows):
    res_cursor = MagicMock()
    res_cursor.fetchone.return_value = None
    res_rows = MagicMock()
    res_rows.fetchall.return_value = rows
    session = MagicMock()
    session.execute.side_effect = [res_cursor, res_rows] + [MagicMock()] * 20
    return session


def _patched(*, facts, reconcile):
    return (
        patch("app.curator.generate_summary", AsyncMock(return_value=None)),
        patch(
            "app.curator.extract_graph",
            AsyncMock(
                return_value={
                    "memories": [ExtractedMemory("fact", f) for f in facts],
                    "entities": [],
                    "edges": [],
                }
            ),
        ),
        patch(
            "app.curator.write_scoped_memory",
            AsyncMock(return_value={"stored": True, "id": "new-id"}),
        ),
        patch("app.curator.embed_text", AsyncMock(return_value=[0.1] * 1024)),
        patch("app.curator.reconcile_fact", reconcile),
    )


def _four_partitions():
    """Twelve rows over four projects: three each, so every one qualifies."""
    rows = []
    idx = 0
    for project in ("p-a", "p-b", "p-c", "p-d"):
        for _ in range(3):
            rows.append(_row(idx, project_id=project))
            idx += 1
    return rows


@pytest.mark.asyncio
async def test_the_reconcile_budget_is_spent_once_per_tick(monkeypatch):
    monkeypatch.setenv("CURATOR_RECONCILE_MAX", "3")
    # Every judged pair costs one unit.
    reconcile = AsyncMock(return_value=(0, 1))

    gen, ext, wsm, emb, rec = _patched(facts=["f1", "f2"], reconcile=reconcile)
    with gen, ext, wsm, emb, rec:
        await curator_tick(_session(_four_partitions()), "org-1", USER, min_inputs=3)

    assert reconcile.await_count <= 3, (
        f"the tick made {reconcile.await_count} judge calls against a "
        "documented per-tick cap of 3; each partition was handed a fresh "
        "allowance, so the operator's number is multiplied by the partition "
        "count"
    )


@pytest.mark.asyncio
async def test_a_later_partition_sees_the_budget_the_earlier_one_left(monkeypatch):
    """The mechanism, not just the total: the cap must be handed down.

    Asserting only the total leaves a fix that resets per partition but caps
    the total elsewhere. This pins that the allowance passed into each
    ``reconcile_fact`` call actually shrinks.
    """
    monkeypatch.setenv("CURATOR_RECONCILE_MAX", "10")
    seen: list[int] = []

    async def reconcile(db, **kwargs):
        seen.append(kwargs["max_judgments"])
        return (0, 2)

    gen, ext, wsm, emb, rec = _patched(
        facts=["f1"], reconcile=AsyncMock(side_effect=reconcile)
    )
    with gen, ext, wsm, emb, rec:
        await curator_tick(_session(_four_partitions()), "org-1", USER, min_inputs=3)

    assert seen == [10, 8, 6, 4], (
        f"each partition should inherit what the last one left; saw {seen}. "
        "Four 10s means four fresh allowances"
    )


@pytest.mark.asyncio
async def test_reconciliation_still_runs_within_the_budget(monkeypatch):
    """Positive control: the budget bounds reconciliation, it does not kill it.

    Both assertions above are satisfied by a tick that never reconciles at
    all. This one fails if the threading turned the feature off.
    """
    monkeypatch.setenv("CURATOR_RECONCILE_MAX", "50")
    reconcile = AsyncMock(return_value=(1, 1))

    gen, ext, wsm, emb, rec = _patched(facts=["f1", "f2"], reconcile=reconcile)
    with gen, ext, wsm, emb, rec:
        await curator_tick(_session(_four_partitions()), "org-1", USER, min_inputs=3)

    assert reconcile.await_count == 8, "two facts in each of four partitions"
