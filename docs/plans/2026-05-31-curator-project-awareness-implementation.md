# Curator Project-Awareness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the org-brain curator project-aware so distilled facts only resurface in their own project (stopping cross-project memory bleed), and delete the legacy `project_id=NULL` curator facts already leaking.

**Architecture:** Inside `curator_tick`, partition each batch by `(space_id, project_id)` and distill each partition `>= min_inputs` independently, tagging its `summary`/`fact` writes with that project/space. The per-org keyset cursor is unchanged (advances over the whole batch). Plus a one-time additive migration deleting the legacy unscoped curator output. Recall is untouched.

**Tech Stack:** Python, SQLAlchemy (sync `db_session.execute`), Alembic, pytest. Run Python via `./.venv/bin/python`; tests need `PYTHONPATH=services/shared`; lint via the standalone `ruff` binary.

**Repo / branch:** `/Users/thakurg/Hive/Research/AgentGuard/vex_private/vex_engine`, branch `feat/curator-project-awareness`. Local commits only — **NO push**.

**Reference spec:** `docs/plans/2026-05-31-curator-project-awareness-design.md`

---

## File Structure

**Modify:**
- `services/mcp-server/app/curator.py` — `_read_new_raw_rows` (select scope cols), `curator_tick` (partition + per-partition distill), `_synthesize_and_store` + `_store_one` (thread `space_id`/`project_id`).
- `services/mcp-server/tests/test_curator.py` — update `_row` helper (+`space_id`/`project_id`); add project-awareness tests.

**Create:**
- `services/migrations/alembic/versions/023_curator_facts_project_cleanup.py` — one-time data cleanup.

No other files change. `write_scoped_memory` already accepts `space_id`/`project_id` kwargs; `memory.py` recall is unchanged.

---

## Task 1: Validate the test environment (baseline green)

**Files:** none (verification only)

- [ ] **Step 1: Confirm the curator suite passes today**

Run:
```bash
cd /Users/thakurg/Hive/Research/AgentGuard/vex_private/vex_engine
PYTHONPATH=services/shared ./.venv/bin/python -m pytest services/mcp-server/tests/test_curator.py -q
```
Expected: all tests pass (green). If red, STOP — the environment is broken before any change.

- [ ] **Step 2: Confirm branch**

Run: `git branch --show-current`
Expected: `feat/curator-project-awareness`.

---

## Task 2: New-rows SELECT carries `space_id` + `project_id`

**Files:**
- Modify: `services/mcp-server/app/curator.py` (`_read_new_raw_rows`, the SELECT column list)
- Test: `services/mcp-server/tests/test_curator.py`

- [ ] **Step 1: Write the failing test**

Add to `test_curator.py`:

```python
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
```

- [ ] **Step 2: Run it — expect FAIL**

Run:
```bash
PYTHONPATH=services/shared ./.venv/bin/python -m pytest \
  services/mcp-server/tests/test_curator.py::test_new_rows_select_includes_scope_columns -q
```
Expected: FAIL (`assert "project_id" in new_rows_sql` — current SELECT is `SELECT id, content, created_at`).

- [ ] **Step 3: Implement — add the columns to the SELECT**

In `_read_new_raw_rows`, change the SELECT line only:

```python
            SELECT id, content, created_at, space_id, project_id
            FROM session_memories
```
(Everything else in that query — WHERE, keyset predicate, `ORDER BY created_at ASC, id ASC`, `LIMIT` — is unchanged.)

- [ ] **Step 4: Run it — expect PASS**

Run the same command as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/mcp-server/app/curator.py services/mcp-server/tests/test_curator.py
git commit -m "feat(curator): read space_id/project_id for project-aware distillation"
```

---

## Task 3: Partition by project + thread scope into writes (the core change)

**Files:**
- Modify: `services/mcp-server/app/curator.py` (`curator_tick`, `_synthesize_and_store`, `_store_one`)
- Test: `services/mcp-server/tests/test_curator.py` (update `_row`; add tests)

- [ ] **Step 1: Update the `_row` test helper to carry scope columns**

Replace the `_row` helper so fake rows mirror the real SELECT (default `None` = unfiled):

```python
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
```

- [ ] **Step 2: Write the failing tests (project-awareness behaviors)**

Add to `test_curator.py`:

```python
@pytest.mark.asyncio
async def test_distills_each_project_separately_and_tags_scope():
    """A batch spanning two projects (each >= min_inputs) distills per project,
    tagging every write with that partition's project_id/space_id."""
    rows = (
        [_row(i, f"a{i}", project_id="proj-A", space_id="sp-A") for i in range(1, 4)]
        + [_row(i, f"b{i}", project_id="proj-B", space_id="sp-B") for i in range(4, 7)]
    )
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
    # processed counts the whole batch; cursor advanced over the last row.
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
    assert written_projects == {"proj-A"}          # proj-B skipped
    assert result["processed"] == 5                # whole batch counted
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
    gen, ext, wsm = _patch_curator(summary="s", facts=[])  # summary-only keeps it simple

    with gen, ext, wsm as mock_write:
        await curator_tick(session, "org-1", min_inputs=3)

    counts = {
        c.kwargs["project_id"]: c.kwargs["metadata"]["input_count"]
        for c in mock_write.await_args_list
    }
    assert counts == {"proj-A": 3, "proj-B": 4}
```

- [ ] **Step 3: Run them — expect FAIL**

Run:
```bash
PYTHONPATH=services/shared ./.venv/bin/python -m pytest \
  services/mcp-server/tests/test_curator.py -q -k "distills_each_project or null_project_partition or subthreshold_partition or input_count_is_per_partition"
```
Expected: FAIL (current `curator_tick` distills the whole batch once with no `project_id`/`space_id` kwargs).

- [ ] **Step 4: Implement — thread scope through the writers**

In `curator.py`, update `_store_one` to accept + forward the scope kwargs:

```python
async def _store_one(
    db_session: Any,
    *,
    org_id: str,
    agent_id: str,
    content: str,
    memory_type: str,
    session_id: str,
    metadata: dict,
    space_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> bool:
    """Write a single synthesised memory; return True only if newly stored."""
    try:
        result = await write_scoped_memory(
            db_session,
            org_id=org_id,
            agent_id=agent_id,
            content=content,
            memory_type=memory_type,
            session_id=session_id,
            space_id=space_id,
            project_id=project_id,
            metadata=metadata,
        )
    except Exception:
        logger.warning(
            "curator: write_scoped_memory raised for org %s (%s); skipping",
            org_id,
            memory_type,
            exc_info=True,
        )
        return False
    return bool(result.get("stored"))
```

Update `_synthesize_and_store` to accept + forward the scope kwargs to both `_store_one` calls (summary and each fact):

```python
async def _synthesize_and_store(
    db_session: Any,
    *,
    org_id: str,
    agent_id: str,
    combined: str,
    input_count: int,
    space_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> int:
    """Run the LLM synthesis over ``combined`` and store the results, tagged with
    the partition's space_id/project_id. Returns the number newly stored."""
    session_id = f"{agent_id}:{org_id}"
    base_meta = {"source": "curator", "input_count": input_count}
    stored = 0

    try:
        summary = await generate_summary(combined)
    except Exception:
        logger.warning(
            "curator: generate_summary raised for org %s; skipping summary",
            org_id,
            exc_info=True,
        )
        summary = None

    if summary:
        if await _store_one(
            db_session,
            org_id=org_id,
            agent_id=agent_id,
            content=summary,
            memory_type="summary",
            session_id=session_id,
            metadata=base_meta,
            space_id=space_id,
            project_id=project_id,
        ):
            stored += 1
    else:
        logger.info("curator: no summary produced for org %s", org_id)

    try:
        facts = await extract_facts(combined)
    except Exception:
        logger.warning(
            "curator: extract_facts raised for org %s; no facts stored",
            org_id,
            exc_info=True,
        )
        facts = []

    for fact in facts:
        if not fact:
            continue
        if await _store_one(
            db_session,
            org_id=org_id,
            agent_id=agent_id,
            content=fact,
            memory_type="fact",
            session_id=session_id,
            metadata=base_meta,
            space_id=space_id,
            project_id=project_id,
        ):
            stored += 1

    return stored
```

- [ ] **Step 5: Implement — partition the batch in `curator_tick`**

Replace the synthesis block in `curator_tick` (the part after the `if len(rows) < min_inputs:` early-return, currently building one `combined` and calling `_synthesize_and_store` once) with per-partition distillation. The cursor-advance block is unchanged:

```python
    # Partition the batch by (space_id, project_id), preserving order so each
    # partition's combined text and the cursor advance stay deterministic.
    # NULL space/project => a genuinely org-wide partition (surfaces everywhere).
    partitions: dict[tuple, list] = {}
    for r in rows:
        partitions.setdefault((r.space_id, r.project_id), []).append(r)

    synthesized = 0
    for (space_id, project_id), part_rows in partitions.items():
        # Per-partition threshold: a project needs real signal before it gets a
        # distilled fact. Sub-threshold rows stay recallable as raw observations.
        if len(part_rows) < min_inputs:
            continue
        combined = "\n".join(r.content for r in part_rows if r.content)
        synthesized += await _synthesize_and_store(
            db_session,
            org_id=org_id,
            agent_id=agent_id,
            combined=combined,
            input_count=len(part_rows),
            space_id=space_id,
            project_id=project_id,
        )

    # Cursor advance is unchanged: rows are globally ordered (created_at ASC,
    # id ASC), so the LAST row carries the batch's composite max.
    batch_max_at = rows[-1].created_at
    batch_max_id = rows[-1].id
    _advance_cursor(db_session, org_id, batch_max_at, batch_max_id, len(rows))

    return {
        "synthesized": synthesized,
        "processed": len(rows),
        "cursor": batch_max_at.isoformat(),
    }
```

- [ ] **Step 6: Run the new tests + the FULL curator suite — expect PASS**

Run:
```bash
PYTHONPATH=services/shared ./.venv/bin/python -m pytest services/mcp-server/tests/test_curator.py -q
```
Expected: all pass — the new project-awareness tests AND the pre-existing tests (the latter use `_row` with `project_id=None` → a single org-wide partition → identical to prior behavior: e.g. the happy-path still asserts 4 writes / `input_count == 5`).

- [ ] **Step 7: Commit**

```bash
git add services/mcp-server/app/curator.py services/mcp-server/tests/test_curator.py
git commit -m "feat(curator): distill per (space_id, project_id) so facts stay project-scoped"
```

---

## Task 4: Migration `023` — delete legacy unscoped curator facts

**Files:**
- Create: `services/migrations/alembic/versions/023_curator_facts_project_cleanup.py`

- [ ] **Step 1: Create the migration**

```python
"""Delete legacy unscoped curator output (cross-project bleed cleanup).

ADDITIVE / DATA-ONLY. The curator historically wrote its distilled
``summary``/``fact`` rows with ``project_id = NULL`` (no project scope), and
recall surfaces NULL-project rows in every project — so those facts leaked into
unrelated projects' sessions. The curator is now project-aware (it tags output
with the source partition's project). This one-time cleanup removes the legacy
unscoped curator output that is already bleeding.

Strictly scoped to curator output: only ``metadata->>'source' = 'curator'`` rows
of type ``summary``/``fact`` with ``project_id IS NULL`` are deleted. Raw
observations (any project) and genuine user ``remember`` notes
(``source != 'curator'``) are untouched. Idempotent: a re-run or a fresh DB
matches nothing. Downgrade is a no-op — the deleted rows are regenerable derived
data, so there is nothing to restore.

Revision ID: 023
Revises: 022
Create Date: 2026-05-31
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "023"
down_revision: Union[str, None] = "022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        DELETE FROM session_memories
        WHERE metadata->>'source' = 'curator'
          AND project_id IS NULL
          AND memory_type IN ('summary', 'fact')
        """
    )


def downgrade() -> None:
    # No-op: deleted rows are regenerable derived data; nothing to restore.
    pass
```

- [ ] **Step 2: Verify the migration imports + chains cleanly (offline check)**

Run:
```bash
cd /Users/thakurg/Hive/Research/AgentGuard/vex_private/vex_engine
./.venv/bin/python -c "import ast; ast.parse(open('services/migrations/alembic/versions/023_curator_facts_project_cleanup.py').read()); print('parse OK')"
grep -n 'down_revision' services/migrations/alembic/versions/023_curator_facts_project_cleanup.py
```
Expected: `parse OK` and `down_revision: ... = "022"` (chains onto the latest migration). Do **not** run `alembic upgrade` against prod here — that happens at deploy.

- [ ] **Step 3: Commit**

```bash
git add services/migrations/alembic/versions/023_curator_facts_project_cleanup.py
git commit -m "feat(migrations): 023 delete legacy unscoped (project_id NULL) curator facts"
```

---

## Task 5: Full verification + lint + wrap

**Files:** none (verification only)

- [ ] **Step 1: Run the full mcp-server + shared test suites**

Run:
```bash
cd /Users/thakurg/Hive/Research/AgentGuard/vex_private/vex_engine
PYTHONPATH=services/shared ./.venv/bin/python -m pytest services/mcp-server/tests services/shared/tests -q
```
Expected: green (DB-backed tests may skip as usual; no failures).

- [ ] **Step 2: Lint the changed Python**

Run:
```bash
ruff check services/mcp-server/app/curator.py services/mcp-server/tests/test_curator.py services/migrations/alembic/versions/023_curator_facts_project_cleanup.py
```
Expected: no errors. Fix any reported issues and re-run.

- [ ] **Step 3: Confirm recall was NOT touched (regression guard)**

Run: `git diff --name-only main...HEAD`
Expected: only `curator.py`, `test_curator.py`, the `023_*` migration, and the two `docs/plans/*` files. **`services/shared/shared/memory.py` must NOT appear** (recall is intentionally unchanged).

- [ ] **Step 4: Rollout notes (manual, gated on user approval — do NOT push here)**

Deploying `mcp_server` to Railway runs migration `023` (removes the bleeding NULL facts) and ships the project-aware curator. After deploy, verify via the `recall` MCP tool: a query that previously surfaced cross-project trivia (e.g. Klio/Railway facts in an unrelated project) no longer does, and newly curated facts carry a `project_id`.

---

## Self-Review (author checklist)

**Spec coverage:** §1 curator distillation → Tasks 2+3; §2 threshold (per-partition `min_inputs`, sub-threshold skipped, cursor over whole batch) → Task 3 (`test_subthreshold_partition_is_not_distilled`); §3 migration 023 → Task 4; §4 recall unchanged → Task 5 Step 3 guard; §5 testing → Tasks 2/3 tests. All covered.

**Placeholder scan:** none — every code step shows complete code and exact commands.

**Type/name consistency:** `_synthesize_and_store` and `_store_one` both gain `space_id`/`project_id` keyword params (default `None`) forwarded to `write_scoped_memory(..., space_id=, project_id=)` — matching its real signature (`brain.py:439`). Partition key `(space_id, project_id)` is used consistently. `Optional` is already imported in `curator.py` (used by `CursorState`). `_row` helper change is in the same task (3) that first reads `r.space_id`/`r.project_id`, so no test breaks.
