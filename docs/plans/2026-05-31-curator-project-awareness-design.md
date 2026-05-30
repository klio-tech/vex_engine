# Curator Project-Awareness (Cross-Project Memory Bleed Fix) — Design

**Date:** 2026-05-31
**Status:** Approved (brainstorming complete; ready for implementation plan)
**Repo:** `vex_engine` (`services/mcp-server`, `services/migrations`)
**Scope:** Backend curator + a one-time data cleanup migration. No recall change.

---

## Problem

Klio Cloud's org brain leaks memory across unrelated projects. Opening Claude
Code in one project (e.g. "Growth App") surfaces durable "facts" from a totally
different project (e.g. Klio/Vex deployment trivia) via the SessionStart recall.

### Confirmed root cause (code-verified)

1. **The curator discards project scope.** `services/mcp-server/app/curator.py`
   reads an org's RAW observations (`_read_new_raw_rows` selects only
   `id, content, created_at`), mashes the whole batch into one `combined` string,
   and writes the distilled `summary`/`fact` rows via `write_scoped_memory(...)`
   **without a `project_id`** → every curated fact lands `project_id = NULL`.
2. **Recall surfaces NULL everywhere.** `services/shared/shared/memory.py`
   (retrieval) matches `(project_id = :project_id OR project_id IS NULL)`, so any
   `project_id = NULL` memory appears in **every** project's recall.
3. **SessionStart injects them.** The Claude Code SessionStart hook does a
   recency recall and injects those org-wide NULL facts into unrelated sessions.
4. **Compounding (not in scope here):** all the user's projects share one org via
   one `X-Vex-Key`, so they pool into a single brain. Project isolation is *meant*
   to be the separation axis — but the curator defeats it.

The raw observations themselves are already correctly `project_id`-tagged (by the
capture hooks). Only the curator's *distilled output* is unscoped.

## Goal

Make the curator project-aware so a project's distilled facts only resurface in
that project, and clean up the legacy NULL facts already bleeding. `NULL` becomes
reserved for *genuinely* org-wide memories (e.g. an explicit `remember` written
with no project context).

## Non-goals (explicitly out of scope)

- **Over-capture reduction** (the PostToolUse hook dumping raw tool echoes) — a
  separate spec (B) in the `klio` repo.
- **Per-project cursor** (re-keying `brain_curator_state`) — not needed; raw
  observations already recall per-project, and the global keyset cursor is fine.
- **Recall changes** (dropping NULL-surfacing) — rejected; that mis-models intent
  and risks space/agent scoping. Recall is deliberately unchanged.
- **Re-curating the backlog** — the legacy NULL facts are deleted, not
  regenerated (no cursor reset).

---

## Approach (chosen: A — group-by-project inside the existing tick)

Partition each curator tick's batch by `(space_id, project_id)` and distill each
partition independently, tagging its output with that partition's project/space.
The per-org keyset cursor is untouched — partitioning is purely internal to a
tick. Contained to `curator.py` plus one data-cleanup migration. No schema change.

Rejected alternatives: **B** (per-project cursor — needs a schema migration +
more complexity, not justified); **C** (strip NULL-surfacing at recall — wrong
layer, breaks genuine org-wide knowledge).

---

## Component design

### 1. `_read_new_raw_rows` — carry the scope columns

Add `space_id, project_id` to the SELECT (both already exist on
`session_memories`). Everything else (tenant scoping, `scope='org'`, RAW types,
`status='active'`, the keyset predicate, `ORDER BY created_at ASC, id ASC`,
`LIMIT`) is unchanged.

### 2. `curator_tick` — partition then distill

After reading the batch:

- Short-circuit unchanged: if `len(rows) < min_inputs`, `_touch_last_run` and
  return `below_threshold` (cheap pre-check on total new activity).
- Otherwise **partition** rows into an ordered map keyed by
  `(space_id, project_id)`, preserving each row's `created_at ASC, id ASC` order.
- For each partition **with `>= min_inputs` rows**, build that partition's
  `combined` text and call `_synthesize_and_store(..., space_id=key.space_id,
  project_id=key.project_id, input_count=len(partition_rows))`.
- Partitions **below `min_inputs`** are skipped (not distilled). Their raw rows
  remain `status='active'` and recallable as project-tagged observations — no
  information loss for recall, just no consolidated fact this tick.
- **Cursor advances over the whole batch** exactly as today: the batch is ordered
  globally by `(created_at, id)`, so `rows[-1]`'s `(created_at, id)` is the
  correct new keyset cursor regardless of partitioning. `_advance_cursor` is
  called once, with `processed_delta = len(rows)` (the full batch).

Threshold rationale: per-partition `min_inputs` means the curator only produces a
fact when a *single project* has real signal — avoiding both noise (distilling
singletons) and a per-project cursor. Sub-threshold rows being passed by the
cursor is acceptable because raw observations still recall per-project.

### 3. `_synthesize_and_store` / `_store_one` — thread the scope through

Both gain keyword params `space_id: Optional[str] = None`,
`project_id: Optional[str] = None`, forwarded to
`write_scoped_memory(..., space_id=space_id, project_id=project_id)` (which
already accepts them). `base_meta` keeps `{"source":"curator","input_count":N}`
where `N` is the **per-partition** count.

Backward compatibility: callers that pass neither get `None` → `NULL`, i.e. a
genuinely org-wide memory — the now-correct meaning of NULL.

### 4. Legacy cleanup — additive migration `023`

`services/migrations/alembic/versions/023_curator_facts_project_cleanup.py`:

```sql
DELETE FROM session_memories
WHERE metadata->>'source' = 'curator'
  AND project_id IS NULL
  AND memory_type IN ('summary','fact');
```

- **Idempotent:** no-op on a fresh DB or a re-run (nothing matches).
- **All orgs:** the bug was global; clean every org with this pattern.
- **Strictly scoped:** only curator-sourced `summary`/`fact` rows with no project.
  Raw observations (any project) and genuine user `remember` notes
  (`source != 'curator'`) are untouched.
- **Downgrade:** a no-op — the deleted rows are regenerable derived data; there is
  nothing to restore.

### 5. Recall — unchanged (documented decision)

`memory.py`'s `(project_id = :project_id OR project_id IS NULL)` stays. After this
fix, NULL is produced only by deliberate org-wide writes, so "NULL surfaces
everywhere" finally behaves as intended. No read-path change keeps the blast
radius small and leaves space/agent scoping semantics intact.

---

## Data flow (after the fix)

```
hooks → raw observations (project_id tagged)  ──┐
                                                 │  curator tick (per org)
                                                 ▼
                          read batch (now incl. project_id, space_id)
                                                 │
                              partition by (space_id, project_id)
                                                 │
                  ┌──────────────┬───────────────┴───────────────┐
              project A       project B                    project_id = NULL
            (≥min_inputs)   (≥min_inputs)               (genuine org-wide)
                  │              │                              │
            distill→facts   distill→facts                 distill→facts
            tagged proj A   tagged proj B                 tagged NULL
                  │              │                              │
                  ▼              ▼                              ▼
        recall in proj A   recall in proj B          recall everywhere (intended)
```

## Error handling

Unchanged from today's curator: `generate_summary`/`extract_facts` failures are
logged and skipped per partition; a `write_scoped_memory` exception in
`_store_one` is swallowed (returns `False`) so one bad write never aborts the
tick. A failure in one partition does not abort the others. The cursor still only
advances after the batch is processed.

## Testing

Curator unit tests (mocked `db_session.execute` + mocked `extract_facts` /
`generate_summary`, matching the existing suite — no live DB):

1. **Per-project tagging:** a batch with rows across two projects →
   `write_scoped_memory` is called for each project with the matching
   `project_id`/`space_id`; facts are tagged per project.
2. **Genuine org-wide:** a partition whose rows have `project_id = NULL` →
   distilled with `project_id=None` (still surfaces everywhere — intended).
3. **Sub-threshold partition:** a partition with `< min_inputs` rows → no
   `write_scoped_memory` call for it; its rows are still counted in the cursor
   advance.
4. **Cursor unchanged:** `_advance_cursor` called once with the whole batch's max
   `(created_at, id)` and `processed_delta = len(rows)`.
5. **`input_count` is per-partition**, not the whole batch.
6. Update existing single-blob curator tests to the partitioned shape.

Migration `023`: verify the DELETE predicate is scoped exactly to
`source='curator' AND project_id IS NULL AND memory_type IN ('summary','fact')`
(inspect SQL / run against a seeded fixture if the migration test harness exists).

## Rollout

1. Land curator change + migration `023` on a branch; local commits only.
2. Tests green (`PYTHONPATH=services/shared` pytest for the curator + shared).
3. Deploy `mcp_server` to Railway (migration runs on deploy); the cleanup removes
   the bleeding NULL facts and the curator produces project-scoped facts onward.
4. Verify via `recall`: a query that previously returned cross-project trivia no
   longer does; new curated facts carry a `project_id`.
