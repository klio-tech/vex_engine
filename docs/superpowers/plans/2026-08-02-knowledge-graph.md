# Knowledge Graph Implementation Plan — phases 1–3

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. The spec (`docs/superpowers/specs/2026-08-02-knowledge-graph-design.md`) is the contract; where this plan and the spec disagree, the spec wins.

**Goal:** Entities and typed edges as first-class rows behind a `GraphStore` seam, the curator emitting the graph, and recall walking it — with the visibility ladder enforced on both endpoints of every edge.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy Core (`text()` + bound params), Alembic, pytest, ruff. Migration head is `034`; this adds `035`.

## Global Constraints

- **Traversal obeys the ladder on both endpoints, in SQL.** Never fetch wide and filter in Python. This is a tenancy boundary with the same rigor as `org_id`.
- **Entity visibility is derived from evidence** — an entity is visible iff the caller can read ≥1 memory evidencing it. `not_found` (never an empty brief) when they cannot: an empty brief confirms the name exists.
- **Everything the curator draws is `INFERRED`.**
- **Relations and kinds are CHECK-constrained closed sets.** `about|supersedes|derived_from|relates_to`; `system|person|concept|decision_subject|tool|other`.
- **No SQL against `entities`/`memory_edges` outside `graph_store.py`.** That module is the future engine-swap boundary.
- **`forget` must never fail because of an edge.** Orphan edges are swept, not FK-cascaded.
- **project/author remain columns, not edges.** They are authorization inputs; two sources of truth is a security bug.
- Existing counts must not drop: shared 436+1skip, mcp-server 350+9skip, migrations 34.
- TDD with observed RED; ruff clean per commit; bound params everywhere.

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `services/migrations/alembic/versions/035_knowledge_graph.py` | entities, memory_edges, indexes, supersedes backfill | Create |
| `services/migrations/tests/test_migration_035.py` | schema + backfill honesty assertions | Create |
| `services/shared/shared/graph_store.py` | the seam: add_entity/add_edge/neighbors/entity_brief | Create |
| `services/shared/tests/test_graph_store.py` | contract tests incl. upsert dedup | Create |
| `services/shared/tests/test_graph_isolation.py` | adversarial ladder tests, both endpoints | Create |
| `services/mcp-server/app/curator.py` | emit entities+edges; reconciliation writes supersedes edges | Modify |
| `services/mcp-server/tests/test_curator_graph.py` | extraction contract, god-entity guards | Create |
| `services/shared/shared/memory.py` | stage-B one-hop expansion in `retrieve_memories_scoped` | Modify |
| `services/mcp-server/app/mcp_app.py` | `explore` tool | Modify |
| `services/mcp-server/tests/test_explore_tool.py` | brief assembly + leak refusals | Create |

## Tasks

### Task 1: Migration 035
Schema exactly as the spec's SQL block. Backfill `supersedes` edges only where the successor id is recoverable from reconciliation metadata — where it is not, **no edge is invented**; the test asserts edge count ≤ superseded-row count and every backfilled edge's endpoints both exist. Exercise the downgrade (`035→034→035`), then re-run tests. Mirror `test_migration_031.py` conventions (`DATABASE_URL`, real Postgres, skip-is-not-pass).

### Task 2: GraphStore
The four functions from the spec. `add_entity` upserts on `(org_id, norm_name, kind)` via `INSERT ... ON CONFLICT ... RETURNING` (race-safe, same pattern as `ensure_project`). `neighbors(db, *, org_id, viewer_user_id, memory_ids, depth=1)` returns only edges whose destination memory passes the caller's ladder — build the ladder predicate once, shared with `entity_brief`. God-entity blocklist + kind requirement live here (`add_entity` refuses), so no writer can bypass them.

**Adversarial tests (the acceptance criterion), mirroring `test_user_silo_isolation.py`'s real-Postgres fixture:**
- Priya's private memory about an org entity: absent from Dev's `neighbors`/brief; **control:** present in Priya's.
- Entity evidenced only by Priya's private rows: `entity_brief` → `not_found` for Dev, for an org admin, and for another org; **control:** full brief for Priya.
- A cross-org edge (seeded directly in SQL, simulating a bug or tamper): grants nothing — the org join on the destination row wins.
- An edge whose destination was `forget`-ed: silently absent, and the sweep removes it.
- **Mutation checks, mandatory:** deleting the viewer ladder predicate from `neighbors` must fail ≥3 tests; deleting the destination `org_id` join must fail ≥2. Report both observations.

### Task 3: Curator emits the graph
Extend the per-user tick's extraction prompt to return `entities` and `edges` alongside facts (spec JSON shape). Upserts via GraphStore only. Per-tick cap on new entities (default 12) — overflow logged, not stored. Reconciliation (`reconcile_fact`) additionally writes a `supersedes` edge at the moment it flips status: it already made the judgment; recording it costs one insert. All curator writes `INFERRED`. Tests pin: emitted SQL goes through GraphStore, cap enforced, blocklisted names refused, edges land `INFERRED`, and a curator tick for user A creates nothing readable by user B (reuse the isolation fixture).

### Task 4: Recall stage B + `explore`
In `retrieve_memories_scoped`: after the vector stage, one `neighbors()` call on the entry-point ids; merge, re-rank `0.70·similarity + 0.15·recency + 0.15·connectedness` (connectedness = normalized edge count into the entry set); cap expansion at +5 rows so payloads stay bounded; structural compression unchanged downstream. **Expansion must never introduce a row the ladder wouldn't return directly** — property-style test seeding a connected-but-forbidden row and asserting absence at every scope argument.

`explore(entity, ...)` MCP tool: resolve entity by `norm_name` within org → derived-visibility check → brief per the spec shape (decisions in temporal order, supersessions with dates, contributor *counts* never names-of-private-authors, provenance tally). Register in the tool inventory tests (count 17→18, names asserted).

### Task 5: Ship
Full suites + ruff + container import on `--network none`. Run migration `035` against local Postgres and exercise the downgrade. **Do not touch production** — the Neon migration runs after review, as with `031`–`034`. Push branch `feat/knowledge-graph`, **no PR, no merge**.

## Report back
Branch, commits, counts before/after, RED evidence per task, both mutation-check observations, backfilled edge count vs the 26 superseded rows, a sample `explore` brief from seeded data, and anything in the spec that proved wrong or impossible — reported, not worked around.
