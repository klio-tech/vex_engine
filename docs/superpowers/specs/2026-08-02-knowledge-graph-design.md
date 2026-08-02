# Klio Knowledge Graph — memory that understands relationships

**Date:** 2026-08-02
**Status:** Design, approved in conversation. Supersedes nothing.
**Builds on:** the user silo (spec 2026-08-01), provenance labels, structural compression.

---

## The problem

Klio holds 41,765 memories whose relationships exist only as columns:

| Relationship | Where it hides today | Traversable at recall? |
|---|---|---|
| supersession | `status='superseded'` (26 rows) | no — excluded, never explained |
| derivation | `metadata.source='curator'` (1,874 rows) | no |
| project grouping | `project_id` (36,602 rows, 94 projects) | filter only |
| authorship | `user_id` (1,450 rows) | filter only |

Recall is pure similarity + recency. It cannot answer "what led to this
decision", "what do we know about the auth service", or "what replaced this
fact and why". Ask about a topic and you get ten lookalike rows, not a
structured answer.

Graphify proved relationships are the right model — but its extractor is
tree-sitter, so it can only ever graph *code*. Klio's knowledge is decisions,
systems, people, facts, and code. The extractor for that class of content is
an LLM pass, and **Klio already runs one on every memory: the curator.**

## What this is

Three additions, in dependency order:

1. **Entities and typed edges as first-class rows** (migration `035`), behind
   a `GraphStore` seam.
2. **The curator becomes the graph builder** — it emits entities and edges as
   it distils, every one stamped `INFERRED`.
3. **Recall learns to walk** — vector entry points, one-hop expansion within
   the visibility ladder, and a new `explore` tool that returns an entity
   brief instead of disconnected rows.

## Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | **Graph model on Postgres; engine behind a seam** | The model (entities + typed edges + vectors) is the asset. A graph-native engine (HelixDB-class) is a deliberate later swap when traversal depth demands it — the `GraphStore` module is the only file that changes. We do not build a storage engine. |
| 2 | **The curator is the sole automatic graph writer in v1** | It already reads every memory, already runs per-user, and its output is already `INFERRED`. No new LLM pass, no new cost centre — an extended prompt on an existing one. |
| 3 | **Entity visibility is DERIVED, never stored** | An entity is visible to a caller iff the ladder grants them ≥1 memory evidencing it. No stored entity scope to drift out of sync, no promotion bookkeeping — share a memory and every entity it evidences becomes visible downstream automatically. |
| 4 | **Traversal obeys the ladder on both endpoints of every edge** | Edges are the new leak class. A private memory linked to an org entity must never surface through a walk to someone who could not read it directly. |
| 5 | **Relations are a closed set, CHECK-constrained** | `about`, `supersedes`, `derived_from`, `relates_to`. Scope had no CHECK once and typos created invisible rows; edges start life constrained. |
| 6 | **project/author stay as columns, not edges** | They are already sound SQL predicates and part of the security ladder. Duplicating them as edges creates two sources of truth for authorization inputs. |

## Schema (migration `035`)

```sql
entities (
  id          UUID PK,
  org_id      VARCHAR(64) NOT NULL,          -- tenancy, as everywhere
  name        VARCHAR(200) NOT NULL,
  norm_name   VARCHAR(200) NOT NULL,          -- lower/trimmed, for dedup
  kind        VARCHAR(32) NOT NULL,           -- CHECK: system|person|concept|decision_subject|tool|other
  embedding   vector,                         -- entity-level semantic entry point
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (org_id, norm_name, kind)
)

memory_edges (
  id          UUID PK,
  org_id      VARCHAR(64) NOT NULL,
  src_id      UUID NOT NULL,                  -- always a session_memories.id
  dst_kind    VARCHAR(8) NOT NULL,            -- CHECK: memory|entity
  dst_id      UUID NOT NULL,
  relation    VARCHAR(16) NOT NULL,           -- CHECK: about|supersedes|derived_from|relates_to
  provenance  VARCHAR(16) NOT NULL DEFAULT 'INFERRED',  -- CHECK, same labels as memories
  created_by  VARCHAR(64),                    -- author when human-drawn (v2)
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (org_id, src_id, dst_kind, dst_id, relation)
)
-- indexes: (org_id, src_id), (org_id, dst_kind, dst_id) — traversal both ways
```

**No FK from edges to `session_memories`** on `dst_id` (it is polymorphic);
integrity is enforced by the store layer and asserted by tests. Orphan edges
(memory forgotten) are swept, not cascaded — an edge must never be the reason
a `forget` fails.

### Backfill (the only ones)

- `supersedes` edges from the 26 `status='superseded'` rows **where the
  successor id is recoverable** from reconciliation metadata. Where it is not,
  no edge — we do not invent history.
- `derived_from` for past curator rows is **not** backfilled: input ids were
  never recorded. It accrues from the first post-deploy curator tick.

## The GraphStore seam

`services/shared/shared/graph_store.py` — the single module that knows the
graph is stored in Postgres:

```
add_entity(db, *, org_id, name, kind) -> entity_id          # upsert on norm_name
add_edge(db, *, org_id, src_id, dst_kind, dst_id, relation, provenance)
neighbors(db, *, org_id, viewer_user_id, memory_ids, ...) -> edges+rows  # LADDER-SCOPED
entity_brief(db, *, org_id, viewer_user_id, entity) -> brief             # LADDER-SCOPED
```

Every read function takes the viewer and applies the ladder **in SQL**. This
module is the future HelixDB boundary: its contract is graph+vector, its
implementation is Postgres, and nothing outside it may write SQL against
`entities` or `memory_edges`. A later engine swap reimplements one file.

## The curator as graph builder

The per-user curator tick already reads a batch of RAW private memories and
emits facts/summaries. Extended output per batch:

```json
{ "facts": [...],
  "entities": [{"name": "auth service", "kind": "system"}],
  "edges":    [{"src": "<fact idx>", "about": "auth service"},
               {"src": "<fact idx>", "relates_to": "rate limiter"}] }
```

- Entities upsert on `(org_id, norm_name, kind)` — the org-level entity row is
  shared; **visibility still derives from evidence**, so an entity evidenced
  only by Priya's private memories is visible only to Priya even though the
  row exists once per org.
- Every curator edge and entity association is `INFERRED`. The existing
  `supersedes` reconciliation now ALSO writes a `supersedes` edge (it already
  makes the judgment; today it only flips status).
- **God-entity guard** (graphify's hardest-won lesson, issue #726 there): a
  blocklist of trivial names ("the code", "the user", "the system", days,
  languages), a `kind` requirement, and a per-tick cap on new entities. A
  degree cap alarms — an entity accumulating edges from everywhere is a
  extraction bug, not knowledge.

## Recall learns to walk

**Stage A (unchanged):** vector search returns entry-point memories under the
existing ladder.

**Stage B (new):** one-hop expansion — `neighbors()` pulls memories connected
to the entry points via edges, **already ladder-filtered in SQL**. Depth is
hard-capped at 1 in v1.

**Stage C:** blended re-rank (similarity + edge-connectedness + recency),
then structural compression as today. Connected memories that would never
surface on similarity alone — the *why* behind a decision — now can.

**The `explore` tool (the demo):**

```
explore(entity: "pricing") →
{ "entity": "pricing", "kind": "concept",
  "summary": "...",                         -- synthesized, INFERRED
  "decisions": [...in temporal order...],
  "superseded": [{"old": ..., "by": ..., "when": ...}],
  "contributors": ["you", "2 teammates"],   -- counts, never private contents
  "provenance": {"EXTRACTED": 12, "INFERRED": 5} }
```

Assembled from the caller's **full cascade — private + projects + org —
evaluated together**, showing each caller only their visible slice. Two
teammates exploring the same entity may legitimately see different briefs.
That is the feature, not a bug.

## Security invariants (tested, mutation-checked)

1. **Both-endpoint ladder check.** `neighbors()` and `entity_brief()` filter
   the *destination* memories by the caller's ladder, in SQL, always.
   Adversarial tests: Priya's private memory about an org entity must never
   appear in Dev's brief; a control proves it appears in Priya's.
2. **Entity existence must not leak.** An entity evidenced solely by another
   user's private memories returns `not_found` to everyone else — not an
   empty brief, which would confirm the name exists.
3. **Cross-org edges cannot grant.** Same shape as the project_members hole
   the silo tests caught: every traversal join carries `org_id` on both the
   edge and the destination row.
4. **Mutation checks required:** deleting the ladder predicate from
   `neighbors()` must fail ≥3 tests; deleting the org join must fail ≥2.

## Risks

| Risk | Mitigation |
|---|---|
| Traversal leaks a private memory | Invariants 1–4 above; same adversarial pattern as the silo, before merge |
| God-entities hairball the graph | Blocklist + kind requirement + per-tick cap + degree alarm |
| Curator extraction quality is poor | Everything is `INFERRED` and per-user private until promoted; bad edges pollute one person's view, not the team's. Measure precision on our own corpus before enabling for the org scope. |
| Entity dedup splits ("auth" vs "auth service") | Accepted in v1; `norm_name` catches trivial variants; alias merging is v2 |
| Postgres traversal too slow someday | Depth-1 cap + indexes now; the GraphStore seam is the engine swap when a real latency signal appears |

## Out of scope (v1)

Multi-hop traversal, entity merge/alias UI, human-drawn edges via MCP,
dashboard graph visualization, HelixDB embedding, AST/code graphing (artifacts
+ graphify-style extraction is its own later spec).

## Phasing — each independently shippable

1. **Schema + GraphStore + supersedes backfill.** No behaviour change;
   reconciliation starts writing edges. Verifiable by tests alone.
2. **Curator emits the graph.** Private, per-user, `INFERRED`. The graph
   grows silently; nothing reads it yet.
3. **Recall expansion + `explore`.** The visible feature, gated on the
   invariants passing adversarially.
