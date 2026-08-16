# Klio recall relevance — diagnosis (2026-08-16)

Measurement only. No production code changed, no writes, no migrations, no deploys.
All database access used a **read-only session** (`psycopg2 set_session(readonly=True)`
for the raw probes; `SET TRANSACTION READ ONLY` per transaction for the SQLAlchemy
replay, which reuses the engine's own `shared.memory.retrieve_memories_scoped`).
Credentials were injected via `railway run --service async-worker` and never printed
or written to disk.

---

## 1. How recall actually works today

### 1.1 Entry point

`POST /capture/recall` → `capture_recall`
(`services/mcp-server/app/capture.py:862-1003`).

Two completely different code paths depending on whether `query` is empty:

| `query` | path | ranking |
|---|---|---|
| empty (`""`) | `_recent_memories` (`capture.py:546-620`) | `ORDER BY created_at DESC` — **no relevance at all** |
| non-empty | `shared.memory.retrieve_memories_scoped` (`memory.py:2219-2666`) | hybrid, below |

`limit` is clamped to 50 (`_MAX_RECALL_LIMIT`, `capture.py:150`).

### 1.2 The actual scoring formula (non-empty query)

**Stage A1 — vector arm** (`memory.py:2475-2494`):

```sql
SELECT * FROM (
    SELECT <cols>, (1 - (embedding <=> :embedding)) AS relevance
    FROM session_memories
    WHERE {filters}
    ORDER BY embedding <=> :embedding          -- HNSW ANN
    LIMIT :candidate_pool                      -- candidate_pool = limit * 4
) candidates
ORDER BY (0.85 * relevance
          + 0.15 * LEAST(1, GREATEST(0, 1 - EXTRACT(EPOCH FROM (NOW()-created_at))
                                          / (30 * 86400)))) DESC
LIMIT :vector_limit                            -- = candidate_pool if lexical live, else limit
```

So: **`score = 0.85·cosine + 0.15·max(0, 1 − age_days/30)`**, applied *only* as a
re-rank inside the top `4·limit` pure-cosine neighbours.

**Stage A2 — lexical arm** (`memory.py:2512-2526`): `ts_rank(fts_document, tsquery)`
over the *identical* filter set, `LIMIT :candidate_pool`.

**Fusion** (`memory.py:2617-2620`): reciprocal rank fusion, vector weight `1.0`
(`lexical.py:106`), lexical weight `1.0` (`DEFAULT_LEXICAL_WEIGHT`, `lexical.py:112`),
`k = 60` (`rrf_k`, default). Truncated to `limit` *after* fusion.
Lexical arm is on by default (`KLIO_LEXICAL_RECALL_ENABLED` unset in prod).

**Stage B/C — graph expansion** (`memory.py:2055-2140`, `_expand_via_graph`):
adds up to `GRAPH_EXPANSION_CAP = 5` (`memory.py:99`) one-hop neighbours, then
re-ranks the merged set by `0.70·similarity + 0.15·recency + 0.15·connectedness`
(`_RERANK_*`, `memory.py:125-127`). Graph rows carry `relevance = None` and
contribute **0** to the similarity term.

**Returned count can exceed `limit` by up to 5.** This matters — see §5, signal 2.

### 1.3 Every filter applied

Vector path (`memory.py:2392-2440`, `2632-2649`):

- `org_id = :org_id`
- `status = 'active'`
- `superseded_by IS NULL`
- `recall_hidden = FALSE`
- scope ladder: `scope='org' OR (scope='private' AND user_id=:user_id)
  OR (scope='project' AND project_id IN (member projects))`
- space: `space_id = :space_id OR space_id IS NULL` (only when a space resolves)
- project: **two-tier** — tier 1 `project_id = :project_id` fills first, tier 2
  `project_id IS NULL` fills the remainder; other projects never surface.
  **Applied only when `project_id` is not `None`. When `None`, no project predicate exists at all.**

Recency path (`capture.py:600-619`): same tenancy/scope/hidden filters, plus a
*flat* project predicate `(:project_id IS NULL OR project_id = :project_id OR project_id IS NULL)`.

**Not present anywhere:** a minimum-similarity threshold, a memory-type weighting,
a per-project boost, an age cutoff, or any read-time near-duplicate collapse.

---

## 2. Is project scoping applied on recall? Mostly **no**.

Three separate defects stack up.

**(a) The engine's read-side project resolver only accepts a git remote.**
`capture.py:453-474` `_find_project_id` returns `None` immediately when
`git_remote` is falsy, and passes only `git_remote` to `brain.find_project_id`.
But `brain.find_project_id` (`brain.py:413-455`) already resolves by
UUID → `git_remote` → `repo_root_path` → `display_name`, and its own docstring says
*"in production two thirds of projects are path-only"*. The write path
(`_ensure_project_id`, `capture.py:411-450`) passes **both** `git_remote` and
`repo_root` and files the memory. So writes file by path; reads refuse to.

Measured: **79 of 118 projects in org `vex` have `git_remote IS NULL`** — including
`Astrology`, `Agent Memory`, `hireloop`, `apply_for_job`. Every memory written from
those directories is filed, and is un-filterable on recall.

**(b) The Node proxy sends no project at all.**
`npm/src/proxy/recall.ts:302`:

```ts
body: JSON.stringify({ query, limit: RECALL_LIMIT, scope: "org" })
```

No `git_remote`, no `repo_root`, no `session_id`. Every proxy recall — ambient and
per-query — is org-wide across all 118 projects. (The `klio hook` client *does* send
them: `npm/src/commands/hook.ts:171,209,247,269` via `resolveProject`.)

**(c) The result, measured in production `brain_recall_events`:**

| source | project resolved | project NULL | % unfiled |
|---|---|---|---|
| `hook` | 33 | **604** | **94.8 %** |
| `mcp` | 4 | 31 | 88.6 % |

Meanwhile, on the write side **23,895 of 25,101** visible active `vex` memories
(95.2 %) *are* filed under a project.

> **95 % of memories are written with a project. 95 % of recalls ignore it.**

**What happens when project is absent:** nothing narrows. The vector path takes the
`else` branch `_fetch("TRUE", {})` (`memory.py:2649`); the recency path's
`:project_id IS NULL` short-circuit admits every row. An astrology fact competes
with a proxy fact on nothing but generic cosine + freshness.

---

## 3. Reproduction against production (read-only)

### 3.1 The live symptom is the **recency** path, not the vector path

The `klio hook` SessionStart handler sends `query: ""` (`hook.ts:164`), so
SessionStart injection is `ORDER BY created_at DESC LIMIT 12` — pure freshness,
zero relevance.

The four leaked memories are real rows, all `scope='private'`, all filed:

| content | project | created |
|---|---|---|
| "The varga engine correctly handles D30 with unequal Trimsamsa portions…" | Astrology | 2026-08-14 |
| "The Jupiter dasha start date of 22 June 2028 was confirmed…" | Astrology | 2026-08-14 |
| "EXTENSION_READY_MESSAGE_TYPE is defined as the constant 'EXTENSION_READY'." | hirly | 2026-08-16 |
| "A Supabase database container named supabase_db_hirly is running… port 54422" | hirly | 2026-08-16 |

Replaying `_recent_memories` verbatim with the owning `user_id` and
`project_id = NULL` (i.e. exactly what production does 95 % of the time) returns the
same shape today:

```
decision | org     | (unfiled)                     | 2026-08-16 | CAPTCHA solving is in scope for Hirly…
note     | org     | (unfiled)                     | 2026-08-16 | hireloop already kept Steel sessions alive…
plan     | org     | (unfiled)                     | 2026-08-16 | Hirly captcha unstick plan…
fact     | private | Agent Memory                  | 2026-08-16 | The savepoint is used inside the entity extraction loop…
fact     | private | Agent Memory                  | 2026-08-16 | The probe uses URL postgresql://…
fact     | private | Agent Memory                  | 2026-08-16 | The test was edited to also patch the embedding service…
fact     | private | Agent Memory                  | 2026-08-16 | A second probe (pool2.py) exercised concurrent session…
fact     | private | Agent Memory                  | 2026-08-16 | A probe script demonstrated that an un-rolled-back session…
summary  | private | Agent Memory                  | 2026-08-16 | The agent ran a static-analysis script…
summary  | private | Agent Memory                  | 2026-08-16 | The agent ran an AST scan…
fact     | private | Agent Memory                  | 2026-08-16 | The commit message explicitly lists shared/usage.py…
fact     | private | next-supabase-saas-kit-turbo  | 2026-08-16 | `ProfileServiceDependencies` interface is exported…
```

Whatever project you are sitting in, you get the last 12 things written *anywhere*.

### 3.2 Vector path, real queries, `limit=8`, `scope='org'`

Run through `retrieve_memories_scoped` itself. Left column = production today
(`project_id=None`); right = the same call with the `Klio` project resolved.

**"how does the proxy inject memories"**

*project=None (production):* 4 rows returned (HNSW under-fill)
| # | sim | project | age | content |
|---|---|---|---|---|
| 1 | 0.854 | .buzz | 18.3d | "The agent is instructed to ask the user about itself if no core memory is found." |
| 2 | 0.850 | .buzz | 18.3d | "The agent has no core memory set." |
| 3 | 0.846 | .buzz | 18.3d | "The agent has no core memory set and is instructed to ask…" |
| 4 | 0.849 | moonforge-prod | 68.1d | "Assistant committed the history-reconcile placeholder and pushed" |

*project=Klio:* 8 rows, all on-topic — Klio positioning, Klio vs Executor, Klio
contract decisions, Klio onboarding copy (sim 0.78–0.83).

**"what did we decide about capture defaults"**

*project=None:* relio "unresolvable posts marked as named defects" (0.831),
loop_v2 subagent statuses (0.827), moonforge `CaptureException` signatures (0.835),
OpenOutreach BLOCKED-status guidance (0.829). Nothing about Klio capture.

*project=Klio:* Klio Cloud conversion decision, engine feature-freeze decision,
Klio contract decisions, Klio capture smoke test.

**"why did we stop wiring Claude Code"**

*project=None:* `.buzz` "global default pinned to claude-opus-5" (0.846),
"Relio asked which LLM is being used" (0.840), Agent-Reach CLAUDE.md PR rule (0.846).

*project=Klio:* Klio model strategy, engine feature-freeze, Klio contract decisions,
"Grok-runtime agents do not inherit Claude Code MCP config" (0.836).

**"team project context: key decisions, conventions, architecture, and current work"**
(the proxy's `AMBIENT_QUERY`)

*project=None:* linkedin_automation "Focus on understanding: purpose, constraints,
success criteria." (0.871), oppla-main "check current project state first" (0.865),
OpenOutreach model-selection guidance (0.862) — generic process boilerplate from five
unrelated projects.

*project=Klio:* Klio production `project_members` finding, Klio deploy topology,
Klio vs Executor, Klio docs positioning.

**Zero `via=graph` and zero `via=lexical` rows appeared in any of the 8 runs.**
`memory_edges` holds only 1,679 rows for org `vex` against 25,101 visible memories,
so graph expansion almost never fires.

### 3.3 Why an astrology fact beats a proxy fact — the numbers

Similarity of *every* `scope='org'` active memory (n = 18,367) against each query:

| query | max | p99.9 | p99 | p90 | median | min |
|---|---|---|---|---|---|---|
| how does the proxy inject memories | 0.8538 | 0.8376 | 0.8185 | 0.7971 | 0.7728 | 0.6899 |
| what did we decide about capture defaults | 0.8429 | 0.8272 | 0.8154 | 0.7975 | 0.7754 | 0.7093 |
| why did we stop wiring Claude Code | 0.8462 | 0.8329 | 0.8174 | 0.7962 | 0.7715 | 0.6906 |
| team project context (ambient) | 0.8711 | 0.8428 | 0.8237 | 0.8011 | 0.7734 | 0.6779 |
| Vedic astrology dasha calculation | 0.8528 | 0.8231 | 0.7911 | 0.7666 | 0.7431 | 0.6742 |

**The whole corpus lives in a 0.08-wide band.** The single best match out of 18,367
scores ~0.085 above a *randomly chosen* memory, and only ~0.016 above the 0.1 %
tail. There is essentially no signal to rank on.

Now put that against the blend:

- Similarity's total contribution span, best-in-corpus → median:
  `0.85 × (0.854 − 0.773) = 0.069`
- Recency's contribution span, brand-new → 30 days old: **`0.15`**

**Freshness is worth more than the entire dynamic range of relevance.**
Concretely: each day of age costs `0.15/30 = 0.005` blended points, equal to
`0.0059` of cosine. The gap between the single best memory in the corpus and a
90th-percentile one is 0.057 cosine ≈ **9.7 days**. A mediocre memory written last
week outranks the best memory in the database if the latter is ten days older.

Root cause of the compressed band: the production embedding model is
`intfloat/multilingual-e5-large-instruct` (`MEMORY_EMBEDDING_MODEL` on `mcp_server`;
also `DEFAULT_EMBEDDING_MODEL`, `memory.py:76`). E5 models are trained with
mandatory asymmetric prefixes — `query: …` for queries, `passage: …` for documents —
and the `-instruct` variant expects `Instruct: <task>\nQuery: <text>`. The engine
sends raw text with no prefix on either side: `_embed_sync`
(`memory.py:413`) builds `{"model": model, "input": [text_input]}` and nothing else,
and `_embed_text_sync` / `embed_text` do the same. Using E5 unprefixed is the
textbook cause of exactly this symptom: everything cosine-similar to everything,
~0.75–0.87 for arbitrary pairs.

---

## 4. Corpus characterisation

Org `vex`: 49,146 rows total; **25,101 active + visible** (`status='active'`,
`recall_hidden=FALSE`). Everything below is that visible set.

**By type:**

| type | count | share |
|---|---|---|
| fact | 22,805 | 90.9 % |
| summary | 2,176 | 8.7 % |
| memory | 45 | 0.18 % |
| note | 40 | 0.16 % |
| decision | 26 | 0.10 % |
| plan | 4 | 0.02 % |

**115 deliberate memories (0.46 %) against 24,981 auto-mined ones.** The brain is
99.5 % transcript exhaust.

**By project (visible, active — top rows):** loopreel 2,047 · moonforge-prod 3,217
(org+private) · Agent Memory 2,023 · Growth App 1,683 · relio 2,076 · moonforge-qual
1,224 · hirly 860 · Astrology 843. No single project dominates — the tail is what
hurts: **118 projects all pouring into one undifferentiated pool.**

**Filed vs unfiled:** 23,895 filed / 1,206 unfiled (4.8 %). Note the consequence for
the two-tier filter: even when a project *does* resolve, tier-2 backfill pulls from
those 1,206 unfiled rows, which is where nearly all deliberate `note`/`decision`/`plan`
memories live (they are written through MCP without a project).

**Junk and duplicates:**
- 461 distinct content prefixes appear more than once; **737 redundant rows**.
- Exact-duplicate content up to 10×: *"A Stripe test key was provided."* (×10),
  *"The user requested extraction of discrete, verifiable facts from the provided
  agent turn."* (×6 — the extractor storing its own prompt as a fact),
  *"The engine runs on the host at 127.0.0.1:8000."* (×4).
- **2,563 of 22,805 facts (11.2 %)** match transient-state patterns
  (`is running`, `currently`, `exited`, `tests pass`, `committed`, `PR #…`).
  Examples from the leak itself: *"A Supabase database container named
  supabase_db_hirly is running locally on port 54422"* — true for twenty minutes,
  stored forever, with a recency bonus that makes it *more* likely to be injected
  the fresher it is.
- 3,938 rows are under 60 characters; 22,732 under 120.

**Age distribution** is fine — 6,311 rows under 10 days old — but that is precisely
the fuel for the recency term.

**Growth:** 1,850 facts on 2026-08-03, 1,261 on 08-05, 445 on 08-07. Every session
mines ~15–30 durable "facts" that are mostly session trivia.

**Embeddings:** 0 rows with a NULL embedding — not an embedding-coverage problem.

---

## 5. The second signal (ambient 13 beat query-specific 8)

This is not evidence that generic queries score better. Two mechanical explanations,
both measured:

1. `RECALL_LIMIT = 8` (`recall.ts:100`) for both, but
   `retrieve_memories_scoped` may return **up to `limit + GRAPH_EXPANSION_CAP` = 13**
   (`memory.py:2344-2352`). 13 is exactly `8 + 5`: the ambient set got a full graph
   expansion, the query-specific set got none. The ambient answer was better because
   it contained **five more memories**, not because it ranked better.
2. Given the §3.3 measurement, a "specific" query has essentially no advantage to
   surrender: the top-8 for *any* query in this corpus is within 0.02 cosine of the
   top-8 for any other. Both sets are near-random draws from the p99 band; the
   larger draw won.

---

## 6. Root causes, ranked

### RC-1 — Project scoping is written but not read (CRITICAL)

**Evidence:** 95.2 % of memories are filed; 94.8 % of hook recalls resolve no
project. `_find_project_id` (`capture.py:453-474`) refuses everything but a git
remote, while `brain.find_project_id` (`brain.py:413-455`) already supports
`repo_root_path` and 79/118 `vex` projects have no remote. The proxy
(`recall.ts:302`) sends no project field at all. Side-by-side replay in §3.2 shows
project-scoped recall returning on-topic Klio memories where unscoped recall returns
`.buzz`, `moonforge-prod`, `loop_v2` and `OpenOutreach`.

**Fix:**
1. `capture.py:_find_project_id` — pass `git_remote or repo_root` to
   `brain.find_project_id` (it already ranks by specificity, so a path match is safe
   and deterministic).
2. `recall.ts` — thread `git_remote`/`repo_root` from the proxy's cwd into the
   ambient *and* per-query recall bodies, and key the cache by project too.
3. Consider making the project predicate a hard fence for `scope='private'` rows
   (a private fact filed under project X is almost never wanted in project Y),
   keeping the two-tier softness only for `scope='org'`.

**Expected improvement: large — the single highest-leverage change.** In §3.2 it
turned 0/4 relevant into 8/8 plausibly-relevant on the proxy query, and removed
every cross-project leak of the kind reported.

### RC-2 — SessionStart injection has no relevance component at all (CRITICAL)

**Evidence:** `hook.ts:164` sends `query: ""`; `capture.py:915-924` routes that to
`_recent_memories`, which is `ORDER BY created_at DESC LIMIT 12` with no embedding,
no similarity, no threshold. The reproduction in §3.1 returns exactly the reported
symptom class.

**Fix:** at SessionStart, recall on *something* — the repo's `CLAUDE.md` topic
sentence, the git branch name, the last N session summaries for this project, or at
minimum the project display name — rather than the empty string. Or make the empty
query path project-*fenced* (`project_id = :project_id` only, no `OR NULL`) so
freshness at least stays inside the working repo.

**Expected improvement: large for the reported symptom specifically** — this is the
path the four astrology/hirly facts actually came through.

### RC-3 — Embeddings are used without their required E5 prefixes (HIGH)

**Evidence:** model is `intfloat/multilingual-e5-large-instruct`; `_embed_sync`
(`memory.py:413-421`) sends bare text for both queries and stored content. Measured
consequence: 18,367 memories compressed into a 0.08-wide similarity band; best match
only 0.085 above the median memory (§3.3).

**Fix:** prefix stored content with `passage: ` and queries with `query: ` (or the
instruct template). **This requires re-embedding the corpus** — mixing prefixed and
unprefixed vectors in one index is worse than either. Alternatively evaluate a
retrieval-tuned model that needs no prefixes (e.g. BGE-M3, `text-embedding-3-large`)
and re-embed once.

**Expected improvement: high, and it is the prerequisite for RC-4 mattering.** With
a properly-prefixed E5, the top-1-vs-median gap on these queries should widen from
~0.08 to ~0.25–0.35, which is what makes a threshold meaningful.

### RC-4 — Recency outweighs the entire relevance range (HIGH)

**Evidence:** recency span 0.15 vs similarity span 0.069 (§3.3). Break-even is ~10
days of age against a top-1-vs-p90 relevance gap.

**Fix:** two options, in order of preference —
(a) after RC-3, keep 0.85/0.15 (it becomes reasonable once similarity has range);
(b) until RC-3 lands, drop the recency weight to ~0.05 and *normalise* similarity
within the candidate pool (`(sim − min)/(max − min)`) so the 0.85 term actually
discriminates. Do not raise recency.

**Expected improvement: moderate on its own, high in combination with RC-3.**

### RC-5 — No relevance floor; recall always returns `limit` rows (HIGH)

**Evidence:** there is no threshold anywhere in `retrieve_memories_scoped`. The
"how does the proxy inject memories" run returned 4 memories about an agent's core
memory being unset, at 0.85 — indistinguishable from a real match. `SIMILARITY_THRESHOLD`
(`memory.py:66`) exists but is the *dedup* threshold on the write path only.

**Fix:** introduce a recall floor (post-RC-3, something like 0.55–0.65 absolute, or a
relative floor `sim >= max_sim − δ`), and let recall return **fewer** memories —
including zero. Injecting nothing is strictly better than injecting four wrong things.

**Expected improvement: high for perceived quality**, because it converts "confidently
wrong context" into "no context".

### RC-6 — The corpus is 99.5 % auto-mined session exhaust (MEDIUM)

**Evidence:** 22,805 facts vs 115 deliberate memories; 11.2 % of facts match
transient-state patterns; 737 redundant rows; the fact extractor has stored its own
prompt 6 times.

**Fix:**
- Tighten `extract_facts` prompting toward durable claims (decisions, constraints,
  conventions, API contracts) and away from run-state observations.
- Add a TTL / decay for `memory_type='fact'` from the hook-transcript source, or mark
  them `recall_hidden` as curator fuel and let the curator promote the durable subset
  (`recall_hidden` machinery already exists — `capture.py:703` uses it for
  observations only).
- Weight `decision`/`note`/`plan`/`memory` above `fact` in the blend.
- Read-time near-duplicate collapse on the returned set.

**Expected improvement: moderate, compounding.** It also reduces cost.

### RC-7 — `limit` / candidate-pool values (LOW–MEDIUM)

**Evidence:** `candidate_pool = limit × 4` = 32 for `limit=8`. Recency can only
re-rank inside those 32 pure-cosine neighbours, so the blend does much less than the
formula suggests. Separately, the "how does the proxy inject memories"
`project=None` run under-filled to 4 rows — HNSW returning fewer candidates than
requested under the scope filters.

**Fix:** raise the pool (e.g. `max(50, limit × 8)`) and/or raise `hnsw.ef_search` for
recall queries. Cheap, low risk, but only worth doing after RC-1/RC-3.

### Not a root cause

- **Embedding coverage** — 0 rows with NULL embeddings.
- **Cross-org leakage** — `org_id` is a SQL predicate on every arm; nothing crossed
  tenants.
- **`recall_hidden`** — working; 16,663 org and 7,086 private observations correctly
  excluded.
- **Graph expansion polluting results** — it barely runs (1,679 edges for 25,101
  memories) and contributed nothing in 8/8 replays.

---

## 7. Recommended order

1. **RC-1** (project on recall — engine `_find_project_id` + proxy body). Days of
   work at most, removes the reported class of leak outright.
2. **RC-2** (stop injecting on an empty query, or fence it to the project).
3. **RC-5** (relevance floor + allow empty results). Cheap, immediately visible.
4. **RC-3** (E5 prefixes + corpus re-embed). The big one for ranking quality;
   needs a migration plan since it invalidates existing vectors.
5. **RC-4** (rebalance the blend once similarity has range).
6. **RC-6** (fact-extraction discipline, decay, type weighting).
7. **RC-7** (pool / `ef_search` tuning).

RC-1 and RC-2 together address the reported symptom. RC-3/RC-4/RC-5 address the
underlying "why does anything score 0.85 against anything".
