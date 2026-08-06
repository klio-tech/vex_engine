# Lexical recall arm — design

**Date:** 2026-08-06
**Status:** Design.
**Baseline:** `main` @ `8dd5dc5`.

---

## Why

Recall today is a single retrieval strategy: an HNSW cosine ANN fetch over
`embedding`, re-ranked by `0.85 * similarity + 0.15 * recency`, then expanded one
hop through the knowledge graph. There is no lexical arm anywhere in the engine.
Confirmed by grep: no `tsvector`, no `to_tsquery`, no GIN index.

Embeddings are weakest exactly where an agent's queries concentrate. A caller
searching for `KLIO_METRICS_TOKEN`, `xmax = 0`, `unsupported startup parameter`,
or `pool_pre_ping` is looking for a rare literal string, not a paraphrase. Dense
vectors compress rare tokens toward their neighbourhood, so the one row that
contains the exact identifier can rank below several rows that are merely about
the same topic.

This is not a hypothesis we invented. Cerebras shipped an internal knowledge base
on the same shape we have, a single Postgres table holding embeddings plus
metadata for Slack, GitHub, Jira and design files, serving over 15,000 questions
a day. Their reported negative result is that **vector search alone was
insufficient**, and their fix was to blend four signals: full text for exact
tokens, embeddings for paraphrase, IDF to surface rare terms, and age decay.

We already have the equivalent of their age decay and, through the graph, a
signal they do not have. We are missing the lexical arm.

## What this is not

Not a rewrite of ranking. Not a change to what any caller is entitled to read.
The tenancy predicates, the scope ladder, the project two-tier fill and the graph
expansion all stay exactly as they are.

## Decisions

### 1. Fuse by Reciprocal Rank Fusion, not by weighting the scores

Cosine relevance is bounded in [0, 1]. `ts_rank` is unbounded, corpus dependent
and shifts as the corpus grows. Blending two incomparable scores with fixed
weights requires a normalisation that silently drifts as data accumulates, which
shows up as a slow ranking regression nobody can date.

RRF ranks instead of scores:

```
fused(row) = Σ_arms  weight_arm / (k + rank_arm(row))
```

Only ordinal position matters, so the two arms never need a common scale.
`k = 60` is the conventional constant and damps the tail so a row ranked first in
one arm cannot be swamped by a row ranked tenth in both.

**The invariant that makes this safe to review.** When the lexical arm returns no
rows, RRF over a single ranked list is a strictly decreasing function of that
list's rank, so the fused order is *identical* to today's order. Every existing
recall behaviour is preserved by construction whenever lexical contributes
nothing, which is the case for an empty query, a stopword-only query, a
disabled feature flag, or a lexical failure. This is an invariant we test, not a
claim we make.

### 2. An expression index, not a generated column

A `GENERATED ALWAYS AS (...) STORED` column takes an `ACCESS EXCLUSIVE` lock and
rewrites the whole table. Against a live production table behind a pooled
endpoint that is real risk for no benefit here.

```sql
CREATE INDEX ix_session_memories_content_fts
    ON session_memories USING GIN (to_tsvector('english', left(content, 100000)));
```

Index build only, no rewrite. The cost is that the query must repeat the
expression exactly, so it is defined once in Python and used in both places.

### 3. `left(content, N)`, and this one is load bearing

`content` is `sa.Text()` with no length cap, and the existing truncation applies
**only to embedding input**. `embed_text`'s docstring is explicit that stored
content is never truncated.

Postgres refuses a `tsvector` larger than 1 MB. With a plain
`to_tsvector('english', content)` expression index, the first `INSERT` of a large
artifact or a long note would fail with `string is too long for tsvector` — the
index would convert an oversized write from something that works today into a
hard write failure. **Adding lexical search would break writing.**

Bounding the indexed expression at 100,000 characters removes that class of
failure entirely. The trade is that a literal appearing only beyond 100k
characters of a single memory is not lexically searchable. That is the correct
trade: the vector arm still covers the row, and a write that fails is worse than
a tail of one document that is only semantically searchable.

A regression test writes a memory far larger than the bound and asserts the write
still succeeds.

### 4. `websearch_to_tsquery`, not `to_tsquery` or `plainto_tsquery`

`to_tsquery` raises a syntax error on arbitrary input. A user-facing search box
that 500s on a stray `&` is not acceptable, and the query text here comes
straight from an LLM.

`websearch_to_tsquery` never raises, understands quoted phrases and `OR`, and
degrades to an empty query rather than an error. An empty tsquery matches nothing,
which lands us on the invariant in decision 1.

### 5. Text search configuration is measured, not assumed

Whether `'english'` or `'simple'` better preserves identifiers like
`KLIO_METRICS_TOKEN`, `pool_pre_ping`, `/healthz` and `xmax = 0` is an empirical
question about the Postgres parser, and stemming behaviour differs between them.
The choice is made against a real Postgres with the exact strings above, and the
measurement is recorded in the plan before the index is written. Guessing here
would decide the feature's core value by coin flip.

### 6. Tenancy is inherited, never restated

This is the highest risk part of the change, and the risk is precise: **a new
query path is a new place to leak.** The Wave 1 defect was exactly this shape —
`retrieve_memories` filtered on session and org but not on scope or
`recall_hidden`, and the boundary held everywhere else in the file.

The lexical arm therefore does not compose its own `WHERE`. It is built from the
same `base_clauses` list and the same `params` dict that the vector arm uses, in
the same function, so a predicate cannot be present on one arm and missing on the
other. The two-tier project fill wraps both arms rather than the vector arm alone,
so `project_id` semantics are structurally identical.

Adversarial tests seed another user's private row and a `recall_hidden` row whose
content contains the exact search literal, then assert the lexical arm returns
zero rows while a positive control proves the literal is otherwise findable. A
mutation that drops the shared clauses from the lexical arm must fail those tests.

### 7. Bounded, guarded configuration

Following the existing pattern in `shared/env_config.py`:

- `KLIO_LEXICAL_RECALL_ENABLED` — kill switch, default on.
- `KLIO_LEXICAL_WEIGHT` — the lexical arm's RRF weight, default below the vector
  arm's. Lexical is a corrective signal, not the primary one.
- `KLIO_RRF_K` — the RRF constant.

All parsed through the guarded parsers, so a typo degrades to the default with a
warning rather than raising into every recall.

### 8. Instrumented from day one

`shared/metrics.py` shipped yesterday. The arm reports how often lexical
contributes at all, and how often it contributes a row the vector arm did not
return. Without that second number we cannot tell whether this feature earns its
place, and "it feels better" is not a measurement.

Labels stay bounded by construction, per the existing registry rules.

## Risks

| Risk | Mitigation |
|---|---|
| Oversized content breaks writes once indexed | `left(content, 100000)`, with a regression test that writes past the bound |
| A new query path leaks across a tenancy boundary | Shared `base_clauses`, adversarial tests, mutation proof |
| Ranking regresses for queries lexical should not affect | RRF single-list invariant, pinned by test |
| Index build blocks production | GIN build on ~46k rows; no table rewrite. Measured locally first |
| Lexical arm errors take recall down | Failure falls back to the vector result, matching how graph expansion already degrades |

## Success criteria

- A query for an exact rare literal returns the row containing it, where the
  vector-only path did not. Demonstrated as an observed RED before the fix.
- Fused order is byte-identical to today's when the lexical arm returns nothing.
- Adversarial tenancy tests pass, and fail under mutation.
- A write larger than the index bound still succeeds.
- No existing test count drops.
