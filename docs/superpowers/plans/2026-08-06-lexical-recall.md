# Lexical recall arm — implementation plan

> The spec (`docs/superpowers/specs/2026-08-06-lexical-recall-design.md`) is the
> contract. Where this plan and the spec disagree, the spec wins.

**Baseline:** `main` @ `8dd5dc5`. **Test counts that must not drop:** shared
`989 passed, 2 skipped`; mcp-server `634 passed, 9 skipped`; migrations `64 passed`.

## Measurements taken before writing any code

All against `timescale/timescaledb-ha:pg16`, the production major version.

### Tokenisation: `english` and `simple` are identical for identifiers

| Input | `english` | `simple` |
|---|---|---|
| `KLIO_METRICS_TOKEN` | `'klio':1 'metric':2 'token':3` | `'klio':1 'metrics':2 'token':3` |
| `pool_pre_ping` | `'ping':3 'pool':1 'pre':2` | `'ping':3 'pool':1 'pre':2` |
| `xmax = 0` | `'0':2 'xmax':1` | `'0':2 'xmax':1` |
| `/healthz` | `'/healthz':1` | `'/healthz':1` |
| `statement_timeout=30000` | `'30000':3 'statement':1 'timeout':2` | same |
| `recall_hidden` | `'hidden':2 'recal':1` | `'hidden':2 'recall':1` |

Both configurations split identifiers on `_` and `=` into the same component
lexemes, and both keep `/healthz` whole. They differ **only** in stemming
(`metrics`→`metric`, `recall`→`recal`).

**Decision: `english`.** Identifier matching is unaffected because tokenisation is
symmetric across query and document, so the split costs nothing. Stemming is a
pure gain on the prose half of the corpus, where `decision` should match
`decisions`. `simple` would buy nothing and lose that.

Note the consequence to document: an identifier matches as a **conjunction of its
parts**, not as an opaque string. `KLIO_METRICS_TOKEN` requires `klio & metric &
token`, which is precise enough that it matched only the row containing the
literal and not a topically similar row about observability.

### The oversized-content hazard is real, and reproduced

```
CREATE INDEX ... GIN (to_tsvector('english', content));
INSERT ... 120k distinct tokens;
ERROR:  string is too long for tsvector (1477986 bytes, max 1048575 bytes)
```

The same INSERT with `left(content, 100000)` in the index expression: `INSERT 0 1`,
storing `max_content_len = 1088894`.

So an unbounded expression index converts a write that succeeds today into a hard
failure, and the bound removes it while still storing the full content. Note the
limit is on **distinct** lexemes: `repeat('alpha bravo charlie delta ', 200000)`
produced a 4,025-byte tsvector, because repetition dedupes.

### Query behaviour

| Probe | Result |
|---|---|
| `websearch_to_tsquery('english','KLIO_METRICS_TOKEN')` | matched only the row containing the literal |
| `'unsupported startup parameter'` | matched the intended row |
| `'/healthz'` | matched |
| `'&& \|\| !! ((( "unclosed'` | returned `'unclos'`, **did not raise** |
| `'the a of and'` | empty tsquery, 0 matches, NOTICE not ERROR |

The last two are what make the fallback invariant real rather than hoped for.

## Work

### 1. Migration

New revision after `036`. Create the GIN index:

```sql
CREATE INDEX ix_session_memories_content_fts
    ON session_memories USING GIN (to_tsvector('english', left(content, 100000)));
```

`CREATE INDEX CONCURRENTLY` is **not** available: Alembic runs migrations inside a
transaction and `CONCURRENTLY` cannot run there. At ~46k production rows a plain
build is short; measure it locally and record the duration in the migration
docstring. Downgrade drops the index.

The expression must be byte-identical to the one the query uses. Define the
bound once as a Python constant and reference it from both.

### 2. The lexical arm in `retrieve_memories_scoped`

`services/shared/shared/memory.py`, around the existing `_fetch` at ~line 1706.

- **Reuse `base_clauses` and `params` verbatim.** The lexical arm must not compose
  its own tenancy predicates. This is the Wave 1 defect shape and the single
  highest risk in the change.
- Keep the existing vector SQL exactly as it is.
- Add a lexical SELECT with the same column list and the same
  `{where}` substitution, ordered by `ts_rank(...) DESC`, bounded by the same
  candidate pool.
- Skip the lexical query entirely when the tsquery is empty or the feature is
  disabled, rather than issuing a query that cannot match.
- Fuse inside `_fetch` so the two-tier `project_id` fill wraps both arms and tier
  semantics are structurally unchanged.

### 3. RRF fusion

New helper, unit-testable without a database:

```
fused(row) = Σ_arms  weight_arm / (k + rank_arm(row))
```

- `k` from `KLIO_RRF_K`, default 60.
- Vector weight 1.0; lexical weight from `KLIO_LEXICAL_WEIGHT`, default below 1.0.
- **Invariant to test directly:** with only one non-empty arm, the fused order
  equals that arm's order. Assert on a permutation, not a two-element list.
- Ties broken deterministically so results are stable across calls.

### 4. Configuration

Through `shared/env_config.py`, never bare `int()`/`float()`:

| Variable | Default | Bounds |
|---|---|---|
| `KLIO_LEXICAL_RECALL_ENABLED` | on | bool |
| `KLIO_LEXICAL_WEIGHT` | 0.5 | 0.0 – 1.0 |
| `KLIO_RRF_K` | 60 | 1 – 1000 |

A malformed value degrades to the default with a warning. Weight `0.0` must be
honoured as "off", not treated as falsy-and-defaulted.

### 5. Metrics

Using the existing registry, bounded labels only:

- lexical arm invoked / skipped (with a bounded reason label)
- queries where lexical returned rows
- **queries where lexical contributed a row the vector arm did not return** —
  this is the number that says whether the feature earns its place
- lexical query latency

### 6. Failure isolation

A lexical failure returns the vector result unchanged and logs, mirroring how
`_expand_via_graph` already degrades. A test forces the lexical query to raise and
asserts recall still returns the vector rows.

## Tests

**Observed RED first.** Seed a corpus where a rare literal appears in one row and
several topically-similar rows do not contain it, and show the vector-only path
fails to return it at rank 1. That failing output is the justification for the
feature and must be pasted, not described.

Required:

1. Exact-literal retrieval succeeds after the change (the RED above, now green).
2. **Single-arm invariant:** lexical empty → order identical to pre-change, over a
   list long enough that coincidental agreement is implausible.
3. **Adversarial tenancy**, each with a positive control proving the literal is
   otherwise findable:
   - another user's `private` row containing the literal → not returned
   - a `recall_hidden` row containing the literal → not returned
   - a non-member's `project` row containing the literal → not returned
   - a row in another `org_id` containing the literal → not returned
4. Project two-tier fill still holds with lexical active.
5. Oversized write (past the index bound) still succeeds.
6. Hostile query strings do not raise.
7. Config: malformed values degrade; `0.0` weight honoured.
8. Lexical failure degrades to vector results.

**Mutation checks, reported verbatim:**

- drop the shared tenancy clauses from the lexical arm → the adversarial tests fail
- remove `left(content, N)` from the index expression → the oversized-write test fails
- replace RRF with a naive weighted sum of `relevance` and `ts_rank` → the ranking
  test fails
- return the lexical arm's rows without fusion → the single-arm invariant fails

## Discipline

- TDD with observed RED; paste actual output.
- Every adversarial test carries a positive control.
- Tenancy predicates stay in SQL. Never fetch wide and filter in Python.
- `ruff check services/` and `ruff format --check services/` clean per commit.
- Never `git add -A`; stage by name.
- Push the branch. No PR, no merge.
