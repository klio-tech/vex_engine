# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **A lexical arm on recall, fused with the dense arm by reciprocal rank fusion.**
  Recall was vector-only: no `tsvector`, no GIN index anywhere in the engine.
  Embeddings are weakest on exactly the queries agents make most, where the target
  is a rare literal rather than a paraphrase. Against a seeded corpus the dense arm
  ranked the row containing `KLIO_METRICS_TOKEN` sixth of six, so `limit=5` dropped
  it entirely. That failure reproduces under six different embedding seeds, so it is
  a property of the geometry rather than of one fixture.

  RRF rather than a weighted blend of the two scores: cosine is bounded in [0, 1]
  and `ts_rank` is unbounded and corpus-dependent, so a fixed normalisation drifts
  as the corpus grows and shows up as a ranking regression nobody can date. Because
  RRF over a single ranked list is strictly decreasing in rank, an empty lexical arm
  leaves the fused order **identical** to the previous behaviour. An empty query, a
  stopword-only query, the kill switch, and a lexical failure all land there by
  construction.

  The lexical arm does not compose its own `WHERE`. It is handed the same
  `where_sql` string and the same params the dense arm uses, so a tenancy predicate
  cannot be present on one arm and missing from the other. Mutating the shared
  clauses away fails twelve tests, including cross-org leakage.

  Configuration through the guarded parsers: `KLIO_LEXICAL_RECALL_ENABLED`,
  `KLIO_LEXICAL_WEIGHT` (default 1.0), `KLIO_RRF_K` (default 60). A weight of `0.0`
  is honoured as "off" rather than treated as falsy and replaced by the default.

- **Migration 037**: a GIN index on `to_tsvector('english', left(content, 100000))`.
  The bound is load bearing. `content` is unbounded `Text` and the existing
  truncation applies only to embedding input, so an unbounded expression index turns
  the first write of a large memory into `string is too long for tsvector`. That
  is reproduced, and it would have meant that adding search broke writing. Built on
  production in 7 seconds over 51,421 rows; no rows currently exceed the bound.

- **Lexical metrics**: arm skipped by reason, queries by outcome, latency, and rows
  the lexical arm returned that the dense arm did not. The last is the number that
  says whether the arm earns its place, and it is measured before truncation so it
  stays meaningful at any weight.

### Fixed

- **`recall`'s tool description omitted `private` from its `scope` enum.** The
  engine has accepted `Literal["org", "agent", "private"]` since private scope
  shipped, but the description agents read advertised only `org` and `agent`, so
  private scope was undiscoverable through `recall`.

### Security

- **Session retrieval leaked private and hidden memories across users.**
  `retrieve_memories` filtered on `session_id` and `org_id` but not on `scope` or
  `recall_hidden`. The verification path accepts a client-supplied `session_id`, so a
  caller who named a reserved `klio-brain:<org>:<agent>` session read another user's
  private rows, and their contents were sent to the consistency LLM. Both queries now
  require `scope = 'session'` and `recall_hidden = FALSE`.
- **Artifact tools did not check project membership.** `artifact_get`, `artifact_put`
  and `forget` now call `projects.is_member` when the artifact carries a `project_id`.
  Strict, with no org-admin override. A non-member gets the same `not_found` as every
  other refusal, so the tools cannot be used to probe for the existence of an artifact.
- **`retract_memory` had no project arm**, leaving a project row retractable by any
  same-org caller who knew its id. A project row is now retractable by project members
  only.
- **Prose carried in `metadata` went unscreened.** `write_scoped_memory` redacted
  `content` and nothing else, while `decide` files its caller-supplied `rationale` into
  `metadata`. A credential pasted into a rationale was stored in the clear and passed to
  the curator's LLM. A declared set of prose-bearing keys is now screened on the same
  fail-closed contract as `content`. Structural values such as checksums and storage
  URIs are deliberately excluded, because they are high-entropy by construction and the
  entropy gate would corrupt them.
- **A swallowed rate-limit error widened scope.** `identity.py` turned a 429 into
  `user_id=None`, which landed a project write at org scope. A rate-limit trip now
  surfaces as a rate-limit error.
- **Redaction gaps closed.** The AWS secret value, quoted JSON password, and Redis URL
  password patterns did not match, and there was no AWS secret-value pattern at all.
  Values under key names such as `password` or `secret` are now redacted regardless of
  entropy. `sync-gateway` and `async-worker` imported no redactor, so LLM-generated
  summaries and facts reached storage unscreened on those paths; both now redact.

### Fixed

- **The curator abandoned rows permanently.** When a partition fell below the synthesis
  threshold the tick skipped it and then advanced the cursor past every row anyway. The
  curator never reads behind its cursor and retention deletes raw observations after
  seven days, so those rows were unrecoverable. The cursor now stops before the earliest
  undistilled row.
- **The curator's entity budget was charged for references, not creations.** `add_entity`
  returned only an id, so an upsert that merely matched an existing entity still consumed
  the per-tick budget. A tick mentioning one entity twenty times spent twenty of its
  twelve, hit the cap, and dropped every later entity along with every edge pointing at
  it. `add_entity` now reports whether it inserted, via `RETURNING id, (xmax = 0)`.
- **Ten handlers swallowed a database error without rolling back**, poisoning a session
  the caller kept using. Seven take `db.rollback()`. Three in the curator take a
  savepoint instead, because prior writes in the batch are still in flight and a bare
  rollback would discard them. These two fixes had to ship together: the missing rollback
  was the only thing preventing the cursor bug above from losing data.
- **`entity_brief` returned the oldest 50 memories**, so `explore` froze in the past once
  an entity passed 50 memories while still reporting derived counts as current. It now
  returns the newest bounded set and reports `truncated`.
- **Embedding calls had no timeout and pinned a pooled connection across the network
  call.** With a pool of 15 and a one-second embed, the sixteenth concurrent write queued
  and failed; at concurrency 40, 25 of 40 failed. The connection is released before the
  await, the timeout is bounded and configurable, and identical content is embedded once
  rather than twice per write. 200 concurrent writes now complete.
- **MCP tool bodies blocked the event loop**, ignoring the threading contract the capture
  path already followed. Throughput was capped near 12 requests per second regardless of
  pool size. Offloading the blocking work raised it to 126, with worst-case loop lag
  falling from 4,092ms to 1.4ms.
- **One malformed plan override took down a whole tenant.** An unknown key or wrongly
  typed value raised a `TypeError` past `verify_api_key`, which catches only `AuthError`,
  and the failure happened before the cache write so every retry failed identically.
  Overrides are now filtered and type-checked, and a bad one is ignored with a warning.
- **Pool exhaustion surfaced as `403 not_connected`**, telling a correctly connected user
  to re-authorize their workspace. Database unavailability now returns a retryable 503.
- **Scheduler sweeps blocked the event loop and were unbounded.** With a slow object
  store a single sweep could wedge the loop for roughly 90 minutes. Sweeps are now row
  bounded and offloaded.
- **`RAW_OBSERVATION_TTL_DAYS` was parsed with a bare `int()`** and feeds a DELETE with no
  `org_id` predicate. Setting it to `0` put the cutoff at `NOW()` and a negative value put
  it in the future, so the plausible operator action of disabling the sweep was the
  destructive one. It is now validated before any DELETE runs, as are the other
  environment knobs that had unguarded parsers.
- **`statement_timeout` was applied as a libpq startup parameter**, which Neon's pooled
  endpoint refuses outright. Every database connection failed in production while
  `/healthz` continued to return 200. It is now applied as a `SET` on connect, and a test
  pins the form because no test against a direct local Postgres can catch it.
- **Long-lived engines had no connect timeout or recycle.** Six engines across four
  modules were affected, three of them on the request path, so an unreachable database
  presented as hung requests rather than errors. Bounds are centralised in
  `shared/engine_bounds.py`.

### Added

- **`/readyz`**, a dependency-aware readiness probe that issues a bounded `SELECT 1` and
  reports embedder configuration. `/healthz` remains a pure liveness probe and still
  never touches the database.
- **`/metrics`**, Prometheus text exposition covering writes, recalls, embedding, pool
  pressure, curator progress and capture. Gated by `KLIO_METRICS_TOKEN` and returning 404
  when that is unset. Label cardinality is bounded by construction: identifier-shaped
  label names are refused at declaration and every label must enumerate its value set.
  `prometheus_client` is not a dependency; the registry and exposition format are
  implemented against the standard library.
- **Three reconciliation scripts** under `services/scripts/` that measure what the fixed
  defects left behind: curator-abandoned rows, unredacted metadata, and orphaned object
  storage. All are dry-run by default. Executing requires `--execute`, an `--org`, and a
  confirmation token derived from the candidate set a dry-run measured, so the execute
  path cannot be reached without first reading a dry-run.
- **Migration 036**: artifact cards filed under a project stop at that project.

### Changed

- `add_entity` returns `EntityUpsert(id, inserted)` rather than a bare id.
- `supersede_memory` requires `org_id` and includes it in the WHERE clause.
- Documented that Railway builds from GitHub source on push to `main`. The
  `docker-publish.yml` workflow is not the deploy path and currently fails for lack of
  credentials.

## Earlier

### Added

- **Klio memory-brain MCP service** (`services/mcp-server`): a new FastMCP service over
  Streamable HTTP exposing org-scoped memory tools, gated by a dedicated `memory`
  API-key scope. Reserved host port 8085; wired into `docker-compose.yml`, the Docker
  publish matrix, and the CI test matrix.
- **Migration 020**: spaces and projects tables plus nullable `space_id` / `project_id`
  columns on `session_memories` to support scoped brain memory.
- **Migration 021**: `brain_curator_state` table backing the org-scoped curator synthesis tick.
- **`memory` API-key scope**: a dedicated scope for the Klio brain, separate from `ingest`.

This change is purely additive: the existing verification, ingestion, and sync paths are
unchanged.
