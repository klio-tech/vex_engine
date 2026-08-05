# Review remediation — implementation plan

> **For agentic workers:** the spec
> (`docs/superpowers/specs/2026-08-04-review-remediation-design.md`) is the contract. Where this plan
> and the spec disagree, the spec wins. Read the spec before writing anything.

**Baseline:** `main` @ `218e0c5`. **Baselines that must not drop:** shared `593 passed, 1 skipped`;
mcp-server `442 passed, 9 skipped`; migrations `64 passed`.

## Waves

Work packages are grouped so that concurrent agents never own the same file. **Do not start a later
wave before the earlier one has merged.**

### Wave 1 — parallel, disjoint files

| WP | Shape | Owns |
|---|---|---|
| **A** | Tenancy and visibility | `shared/memory.py`, `mcp-server/app/mcp_app.py`, `shared/promotion.py`, `mcp-server/app/identity.py`, `mcp-server/app/capture.py`, `shared/projects.py` |
| **B** | Redaction and egress | `shared/redaction.py`, `sync-gateway/app/*`, `async-worker/app/*` |
| **C** | Config hardening | `mcp-server/app/config.py`, `shared/memory_config.py`, `scripts/run_retention.py`, `mcp-server/app/session.py`, `verification-engine/engine/llm_client.py`, `shared/compression/prose.py`, `shared/auth.py` (agent_id truncation only) |

**Collision note:** B owns whole files in `sync-gateway`/`async-worker`, including the bare `float()`
parsing at `routes.py:45-46`. C does **not** touch those two services.

### Wave 2 — parallel, after Wave 1 merges

| WP | Shape | Owns |
|---|---|---|
| **D** | Rollback pattern **+** curator correctness | `shared/usage.py`, `mcp-server/app/capture.py`, `mcp-server/app/onboarding.py`, `mcp-server/app/auth.py`, `shared/projects.py`, `shared/graph_store.py`, `mcp-server/app/curator.py` |
| **E** | Availability | `shared/memory.py`, `mcp-server/app/db.py`, `mcp-server/app/scheduler.py`, `shared/artifacts.py`, `shared/plan_limits.py`, `shared/auth.py`, `mcp-server/app/oauth.py`, `mcp-server/app/oauth_middleware.py` |

**Shapes 2 and 3 are one work package on purpose.** Splitting them reintroduces the abandonment bug.

### Wave 3 — after Wave 2 merges

| WP | Shape | Owns |
|---|---|---|
| **F** | Observability | new `shared/metrics.py`, `mcp-server/app/main.py`, wiring at write/recall/capture/curator call sites |
| **G** | Data reconciliation | new scripts under `services/scripts/` |

## Work packages

### WP-A — Tenancy and visibility

1. **`retrieve_memories` (`memory.py:630`) — do this first, it is live.** Add `AND scope = 'session'`
   and `AND recall_hidden = FALSE` to **both** queries. Regression test: seed another user's private
   row and a `recall_hidden` row under a reserved `klio-brain:<org>:<agent>` session id, call through
   the sync-gateway entry point, assert zero rows. Positive control: the caller's own session rows
   still return.
2. **Artifact tools** — `artifact_get`, `artifact_put`, `forget` must call `projects.is_member` when
   the artifact carries a `project_id`. Strict, **no admin override**. Non-member gets the same
   `not_found` every other refusal returns — do not introduce an existence oracle.
3. **`retract_memory`** — add the project arm. A project row is retractable by project members only.
4. **`promotion.py:104`** — add `AND scope = 'private' AND status = 'active'` to the UPDATE and return
   `already_shared` on `rowcount == 0`.
5. **`identity.py:87`** — do not swallow a 429 into `user_id=None`. Resolve `KeyInfo` once per request
   and reuse it (this also fixes the double rate-limit consumption). A rate-limit trip must surface as
   a rate-limit error, never as a silent scope widening.
6. **`capture.py:264`** — namespace the client-supplied `session_id` by principal so an unattributed
   caller cannot share a dedup bucket with an attributed user's private rows.
7. **`supersede_memory` (`memory.py:830`)** — require `org_id` as a keyword and add it to the WHERE.

**Mutation checks required, each reported verbatim:** removing either predicate in (1); removing the
membership check in (2); removing the project arm in (3); removing the scope guard in (4).

### WP-B — Redaction and egress

1. Fix all four patterns in `redaction.py`, driven by tests that use the **exact strings in the spec**.
   Add a dedicated AWS secret-value pattern — the access-key-id pattern does not cover it.
2. Entropy gate: redact unconditionally when the key name is `password`/`passwd`/`pwd`/`secret`; keep
   the entropy gate for prose-shaped values only.
3. Route `sync-gateway` and `async-worker` LLM output through the redactor before `write_memory`.
4. Correct the self-contradictory comment at `capture.py:355-373`.

**Test discipline:** every gap gets a test asserting `redacted=True` for the exact string that
currently fails. Mutation: revert each pattern, report the failure count.

### WP-C — Config hardening

1. **`RAW_OBSERVATION_TTL_DAYS` first.** Validate after parsing: reject non-positive and
   unreasonably large values with a fatal error before any DELETE runs.
2. Apply the existing guarded parsers everywhere they are missing: `MCP_OAUTH_ENABLED`,
   `VERIFICATION_TIMEOUT_S` (**two sites** — `memory.py` and `verification-engine/llm_client.py`),
   `CURATOR_ENTITY_CAP`, `CURATOR_RECONCILE_MAX`, `MEMORY_ENABLED`, `KLIO_PROSE_ALLOW_DOWNLOAD`,
   and the module-scope `float()` calls in `sync-gateway/app/routes.py` — **coordinate: WP-B owns that
   file this wave, so leave those two to B and say so in your report.**
3. `_parse_bool` must return `default` for unrecognised tokens, not `False`.
4. Truncate `agent_id` at the trust boundary; bound the `mcp-session-id` header before it reaches a
   `VARCHAR(256)` column.

### WP-D — Rollback pattern + curator correctness

**Read the spec's ordering constraint before starting.** Land both shapes in one branch.

1. Seven `db.rollback()` sites; three `begin_nested()` sites. Do not paste one fix into all ten.
2. `add_entity` returns `(id, inserted)` via `RETURNING id, (xmax = 0) AS inserted`; decrement the
   entity budget **only on insert**.
3. Cursor: leave it unchanged when no qualifying partition synthesized; bump `last_run_at` only.
   Preserve the correct whole-batch behaviour — **that branch is not a bug.**
4. `entity_brief`: newest bounded set, reordered for presentation. Add a `truncated` field.
5. Reconcile budget: one mutable budget created in `curator_tick`, threaded through, mirroring
   `EntityBudget`.
6. Fact filter: replace invalid elements rather than filtering, to preserve LLM `src` indices.
7. Correct the false comment at `curator.py:878`.

**Mandatory mutation check:** revert the rollback fix alone and demonstrate that the cursor test
**still passes** — proving the two fixes are genuinely coupled — then revert the cursor fix and show
it fails. Report both.

### WP-E — Availability

1. Bounded timeout on `aembedding`, configurable, plus a concurrency cap.
2. **Release the pooled connection before awaiting the embed.** Note there are two embeds per write;
   memoize the second where the content is identical.
3. Offload MCP tool-body DB statements per the threading contract `capture.py` already follows.
4. Bound the scheduler sweeps with a row limit and `anyio.to_thread`.
5. `get_plan_config`: filter to known fields and type-check; degrade to "override ignored + WARNING",
   never a tenant-wide failure.
6. Map pool exhaustion to a retryable 503, not `403 not_connected`.
7. Cache the S3 client rather than rebuilding it per call.

### WP-F — Observability

Prometheus `/metrics` + `remote_write` to Grafana Cloud. Counters and histograms in one module.
Dependency-aware `/readyz`. Log the currently-silent failures (`recall_log.py:39`, `identity.py:91`).

### WP-G — Data reconciliation

Three scripts, each **dry-run by default**, printing counts and sample ids, requiring an explicit
flag to execute. **Do not execute any of them** — report the dry-run output and stop.

## Discipline (all packages)

- TDD with **observed RED** — show the actual failing output, do not assert you saw it.
- Every adversarial test needs a **positive control** proving the fixture is not vacuous.
- Report every mutation result **verbatim**.
- Tenancy predicates stay in SQL; never fetch wide and filter in Python.
- `ruff check services/` and `ruff format --check services/` clean per commit.
- Push your branch. **No PR, no merge** — the coordinator merges and deploys.
- If a finding proves wrong or impossible, **report it and stop**; do not work around it silently.
