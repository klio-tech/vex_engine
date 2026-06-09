# Klio Memory Engine Hardening — Design

**Date:** 2026-06-10
**Status:** Approved (design review with product owner)
**Scope:** vex_engine memory substrate (mcp-server, shared, migrations, scripts)

## Problem

A live review of the Klio memory product (dogfooded against the production org
brain) found six issues. Five are engine work and covered by this spec; the
sixth (public positioning still describes the legacy Vex reliability product)
is tracked separately.

1. **Secret leakage.** The passive hook capture stored verbatim Stripe secret
   keys as `fact` rows, recallable by any agent in the org. There is no
   redaction anywhere in the write paths.
2. **Cross-project bleed.** `recall(scope='org')` mixes memories from
   unrelated projects because the project filter is opt-in and MCP clients
   have no way to declare their project.
3. **Capture noise.** Raw tool-call observations (often with empty output)
   are embedded and compete with genuine facts in recall results.
4. **No staleness handling.** Contradictory facts co-surface with equal
   weight; `superseded_by` exists in the schema but nothing sets or honors
   it. There is no way to retract a wrong memory.
5. **Shallow retrieval.** Hardcoded 10 results, pure cosine ranking, no
   relevance score in responses, project ids returned as opaque UUIDs.

## Decisions (made in design review)

| Topic | Decision |
|---|---|
| Redaction behavior | Mask and store: replace secret with typed placeholder, keep the memory |
| Detection mechanism | Pattern + entropy at the `write_scoped_memory` chokepoint, plus hardened extraction prompts (no LLM screening in the hot path) |
| Existing data | Backfill: mask in place across **all orgs**, re-embed affected rows |
| Project scoping | New `X-Vex-Project` header; two-tier recall (project rows first, unfiled org rows fill remaining slots) |
| Raw observations | Store but exclude from recall (`recall_hidden`), 7-day TTL; curator still consumes them |
| Staleness | Curator reconciliation marks superseded rows + new `forget` MCP tool |

## 1. Redaction module

New `services/shared/shared/redaction.py` (~200 lines, pure functions,
no I/O, immutable results).

### API

```python
@dataclass(frozen=True)
class Finding:
    kind: str          # e.g. "stripe_secret_key", "email", "generic_token"
    fingerprint: str   # last 4 hex chars of SHA-256 of the matched secret

@dataclass(frozen=True)
class RedactionResult:
    clean_text: str
    findings: tuple[Finding, ...]
    redacted: bool

def redact(content: str) -> RedactionResult: ...
```

### Detection catalog

Credential patterns (regex), mirroring the detect-secrets approach already
used by this repo's pre-commit hook:

- Stripe (`sk_live_`, `sk_test_`, `pk_live_`, `pk_test_`, `rk_`, `whsec_`)
- AWS access keys (`AKIA`/`ASIA` + 16 chars) and secret-key assignments
- GitHub (`ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_`, `github_pat_`)
- OpenAI (`sk-` family), Anthropic (`sk-ant-`)
- Slack (`xox[baprs]-`), Google (`AIza`), Supabase (`sbp_`, service-role JWTs)
- JWTs (`eyJ` three-segment base64)
- PEM private key blocks (`-----BEGIN ... PRIVATE KEY-----`)
- Key-value assignments: `password|passwd|secret|token|api[_-]?key` followed
  by `[:=]` and a value
- Connection strings with embedded credentials (`scheme://user:pass@host`)
- `Authorization: Bearer <token>` headers
- Generic high-entropy detector: base64/hex tokens ≥ 20 chars with Shannon
  entropy above threshold, **only** when adjacent to a key-value or
  credential-shaped context (entropy alone produces too many false positives
  on hashes/UUIDs, which are fine to store)

PII v1 (deliberately conservative):

- Email addresses
- Credit card numbers (13–19 digits, Luhn-validated)
- Phone numbers (E.164-shaped, conservative)

### Placeholder format

`[REDACTED:<kind>:<fingerprint>]`, e.g. `[REDACTED:stripe_secret_key:a1b2]`.
The fingerprint lets two mentions of the same secret correlate without
exposure. Placeholders never re-match the detector (idempotent).

### Integration

- `shared.brain.write_scoped_memory` calls `redact()` on `content` **before
  the embedding call** — neither stored text nor vector ever contains the
  secret. On findings: store `clean_text`, append `metadata.redactions =
  [kinds]`, include `"redacted": true` and finding kinds in the write
  response. Covers every write path: MCP tools, `/capture/event`,
  `/capture/transcript`, curator output.
- `capture._combine_messages` redacts the combined transcript **before** it
  is sent to the external LLM for `extract_facts` / `generate_summary` —
  secrets never leave the infrastructure via the mining path.
- `brain_recall_events.query_text` is redacted before logging (queries can
  contain pasted secrets).

### Prompt hardening

`extract_facts` and `generate_summary` prompts (shared/memory.py) gain an
explicit instruction: never reproduce credentials, keys, tokens, or passwords
even partially; refer to them generically ("a Stripe test key was provided").
Defense in depth for prose-described secrets that patterns cannot catch.

### Failure policy

Fail closed. If `redact()` raises, the write is rejected with
`{"stored": false, "error": "redaction_failed"}` and logged with context —
unscreened content is never stored. Escape hatch
`MEMORY_REDACTION_DISABLED=1` (default off → redaction ON) documented for
self-hosted deployments only.

## 2. Backfill script (all orgs)

New `services/scripts/run_redaction_backfill.py`, modeled on
`run_retention.py`:

- Keyset-paginated sweep (`created_at, id` cursor) over `session_memories`
  content and `brain_recall_events.query_text`, all orgs.
- For each detector hit: rewrite content with placeholders, re-embed the
  cleaned text (batched, rate-limited to respect the embedding provider),
  stamp `metadata.redaction_backfill = <ISO date>`, update `updated_at`.
- `--dry-run` (default): per-org, per-kind report; writes nothing.
- `--org <id>`: restrict to one org for staged rollout.
- Idempotent: placeholders never re-match.

**Manual operational task (tracked in the plan):** rotate the two Stripe
test-mode keys found exposed in the production org brain. Masking the rows
does not un-leak what was already recallable.

## 3. Project scoping: `X-Vex-Project` header + two-tier recall

- New optional `X-Vex-Project` header (sibling of `X-Vex-Agent`), value is a
  git remote URL or project slug. Resolved in the principal dependency via
  the existing `brain.find_project_id`; resolution failure → header ignored
  (fail open to org-wide, matching read-path philosophy). The explicit
  `project` tool argument overrides the header.
- On writes with a resolved header project, rows are filed under it
  (best-effort `ensure_project`, never blocks the write).
- On recall with a project in scope, replace the current
  `(project_id = :pid OR project_id IS NULL)` flat filter with a two-tier
  fetch: fill `limit` slots from same-project rows by blended score first,
  then remaining slots from unfiled (`project_id IS NULL`) org rows. Rows
  filed under *other* projects never surface.
- Recall results include `project: {slug, name}` (or null) instead of only
  the raw UUID.
- No header and no argument → unchanged org-wide behavior. Fully backward
  compatible; all schema changes additive.

## 4. Recall-hidden raw observations

- Migration 023: `recall_hidden BOOLEAN NOT NULL DEFAULT FALSE` on
  `session_memories`; partial index on `recall_hidden = FALSE` paths; data
  migration sets `recall_hidden = TRUE` on existing rows with
  `memory_type = 'observation'` and `metadata->>'source' IN
  ('hook', 'hook-tool')`.
- `/capture/event` writes `observation` rows with `recall_hidden = TRUE`.
  Explicit MCP `observe` calls remain visible (deliberate agent logging is
  signal; the passive tool firehose is not).
- Both recall paths (vector `retrieve_memories_scoped` and the recency
  fallback in capture) add `AND recall_hidden = FALSE`.
- The curator continues to consume hidden rows as synthesis input.
- The retention sweep (`run_retention.py`) gains a fixed TTL for hidden raw
  rows: `RAW_OBSERVATION_TTL_DAYS = 7`, independent of plan retention.

## 5. Staleness: supersession + `forget`

### `forget` MCP tool (the 8th tool)

- Args: `memory_id` (UUID string from a prior recall result).
- Marks the row `status = 'retracted'`, records audit metadata
  (`retracted_by` agent id, timestamp, `via: 'mcp'`).
- Authorization: row must belong to the caller's org (404 otherwise);
  `scope='agent'` rows are retractable only by their owning agent (403);
  `scope='org'` rows by any agent in the org (audit-logged).
- Mirrored REST endpoint `/capture/forget` for the hook client.

### Curator reconciliation pass

- After each synthesis tick, for each newly written fact: vector-search the
  top-K (K=5) most similar **older active** facts in the same org with
  cosine similarity ≥ 0.85; a conservative LLM judgment ("does the new fact
  directly contradict the old one — not merely relate to it?") marks
  contradicted rows `status = 'superseded'`, `superseded_by = <new id>`.
- Bounded per tick (max reconciliation comparisons configurable,
  default 50) so a busy org cannot blow up curator cost.
- Supersession is conservative by design: false negatives (stale fact
  survives) are acceptable; false positives (live fact hidden) are not.

### Recall enforcement

Both recall paths enforce `status = 'active'` (the recency path already
does; the vector path gets it explicitly). Retracted and superseded rows
never surface.

## 6. Retrieval quality

- MCP `recall` gains `limit` (default 10, max 50 — same `_MAX_RECALL_LIMIT`
  cap as the REST path).
- Results include `relevance` (1 − cosine distance, rounded to 3 decimal
  places) so agents can self-filter weak matches.
- Ranking becomes a blended score computed in SQL:
  `0.85 × similarity + 0.15 × recency_norm`, where `recency_norm` is the
  row's age normalized against a 30-day window (clamped to [0, 1]). Newer
  memories win ties and near-ties without recency swamping topical
  relevance. Applies to both tiers of the two-tier fetch.

## 7. Error handling summary

| Path | Failure | Behavior |
|---|---|---|
| Write | Redactor raises | Fail closed: reject write, `redaction_failed` |
| Write | Embedding fails | Unchanged: `embedding_failed`, no store |
| Recall | `X-Vex-Project` unresolvable | Fail open: org-wide recall |
| Recall | Recall-event logging fails | Unchanged: swallowed, never breaks recall |
| Forget | Unknown id / cross-org | 404 |
| Forget | Agent-scoped row, wrong agent | 403 |
| Curator | Reconciliation LLM error | Skip pair, log, continue tick |
| Backfill | Re-embed fails for a row | Leave row unmodified, report, continue |

## 8. Testing

Per house rules (TDD, ≥ 80% coverage):

- **Unit — detection catalog:** true-positive corpus (every pattern kind,
  including the two real leaked key shapes), false-positive corpus (UUIDs,
  git SHAs, pgvector ids, ordinary prose) — false positives here are
  product damage, so this corpus is as important as the true positives.
- **Unit — placeholder idempotency:** `redact(redact(x).clean_text)` finds
  nothing.
- **Integration — write paths:** MCP `remember`/`observe`, `/capture/event`,
  `/capture/transcript` each store masked content; response carries
  `redacted: true`; embeddings generated from clean text.
- **Integration — recall:** hidden observations excluded; superseded /
  retracted excluded; two-tier project scoping (project rows before unfiled,
  other-project rows never); `relevance` present; `limit` respected.
- **Integration — forget:** happy path, 404 cross-org, 403 cross-agent.
- **Integration — curator reconciliation:** seeded contradiction fixtures →
  old row superseded; related-but-not-contradicting fixtures → untouched.
- **Backfill:** dry-run report against seeded fixtures; apply mode masks,
  re-embeds, and is idempotent on second run.

## 9. Rollout order

1. Redaction module + chokepoint integration + prompt hardening (flag on).
2. Backfill dry-run on production → review report → apply, all orgs.
   Rotate the exposed Stripe keys.
3. Migration 023 + recall-hidden capture behavior + TTL.
4. `X-Vex-Project` scoping + two-tier recall + `relevance`/`limit`.
5. `forget` tool + curator reconciliation.

Each step is independently deployable; all MCP schema changes are additive,
so existing clients keep working throughout.

## Out of scope

- Public positioning/docs reconciliation (separate sub-project).
- LLM-based write screening (revisit if prose-secret leakage is observed
  after prompt hardening ships).
- Spaces UX (auto-space creation, default-space policies).
- Pagination/cursor recall and recall caching.
