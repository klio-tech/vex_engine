# Klio Memory Signal Quality — Design

**Date:** 2026-06-11
**Status:** Approved (design review with product owner)
**Scope:** vex_engine — curator, shared embedding, and a one-off cleanup script.

## Problem

The first real OAuth-authenticated `recall` (claude.ai, abhishek@moonforge.co)
returned, in Claude's words, *"mostly technical/development logs … build
artifacts, not your career profile."* The retrieval pipe works; its content
does not. Three coupled causes:

1. **The curator manufactures noise.** Passive hook tool-call captures are
   stored as `observation` rows flagged `recall_hidden=TRUE` (migration 026
   hides them from *direct* recall). But the curator reads ALL RAW
   `scope='org'` rows regardless of `recall_hidden` and distills them into
   `fact` rows like *"Tool Bash was used with command: git show…"*. Those
   synthesized facts are recall-visible — the leak the user saw.
2. **A backlog of junk facts** already exists from before the hardening and
   from the curator's ongoing synthesis.
3. **Oversized content loops the embedder.** Content over the embedding
   model's 512-token limit makes the provider return HTTP 400; the curator
   retries in a tight loop, flooding logs at Railway's 500/sec cap (messages
   dropped) and wasting resources. Same root cause as the residual
   `88e8afbb` row left by the redaction backfill.

## Decisions (design review)

| Topic | Decision |
|---|---|
| Curator input | Exclude `recall_hidden=TRUE` rows — synthesize only from deliberate signal |
| Existing noise | Pattern-retract (reversible) across all orgs, dry-run first |
| Oversized content | Truncate at the `embed_text` boundary (one fix, all callers); stored content stays full |

## 1. Curator input filter

`services/mcp-server/app/curator.py`, `_read_new_raw_rows`: add
`AND recall_hidden = FALSE` to the WHERE clause. The curator already filters
`scope='org'`, `status='active'`, and RAW `memory_type`; this adds the hidden
exclusion so it ingests only deliberate writes.

What remains as curator input after this: the explicit MCP write tools
(`observe`, `note`, `plan`, `decide`, `remember`) — all default
`recall_hidden=FALSE`. Passive `/capture/event` tool-call observations
(`recall_hidden=TRUE`) are skipped. The curator's job narrows to summarizing
and cross-linking *intentional* memories; it no longer mints tool-call facts.

Cursor interaction: the keyset cursor (`brain_curator_state`) advances over
the ordered batch; hidden rows simply never enter the batch. No migration,
no cursor reset needed. Reconciliation (Task-12 supersession) is unaffected —
it operates on `fact` rows.

Test: assert the generated `_read_new_raw_rows` SQL contains
`recall_hidden = FALSE`; an integration test seeding a hidden observation +
a visible note confirms only the note reaches synthesis.

## 2. Embedding truncation

`services/shared/shared/memory.py`, `embed_text`: before calling
`aembedding`, truncate `text_input` to a conservative character budget that
stays under the model's 512-token limit. Use a module constant
`MAX_EMBED_CHARS = 1800` (≈450 tokens at ~4 chars/token — margin below 512 so
multibyte/CJK content can't overshoot). Truncation is head-based
(`text_input[:MAX_EMBED_CHARS]`).

- Applies to EVERY caller (writes, dedup lookups, curator synthesis input,
  recall queries) — a single chokepoint.
- Only the embedding INPUT is truncated; stored `content` is never modified
  (writes pass full content to the INSERT separately from the embedding).
- Eliminates the 400 loop and the log flood; the residual `88e8afbb` row
  becomes embeddable on its next touch (and the redaction backfill's
  re-embed step would now succeed).
- A one-line log at DEBUG when truncation occurs (no per-call INFO spam).

Why char-based, not a tokenizer: the e5 model ships its own tokenizer not
available in this service; a conservative char cap is dependency-free,
deterministic, and safe. Exactness is unnecessary — we only need to stay
under the limit.

Test: a >MAX_EMBED_CHARS input results in a truncated string passed to a
mocked `aembedding`; a short input passes through unchanged.

## 3. Noise-retraction backfill (all orgs)

New `services/scripts/run_fact_cleanup.py`, modeled on
`run_redaction_backfill.py` (keyset pagination, dry-run default, `--apply`,
`--org`, idempotent).

- **Targets:** `session_memories` rows with `status='active'` and
  `memory_type IN ('fact','summary')` whose `content` matches the
  noise-pattern set.
- **Noise patterns** (case-insensitive, conservative — match clear
  tool-call / build-artifact shapes only):
  - `^Used tool\b`
  - `\bTool\s+\w+\s+was used\b`
  - `\bwas used with command\b`
  - `input:\s*\{"command"`
  - `response:\s*\{"stdout"`
  - `\bexecuted (the )?Bash (tool|command)\b`
  These are intentionally narrow: they describe a tool invocation, not a
  durable fact. A curated **false-positive corpus** (e.g. *"The team decided
  to use Stripe for billing"*, *"Bash scripts live in /scripts"*, a real
  career fact) must NOT match.
- **Action:** `status='retracted'`, plus audit metadata
  (`{"cleanup": "<date>", "cleanup_reason": "tool_call_noise"}`).
  Reversible (a future query can flip `retracted`→`active`). Recall already
  excludes non-`active` rows, so cleanup takes effect immediately.
- **Dry-run (default):** per-org, per-pattern hit counts + a sample of
  matched contents (truncated, secret-safe) so the report can be eyeballed
  before `--apply`. No writes.
- **Idempotent:** retracted rows are not active, so a second pass is a no-op.

Test: pattern matcher unit tests (true-positive noise corpus + false-positive
legit-fact corpus); dry-run reports without writing; apply retracts +
stamps metadata; second pass is a no-op.

## Error handling

| Path | Failure | Behavior |
|---|---|---|
| Curator read | (unchanged) | Existing per-tick crash isolation |
| `embed_text` | provider error after truncation | Unchanged: returns `None` (caller handles) |
| Cleanup | DB error mid-sweep | Abort loudly (per-row commit leaves processed rows consistent), like the redaction backfill |
| Cleanup | pattern over-matches | Mitigated by conservative patterns + dry-run review + reversibility |

## Testing

Per house rules (TDD, ≥80%):
- **Unit:** embed truncation (long→truncated, short→unchanged); noise-pattern
  matcher (true-positive + false-positive corpora).
- **Integration:** curator excludes `recall_hidden` (SQL + seeded-rows);
  cleanup dry-run/apply/idempotency.
- **Regression:** existing curator and memory suites stay green; the
  bind-cast SQL scan covers any new SQL.

## Rollout

1. Deploy #1 (curator filter) + #2 (embed truncation) together — stops new
   noise and ends the embedding loop/log-flood immediately.
2. Run `run_fact_cleanup.py` dry-run against production → review the per-org
   report → `--apply`.
3. Re-run a `recall` for "about me" from the connected Claude session to
   confirm the result is now signal, not build logs.

## Out of scope

- LLM-based relevance scoring of memories (revisit if pattern cleanup proves
  insufficient).
- Changing what the passive hook captures (still useful as curator-excluded
  raw history under its 7-day TTL).
- Per-memory importance ranking / decay beyond the existing blended recency.
- Connected-apps client-name display (tracked separately).
