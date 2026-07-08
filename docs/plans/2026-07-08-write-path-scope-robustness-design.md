# Write-Path Scope + Robustness — Design

**Status:** Approved direction, ready for implementation plan
**Date:** 2026-07-08
**Area:** `vex_engine` — MCP write tools (`services/mcp-server/app/mcp_app.py`) + shared brain/memory (`services/shared/shared/{brain,memory}.py`)

## Goal

Fix three defects on the memory **write** path, all surfaced by dogfooding Klio:

1. **Writes don't attach to a space/project** — every write persists `space_id: null`, so scoping only works on reads and everything lands in the flat org brain.
2. **Dedup is too aggressive** — genuinely distinct same-topic notes get collapsed into one.
3. **Long notes fail the embedder** — char-based truncation under-truncates dense text, so a write can error/be lost.

## Root causes (code-grounded)

1. `write_scoped_memory` (brain.py) **already accepts and persists** `space_id` / `project_id`. But the MCP write tools (`remember`, `note`, `observe`, `plan`, `decide`) never pass them, and the `space` tool's `switch` action is **advisory** — MCP is stateless per request, so there is no server-side "current space" to carry into a later write. Result: `space_id` is always `None` on write.
2. `DEDUP_THRESHOLD = 0.92` (cosine) in `memory.py`. Same-topic technical text embeds very tightly, so distinct notes clear 0.92 and `find_similar_memory` treats them as duplicates.
3. `embed_text` truncates to `MAX_EMBED_CHARS` (~2048 chars ≈ 512 tokens at 4 chars/token), but dense code/JSON is < 4 chars/token, so it can still exceed the model's 512-token cap → embedder error; and a failed embed can fail the whole write.

## Design decisions

- **D1 — Space/project on write via explicit params** (mirrors `recall`, which already resolves `space`/`project` filters → ids, and `write_scoped_memory`, which already takes `space_id`/`project_id`). Add optional `space` (slug) and `project` args to the 5 write tools. Resolve `space` → `space_id` via `ensure_space(org_id, slug)` (get-or-create, idempotent) and `project` → `project_id` via the existing project resolver; pass both into `write_scoped_memory`. Rationale: stateless-correct and consistent with how reads already scope. Omitting the params keeps today's behavior (org-brain write) — backward compatible.
  - Tenancy: resolving through `ensure_space(org_id, slug)` guarantees the returned `space_id` belongs to the caller's org, so a write can never target another org's space.
  - Get-or-create on write (vs require-exists): chosen so "write to a new space" just works (matches `space create` idempotency). Typos create a stray space, which is low-harm and listable via the `space` tool.
- **D2 — Raise dedup threshold to `0.95`, env-configurable** via `KLIO_DEDUP_THRESHOLD` (default `0.95`). Keeps near-identical dedup while letting distinct same-topic notes coexist. Single-line tunable, no data-model change.
- **D3 — Embedder: token-safe truncation + never-fail-the-write.** Lower `MAX_EMBED_CHARS` to a conservative token-safe floor (so even dense text stays under 512 tokens), AND make `embed_text` degrade gracefully: on any embedder error, retry once with harder truncation; if it still fails, return `None`. The write path must persist the memory **even when the embedding is `None`** (content stored full-fidelity; that row is simply not semantically recallable until re-embedded). Invariant: **a write is never lost because the content was long.**

## Files

- `services/mcp-server/app/mcp_app.py` — add `space` / `project` params to `remember`, `note`, `observe`, `plan`, `decide`; resolve slugs → ids (reuse the same resolution `recall` uses — `ensure_space` / project resolver); thread `space_id` / `project_id` into `write_scoped_memory`. Update each tool's schema/docstring.
- `services/shared/shared/memory.py` — D2: `DEDUP_THRESHOLD` → `0.95` + `KLIO_DEDUP_THRESHOLD` env override; D3: conservative `MAX_EMBED_CHARS` floor + `embed_text` retry/`None` fallback; confirm `write_memory` persists a row when `embedding is None` (add/verify the null-embedding insert path).
- `services/shared/shared/brain.py` — expected **no change** (write_scoped_memory already threads space_id/project_id); confirm resolvers are importable by the tool layer.

## Data flow (write with a space)

`remember(space="spicychat-platform")` → resolve slug → `space_id` via `ensure_space(org_id, slug)` → `write_scoped_memory(space_id=…, project_id=…)` → `find_similar_memory` (dedup within scope, threshold 0.95) → `write_memory` persists the row with `space_id` (and `project_id`).

## Testing

- **Space-on-write:** `remember(space="x")` → stored row has `space_id` set (not null); `recall(space="x")` returns it; `recall()` (no space) still returns it (org brain); a different org cannot see it. `project` param behaves symmetrically.
- **Dedup:** two distinct same-topic notes that collapse at 0.92 both store at 0.95; a near-identical repeat still dedups; `KLIO_DEDUP_THRESHOLD` override respected.
- **Embedder robustness:** a very long / dense note **stores successfully** (write never raises); embedding present when it fits, `None`-fallback when it doesn't; stored `content` is always full-fidelity; a row written with `embedding IS NULL` is a valid row.
- Existing suite stays green (ruff + pytest), incl. the recall-untouched and org-isolation guards.

## Out of scope (explicit follow-ups)

- Cross-agent conflict auto-resolution / supersede-on-update (separate effort).
- A persisted per-agent "default space" (stateful convenience) — explicit params suffice for now.
- A real tokenizer dependency for token-exact truncation — only if the conservative char floor proves too lossy for prose.
- Backfill re-embedding of any rows stored with a null embedding.

## Risks / to verify during implementation

- **`embedding` column nullability (D3).** The primary robustness guarantee is the conservative `MAX_EMBED_CHARS` floor — it makes every input embeddable, so the embedder should never be handed an oversize string in the first place. The `embedding IS NULL` fallback is a last-resort safety net and only works if the `embedding` column is nullable. Verify the column definition: if it is `NOT NULL`, either (a) rely solely on the token-safe floor + retry (no null rows, no migration), or (b) add a migration to make it nullable. Prefer (a) unless a null-embedding path is clearly needed.
- **Resolver import boundary.** Confirm `ensure_space` / the project resolver are cleanly importable into the mcp-server tool layer (they already are for `recall`); no new coupling.

## Rollout

Backward compatible: the new write params are optional (omitted → current org-brain behavior); the threshold and truncation changes are internal. Likely **no migration** (contingent on the D3 nullability check above).
