# `init` MCP tool — implementation plan

> **For agentic workers:** the spec
> (`docs/superpowers/specs/2026-08-03-klio-init-onboarding-design.md`) is the
> contract; where this plan and the spec disagree, the spec wins.

**Goal:** the sentence "use the Klio init tool to onboard me" causes a
connected agent to run start → submit → finish, seeding the user's private
silo through the same redaction/dedup door as capture, capped server-side.

**Head:** migration `035`. **No new migration expected** — `init` writes
ordinary `session_memories` rows plus one marker note. Run counting (3/day)
derives from marker memories, not a new table; if that proves impossible,
STOP and report rather than adding a migration silently.

## Files

| File | Action |
|---|---|
| `services/mcp-server/app/onboarding.py` | Create — playbook text (versioned constant), caps, phase logic |
| `services/mcp-server/app/mcp_app.py` | Modify — register `init` tool, wire phases to onboarding module |
| `services/mcp-server/tests/test_init_tool.py` | Create — contract + all five spec invariants |
| `services/mcp-server/tests/test_tool_inventory*` | Modify — 18 → 19, name asserted |

## Tasks

1. **Playbook + phase state.** `start` returns the v1 playbook (spec section
   "Phase start", verbatim intent). Stateless across calls except run
   counting; a run is identified by (org, user, UTC day).
2. **Submit path.** Candidates `{content, kind, source}`; validate kind
   against RAW types, source against `conversation|repo|user`. Route every
   candidate through the SAME redaction + dedup + embedding path capture
   uses (call it, do not reimplement). Scope hard-coded `private`;
   provenance EXTRACTED only for `source="user"`, else INFERRED; project
   via `find_project_id` when repo identifiers supplied.
3. **Caps.** 25/call, 60 accepted/run, 2,000 chars/candidate, 3 runs/day —
   all server-side, each with a distinct, actionable refusal message.
4. **Finish.** Counts by kind/source, redaction kinds (never contents),
   the relay paragraph, and the marker note.
5. **Tests + mutation checks** — the spec's five invariants, each with
   observed RED, a positive control, and reported mutation results:
   redaction bypass (≥2 fails), scope passthrough (≥1), payload identity
   (≥1), per-run cap (≥1), idempotence (accepted ≈ 0 on rerun).

## Discipline

- Real Postgres (`DATABASE_URL`, shared `requires_db` gate) for anything
  touching rows; `KLIO_REQUIRE_DB=1` locally to prove they ran.
- `ruff check` / `ruff format --check` clean per commit; counts must not drop
  (mcp-server baseline: 395 passed, 9 skipped).
- Tool description must route "onboard me / set me up / get started" to
  `init` — test asserts the phrases appear in the description.
- Push branch; no PR, no merge.
