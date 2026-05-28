# Klio Brain — MCP memory service on Vex — design

**Status:** approved (brainstorm), pending implementation plan
**Date:** 2026-05-28
**Repo:** `vex_private/vex_engine` (the private Vex engine)
**Related:** local-first Klio (`/Users/thakurg/Me/klio`, shipped v0.7.1) — **untouched** by this work

## Context & strategy

"Klio" shipped as a local-first, MCP-native memory product for AI coding
agents (engine + Go bridge + trust-app + npm CLI). "Vex" is the parent
platform: an AI-agent **verification/reliability** service whose engine
already contains an org-scoped, multi-tenant **session-memory subsystem**
(`shared/memory.py`) — pgvector embeddings, semantic recall, dedup,
supersedes-graph, fact extraction, summaries. Klio's memory engine was
modeled on Vex's; they share DNA.

**Strategic decision:** turn Vex into Klio's growth vehicle. Vex has real
demand and organic search equity, but memory is the universal problem
everyone wants solved. Rather than build "Klio Cloud" greenfield, we add
the **Klio memory brain as a new, additive feature inside the Vex engine**,
reusing Vex's multi-tenancy, auth, storage, workers, and ops. Vex's organic
traffic later 301-redirects into klio.tech.

### Hard scoping constraints (from the owner)

1. **Local Klio is untouched.** It remains the local-first / self-host
   product. No migration, no merge.
2. **Vex's existing features are untouched.** Verification (6-check
   pipeline, sync-gateway, alert-service) keeps working exactly as is.
   Nothing is shed.
3. **This is purely additive.** A new feature built on top of what Vex
   already does — new service + additive functions + one additive
   migration. It coexists; it replaces nothing.

### Locked decisions (from brainstorm)

- **Brain unit:** team / org brain (Vex is `org_id`-scoped natively).
- **Capture:** cloud-explicit via MCP tool calls in v1; optional local
  silent capture (hooks) is a documented fast-follow, not v1.
- **Trust model:** per-tenant, server-readable (Vex stores content
  server-side as plaintext today — which fits, since semantic recall +
  curator tuning *require* the server to read plaintext). Per-tenant
  envelope keys + crypto-shred + BYOK are Phase-2 enterprise hardening
  (see Open Questions), not v1.
- **Foundation:** build on Vex's substrate; full Klio tool surface, not a
  thin wrapper.

## Goal

A hosted, remote MCP server inside the Vex engine that exposes Klio's
memory brain to any agent (Claude Code, Cursor, ChatGPT, web) over one
pasteable URL, authenticated by the existing Vex API key, backed by Vex's
org-scoped pgvector memory — with team-wide ("org" scope) persistent recall,
spaces + projects organization, and server-side curation.

## The reuse map (what is / isn't new)

**♻️ Reused from Vex, unchanged:**
- Multi-tenancy: `org_id` principal, `organizations` table.
- Auth: `shared/auth.py` — `X-Vex-Key` → `(org_id, agent_id)` via
  `KeyValidator.validate()`.
- Memory core: `shared/memory.py` — `embed_text`, `write_memory`,
  `find_similar_memory` (dedup @ 0.92 cosine), `supersede_memory`,
  `extract_facts`, `generate_summary`, over Postgres + pgvector (HNSW,
  `vector(1024)`, multilingual-e5-large).
- The **scope ladder**: `session_memories.scope ∈ {session, agent, org}`
  with indexes `ix_session_memories_{session,agent,org}_lookup` **already
  built** (migration 016, lines 79–97 — the agent/org indexes are even
  labelled "Future").
- Ops: async-worker, storage-worker, Redis, alembic migrations, Docker
  Hub CI, multi-service compose.
- `shared/plan_limits.py` — quota/tiering seed for later billing.

**📦 New, additive (this project):**
- `services/mcp-server/` — new FastAPI + `mcp` SDK service (Streamable
  HTTP remote transport).
- `retrieve_memories_scoped(...)` in `shared/memory.py` — brain recall at
  agent/org scope (the existing session-scoped `retrieve_memories` stays
  untouched; verification keeps using it).
- 5 new `memory_type` values (`memory`, `observation`, `plan`, `decision`,
  `note`) alongside the existing `summary`, `fact`.
- Migration **018**: `spaces` table, `projects` table, nullable
  `space_id` + `project_id` on `session_memories`.
- Curator scheduler — reuses `extract_facts`/`generate_summary`; only the
  schedule is new.

**🚫 Untouched (coexistence guarantee):**
- verification-engine, sync-gateway, alert-service, existing
  `retrieve_memories`, all current Vex behavior — and the entire
  local-Klio repo.

## Architecture — the additive feature

```
                         ┌────────────────────────────────────────┐
  any agent ── MCP ─────►│  services/mcp-server  (NEW)             │
  (Claude Code, Cursor,  │  FastAPI + mcp SDK, Streamable HTTP     │
   ChatGPT, web)         │  ── auth bridge: X-Vex-Key → (org,agent)│
                         │  ── 7 tools → memory fns                │
                         └───────────────┬────────────────────────┘
                                         │ calls
                         ┌───────────────▼────────────────────────┐
                         │  shared/memory.py                       │
                         │  write_memory · retrieve_memories_scoped│ (NEW fn)
                         │  find_similar_memory · supersede_memory │
                         │  extract_facts · generate_summary       │
                         └───────────────┬────────────────────────┘
                                         │ pgvector
                         ┌───────────────▼────────────────────────┐
                         │  session_memories  (+ spaces, projects) │
                         │  scope: session|agent|org  (indexes     │
                         │  already exist)                         │
                         └─────────────────────────────────────────┘

   UNTOUCHED, running alongside:  sync-gateway · verification-engine ·
   alert-service · async/storage workers · dashboard-api
```

## Data model — additive migration 018

```sql
-- New: spaces (org-owned coarse grouping: "Backend", "Design", …)
CREATE TABLE spaces (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          VARCHAR(64) NOT NULL,
    name            VARCHAR(200) NOT NULL,
    slug            VARCHAR(200) NOT NULL,
    embedding_model VARCHAR(200),          -- per-space model (Phase 2; v1 = org default)
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (org_id, slug)
);

-- New: projects (git-remote-derived, mirrors local-Klio 0.7.x)
CREATE TABLE projects (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id         VARCHAR(64) NOT NULL,
    space_id       UUID REFERENCES spaces(id) ON DELETE SET NULL,
    git_remote     TEXT,
    repo_root_path TEXT,
    display_name   VARCHAR(200) NOT NULL,
    created_at     TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at   TIMESTAMPTZ DEFAULT NOW()
);
-- partial-unique semantics identical to local-Klio: git_remote wins,
-- else repo_root_path.
CREATE UNIQUE INDEX ix_projects_org_remote ON projects (org_id, git_remote)
    WHERE git_remote IS NOT NULL;
CREATE UNIQUE INDEX ix_projects_org_path ON projects (org_id, repo_root_path)
    WHERE git_remote IS NULL AND repo_root_path IS NOT NULL;

-- Additive columns on the EXISTING table. Existing verification-memory
-- rows stay NULL — completely undisturbed.
ALTER TABLE session_memories ADD COLUMN space_id   UUID REFERENCES spaces(id)   ON DELETE SET NULL;
ALTER TABLE session_memories ADD COLUMN project_id UUID REFERENCES projects(id) ON DELETE SET NULL;
CREATE INDEX ix_session_memories_space   ON session_memories (space_id);
CREATE INDEX ix_session_memories_project ON session_memories (project_id);
```

Two orthogonal axes, kept clean:
- **`scope`** (session / agent / org) = *who can see it*. Team brain
  defaults writes + recall to `scope="org"` so the team shares one brain.
- **`space_id` / `project_id`** = *which bucket within the org*. Optional
  narrowing. NULL = unfiled (surfaces everywhere, mirroring local-Klio's
  NULL-surfacing rule).

## Tool surface → memory function mapping

The MCP server exposes Klio's 7 tools; each is a thin call into existing
(or one new) memory functions, always scoped to the auth-resolved
`(org_id, agent_id)`.

| MCP tool | memory fn | memory_type | default scope |
|---|---|---|---|
| `recall(query, scope?, space?, project?)` | `retrieve_memories_scoped` (NEW) | — | `org` |
| `remember(content)` | `write_memory` | `memory` | `org` |
| `observe(content)` | `write_memory` | `observation` | `org` |
| `plan(content)` | `write_memory` | `plan` | `org` |
| `decide(content, rationale?)` | `write_memory` | `decision` | `org` |
| `note(content)` | `write_memory` | `note` | `org` |
| `space(action, …)` | spaces CRUD | — | — |

- Writes dedup via `find_similar_memory` (existing 0.92 threshold) and
  link supersedes via `supersede_memory` (existing).
- `recall` accepts optional `scope` ("org"|"agent"|"session"),
  `space`, and `project` filters; defaults to org-wide.
- The existing curator-produced `summary`/`fact` types are recallable too
  (they're the synthesized layer).

## Auth / identity

Reuse `shared/auth.py` verbatim. The MCP server reads the API key from the
connection (header on the Streamable-HTTP request), calls
`KeyValidator.validate(api_key, agent_id)` → `(org_id, agent_id)`, and
scopes every tool call. **No new auth system in v1.** OAuth 2.1 remote-MCP
auth is a later enhancement (Open Questions).

## Capture model

- **v1: cloud-explicit.** Memory is captured when an agent calls a tool
  (`remember`/`observe`/…). Works identically across every client,
  zero local install.
- **Fast-follow (not v1): optional local silent capture.** Port Klio's
  Claude Code hooks + a thin local relay that forwards captured events to
  the remote MCP. Restores the ambient "captures everything" magic for
  clients that support hooks. Deferred deliberately.

## Curator (server-side tuning)

Reuse `extract_facts` + `generate_summary`. Add a scheduled loop (new tick
in `async-worker`, or a small dedicated worker) that periodically
synthesizes `scope="org"` observations into higher-confidence
`summary`/`fact`/`decision` memories — the "tune content for all agents"
behavior, now team-wide. Cursor/bookkeeping per org.

## v1 boundary

**In:**
- `services/mcp-server/` (FastMCP, Streamable HTTP), wired into compose + CI.
- API-key auth bridge (reuse).
- 7 tools mapped to memory functions.
- `retrieve_memories_scoped` (brain recall, org default; existing fn untouched).
- Migration 018 (spaces, projects, additive columns).
- Curator scheduler (reuse primitives).
- Coexistence tests proving verification + existing memory + existing
  retrieve path are unaffected.

**Deferred (documented, not built in v1):**
- Optional local silent capture (hooks port).
- Team-management UI (invites/roles) — sits on existing `permissions`
  primitives.
- Billing/Stripe on `plan_limits`.
- OAuth 2.1 remote-MCP auth (API key suffices for v1).
- Per-tenant envelope keys / crypto-shred / BYOK (see Open Questions).
- Per-space pluggable embedding models (v1 = Vex's single org model;
  per-space needs the shadow-table approach from local-Klio).
- Brain views in dashboard-api.

## Open questions (resolve during plan or early build)

1. **Per-tenant envelope encryption.** Brainstorm chose "per-tenant keys,
   server-readable." Vex stores plaintext today. Recommendation: v1
   inherits Vex's current at-rest posture; add per-tenant envelope keys +
   crypto-shred + BYOK as Phase 2 enterprise hardening. Confirm acceptable.
2. **`session_id` for org-scope writes.** Record the originating
   `session_id` for provenance, but org-scope recall ignores it.
   Confirm the MCP server can always derive a stable session id (e.g.,
   per MCP connection) to satisfy the NOT-NULL `session_id` column — or
   relax that column to nullable for brain writes (additive).
3. **Spaces/projects as tables vs JSONB.** This design uses real tables +
   FK columns (cleaner, indexable). Confirm over the lighter
   metadata-JSONB option.

## Testing strategy

- **Unit:** auth bridge (key → org/agent), tool→fn mapping, scope routing
  (org/agent/session → correct index path), dedup/supersede passthrough.
- **Migration:** 018 applies cleanly; existing `session_memories` rows
  remain valid with NULL space/project; downgrade reverses.
- **Integration:** MCP client → Streamable HTTP → server → memory
  round-trip; **org isolation** (org A cannot recall org B); recall at
  each scope returns the right set.
- **Coexistence (load-bearing):** existing `retrieve_memories` behavior
  unchanged; verification-engine + sync-gateway test suites still green;
  no regression in quota/scope-filtering tests (the latest Vex commit).

## Coexistence guarantees (the contract)

- No edits to verification-engine, sync-gateway, alert-service.
- No edits to the existing `retrieve_memories` signature/behavior.
- Migration 018 is purely additive (new tables + nullable columns).
- The local-Klio repo is not touched by any task in this project.
