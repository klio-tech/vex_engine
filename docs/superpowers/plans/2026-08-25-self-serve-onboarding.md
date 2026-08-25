# Context Network — Self-Serve Onboarding (P1–P5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans, task-by-task.

**Goal:** A brand-new workspace user connects their agent with zero operator involvement: create a token in the UI (P2), run one command / paste one prompt (P3, P4), and the engine authenticates them as their workspace identity with live-derived access (P1). Humans can see and manage the node/project/branch layer from the workspace (P5).

**Architecture:** The workspace's native `api_tokens` (`plane_api_<32hex>`, plaintext+indexed, user FK, `is_active`, `expired_at`) becomes an engine credential: the engine's KeyValidator, on an org-keys miss, looks the raw key up in the workspace DB and derives `(user_id, org)` live — org from the token's workspace, or the user's personal workspace for user-level tokens, lazily ensured via the existing `ensure_org_for_workspace`. Access is never baked into the token: membership, node reach, project reach all resolve per call through the machinery PRs #41–#46 built. P5 reads engine state (link status, memory counts, branches) through a small internal HTTP API on the engine, reachable from Django over the shared docker network, guarded by a shared secret — never by Caddy exposure assumptions.

**Repos:** engine (`vex_private/vex_engine`), workspace (`plane`), CLI (`~/Me/klio` npm package).

## Global Constraints

- **A PAT is a USER credential.** `KeyInfo.user_id` = the token's workspace user; org = token's workspace org (or personal). Everything downstream (membership guard, node selection, project reach) already keys off that — do not duplicate any authz in the PAT arm.
- **Engine-keys lookup stays first**; the workspace-PAT lookup runs only on miss. Unset `WORKSPACE_DATABASE_URL` ⇒ PATs simply don't validate (401), engine keys unaffected.
- **Inactive/expired/soft-deleted tokens are 401 with the same message as unknown keys** — no oracle distinguishing "revoked" from "never existed" for PATs.
- **`last_used` updates are best-effort** (never fail a request over it) and throttled (at most once per cache TTL per token).
- **Internal API endpoints require `X-Klio-Internal: <secret>`** (constant-time compare); wrong/missing secret → 404 shape. They are additive read-only endpoints; nothing existing changes.
- **The cache treats PAT entries like key entries** (same TTL, same revocation latency ~30s).
- Engine test/lint/baseline discipline as all prior tracks (25 pre-existing / 2192 / 11). Plane suite: 565 passed baseline (`--ignore=plane/tests/unit/bg_tasks/test_cleanup_task.py`). CLI: `npm test` 716 passing baseline.
- TDD RED-first; mutation checks on every auth boundary.

---

### Task E1 (engine): PAT validation + internal status API

**Files:** modify `services/shared/shared/auth.py` (PAT arm in `_query_db`-miss path — new `_query_workspace_pat(raw_key)`), `services/mcp-server/app/main.py` or a new `services/mcp-server/app/internal_api.py` (router), env wiring in `app/auth.py`/config; tests `services/shared/tests/test_auth_workspace_pat.py`, `services/mcp-server/tests/test_internal_api.py`.

**PAT arm rules (each a test):**
1. A valid active token authenticates: org = the token's workspace's org (`organizations.workspace_id`), ensured via `membership.ensure_org_for_workspace` if absent (workspace row fetched from the workspace DB for slug/is_personal/owner).
2. Workspace-less token (user-level) → the user's personal workspace org (lookup owner+is_personal, alive), ensured likewise.
3. `is_active=false`, `expired_at` in the past, or `deleted_at` set → same 401 as an unknown key (no oracle).
4. `KeyInfo.user_id` = the token's user; scopes `["memory","read"]`; `key_id` = the token row id; plan resolves through the existing org-plan path.
5. Engine-keys hit short-circuits (a key in `organizations.api_keys` never reaches the PAT arm); PAT lookup only on miss. Mutation: reorder → test fails.
6. No workspace engine ⇒ PAT is 401, engine keys unaffected.
7. Recording-connection guard: the workspace connection is never asked about engine tables and vice versa.
8. `last_used` best-effort: a failing UPDATE never fails validation (mutation: make it raise → validation still passes).

**Internal API (FastAPI router mounted on the mcp app):**
- `GET /internal/project-link?workspace_project=<uuid>` → `{linked, engine_project_id, display_name, memory_counts: {org, project, private}, open_branches: [{name, unmerged_count, last_activity}]}` (reuse `branches.list_branches`; counts by scope for the linked engine project).
- `GET /internal/workspace-node?workspace=<uuid>` → `{org_id, exists, memory_count}`.
- Auth: `X-Klio-Internal` == env `INTERNAL_API_SECRET` (constant-time); missing env ⇒ endpoints return 404 always (feature off). Tests: right secret → data; wrong/missing → 404; env unset → 404.

Commit: `feat(auth): workspace personal access tokens are engine credentials; internal status API`

### Task C1 (CLI, ~/Me/klio npm — runs in PARALLEL with E1):

- New `klio quickstart`: prints (agent-readable, plain text) the concept map (nodes/projects/branches/scopes), every relevant command, and the MCP config snippets for claude-code/claude-desktop/codex/cursor with `X-Vex-Key` header form and `https://mcp.klio.tech/mcp`.
- `klio init --key <token>`: verify it writes MCP config for the present agents (reuse `wireAgents`), installs hooks, and ends with a live `recall` smoke against the key, printing which node answered (server tells org slug? — print HTTP result + org info from a `/capture/event` probe response or a recall via MCP; whatever the existing init verification does, extend it to print the resolved identity).
- `klio status`: key present? which config files carry it? last probe result.
- Follow repo conventions (node:test, HOME-redirect harness — read tests/run.mjs warnings). TDD. Baseline 716 must not regress.

Commit: `feat(cli): quickstart, key-based init verification, status`

### Task W1 (workspace, after E1's contract is fixed): connect surface + memory panel

- **P2 verify/rebrand:** confirm Workspace Settings → API tokens page works in the fork (components exist under `core/components/api-token/`); fix any Plane-branded copy; ensure the create-modal surfaces the token once with a copy button (stock behaviour — verify, don't rebuild).
- **P4 connect step:** a "Connect your agent" surface in workspace settings (mirroring an existing settings page's layout) + a card on the home quickstart guide linking to it. Content: (1) create-token button routing to the API-tokens page (or inline create via the existing service), (2) the one command `npx @klio-tech/klio init --key <token>` with copy, (3) a copyable agent prompt block ("Paste this to your agent…" → runs quickstart, verifies, reports). No new backend.
- **P5 panel:** Project Settings gains an "Agent memory" section: Django proxy endpoints (`GET /api/workspaces/<slug>/projects/<pk>/agent-memory/` → engine internal API via env `ENGINE_INTERNAL_URL` + `ENGINE_INTERNAL_SECRET`, member-gated by existing project permission classes, 200 with `{configured: false}` when env unset) and a web panel showing: link status, engine project name, memory counts, open branches, and — when unlinked — the copyable agent prompt "ask your agent: link this project (`project_link`, workspace_project=<uuid>)". Workspace-level equivalent on the workspace settings connect page showing the node's org/memory count.
- Tests: Django proxy (mocked engine HTTP: configured/unconfigured/secret propagation/member-gating incl. 404-shape for non-members); in-container suite baseline 565 + delta.

Commit: `feat(web): self-serve agent connect + project agent-memory panel`

### Deploy order
E1 → build engine images → VM (`ENGINE_TAG`), add `INTERNAL_API_SECRET` to `~/klio-engine/.env`. W1 → CI → `make deploy` with `ENGINE_INTERNAL_URL=http://mcp:8000` + `ENGINE_INTERNAL_SECRET` in workspace `.env`. C1 merges → npm auto-publish. Then the live teammate test script.

## Out of scope
OAuth provider (P6, explicitly after the teammate test), identity bridge for legacy engine projects, chat capture env (separate switch), any engine-keys deprecation.
