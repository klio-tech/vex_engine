# Context Network — Project Link (workspace ⇄ engine projects) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans, task-by-task.

**Goal:** One project, two faces: a workspace (Plane) project and an engine project joined by `projects.plane_project_id`, with read visibility (and explicit-write permission) following the **workspace** project's membership — the GitHub-repo model: add someone to the project in the UI, their agent can recall its memory.

**Architecture:** Cross-database, joined in Python — the established pattern (see `shared/membership.py`, PR #45): the workspace DB answers "which plane projects is this user in, at what role", the engine DB answers "which of those are linked to my org's projects", and the recall ladder gains one precomputed `ANY(:ids)` arm. Linking is agent-driven (a new `project_link` MCP tool), member-gated on **both** sides, and never re-points — the `6ad63ae` lesson. Resolution: a workspace project UUID becomes a first-class project identifier everywhere (`X-Vex-Project`, `project=`), via a `plane_project_id` fallback in `find_project_id`.

**Baseline:** engine main (`db620b5b`+), suite 25 pre-existing failed / 2151 passed / 11 skipped; lint clean. Same run/lint invocations as all prior tracks. `WORKSPACE_DATABASE_URL` reaches Plane's `projects` (workspace FK, `deleted_at`) and `project_members` (`member_id`… **verify actual column names in Plane's models before writing SQL**: apps/api/plane/db/models/project.py — `project_members` has `project_id`, `member_id`, `role` int, `is_active`, `deleted_at`). Plane roles: 20 admin, 15 member, 5 guest → map 20→"admin", 15→"write", 5→"read".

## Global Constraints

- **Never re-point a link** (either direction). An engine project with a different non-NULL `plane_project_id`, or a plane id already linked to a different engine project, is a refusal — not an update.
- **Linked-membership is a UNION, not a replacement**: engine `project_members` keep working exactly as today; workspace membership ADDS reach on linked projects only.
- **Cross-DB queries never name the other side's tables** — recording-connection guard tests, as in `test_membership_resolution.py`.
- **No workspace engine configured ⇒ today's behaviour byte-identical** (the extra arm is simply empty; `project_link` returns a clear error).
- **Non-members get `not_found`**, the existing convention.
- TDD RED-first; mutation checks on every boundary; 25-failure baseline must not rise; ruff check+format clean.

---

### Task L1 (engine): link primitive + resolution + read/write reach

**Files:**
- Create: `services/shared/shared/project_link.py`
- Modify: `services/shared/shared/brain.py` (`find_project_id`: UUID misses on `id` → try `plane_project_id = :ident` before the ranked query)
- Modify: `services/shared/shared/memory.py` (`retrieve_memories_scoped` gains `extra_project_ids: Optional[list[str]] = None` → one additional visibility arm `(scope = 'project' AND project_id = ANY(:extra_project_ids))`; empty/None = absent, no other ladder change)
- Modify: `services/mcp-server/app/mcp_app.py`:
  - `_perform_recall`: when the caller is attributable AND the workspace engine is configured, compute `extra_project_ids` = engine ids of org projects whose `plane_project_id` is in the caller's active plane memberships (role ≥ 5). Two queries, joined in Python.
  - `_explicit_write_scope`: a caller who is not an engine member may still land a `project` write when the project is linked and their plane role ≥ 15 (helper `can_write_linked`).
  - New MCP tool `project_link(project, workspace_project)`: resolve engine project (existing identifier rules); gates in order — attributable caller; engine-side `projects.can_write` OR creator; caller is an active member (role ≥ 15) of the plane project in the workspace DB; neither side already linked elsewhere. Writes the id with `UPDATE … WHERE plane_project_id IS NULL RETURNING`, lost race → re-read, winner-if-ours else error. Returns `{linked, project_id, workspace_project}`. Non-member/unknown on either side → `not_found` shape.
- Create: `services/shared/tests/test_project_link.py`, `services/mcp-server/tests/test_project_link_tool.py` (+ extend recall wiring tests)

**`shared/project_link.py` produces:**
- `link_engine_project(tsdb, *, org_id, project_id, plane_project_id) -> dict` (the guarded UPDATE only — tool-layer does the gating)
- `plane_project_memberships(workspace_db, *, user_id) -> dict[str, int]` (plane_project_id → role; `is_active`, both `deleted_at IS NULL`, join `projects`→workspace alive too)
- `linked_engine_projects(tsdb, *, org_id, plane_ids) -> dict[engine_id, plane_id]`
- `can_write_linked(workspace_db, tsdb, *, org_id, project_id, user_id) -> bool` (engine `can_write` OR linked-and-plane-role ≥ 15; workspace_db None → engine answer only)

**Tests that carry the task (RED first, each mutation-checked):**
1. Linked + plane-member (role 15) recalls the project's memories via the extra arm; **the same user with the link removed gets nothing** (mutation: drop the extra arm → fails).
2. A plane member of an UNLINKED project gets nothing; a non-member of a linked project gets nothing (the arm must be built from *their* memberships — mutation: build it from all org links → fails).
3. Cross-org: a linked project in org B never enters org A's arm.
4. Never-re-point, both directions (mutation: disable guard → fails).
5. `find_project_id` resolves a plane uuid to the linked engine project; unlinked plane uuid → None (no fallthrough to the ranked query on a uuid).
6. Recording-connection guards for every new cross-DB function.
7. `can_write_linked`: role 15 writes, role 5 does not, workspace_db=None keeps engine-only behaviour.
8. Tool gating: not-plane-member → not_found; already-linked-elsewhere → error naming nothing about the other org.

Commit: `feat(projects): workspace and engine projects link — membership reaches through`

### Task L2 (plane repo, branch feat/chat-v1 base main): bridge addresses by id

`chat_capture_task.py`: project channels send `X-Vex-Project: <plane project uuid>` instead of display name (delete the interim-docstring paragraph; the engine now resolves the uuid via L1). Update its payload-shape tests. Full in-container suite; no regressions. Commit: `fix(chat): capture addresses the engine by workspace project id`

## Out of scope
UI (deliberately), public/per-individual personal-project sharing (the model supports it later via the same membership layer), auto-linking heuristics (explicit agent-driven only in v1).
