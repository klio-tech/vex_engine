# Context Network — Track C (Context Branches) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans, task-by-task.

**Goal:** A project node gets git-shaped context branches: work accumulates on a branch, main stays clean, merge promotes with `supersedes` edges, discard retracts — and inside a git repo the hook auto-binds to the current git branch.

**Architecture:** A `context_branches` registry (branch lifecycle lives in a row, not a naming convention) plus a nullable `session_memories.branch_id`. `branch_id IS NULL` **is** main — every existing row is on main by construction, and every existing reader keeps working the day the column appears. Readers exclude branch rows unless the caller names a branch; then branch + main are both live, tagged. Merge re-scopes rows to main and, where a merged row near-duplicates a main row (the existing `DEDUP_THRESHOLD` embedding similarity), the branch version supersedes it through the existing `supersede_memory` — no new conflict machinery. Discard retracts.

**Tech Stack:** as Tracks A/B. Baseline: 25 failed (pre-existing) / 2066 passed / 11 skipped; lint clean; migrations at 049.

## Global Constraints

- **`branch_id IS NULL` means main.** No sentinel values, no "main" row in the registry.
- **A branch always belongs to a project.** No project resolved → no branch; the write falls through to its normal scope. Branch rows carry `scope='project'` + `project_id` + `branch_id`.
- **Branches are pushed-branch visible:** any project member may read a branch by naming it. The visibility ladder's project-membership arm already gates that; branch adds a row filter, never a new access path.
- **Default reads see main only.** A caller who does not name a branch must never receive branch rows — the failure mode is a half-formed branch conclusion surfacing as project truth.
- **Merge and discard are member-gated** by `projects.can_write`, and idempotent (merging a merged branch is a no-op result, not an error).
- **No header/param ⇒ byte-identical behaviour.** The untouched suite is the proof.
- Test/lint invocations, baseline discipline, TDD, and mutation checks: as Tracks A/B.

---

### Task C1: schema + registry + branch-aware writes

**Files:**
- Create: `services/migrations/alembic/versions/050_context_branches.py`
- Create: `services/shared/shared/branches.py`
- Modify: `services/shared/shared/brain.py` (`write_scoped_memory` gains `branch_id: Optional[str] = None`, threaded into the post-insert UPDATE)
- Modify: `services/shared/shared/memory.py` (`find_similar_memory`: dedup window becomes branch-aware — `branch_id IS NOT DISTINCT FROM :branch_id` in every arm, so a branch dedups against itself, not against main)
- Create: `services/shared/tests/test_context_branches_schema.py`, `services/shared/tests/test_branch_writes.py`

**Migration 050:**
```python
def upgrade() -> None:
    op.create_table(
        "context_branches",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=False), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("project_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="open"),
        sa.Column("created_by", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('open','merged','discarded')", name="ck_context_branches_status"),
    )
    # One OPEN branch per (org, project, name); a merged/discarded name is
    # reusable, exactly like git.
    op.create_index("uq_context_branches_open_name", "context_branches",
                    ["org_id", "project_id", "name"], unique=True,
                    postgresql_where=sa.text("status = 'open'"))
    op.create_index("ix_context_branches_project", "context_branches", ["org_id", "project_id"])
    op.add_column("session_memories",
                  sa.Column("branch_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=True))
    op.create_index("ix_session_memories_branch", "session_memories", ["branch_id"],
                    postgresql_where=sa.text("branch_id IS NOT NULL"))
```
(downgrade drops in reverse; verify the roundtrip as always. No FK on `branch_id` — matches the repo's edges/`superseded_by` convention of unconstrained cross-refs plus sweeps.)

**`shared/branches.py`:**
- `ensure_branch(db, *, org_id, project_id, name, created_by) -> dict` — upsert on the open-name index (`ON CONFLICT` needs the index inference: `INSERT ... ON CONFLICT (org_id, project_id, name) WHERE status = 'open' DO UPDATE SET name = EXCLUDED.name RETURNING id, status, ...`). Name validation: strip; refuse blank and >200; refuse `main` (case-insensitive — `main` is the absence of a branch, and a registry row named main would fork the meaning of NULL).
- `get_branch(db, *, org_id, project_id, name) -> Optional[dict]` (open branches only), `get_branch_by_id(db, *, org_id, branch_id)`.
- `list_branches(db, *, org_id, project_id) -> list[dict]` — each with `unmerged_count` (active memories on the branch) and `last_activity` (max created_at) — this is the stale-branch surfacing from the spec, computed not stored.
- Tests: lifecycle, blank/`main` refusal, name reuse after merge, cross-org isolation (branch in org A invisible via org B — the usual recording/negative shapes).

**Branch-aware writes:** `write_scoped_memory(..., branch_id=...)` sets the column in the existing post-insert UPDATE; `find_similar_memory` arms gain the `IS NOT DISTINCT FROM` predicate. Tests RED-first: a write with a branch lands with `branch_id` set; the same content on branch and on main are NOT deduped against each other; two writes on one branch ARE.

Commit: `feat(branches): registry, schema, and branch-aware writes (050)`

### Task C2: branch-aware reads

**Files:**
- Modify: `services/shared/shared/memory.py` (`retrieve_memories_scoped` gains `branch_id: Optional[str] = None`)
- Modify: `services/mcp-server/app/mcp_app.py` (`recall` tool gains `branch: Optional[str]`; `_perform_recall` resolves it via `branches.get_branch` against the resolved project — unknown branch → warn-and-empty like unknown space; each memory tagged `source_branch: <name>` when its row has the branch_id, and the existing B3 `source` tag is preserved)
- Modify: `services/mcp-server/app/capture.py` (session-start recall: same default exclusion)
- Create: `services/shared/tests/test_branch_reads.py`

Rules, each a RED-first test:
- No `branch_id` argument → `base_clauses` gain `branch_id IS NULL`. **This is the load-bearing default**: seed a branch row, run every no-branch recall shape, assert it never surfaces. Mutation: drop the clause → that test fails.
- With `branch_id` → `(branch_id IS NULL OR branch_id = :branch_id)` — branch and main both live, branch rows distinguishable in the result.
- The B3 personal union composes: the extra (personal) org arm always takes `branch_id IS NULL` (branches live under org projects; a personal-node branch contribution is meaningless and must not leak).
- A member who names another member's branch reads it (pushed-branch visibility); a non-member still gets nothing (existing ladder — assert it still binds with the branch filter present).

Commit: `feat(branches): reads default to main, opt into a branch, tagged`

### Task C3: merge, discard, the `branch` tool, and hook auto-bind

**Files:**
- Modify: `services/shared/shared/branches.py` (add `merge_branch`, `discard_branch`)
- Modify: `services/mcp-server/app/mcp_app.py` (new `branch` MCP tool: `action: Literal["create","list","info","merge","discard"]`, `name`, optional `project`; member-gated like the project tools — non-member gets `not_found`)
- Modify: `services/mcp-server/app/capture.py` (`CaptureTranscriptRequest` + event request gain `git_branch: Optional[str]`; when present AND a project resolved AND the caller is attributable → `ensure_branch` and the capture writes carry its id. `main`/`master` map to NULL — the repo's default branch IS main.)
- Create: `services/shared/tests/test_branch_merge_discard.py`, `services/mcp-server/tests/test_branch_tool_and_autobind.py`

**`merge_branch(db, *, org_id, project_id, branch_id, merged_by) -> dict`:**
1. Load branch FOR UPDATE; `status != 'open'` → `{"merged": False, "reason": "already_<status>"}` (idempotence).
2. For each active memory on the branch (embedding present), `find_similar_memory` against MAIN rows of the same project (`branch_id IS NULL` window — extend the function or query directly); similarity ≥ `DEDUP_THRESHOLD` → `supersede_memory(old=main_row, new=branch_row)` — the branch version is the newer belief and the graph edge follows via the existing machinery pattern (write the `supersedes` edge as `curator.py:669-690` does).
3. `UPDATE session_memories SET branch_id = NULL, updated_at = NOW() WHERE branch_id = :b AND org_id = :org` — the rows are now main.
4. Branch row → `status='merged'`, `closed_at=NOW()`. Return counts: `{"merged": True, "promoted": n, "superseded": m}`.

**`discard_branch`:** branch rows → `status='retracted'` with `metadata.retracted_via='branch_discard'` (mirror `retract_memory`'s metadata discipline); branch row → `discarded`. Idempotent like merge.

Tests, RED first: merge promotes (rows readable without naming the branch afterwards); a near-duplicate main row is superseded with the edge present (query `memory_edges`); discard makes rows invisible everywhere; both idempotent; non-member merge → `not_found`; auto-bind: transcript with `git_branch="feat/x"` lands on that branch, with `git_branch="main"` lands on main (NULL), unattributable caller never auto-creates a branch. Mutation: skip step 2 (no supersede) → the supersede test fails; make discard leave rows `active` → the invisibility test fails.

Commit: `feat(branches): merge via supersedes, discard, the branch tool, hook auto-bind`

### Task C4 (klio CLI repo — ~/Me/klio): send the git branch

The hooks that POST `/capture/transcript` add `git_branch` from `git rev-parse --abbrev-ref HEAD` (best-effort: outside a repo or on error, omit the field). Find the capture POST in the package, add the field + a test matching that repo's conventions, commit on a branch, push, PR. Engine tolerates absence (C3), so ordering is safe either way.

## Out of scope
- LLM-judged merge contradictions (embedding similarity only in v1 — recorded in `merge_branch`'s docstring).
- Branch attach in chat (Track D), stale-branch *notifications* (surfacing exists via `list_branches`).
