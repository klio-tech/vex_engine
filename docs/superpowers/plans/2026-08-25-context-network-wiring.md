# Context Network — Track B (Wire the Node) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans, task-by-task.

**Goal:** `X-Klio-Node` gets wire effect — every MCP tool and capture route acts in the selected node instead of the key's org, and recall can additionally stack the caller's personal node under the active org.

**Architecture:** One new resolver, `resolve_active_principal(request, db)`, wraps the existing `resolve_principal_plan` + `resolve_active_org` pair so a tool body swaps one call for another and cannot rewire half the surface. Sites that hold a `db` session adopt it; the handful that resolve before opening a session open it earlier (SessionLocal is cheap; the connection is lazy). Recall grows an opt-in `include_personal` union so an agent attached to an org still sees its own personal-node memories, tagged by source.

**Tech Stack:** as Track A. Baseline: 25 failed (pre-existing) / 2045 passed / 11 skipped; lint clean.

## Global Constraints

- **No header ⇒ byte-identical behaviour.** Every existing caller must be unaffected. The suite is the proof: outside new tests, nothing may change.
- **One resolver.** No tool body may call `resolve_active_org` + `resolve_principal_plan` separately — that is the two-copies-drift lesson (d6f53ce). Everything goes through `resolve_active_principal`.
- **The active org is a tenancy boundary.** It comes out of the membership check; after that, every downstream `org_id=` binding uses it. A site that keeps the key org while its neighbours use the active org is a cross-tenant write.
- **Fail closed.** `WORKSPACE_DATABASE_URL` unset + header present + header ≠ key org → 404 (already implemented; do not weaken while wiring).
- **Personal-node recall is read-only union.** Writes NEVER fan out; a write lands in exactly one org (the active one).

---

### Task B1: `resolve_active_principal` + wire `mcp_app.py`

**Files:**
- Modify: `services/mcp-server/app/identity.py` (add the wrapper)
- Modify: `services/mcp-server/app/mcp_app.py` (every `resolve_principal*(request)` site — grep gives the authoritative list; ~13 sites here)
- Create: `services/mcp-server/tests/test_active_principal_wiring.py`

**Interfaces:**
- Produces: `resolve_active_principal(request, db, *, workspace=None) -> tuple[str, str, str, Optional[dict]]` — `(active_org_id, agent_id, plan, plan_overrides)`. Same tuple shape as `resolve_principal_plan`, org swapped for the active one.

Rules:
- The wrapper calls `resolve_principal_plan(request)` once, then `resolve_active_org(request, db, workspace=...)` once, returns the active org with the rest of the tuple untouched. It must not double-validate the key (both callees already share the per-request memo).
- Sites that currently call `resolve_principal(request)` (2-tuple) get the 4-tuple wrapper with the extras ignored, or a 2-tuple convenience — pick ONE style and use it everywhere.
- Sites that resolve before `db = SessionLocal()` exists: move the session open above the resolution. Keep the 402/429 refusal ordering where a comment says refusal must precede session work — read `mcp_app.py:846` and preserve that property (open the session, resolve, refuse, close in finally — the existing try/finally patterns already close).
- Tests (RED first): with a header naming a second reachable workspace-org, `remember` stores at the selected org and `recall` reads it there (use the tool-layer test conventions already in `services/mcp-server/tests/`; the membership fixture pattern is in `test_node_selection.py`); without a header, an identical call lands at the key org; with an unreachable header, the tool returns the 404-shaped error, not a 500.
- Mutation check: make the wrapper return the key org unconditionally — the selected-org tests must fail.
- Commit: `feat(wiring): mcp tools act in the selected node`

### Task B2: wire `capture.py`, `proposals.py`, `onboarding.py`

Same treatment for the remaining resolution sites (grep `resolve_principal` in those three files; also the `resolve_principal_dep` FastAPI dependency in `capture.py` — give it an active-org variant that takes the request's db from the route body, or resolve inside the body where the dep only fed org/agent). The hook-capture routes are the sensitive ones: a hook firing with a node header must capture INTO that node.
- Tests mirror B1's three shapes for one capture route and one proposals route.
- Commit: `feat(wiring): capture and proposals act in the selected node`

### Task B3: personal-node union in recall

**Files:**
- Modify: `services/shared/shared/memory.py` (`retrieve_memories_scoped`)
- Modify: `services/mcp-server/app/mcp_app.py` (`_perform_recall`)
- Create: `services/shared/tests/test_recall_personal_union.py`

Rules:
- `retrieve_memories_scoped` gains `extra_org_id: Optional[str] = None`, which when set widens the tenancy predicate to `org_id IN (:org_id, :extra_org_id)` **with the visibility ladder applied per-org** — the extra org contributes ONLY rows the caller could read there (in practice: their personal node, where everything is theirs). Every returned row already carries `org_id`; `_perform_recall` maps it to a `source: "personal" | "active"` tag on the payload.
- `_perform_recall` passes the caller's personal org as `extra_org_id` only when (a) the caller is attributable, (b) a personal org exists (`organizations.personal_owner_user_id = user`), and (c) the active org is not already the personal org.
- The dangerous direction gets the test: another user's private rows in the extra org must NOT surface (the ladder's `user_id` predicate must bind in both orgs). Mutation: drop the per-org ladder on the extra arm → that test fails.
- Writes are untouched. Assert that too: a `remember` with the union active still lands only in the active org.
- Commit: `feat(wiring): recall unions the personal node, tagged by source`

## Out of scope
- `WORKSPACE_DATABASE_URL` in Railway prod: the Plane DB lives in Docker on the GCP VM and is not reachable from Railway today — connectivity (private networking / expose / relocate) is an infra decision recorded for the user, not code.
- Branches (C), chat (D), Supabase retirement.
