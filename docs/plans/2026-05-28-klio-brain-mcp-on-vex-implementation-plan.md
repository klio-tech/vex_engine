# Klio Brain — MCP memory service on Vex — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) to implement this plan task-by-task.

**Goal:** Add a hosted remote-MCP memory brain as an *additive* service inside the Vex engine, exposing Klio's 7 tools over Streamable HTTP, backed by Vex's existing org-scoped pgvector memory, with team-wide ("org" scope) persistent recall plus spaces/projects.

> **CORRECTION (env-validation finding):** the live alembic head is **019** (`018_add_alert_rules`, `019_add_experiments` already exist in Vex). Our migration is therefore **020**, `down_revision="019"` — NOT 018/017 as an earlier draft assumed. All "migration 020" references below are correct; the design doc's "018" is superseded by this.

**Architecture:** New `services/mcp-server/` (FastAPI + official `mcp` SDK) reuses `shared/auth.py` for `X-Vex-Key → (org_id, agent_id)` and `shared/memory.py` for storage. One new memory function (`retrieve_memories_scoped`) and one additive migration (020: `spaces`, `projects`, nullable columns). Verification services, the existing `retrieve_memories`, and the local-Klio repo are **untouched**.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy (sync `create_engine` + `sessionmaker` — see gotcha), alembic, pytest, Postgres + pgvector (HNSW, `vector(1024)`), Redis, `mcp` SDK (Streamable HTTP), docker-compose.

**Design doc:** `docs/plans/2026-05-28-klio-brain-mcp-on-vex-design.md`

**Standing constraints:**
- **Purely additive.** Do NOT modify `verification-engine`, `sync-gateway`, `alert-service`, or the existing `retrieve_memories`. Do NOT touch the local-Klio repo (`/Users/thakurg/Me/klio`).
- **Local commits only. Do NOT push** until the user approves.
- **Sync-session gotcha:** `shared/memory.py` functions are `async def` but call `db_session.execute()`/`.commit()` *synchronously*. Use `sqlalchemy.create_engine` + `sessionmaker` (mirror `services/storage-worker/app/db.py`). Do NOT introduce `async_sessionmaker`.

**Test env:** mirror the existing services' `conftest.py`. DB at the compose Postgres; `DATABASE_URL` env. Run `pytest` from each service dir (the repo uses per-service test suites).

---

## Phase A — Data model (additive)

### Task A1: Migration 020 — spaces, projects, nullable columns

**Files:**
- Create: `services/migrations/alembic/versions/020_klio_brain_spaces_projects.py`
- Test: `services/migrations/tests/test_migration_020.py` (create; mirror any existing migration test, else use the inline pattern below)

**Step 1: Write the failing test**

```python
# services/migrations/tests/test_migration_020.py
import os
import subprocess
import sys
import uuid

import pytest
from sqlalchemy import create_engine, inspect, text

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",
)


def _alembic_upgrade_head():
    env = os.environ.copy()
    env["DATABASE_URL"] = DB_URL
    r = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        env=env,
    )
    assert r.returncode == 0, f"alembic upgrade failed:\n{r.stdout}\n{r.stderr}"


def test_spaces_and_projects_tables_exist():
    _alembic_upgrade_head()
    eng = create_engine(DB_URL)
    with eng.connect() as c:
        tables = set(inspect(c).get_table_names(schema="public"))
    assert "spaces" in tables
    assert "projects" in tables


def test_session_memories_gains_nullable_space_project():
    _alembic_upgrade_head()
    eng = create_engine(DB_URL)
    with eng.connect() as c:
        cols = {col["name"]: col for col in inspect(c).get_columns("session_memories")}
    assert "space_id" in cols and cols["space_id"]["nullable"] is True
    assert "project_id" in cols and cols["project_id"]["nullable"] is True


def test_existing_session_memory_row_undisturbed():
    """Additive contract: a pre-existing row (no space/project) stays valid."""
    _alembic_upgrade_head()
    eng = create_engine(DB_URL)
    with eng.begin() as c:
        sid = str(uuid.uuid4())
        c.execute(text("""
            INSERT INTO session_memories
              (id, session_id, org_id, agent_id, memory_type, content, scope, status)
            VALUES (gen_random_uuid(), :sid, 'org_test', 'agent_test',
                    'fact', 'legacy row', 'session', 'active')
        """), {"sid": sid})
        row = c.execute(text(
            "SELECT space_id, project_id FROM session_memories WHERE session_id=:sid"
        ), {"sid": sid}).one()
    assert row.space_id is None and row.project_id is None
```

**Step 2: Run — verify it fails**

```
cd services/migrations && DATABASE_URL=... uv run pytest tests/test_migration_020.py -v
```
Expected: FAIL (`spaces`/`projects` missing).

**Step 3: Write migration 020**

```python
# services/migrations/alembic/versions/020_klio_brain_spaces_projects.py
"""Klio brain: spaces + projects tables, nullable space/project on session_memories.

ADDITIVE ONLY. Existing session_memories rows are unaffected (new columns
default NULL). Verification memory paths never set these columns.

Revision ID: 020
Revises: 019
Create Date: 2026-05-28
"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "020"
down_revision: Union[str, None] = "019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "spaces",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("slug", sa.String(200), nullable=False),
        sa.Column("embedding_model", sa.String(200), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.UniqueConstraint("org_id", "slug", name="uq_spaces_org_slug"),
    )
    op.create_table(
        "projects",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("space_id", sa.UUID(), sa.ForeignKey("spaces.id", ondelete="SET NULL"), nullable=True),
        sa.Column("git_remote", sa.Text(), nullable=True),
        sa.Column("repo_root_path", sa.Text(), nullable=True),
        sa.Column("display_name", sa.String(200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index(
        "ix_projects_org_remote", "projects", ["org_id", "git_remote"],
        unique=True, postgresql_where=sa.text("git_remote IS NOT NULL"),
    )
    op.create_index(
        "ix_projects_org_path", "projects", ["org_id", "repo_root_path"],
        unique=True, postgresql_where=sa.text("git_remote IS NULL AND repo_root_path IS NOT NULL"),
    )
    op.add_column("session_memories",
        sa.Column("space_id", sa.UUID(), sa.ForeignKey("spaces.id", ondelete="SET NULL"), nullable=True))
    op.add_column("session_memories",
        sa.Column("project_id", sa.UUID(), sa.ForeignKey("projects.id", ondelete="SET NULL"), nullable=True))
    op.create_index("ix_session_memories_space", "session_memories", ["space_id"])
    op.create_index("ix_session_memories_project", "session_memories", ["project_id"])


def downgrade() -> None:
    op.drop_index("ix_session_memories_project", table_name="session_memories")
    op.drop_index("ix_session_memories_space", table_name="session_memories")
    op.drop_column("session_memories", "project_id")
    op.drop_column("session_memories", "space_id")
    op.drop_index("ix_projects_org_path", table_name="projects")
    op.drop_index("ix_projects_org_remote", table_name="projects")
    op.drop_table("projects")
    op.drop_table("spaces")
```

**Step 4: Run — verify pass.** `uv run pytest tests/test_migration_020.py -v` → 3 PASS.

**Step 5: Commit**

```
git add services/migrations/alembic/versions/020_klio_brain_spaces_projects.py \
        services/migrations/tests/test_migration_020.py
git commit -m "feat(migrations): 020 additive spaces+projects+nullable cols for Klio brain"
```

---

### Task A2: Space + Project access helpers (org-scoped get-or-create)

**Files:**
- Create: `services/shared/shared/brain.py` (new module — keeps brain additions out of `memory.py` so the existing file's surface is unchanged)
- Test: `services/shared/tests/test_brain.py`

Implement `ensure_space(db, org_id, name, slug) -> space_id` and `ensure_project(db, org_id, git_remote, repo_root_path, display_name, space_id=None) -> project_id`, mirroring local-Klio's `ProjectService.ensure` (dedupe by git_remote when present else repo_root_path; bump `last_seen_at`; raise `ValueError` if both keys None). Sync session. Tests: create-on-first, dedupe-by-remote, dedupe-by-path, tenant isolation (org A ≠ org B), ValueError on both-None.

**Commit:** `feat(shared): brain.py — org-scoped space/project ensure helpers`

---

## Phase B — Brain recall (additive memory fn)

### Task B1: `retrieve_memories_scoped`

**Files:**
- Modify: `services/shared/shared/memory.py` (ADD a new function at the end; do NOT touch `retrieve_memories`)
- Test: `services/shared/tests/test_brain_recall.py`

New signature:
```python
async def retrieve_memories_scoped(
    db_session, org_id: str, query_text: str, *,
    scope: str = "org", agent_id: str | None = None,
    space_id=None, project_id=None, limit: int = 10,
) -> list[MemoryRecord]:
```
Behavior (mirror `retrieve_memories`'s embed + pgvector cosine, but):
- Filter `org_id = :org_id AND status='active' AND superseded_by IS NULL`.
- `scope="org"` → no session/agent narrowing (uses `ix_session_memories_org_lookup`). `scope="agent"` → `AND agent_id = :agent_id`. `scope="session"` → keep current session behavior (require a session_id param if you support it; otherwise document org/agent only for v1).
- If `project_id` given: `AND (project_id = :project_id OR project_id IS NULL)` (NULL-surfacing, matching local-Klio B2). Same shape for `space_id`.

Tests: org isolation; scope=org returns across agents; scope=agent narrows; project filter + NULL-surfacing; existing `retrieve_memories` import still works unchanged (regression guard).

**Commit:** `feat(shared): retrieve_memories_scoped (brain recall; existing fn untouched)`

---

## Phase C — MCP service skeleton + auth

### Task C1: `services/mcp-server/` scaffold + health + db + deps

**Files:** Create `services/mcp-server/{app/__init__.py,app/main.py,app/db.py,app/config.py,requirements.txt,Dockerfile,tests/__init__.py,tests/conftest.py,tests/test_health.py}`

- `app/db.py`: mirror `services/storage-worker/app/db.py` exactly — `create_engine(DATABASE_URL, pool_pre_ping=True)` + `SessionLocal = sessionmaker(bind=engine)` + a `get_db()` FastAPI dependency that yields a `SessionLocal()` and closes it.
- `requirements.txt`: the service deps + **`mcp`** (the official SDK) + `fastapi`, `uvicorn`, the shared lib.
- `app/main.py`: `create_app()` factory + lifespan (mirror ingestion-api) with a `/healthz` route.
- Test `test_health.py`: `GET /healthz` → 200.

**Commit:** `feat(mcp-server): service scaffold (app factory, sync db, health, deps)`

### Task C2: Auth bridge

**Files:** Create `services/mcp-server/app/auth.py`; Test `tests/test_auth.py`

Mirror `services/ingestion-api/app/auth.py`: `get_validator()` singleton with `KeyValidator(database_url=DATABASE_URL, required_scope="memory", ...)`, `verify_api_key(request) -> KeyInfo` reading `X-Vex-Key`. Add a helper `resolve_principal(request) -> tuple[str, str]` returning `(org_id, agent_id)` (agent_id from header `X-Vex-Agent` or KeyInfo). Tests: missing key→401, invalid→401, valid→org_id; agent resolution.

> Note: `required_scope="memory"` — confirm whether to reuse `"ingest"` or add a `"memory"` scope to the org api_keys. If adding a scope is non-trivial, reuse `"ingest"` for v1 and flag it. (Open Question — do not block.)

**Commit:** `feat(mcp-server): X-Vex-Key auth bridge → (org_id, agent_id)`

---

## Phase D — MCP tools

### Task D1: MCP server over Streamable HTTP + tools/list

**Files:** Create `services/mcp-server/app/mcp_app.py`; mount into `app/main.py`; Test `tests/test_mcp_protocol.py`

Use the `mcp` SDK's `FastMCP` (or low-level Server) with the **Streamable HTTP** transport, mounted as an ASGI sub-app under the FastAPI `app` (e.g. at `/mcp`). Register the 7 tool stubs so `tools/list` returns them. Test: an MCP client (the SDK's in-memory/HTTP client) `initialize` + `tools/list` → the 7 tool names present.

> Pin the `mcp` SDK version in requirements; consult Context7/official MCP Python SDK docs for the exact Streamable-HTTP + FastAPI mount API (it has shifted across versions). Reference the docs, don't guess the transport wiring.

**Commit:** `feat(mcp-server): Streamable HTTP transport + 7-tool registration`

### Task D2: write tools (remember/observe/plan/decide/note)

Map each to `write_memory(db, session_id=<per-connection id>, org_id, agent_id, execution_id=None, memory_type=<kind>, content, metadata)` with `scope="org"` (set via the row's scope — extend `write_memory` call path to pass scope, or write then update scope; prefer adding a `scope` kwarg to a thin brain-write wrapper in `brain.py` rather than editing `write_memory`). Dedup via existing `find_similar_memory`. Per-tool memory_type: memory/observation/plan/decision/note. Tests: each tool writes a row with correct memory_type + scope=org + org_id; dedup suppresses a near-duplicate.

**Commit:** `feat(mcp-server): remember/observe/plan/decide/note write tools`

### Task D3: recall tool

`recall(query, scope?, space?, project?)` → `retrieve_memories_scoped(...)`. Resolve `space`/`project` names→ids via `brain.py`. Tests: recall returns org-scoped results; project filter + NULL-surfacing; org isolation through the full tool path.

**Commit:** `feat(mcp-server): recall tool (scope/space/project aware)`

### Task D4: space tool

`space(action: list|create|info|switch, ...)` over `brain.ensure_space` + a list query. Tests: create→list shows it; tenant-scoped.

**Commit:** `feat(mcp-server): space management tool`

---

## Phase E — Curator scheduler (server-side tuning)

### Task E1: org-scoped curator tick

**Files:** extend `services/async-worker/app/worker.py` (it already imports `write_memory`) with an additive scheduled job, OR a new `services/mcp-server/app/curator.py` invoked by a periodic trigger. Reuse `extract_facts` + `generate_summary`. Per-org cursor (a small `brain_curator_state` table, or reuse metadata). Synthesize `scope="org"` observations → summary/fact/decision. Tests: given N org observations, a tick writes synthesized memory; cursor advances; idempotent.

> Keep this additive — if extending async-worker risks touching verification behavior, put the curator in its own module/worker. Decide during the task.

**Commit:** `feat(brain): org-scoped curator synthesis tick`

---

## Phase F — Coexistence, integration, release prep

### Task F1: Coexistence guard tests
Assert: existing `retrieve_memories` signature/behavior unchanged (call it, verify session-scoped result); the `sync-gateway` + `verification-engine` test suites still pass; migration 020 is additive (re-run their memory tests). **Commit:** `test: coexistence — verification + existing memory paths unaffected`

### Task F2: End-to-end integration
MCP client → Streamable HTTP → `remember` then `recall` round-trip against a real test org; org-isolation (org B cannot recall org A). **Commit:** `test(mcp-server): e2e memory round-trip + org isolation`

### Task F3: compose + CI + docs
Add `mcp-server` to `docker-compose.yml` (mirror ingestion-api service block: env, depends_on postgres+redis, port). Add to the Docker Hub publish matrix in `.github/workflows`. Update `vex_engine/README.md` services table + a CHANGELOG note. **Do NOT push.** **Commit:** `chore(mcp-server): compose + CI wiring + docs`

---

## Open questions (resolve in-flight, don't block)
1. **`memory` scope on api_keys** vs reuse `ingest` for v1 (C2).
2. **`session_id` for org writes** — generate a stable per-MCP-connection id (recommended) vs relax the NOT-NULL column (additive). Confirm in D2.
3. **`scope` kwarg plumbing** — add a thin brain-write wrapper in `brain.py` (preferred, keeps `write_memory` untouched) vs add a `scope` param to `write_memory`.
4. **Curator placement** — extend async-worker vs dedicated module (E1).
5. **Per-tenant envelope keys / BYOK** — Phase 2, per design doc.

## Verification matrix
| Layer | Test file | Tasks |
|---|---|---|
| Migration 020 | `migrations/tests/test_migration_020.py` | A1 |
| Space/project helpers | `shared/tests/test_brain.py` | A2 |
| Brain recall | `shared/tests/test_brain_recall.py` | B1 |
| Service health/db | `mcp-server/tests/test_health.py` | C1 |
| Auth bridge | `mcp-server/tests/test_auth.py` | C2 |
| MCP protocol | `mcp-server/tests/test_mcp_protocol.py` | D1 |
| Write/recall/space tools | `mcp-server/tests/test_tools_*.py` | D2-D4 |
| Curator | `*/tests/test_curator.py` | E1 |
| Coexistence | existing suites + new guard | F1 |
| E2E + isolation | `mcp-server/tests/test_e2e.py` | F2 |
