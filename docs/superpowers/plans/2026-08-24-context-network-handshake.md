# Context Network — Track A2 (Plane ⇄ Engine Handshake) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One-to-one match between Plane workspaces and engine orgs — the workspace DB replaces Supabase as the identity source, the first workspace a user creates is their personal workspace, and its slug is their username.

**Architecture:** Two repos. Plane (`~/Hive/Research/AgentGuard/plane`, Django) gains `workspaces.is_personal`, set on the first workspace a user creates, with a partial unique index enforcing one personal workspace per living owner. The engine (`~/Hive/Research/AgentGuard/vex_private/vex_engine`) gains `organizations.workspace_id` as the immutable join key (slug is mutable: soft-delete rewrites it to `slug__<epoch>`), swaps `membership.py`'s external side from Supabase to the workspace DB, and gains an idempotent ensure that provisions/links an engine org for a workspace on first selection.

**Tech Stack:** Plane: Django 5.2 / DRF / Postgres, migrations at `apps/api/plane/db/migrations/` (latest: 0122), pytest with `plane.settings.test`. Engine: Python 3.11 / SQLAlchemy Core / Alembic (latest: 048), pytest.

## Global Constraints

- **Join by `workspace_id`, never by slug.** Slug is the *address* (username / org handle); id is the identity. Plane soft-delete mutates slug.
- **The user UUID is the same value in both systems.** Plane `users.id` is what the engine stores in `user_id` columns. No mapping table.
- **Soft-delete awareness everywhere.** Plane rows persist with `deleted_at` set. Every query against Plane tables filters `deleted_at IS NULL`; every uniqueness that must survive delete-and-recreate is partial on it.
- **Two databases, joined in Python.** The workspace DB (`WORKSPACE_DATABASE_URL`) and the engine DB (`DATABASE_URL`) are separate servers. No SQL statement names tables from both. The recording-connection guard test pattern from `test_membership_resolution.py` applies.
- **Engine baseline:** 25 failed (pre-existing, outside CI's paths) / 2017 passed / 11 skipped. Do not exceed 25. Engine lint: `ruff check services/` and `ruff format --check services/` both pass — keep them passing.
- **Plane test invocation:** `docker exec klio-api-1 sh -c 'cd /code && python -m pytest <path> -q'` after `docker cp`-ing changed files in — or run migrations/tests via the compose stack. Plane tests live under `apps/api/plane/tests/` (see `unit/templatetags` for the convention).
- **Engine test invocation:** `DATABASE_URL="postgresql://agentguard:agentguard_dev@localhost:5432/agentguard" .venv/bin/python -m pytest ...` from the engine repo root.
- **Behaviour off = today's behaviour.** With `WORKSPACE_DATABASE_URL` unset, everything works exactly as it does now (fail-closed on node selection, key-org default unchanged).

---

### Task P1 (Plane repo): `is_personal`, set on first creation

**Files:**
- Modify: `apps/api/plane/db/models/workspace.py` (Workspace model, ~line 119)
- Create: `apps/api/plane/db/migrations/0123_workspace_is_personal.py` (via `makemigrations`, then add the partial constraint)
- Modify: `apps/api/plane/app/views/workspace/base.py:83-130` (create view)
- Create: `apps/api/plane/tests/unit/workspace/test_personal_workspace.py` (+ `__init__.py` files as needed)

**Interfaces:**
- Produces: `workspaces.is_personal boolean NOT NULL DEFAULT false` column; invariant "at most one living personal workspace per owner" enforced by a partial unique index `WHERE is_personal AND deleted_at IS NULL`.

Steps:

- [ ] Model: add to `Workspace`:
```python
    is_personal = models.BooleanField(
        default=False,
        help_text=(
            "The first workspace a user creates is their personal workspace; "
            "its slug is their username. At most one living personal "
            "workspace per owner, enforced by a partial unique index -- "
            "partial on deleted_at so deleting it does not lock the user out "
            "of ever having one again."
        ),
    )
```
  and in `Workspace.Meta`, add:
```python
        constraints = [
            models.UniqueConstraint(
                fields=["owner"],
                condition=models.Q(is_personal=True, deleted_at__isnull=True),
                name="uq_workspace_personal_owner",
            )
        ]
```
- [ ] Generate the migration (`python manage.py makemigrations db` inside the api container), verify it contains both the field and the constraint, and that `migrate` and `migrate db 0122` (rollback) both succeed.
- [ ] View: in `WorkSpaceViewSet.create`, before `serializer.save(owner=request.user)`, compute:
```python
            # The first workspace a user creates is their personal workspace
            # (its slug becomes their username). Explicitly their call to
            # change later; this only picks the default.
            is_first = not Workspace.objects.filter(owner=request.user).exists()
```
  and pass `serializer.save(owner=request.user, is_personal=is_first)`.
- [ ] Tests (TDD: write first, watch them fail against the un-modified view):
  - first workspace created by a user has `is_personal=True`
  - second workspace by the same user has `is_personal=False`
  - the DB constraint rejects a second living personal workspace for one owner (`IntegrityError` on direct `objects.create`)
  - soft-deleting the personal workspace then creating a new first workspace succeeds (the partial index permits it)
- [ ] Run the full existing Plane test suite you can reach (`plane/tests/`), lint if the repo has it wired, commit:
  `feat: first-created workspace is the user's personal workspace`

---

### Task E1 (engine): migration 049 — the join keys

**Files:**
- Create: `services/migrations/alembic/versions/049_workspace_handshake_keys.py`
- Create: `services/shared/tests/test_workspace_handshake_keys.py`

**Interfaces:**
- Produces: `organizations.workspace_id UUID NULL` + unique partial index `WHERE workspace_id IS NOT NULL`; `projects.plane_project_id UUID NULL` + unique partial index `WHERE plane_project_id IS NOT NULL`.

Steps:

- [ ] Test first (against the live DB, like `test_scopes_single_definition.py` does): assert both columns exist with the right type and that inserting two orgs with the same non-NULL `workspace_id` raises, while two NULLs coexist (that is what partial gives us — pre-handshake rows are legal).
- [ ] Migration:
```python
"""Workspace handshake keys: the immutable Plane <-> engine join.

Revision ID: 049
Revises: 048
"""

import sqlalchemy as sa
from alembic import op

revision = "049"
down_revision = "048"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # UUID, not String: Plane ids are uuid4 and the column should refuse
    # anything that is not one. Nullable because every existing org predates
    # the handshake; unique where present because the match is one-to-one.
    op.add_column("organizations", sa.Column("workspace_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=True))
    op.create_index(
        "uq_organizations_workspace_id",
        "organizations",
        ["workspace_id"],
        unique=True,
        postgresql_where=sa.text("workspace_id IS NOT NULL"),
    )
    op.add_column("projects", sa.Column("plane_project_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=True))
    op.create_index(
        "uq_projects_plane_project_id",
        "projects",
        ["plane_project_id"],
        unique=True,
        postgresql_where=sa.text("plane_project_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_projects_plane_project_id", table_name="projects")
    op.drop_column("projects", "plane_project_id")
    op.drop_index("uq_organizations_workspace_id", table_name="organizations")
    op.drop_column("organizations", "workspace_id")
```
- [ ] Verify `alembic downgrade 048` then `upgrade head` both run. Full suite, lint, commit:
  `feat(handshake): workspace_id and plane_project_id join keys (049)`

---

### Task E2 (engine): membership reads the workspace DB

**Files:**
- Create: `services/mcp-server/app/workspace_db.py` (mirror `directory.py`'s lazy pooled engine, env var `WORKSPACE_DATABASE_URL`)
- Modify: `services/shared/shared/membership.py`
- Modify: `services/mcp-server/app/identity.py` (`resolve_active_org` borrows the workspace engine instead of the Supabase one)
- Modify: `services/shared/tests/test_membership_resolution.py`, `services/mcp-server/tests/test_node_selection.py`
- Modify: `services/shared/tests/test_personal_node.py` only if signatures require it

**Interfaces:**
- Produces: `reachable_orgs(workspace_db, tsdb, *, user_id) -> list[dict]` with keys `org_id | workspace_id | slug | is_personal`; `may_reach(workspace_db, tsdb, *, user_id, org_id) -> bool`. (Parameter renamed from `supabase`; call sites in tests updated.)

Steps:

- [ ] Rewrite the external-side SQL:
```python
#: Workspace DB side: which living workspaces is this user an active member of?
_WORKSPACES_SQL = text(
    """
    SELECT w.id AS workspace_id,
           w.slug AS slug,
           w.is_personal AS is_personal
    FROM workspace_members m
    JOIN workspaces w ON w.id = m.workspace_id
    WHERE m.member_id = :user_id
      AND m.is_active = true
      AND m.deleted_at IS NULL
      AND w.deleted_at IS NULL
    """
)

#: Engine side: which of those workspaces have an org row? Joined by the
#: immutable id -- slug is the address, not the identity.
_ORGS_SQL = text(
    """
    SELECT org_id, workspace_id
    FROM organizations
    WHERE workspace_id = ANY(:workspace_ids)
    """
)
```
  Join in Python as before; result dicts carry `org_id`, `workspace_id`, `slug`, `is_personal`; sort personal-first then slug. A workspace with no org row is omitted here (Task E3's ensure closes that gap at selection time).
- [ ] Keep/extend the recording-connection guard test: the workspace connection is never asked about `organizations`; the tsdb connection is never asked about `workspaces` or `workspace_members`.
- [ ] Fixture change: the local fixture now creates `workspaces(id uuid, slug text, is_personal boolean, deleted_at timestamptz)` and `workspace_members(workspace_id uuid, member_id uuid, is_active boolean, deleted_at timestamptz)` instead of the accounts tables, and `organizations` rows carry `workspace_id`.
- [ ] Kill-switch tests (the ones that matter):
  - `is_active = false` → workspace gone from `reachable_orgs` on the next call
  - `deleted_at` set on the workspace → gone
  - `deleted_at` set on the membership row → gone
- [ ] `resolve_active_org`: swap `directory.supabase_engine()` for `workspace_db.workspace_engine()`; unset env still fails closed to 404 on selection and key-org default without one. Update `test_node_selection.py`'s fixtures to the new tables.
- [ ] Remove nothing else yet: `oauth.py` and `directory.py` keep working until their retirement task — this task only moves membership resolution.
- [ ] Mutation check: force `may_reach` → `True`; the non-member/unknown/blank tests must fail. Full suite (do not exceed 25 pre-existing failures), lint, commit:
  `feat(handshake): membership resolves against the workspace DB`

---

### Task E3 (engine): the ensure — a selected workspace gets its org row

**Files:**
- Modify: `services/shared/shared/membership.py` (add `ensure_org_for_workspace`)
- Modify: `services/mcp-server/app/identity.py` (`resolve_active_org` provisions on first selection)
- Create: `services/shared/tests/test_workspace_org_ensure.py`

**Interfaces:**
- Produces: `ensure_org_for_workspace(tsdb, *, workspace_id: str, slug: str, is_personal: bool, owner_user_id: Optional[str]) -> str` returning the `org_id`.

Rules (each one is a test):

- [ ] **Already linked** (`workspace_id` matches) → return that `org_id`, touch nothing. Idempotent.
- [ ] **Legacy adoption**: an org whose `account_slug` equals the slug and whose `workspace_id` IS NULL gets stamped with the workspace id (this is how every existing MoonForge-era org joins the handshake without data migration). The stamp's `UPDATE ... WHERE workspace_id IS NULL` guard makes a concurrent claim lose rather than overwrite.
- [ ] **Slug taken by a different workspace** (org has a different non-NULL `workspace_id`) → `ValueError`. Never re-point a link; that is the team-org-hijack lesson from `6ad63ae` in a new coat.
- [ ] **Nothing exists** → insert `org_id = f"ws_{workspace_id}"`, `account_slug = slug`, `name = slug`.
- [ ] **Personal mirroring**: when `is_personal` and `owner_user_id` is set, set `personal_owner_user_id = owner_user_id` — respecting 048's partial unique index; if the owner already has a different personal org, `ValueError` rather than silently minting a second.
- [ ] Wire into `resolve_active_org`: when the header names a workspace the caller can reach (per `reachable_orgs` from the workspace DB) but no org row exists, call the ensure and return the fresh org_id. Selection is the provisioning moment — a read (`reachable_orgs`) stays a pure read.
- [ ] Mutation check on the never-re-point rule: disable the different-workspace guard, its test must fail. Full suite, lint, commit:
  `feat(handshake): selecting a workspace provisions its engine org`

---

## Out of scope (so the next plan does not assume otherwise)

- Retiring `oauth.py` / `directory.py` / Supabase env vars — after this lands and soaks.
- The Plane→engine *push* direction (webhook/signal on workspace create). Lazy ensure-on-selection covers correctness; push is an optimization.
- `projects.plane_project_id` is created but not yet written — project-level linking is Track B.
- Onboarding UX (defaulting the personal slug from the user's handle) — UI work, later by the user's own call.

## Self-review

Placeholders: none — every step names files, code, and commands. Type consistency: `reachable_orgs`/`may_reach` renamed first param `workspace_db` used consistently in E2/E3; `ensure_org_for_workspace` signature identical in E3's two mentions. Coverage: personal-workspace rule (P1), join keys (E1), identity-source swap + kill-switches (E2), 1:1 provisioning + adoption (E3) — matches everything agreed in conversation, including slug-as-username (no schema needed; P1's flag + existing unique slug carry it).
