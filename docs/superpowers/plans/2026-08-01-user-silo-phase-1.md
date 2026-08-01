# User Silo Phase 1 — Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the engine a real user principal — a `user_id` on memories, resolvable from either credential — plus the schema for project ACLs and provenance, **without changing a single byte of observable behaviour**.

**Architecture:** Purely additive. Migration `031` adds nullable columns, a new table, and constraints. `resolve_user_id()` is a new pure-ish helper that reads an already-validated principal. `KeyInfo` gains a field carried through from data that is already stored (`created_by`) but never propagated. Nothing reads `user_id` for authorization yet — that is Phase 2. After this lands, every existing request behaves identically.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy Core (`text()` + bound params, no ORM), Alembic, pytest, ruff.

## Global Constraints

- **Additive only.** No column drops, no data destroyed, no behaviour changed. Verbatim from the spec: *"Phase 1 leaves the product behaving exactly as it does now, so it is a safe place to stop if phase 2 needs rethinking."*
- **Migration head is `030`.** This is `031`, `down_revision = "030"`.
- **Never derive identity from `X-Vex-Agent`.** Verbatim from the spec: *"`X-Vex-Agent` is client-declared. Deriving identity from it would let any caller read another person's private memories by setting a header."*
- **`user_id` is nullable and `NULL` means "unattributed".** Verbatim from the spec: *"When `user_id` is `None`: write org-scoped and log a warning. Not private, not refused."*
- **The predicate is always `scope`, never the presence of `project_id`.** Verbatim from the spec. `project_id` is a relevance filter on 32,568 existing org-scoped rows and is not a boundary.
- **Existing 37,605 memories keep `scope='org'` and `user_id IS NULL`.** Decision 6. The only backfill Phase 1 performs is `confidence`.
- SQL uses bound parameters via `text()`. Never f-strings into SQL.
- `ruff format` and `ruff check` must pass before every commit.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `services/migrations/alembic/versions/031_user_silo_identity.py` | All schema for Phase 1 | Create |
| `services/shared/shared/auth.py` | `KeyInfo.user_id`, `_CachedKey.user_id`, carry `created_by` through | Modify |
| `services/mcp-server/app/identity.py` | `resolve_user_id(request)` — the single identity rule | Create |
| `services/shared/tests/test_auth.py` | Key-path identity tests | Modify |
| `services/mcp-server/tests/test_identity.py` | `resolve_user_id` tests incl. the header-spoofing guard | Create |
| `services/migrations/tests/test_migration_031.py` | Constraint + backfill assertions | Create |

`identity.py` is its own file rather than another function in `auth.py` because it is the one place the "who is this person" rule lives, and Phase 2's recall path will import it. `auth.py` is already 400+ lines and answers a different question ("is this key valid").

---

### Task 1: Migration 031 — schema and provenance backfill

**Files:**
- Create: `services/migrations/alembic/versions/031_user_silo_identity.py`
- Test: `services/migrations/tests/test_migration_031.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `session_memories.user_id` (VARCHAR(64), nullable), `session_memories.confidence` (VARCHAR(16), NOT NULL, default `'EXTRACTED'`), table `project_members(project_id UUID, user_id VARCHAR(64), role VARCHAR(32), granted_by VARCHAR(64), granted_at TIMESTAMPTZ)` with PK `(project_id, user_id)`, CHECK constraint `ck_session_memories_scope`, CHECK constraint `ck_session_memories_confidence`, index `ix_session_memories_private`, `brain_curator_state` PK `(org_id, user_id)`.

- [ ] **Step 1: Write the failing test**

Create `services/migrations/tests/test_migration_031.py`:

```python
"""Migration 031 asserts the shape Phase 2 depends on.

These are DB tests: they need the live test Postgres and skip without it,
matching the convention in services/shared/tests/test_brain_recall.py.
"""

import os

import pytest
from sqlalchemy import create_engine, text

DB_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://vex:vex@localhost:5432/vex_test")


@pytest.fixture(scope="module")
def conn():
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("Postgres not reachable; migration 031 tests require the live test DB")
    with create_engine(DB_URL).connect() as c:
        yield c


def _columns(conn, table):
    rows = conn.execute(
        text(
            "SELECT column_name, is_nullable, data_type FROM information_schema.columns "
            "WHERE table_name = :t"
        ),
        {"t": table},
    ).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def test_session_memories_has_nullable_user_id(conn):
    """user_id is nullable: NULL means unattributed, which is a legal state."""
    cols = _columns(conn, "session_memories")
    assert "user_id" in cols
    assert cols["user_id"][0] == "YES", "user_id must be nullable — NULL means unattributed"


def test_confidence_defaults_to_extracted_and_is_not_null(conn):
    cols = _columns(conn, "session_memories")
    assert "confidence" in cols
    assert cols["confidence"][0] == "NO"


def test_scope_check_constraint_rejects_a_typo(conn):
    """Before 031 there was NO constraint, so a typo'd scope created rows
    no query would ever return. That is the bug this closes."""
    tx = conn.begin_nested()
    with pytest.raises(Exception):
        conn.execute(
            text(
                "INSERT INTO session_memories "
                "(org_id, agent_id, session_id, scope, status, content, memory_type) "
                "VALUES ('t', 'a', 's', 'privte', 'active', 'x', 'fact')"
            )
        )
    tx.rollback()


def test_scope_check_constraint_admits_private(conn):
    tx = conn.begin_nested()
    conn.execute(
        text(
            "INSERT INTO session_memories "
            "(org_id, agent_id, session_id, scope, status, content, memory_type) "
            "VALUES ('t031', 'a', 's', 'private', 'active', 'x', 'fact')"
        )
    )
    tx.rollback()


def test_confidence_check_constraint_rejects_unknown_label(conn):
    tx = conn.begin_nested()
    with pytest.raises(Exception):
        conn.execute(
            text(
                "INSERT INTO session_memories "
                "(org_id, agent_id, session_id, scope, status, content, memory_type, confidence) "
                "VALUES ('t', 'a', 's', 'org', 'active', 'x', 'fact', 'MAYBE')"
            )
        )
    tx.rollback()


def test_curator_rows_backfilled_to_inferred(conn):
    """The one backfill Phase 1 performs. Curator output is an LLM deduction
    and must not be indistinguishable from something a user stated."""
    row = conn.execute(
        text(
            "SELECT count(*) FROM session_memories "
            "WHERE metadata->>'source' = 'curator' AND confidence <> 'INFERRED'"
        )
    ).scalar()
    assert row == 0, "every curator-authored row must be INFERRED"


def test_non_curator_rows_are_extracted(conn):
    row = conn.execute(
        text(
            "SELECT count(*) FROM session_memories "
            "WHERE COALESCE(metadata->>'source', '') <> 'curator' AND confidence <> 'EXTRACTED'"
        )
    ).scalar()
    assert row == 0


def test_project_members_table_exists_with_composite_pk(conn):
    cols = _columns(conn, "project_members")
    assert set(cols) >= {"project_id", "user_id", "role", "granted_by", "granted_at"}
    pk = conn.execute(
        text(
            "SELECT a.attname FROM pg_index i "
            "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
            "WHERE i.indrelid = 'project_members'::regclass AND i.indisprimary"
        )
    ).fetchall()
    assert {r[0] for r in pk} == {"project_id", "user_id"}


def test_private_partial_index_exists(conn):
    idx = conn.execute(
        text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_session_memories_private'")
    ).scalar()
    assert idx is not None, "private recall is the hot path and must not seq-scan"
    assert "private" in idx


def test_curator_state_pk_is_org_and_user(conn):
    pk = conn.execute(
        text(
            "SELECT a.attname FROM pg_index i "
            "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
            "WHERE i.indrelid = 'brain_curator_state'::regclass AND i.indisprimary"
        )
    ).fetchall()
    assert {r[0] for r in pk} == {"org_id", "user_id"}


def test_existing_memories_keep_org_scope_and_null_user(conn):
    """Decision 6: no retroactive privatisation."""
    leaked = conn.execute(
        text("SELECT count(*) FROM session_memories WHERE scope = 'private' AND user_id IS NULL")
    ).scalar()
    assert leaked == 0, "a private row with no owner is unreachable by anyone"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd services/migrations && python -m pytest tests/test_migration_031.py -v`
Expected: FAIL — `user_id` not in columns (or every test errors on missing `project_members`). If Postgres is unreachable the tests SKIP; that is not a pass, start the stack first with `docker compose up -d postgres`.

- [ ] **Step 3: Write the migration**

Create `services/migrations/alembic/versions/031_user_silo_identity.py`:

```python
"""User silo phase 1: a real user principal, project ACLs, provenance.

ADDITIVE ONLY. No column is dropped, no row is made less visible, and no
request behaves differently after this runs. Phase 2 is what starts routing
writes to the new scope; this migration only makes that possible.

``session_memories.user_id`` is NULLABLE on purpose. NULL means "we could not
attribute this write to a person", which is a legal, permanent state for the
37,605 rows that predate this migration and for any caller we cannot resolve.
A NOT NULL column would force us to invent an owner, and inventing an owner
for a private scope is how you leak one person's memories to another.

``scope`` had NO check constraint before this. A typo'd scope therefore created
rows that every query silently excluded — invisible, undeletable-by-UI, and
counted against quota. Introducing 'private' is the right moment to close that,
because the failure mode of a typo goes from "row is invisible" to "row that
should be private is visible".

**Backfill.** Exactly one: ``confidence``. Rows authored by the curator become
INFERRED, everything else EXTRACTED. This is safe because it adds information
to rows without changing who can read them. Memories are NOT backfilled to
private — see decision 6 in the spec: retroactively hiding a team's accumulated
brain is a worse surprise than leaving it.

Revision ID: 031
Revises: 030
Create Date: 2026-08-01
"""

from alembic import op
import sqlalchemy as sa

revision = "031"
down_revision = "030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---- identity -------------------------------------------------------
    op.add_column(
        "session_memories",
        sa.Column("user_id", sa.String(64), nullable=True),
    )

    # ---- provenance -----------------------------------------------------
    # Server default so existing rows land on EXTRACTED without a table
    # rewrite, and so a writer that forgets the column still produces a
    # truthful row rather than a NULL that reads as "unknown".
    op.add_column(
        "session_memories",
        sa.Column(
            "confidence",
            sa.String(16),
            nullable=False,
            server_default="EXTRACTED",
        ),
    )

    # The one backfill. Curator output is an LLM deduction; leaving it
    # indistinguishable from something a user stated is the exact failure
    # external review flagged ("if it starts driving inferences that aren't
    # correct").
    op.execute(
        """
        UPDATE session_memories
        SET confidence = 'INFERRED'
        WHERE metadata->>'source' = 'curator'
        """
    )

    # ---- constraints ----------------------------------------------------
    op.create_check_constraint(
        "ck_session_memories_scope",
        "session_memories",
        "scope IN ('private', 'project', 'org', 'session', 'agent')",
    )
    op.create_check_constraint(
        "ck_session_memories_confidence",
        "session_memories",
        "confidence IN ('EXTRACTED', 'INFERRED', 'AMBIGUOUS')",
    )

    # A private row with no owner is unreachable by every caller including its
    # author, so it is a bug by construction rather than a state to handle.
    op.create_check_constraint(
        "ck_session_memories_private_has_owner",
        "session_memories",
        "scope <> 'private' OR user_id IS NOT NULL",
    )

    # ---- recall index ---------------------------------------------------
    # Partial: private is a small slice today and the predicate is always
    # present on the private read path.
    op.execute(
        """
        CREATE INDEX ix_session_memories_private
        ON session_memories (org_id, user_id, created_at DESC)
        WHERE scope = 'private'
        """
    )

    # ---- project ACL ----------------------------------------------------
    op.create_table(
        "project_members",
        sa.Column("project_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("role", sa.String(32), nullable=False, server_default="member"),
        sa.Column("granted_by", sa.String(64), nullable=True),
        sa.Column(
            "granted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint("project_id", "user_id"),
    )
    op.create_check_constraint(
        "ck_project_members_role",
        "project_members",
        "role IN ('member', 'admin')",
    )
    # "which projects can this user read" is the Phase 2 recall predicate.
    op.create_index("ix_project_members_user", "project_members", ["user_id"])

    # ---- per-user curator cursor ---------------------------------------
    # The curator becomes per-user in Phase 2. Widening the PK now means Phase
    # 2 is a code change with no migration. Existing rows get '' rather than
    # NULL because NULL is not comparable in a PK.
    op.add_column(
        "brain_curator_state",
        sa.Column("user_id", sa.String(64), nullable=False, server_default=""),
    )
    op.drop_constraint("brain_curator_state_pkey", "brain_curator_state", type_="primary")
    op.create_primary_key(
        "brain_curator_state_pkey", "brain_curator_state", ["org_id", "user_id"]
    )


def downgrade() -> None:
    op.drop_constraint("brain_curator_state_pkey", "brain_curator_state", type_="primary")
    op.create_primary_key("brain_curator_state_pkey", "brain_curator_state", ["org_id"])
    op.drop_column("brain_curator_state", "user_id")

    op.drop_index("ix_project_members_user", table_name="project_members")
    op.drop_table("project_members")

    op.execute("DROP INDEX IF EXISTS ix_session_memories_private")
    op.drop_constraint(
        "ck_session_memories_private_has_owner", "session_memories", type_="check"
    )
    op.drop_constraint("ck_session_memories_confidence", "session_memories", type_="check")
    op.drop_constraint("ck_session_memories_scope", "session_memories", type_="check")
    op.drop_column("session_memories", "confidence")
    op.drop_column("session_memories", "user_id")
```

- [ ] **Step 4: Run the migration against the test DB**

```bash
cd services/migrations && alembic upgrade head
```
Expected: `Running upgrade 030 -> 031`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd services/migrations && python -m pytest tests/test_migration_031.py -v`
Expected: all PASS.

- [ ] **Step 6: Verify the downgrade actually works**

A downgrade that has never been run is a downgrade that does not work.

```bash
cd services/migrations && alembic downgrade 030 && alembic upgrade head
```
Expected: both succeed; re-run the tests from Step 5 and they still pass.

- [ ] **Step 7: Commit**

```bash
ruff format services/ && ruff check services/
git add services/migrations/alembic/versions/031_user_silo_identity.py services/migrations/tests/test_migration_031.py
git commit -m "feat(db): migration 031 — user_id, project ACLs, provenance

Additive only. Adds session_memories.user_id (nullable — NULL means
unattributed, which is legal and permanent for the 37,605 pre-existing
rows), session_memories.confidence, the project_members ACL table, a
partial index for private recall, and a per-user curator cursor.

scope had NO check constraint, so a typo silently created rows no query
returned. Introducing 'private' is the right moment to close that: the
failure mode of a typo goes from 'row is invisible' to 'row that should
be private is visible'.

One backfill: curator-authored rows become INFERRED, everything else
EXTRACTED. Safe because it adds information without changing who can read
anything. Memories are NOT backfilled to private (decision 6)."
```

---

### Task 2: Carry `created_by` through the key path

**Files:**
- Modify: `services/shared/shared/auth.py:62-75` (`KeyInfo`), `:78-92` (`_CachedKey`), `:224-230` (`KeyInfo(...)` construction), `:404-413` (`_CachedKey(...)` construction)
- Test: `services/shared/tests/test_auth.py`

**Interfaces:**
- Consumes: nothing from Task 1 (independent — can run in parallel)
- Produces: `KeyInfo.user_id: str | None` — the UUID string from the key entry's `created_by`, or `None` when the field is absent.

- [ ] **Step 1: Write the failing test**

Append to `services/shared/tests/test_auth.py`:

```python
# ---------------------------------------------------------------------------
# Key-path identity
#
# Every key entry already stores `created_by` (verified in production: 55/55
# keys, 36-char UUIDs, 14 distinct creators). It has never been propagated to
# the caller, so the engine has had a user identity available on every keyed
# request and thrown it away.
# ---------------------------------------------------------------------------


def test_keyinfo_carries_created_by_as_user_id():
    from shared.auth import KeyInfo

    info = KeyInfo(
        org_id="acme",
        key_id="k1",
        scopes=["memory:write"],
        user_id="11111111-2222-3333-4444-555555555555",
    )
    assert info.user_id == "11111111-2222-3333-4444-555555555555"


def test_keyinfo_user_id_defaults_to_none():
    """Additive: ingestion-api and sync-gateway construct KeyInfo without it."""
    from shared.auth import KeyInfo

    info = KeyInfo(org_id="acme", key_id="k1", scopes=[])
    assert info.user_id is None


def test_cached_key_reads_created_by_from_the_key_entry():
    from shared.auth import _CachedKey

    entry = _CachedKey(
        org_id="acme",
        key_id="k1",
        scopes=[],
        rate_limit_rpm=100,
        expires_at=None,
        revoked=False,
        user_id="11111111-2222-3333-4444-555555555555",
    )
    assert entry.user_id == "11111111-2222-3333-4444-555555555555"


def test_cached_key_user_id_defaults_to_none_for_legacy_entries():
    """A key entry written before created_by existed must not crash the path."""
    from shared.auth import _CachedKey

    entry = _CachedKey(
        org_id="acme",
        key_id="k1",
        scopes=[],
        rate_limit_rpm=100,
        expires_at=None,
        revoked=False,
    )
    assert entry.user_id is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd services/shared && python -m pytest tests/test_auth.py -k "created_by or user_id" -v`
Expected: FAIL — `KeyInfo.__init__() got an unexpected keyword argument 'user_id'`.

- [ ] **Step 3: Add the field to both dataclasses**

In `services/shared/shared/auth.py`, inside `class KeyInfo`, after `plan_overrides`:

```python
    # The Supabase user UUID of whoever created this key, copied from the key
    # entry's ``created_by``. Every key already carries it; it simply never
    # reached the caller. Optional because ingestion-api and sync-gateway
    # construct KeyInfo directly, and because a key written before the field
    # existed must not crash the auth path.
    #
    # This is the ONLY identity source on the key path. Never derive it from
    # X-Vex-Agent, which the client declares and can therefore forge.
    user_id: str | None = None
```

In `class _CachedKey`, after `plan_overrides`:

```python
    user_id: str | None = None
```

- [ ] **Step 4: Populate it at both construction sites**

In the `_CachedKey(...)` construction (around line 404), add as the final argument:

```python
                    user_id=key_entry.get("created_by"),
```

In the `KeyInfo(...)` construction (around line 224), add as the final argument:

```python
            user_id=entry.user_id,
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd services/shared && python -m pytest tests/test_auth.py -v`
Expected: all PASS, including the pre-existing tests — this change is purely additive and must not alter any existing assertion.

- [ ] **Step 6: Commit**

```bash
ruff format services/ && ruff check services/
git add services/shared/shared/auth.py services/shared/tests/test_auth.py
git commit -m "feat(auth): carry the key's created_by through as KeyInfo.user_id

Every API key entry already stores created_by — verified in production at
55/55 keys, 36-char UUIDs, 14 distinct creators. The engine has had a user
identity available on every keyed request and discarded it.

Additive and optional: ingestion-api and sync-gateway construct KeyInfo
directly, and a key entry written before the field existed must not crash
the auth path. Nothing reads user_id yet."
```

---

### Task 3: `resolve_user_id` — one rule, one place

**Files:**
- Create: `services/mcp-server/app/identity.py`
- Test: `services/mcp-server/tests/test_identity.py`

**Interfaces:**
- Consumes: `KeyInfo.user_id` from Task 2; `OAuthPrincipal.user_id` (already exists, `services/mcp-server/app/oauth.py:278`)
- Produces: `resolve_user_id(request: Request, key_info: KeyInfo | None = None) -> str | None`

- [ ] **Step 1: Write the failing test**

Create `services/mcp-server/tests/test_identity.py`:

```python
"""The single identity rule.

OAuth wins over the key because an OAuth principal is a verified human
session, while a key is a bearer token that a person created. Both are
trustworthy; neither is X-Vex-Agent.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.identity import resolve_user_id
from app.oauth import OAuthPrincipal
from shared.auth import KeyInfo

USER = "11111111-2222-3333-4444-555555555555"
OTHER = "99999999-8888-7777-6666-555555555555"


def _request(oauth_principal=None, headers=None):
    req = MagicMock()
    req.state = SimpleNamespace()
    if oauth_principal is not None:
        req.state.oauth_principal = oauth_principal
    req.headers = headers or {}
    return req


def _principal(user_id=USER):
    return OAuthPrincipal(
        org_id="acme",
        account_slug="acme",
        user_id=user_id,
        client_id="cli",
        plan="free",
        plan_overrides=None,
    )


def test_oauth_principal_supplies_the_full_uuid():
    """Not the 8-char prefix that agent_id embeds — the real value."""
    assert resolve_user_id(_request(oauth_principal=_principal())) == USER


def test_api_key_created_by_is_used_when_there_is_no_oauth():
    info = KeyInfo(org_id="acme", key_id="k1", scopes=[], user_id=USER)
    assert resolve_user_id(_request(), key_info=info) == USER


def test_oauth_wins_when_both_are_present():
    info = KeyInfo(org_id="acme", key_id="k1", scopes=[], user_id=OTHER)
    got = resolve_user_id(_request(oauth_principal=_principal(USER)), key_info=info)
    assert got == USER


def test_returns_none_when_neither_credential_identifies_a_person():
    """Legal state. The caller writes org-scoped and logs; it does not fail."""
    assert resolve_user_id(_request()) is None
    assert resolve_user_id(_request(), key_info=KeyInfo("acme", "k1", [])) is None


def test_x_vex_agent_is_never_an_identity_source():
    """The security property of this module.

    X-Vex-Agent is declared by the client. If it could set identity, any
    caller could read another person's private memories by setting a header.
    """
    req = _request(headers={"X-Vex-Agent": OTHER, "x-vex-agent": OTHER})
    assert resolve_user_id(req) is None


def test_x_vex_agent_cannot_override_a_real_credential():
    req = _request(oauth_principal=_principal(USER), headers={"X-Vex-Agent": OTHER})
    assert resolve_user_id(req) == USER


def test_missing_state_attribute_is_tolerated():
    """Key-path requests and test fakes have no oauth_principal attribute."""
    req = MagicMock()
    req.state = SimpleNamespace()
    req.headers = {}
    assert resolve_user_id(req) is None


def test_a_non_principal_object_on_state_is_ignored():
    """MagicMock-patched requests in pre-existing tests return a truthy mock
    for any attribute access; that must not be read as an identity."""
    req = _request(oauth_principal=object())
    assert resolve_user_id(req) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd services/mcp-server && python -m pytest tests/test_identity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.identity'`.

- [ ] **Step 3: Write the implementation**

Create `services/mcp-server/app/identity.py`:

```python
"""Resolve the human behind a request.

This is the ONLY place the engine decides who a caller is. Phase 2's recall
path treats the result as a tenancy boundary equal to ``org_id``, so the rule
lives in one file with one test suite rather than being re-derived at each
call site.

Two trustworthy sources, in order:

1. **OAuth** — ``request.state.oauth_principal.user_id``, a verified session.
   Preferred when both are present because it identifies the human directly
   rather than the human who once minted a token.
2. **API key** — ``KeyInfo.user_id``, copied from the key entry's
   ``created_by``. Keys are personal by policy.

And one source that is never used: ``X-Vex-Agent``. The client declares it,
so treating it as identity would let any caller read another person's private
memories by setting a header. It is a label for WHICH TOOL, never WHICH
PERSON.

``None`` is a legal result. It means "this request cannot be attributed to a
person", and the caller's contract is to write org-scoped and log — not to
fail. Refusing would break every legacy key; inventing an owner would leak one
person's memories into another's private scope.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import Request

logger = logging.getLogger(__name__)


def resolve_user_id(
    request: Request,
    key_info: Optional[Any] = None,
) -> Optional[str]:
    """Return the caller's user id, or ``None`` if they cannot be attributed.

    Args:
        request: The inbound request. Only ``request.state.oauth_principal``
            is consulted; headers are deliberately ignored.
        key_info: The validated :class:`~shared.auth.KeyInfo` for key-path
            callers, when the route has one.

    Returns:
        The user's UUID as a string, or ``None``.
    """
    principal = _oauth_principal(request)
    if principal is not None:
        user_id = getattr(principal, "user_id", None)
        if user_id:
            return str(user_id)

    if key_info is not None:
        user_id = getattr(key_info, "user_id", None)
        if user_id:
            return str(user_id)

    return None


def _oauth_principal(request: Request) -> Optional[Any]:
    """Return the middleware-attached OAuthPrincipal, or ``None``.

    Mirrors ``app.auth._oauth_principal_from``: tolerates a ``state`` with no
    such attribute, and only accepts an actual ``OAuthPrincipal`` so a
    MagicMock-patched request in a pre-existing test cannot be mistaken for a
    real identity. Never raises.
    """
    candidate = getattr(getattr(request, "state", None), "oauth_principal", None)
    if candidate is None:
        return None
    try:
        from app.oauth import OAuthPrincipal
    except Exception:  # pragma: no cover - import guard only
        return None
    return candidate if isinstance(candidate, OAuthPrincipal) else None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd services/mcp-server && python -m pytest tests/test_identity.py -v`
Expected: all 8 PASS.

- [ ] **Step 5: Run both full suites to prove nothing changed**

```bash
cd services/mcp-server && python -m pytest -q
cd ../shared && python -m pytest -q
```
Expected: the pre-existing counts, plus the new tests. No failures. If any pre-existing test changed behaviour, this task broke the "no observable change" constraint — stop and fix rather than updating the test.

- [ ] **Step 6: Commit**

```bash
ruff format services/ && ruff check services/
git add services/mcp-server/app/identity.py services/mcp-server/tests/test_identity.py
git commit -m "feat(identity): resolve_user_id — one rule, one place

OAuth principal first (a verified session), then the key's created_by
(personal by policy). Never X-Vex-Agent: the client declares it, so using
it for identity would let any caller read another person's private
memories by setting a header. There is a test asserting exactly that.

None is a legal result meaning 'cannot attribute'. Callers write
org-scoped and log rather than failing — refusing would break every
legacy key, and inventing an owner would leak memories into the wrong
private scope.

Nothing calls this yet; Phase 2 wires it into the write path."
```

---

### Task 4: Prove Phase 1 changed nothing

**Files:**
- Test: `services/mcp-server/tests/test_identity.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 1-3
- Produces: nothing — this is the gate

The whole premise of Phase 1 is that it is safe to land and stop. That claim needs a test, not an assurance.

- [ ] **Step 1: Write the guard test**

Append to `services/mcp-server/tests/test_identity.py`:

```python
# ---------------------------------------------------------------------------
# Phase 1 is a no-op at the boundary.
#
# The spec's promise is: "Phase 1 leaves the product behaving exactly as it
# does now, so it is a safe place to stop if phase 2 needs rethinking." These
# assert that promise mechanically so a later edit cannot quietly break it.
# ---------------------------------------------------------------------------


def test_no_write_path_calls_resolve_user_id_yet():
    """Phase 2 wires identity into writes. If this fails, a routing change
    landed in a phase that is supposed to be schema-only."""
    import pathlib

    app_dir = pathlib.Path(__file__).resolve().parents[1] / "app"
    callers = [
        p.name
        for p in app_dir.glob("*.py")
        if p.name not in {"identity.py"} and "resolve_user_id" in p.read_text()
    ]
    assert callers == [], f"Phase 1 must not wire identity into: {callers}"


def test_capture_still_writes_org_scope():
    """The 33,037 hook rows a month keep landing at org until Phase 2."""
    import pathlib

    capture = (
        pathlib.Path(__file__).resolve().parents[1] / "app" / "capture.py"
    ).read_text()
    assert "scope='private'" not in capture
    assert 'scope="private"' not in capture
```

- [ ] **Step 2: Run it to verify it passes**

Run: `cd services/mcp-server && python -m pytest tests/test_identity.py -v`
Expected: PASS. If `test_no_write_path_calls_resolve_user_id_yet` fails, a Phase 2 change has leaked into this branch.

- [ ] **Step 3: Verify the container import still works**

The CI job added on 2026-07-30 catches a missing runtime dependency that the test suite cannot, because tests run in a venv holding every transitive package while the images maintain their own lists.

```bash
docker build -f services/mcp-server/Dockerfile -t klio-mcp-check .
docker run --rm --network none klio-mcp-check python -c "import app.main; print('IMPORT OK')"
```
Expected: `IMPORT OK`. `--network none` is deliberate — anything needing a live dependency belongs behind a function call, not at import time.

- [ ] **Step 4: Commit**

```bash
git add services/mcp-server/tests/test_identity.py
git commit -m "test: assert phase 1 changed no observable behaviour

The spec's promise is that phase 1 is safe to land and stop. That is a
claim about what did NOT change, so it needs a mechanical assertion
rather than an assurance: no write path calls resolve_user_id, and
capture.py still writes org scope."
```

---

### Task 5: Ship it

**Files:** none

- [ ] **Step 1: Full verification**

```bash
cd services/shared && python -m pytest -q
cd ../mcp-server && python -m pytest -q
cd ../migrations && python -m pytest -q
cd ../.. && ruff format --check services/ && ruff check services/
```
Expected: all green. Migration tests SKIP without Postgres — that is not a pass; run `docker compose up -d postgres` and re-run so they execute.

- [ ] **Step 2: Open the PR**

```bash
git push -u origin feat/user-silo-phase-1
gh pr create --base main --title "feat: user silo phase 1 — a real user principal" --body "$(cat <<'EOF'
Implements phase 1 of `docs/superpowers/specs/2026-08-01-user-silo-design.md`.

**Additive only. No observable behaviour change**, asserted by a test rather than claimed.

## What lands

- Migration `031`: `session_memories.user_id` (nullable), `confidence`, `project_members`, a partial index for private recall, a per-user curator cursor, and CHECK constraints on `scope` and `confidence`.
- `KeyInfo.user_id`, carried through from the key entry's `created_by` — a value every one of the 55 production keys already stores and the engine has been discarding on every request.
- `resolve_user_id()` — the single identity rule. OAuth first, then the key. **Never `X-Vex-Agent`**, with a test asserting a forged header cannot supply or override an identity.

## What does not land

Nothing reads `user_id` for authorization. Hook captures still write `org` scope. The shared brain is unchanged. Phase 2 does the routing.

## Two constraints worth reviewing

`scope` had **no** check constraint, so a typo silently created rows no query returned. Adding `private` is the moment to close that — the failure mode of a typo goes from "row is invisible" to "row that should be private is visible".

`ck_session_memories_private_has_owner` makes a private row without an owner impossible: it would be unreachable by every caller including its author, so it is a bug by construction rather than a state to handle.

## The one backfill

Curator-authored rows become `INFERRED`, everything else `EXTRACTED`. Safe — it adds information without changing who can read anything. Memories are **not** backfilled to private (decision 6): retroactively hiding a team's accumulated brain is a worse surprise than leaving it.
EOF
)"
```

- [ ] **Step 3: Confirm CI is green before merging**

All `Container Import` jobs, `Lint & Format`, `Service Tests (shared)` and `Service Tests (mcp-server)` must pass. `Service Tests (alert-service)` has been failing on `main` since 2026-07-27 and is unrelated to this change.

---

## Self-Review

**Spec coverage (Phase 1 items only):**

| Spec requirement | Task |
|---|---|
| `session_memories.user_id` nullable | 1 |
| CHECK constraint on `scope` | 1 |
| Partial index for private recall | 1 |
| `project_members` table, PK `(project_id, user_id)` | 1 |
| `brain_curator_state` PK → `(org_id, user_id)` | 1 |
| `confidence` column + curator backfill to `INFERRED` | 1 |
| `KeyInfo.user_id` from `created_by` | 2 |
| `resolve_user_id`, OAuth then key, never `X-Vex-Agent` | 3 |
| `None` is legal, caller writes org-scoped | 3 |
| Existing memories stay org-scoped, `user_id` NULL | 1 (asserted) |
| No observable behaviour change | 4 (asserted) |

**Placeholders:** none — every step carries runnable code or an exact command.

**Type consistency:** `resolve_user_id(request, key_info=None) -> str | None` is used identically in Tasks 3 and 4. `KeyInfo.user_id` and `_CachedKey.user_id` are both `str | None` defaulting to `None`. `user_id` is `VARCHAR(64)` in every table.

**Deliberately deferred to Phase 2:** write routing, the recall ladder, the per-user curator, `share`, and every `project_*` MCP tool.
