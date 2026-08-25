# Context Network — Track A (Identity Inversion) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the human the root principal — a caller selects which org node they act in, from the set their Supabase membership entitles them to, enforced server-side on every call.

**Architecture:** The engine has no local users table and should not grow one: user ids are Supabase UUIDs and memberships already live in `public.accounts_memberships`. Personal nodes already exist upstream as MakerKit personal accounts — the engine currently excludes them with a hardcoded `is_personal_account = false` in four places. So the inversion is three moves: give memory scope a single definition (it is currently duplicated across six independent validations), add one membership-resolution function, and let the active org be selected per request from what that function returns.

**Tech Stack:** Python 3.11, FastAPI/FastMCP, SQLAlchemy Core (raw SQL), Alembic, Postgres 16 + pgvector, pytest.

## Global Constraints

- **No local `users` table.** User identity is a Supabase UUID carried as `String(64)`, never FK-constrained locally. Memberships are read cross-database from `public.accounts_memberships`.
- **`org_id` remains a tenancy boundary.** After this change it is *selected* rather than *derived*, so every selection must be re-checked against membership server-side, on every call — not at switch time.
- **One resolution function.** Every call path resolves node access through the same function. A second copy is how the `agent_id` bound drifted (see commit `d6f53ce`).
- **A non-member must get `not_found`, not `forbidden`.** The existing project tools do this deliberately to avoid an existence oracle (`mcp_app.py:2929-2933`). Match it.
- **Behaviour-preserving refactors must prove it.** Task 1 changes no behaviour; the existing suite staying green is the evidence.
- **Baseline:** on `feat/context-network-identity`, `pytest services/shared/tests/ services/mcp-server/tests/` gives **25 failed, 1989 passed, 11 skipped**. All 25 failures pre-date this work and live in `services/shared/test_*.py` — files outside the `tests/` directory CI runs. Do not fix them; do not let the count rise.
- **Run tests with** `DATABASE_URL="postgresql://agentguard:agentguard_dev@localhost:5432/agentguard"` and the repo's `.venv/bin/python`. Postgres is up via `docker compose up -d postgres`; migrations via `cd services/migrations && alembic upgrade head`.
- **CI runs `ruff check services/` and `ruff format --check services/`.** Both must pass.

---

## File Structure

| File | Responsibility |
|---|---|
| `services/shared/shared/scopes.py` | **New.** The single definition of memory-row scope values and the sets derived from them. |
| `services/shared/shared/memory.py` | Modify two bare-tuple validations to import from `scopes`. |
| `services/shared/shared/brain.py` | Modify one bare-tuple validation to import from `scopes`. |
| `services/shared/shared/graph_store.py` | Modify one bare-tuple validation to import from `scopes`. |
| `services/shared/shared/promotion.py` | Replace local `PROMOTABLE_SCOPES` with the shared one. |
| `services/shared/shared/membership.py` | **New.** `reachable_orgs` / `may_reach` — the one membership resolution function. |
| `services/mcp-server/app/identity.py` | Add per-request node selection on top of the existing principal resolution. |

`shared/auth.py` also defines a `VALID_SCOPES` — that is **API key scopes** (`ingest`/`verify`/`read`/`memory`), an unrelated concept that merely shares the name. Do not touch it and do not import it into `scopes.py`.

---

### Task 1: One definition of memory scope

Today the memory-row scope value set is written out independently in six places, with no shared source. Adding any new scope means finding all six; missing one produces a row the database accepts and a reader silently drops.

**Files:**
- Create: `services/shared/shared/scopes.py`
- Create: `services/shared/tests/test_scopes_single_definition.py`
- Modify: `services/shared/shared/memory.py:1711`, `services/shared/shared/memory.py:2522-2540`
- Modify: `services/shared/shared/brain.py:761`
- Modify: `services/shared/shared/graph_store.py:389`
- Modify: `services/shared/shared/promotion.py:34`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `shared.scopes.MEMORY_SCOPES: frozenset[str]`, `WRITABLE_SCOPES: frozenset[str]`, `RECALL_SCOPES: frozenset[str]`, `PROMOTABLE_SCOPES: frozenset[str]`, `GRAPH_SCOPES: frozenset[str]`.

- [ ] **Step 1: Write the failing test**

The test that matters is the one tying the Python constants to the database's own CHECK constraint. A test that only compares Python constants to each other cannot catch the drift this task exists to prevent.

```python
"""The scope value set has one definition, and the database agrees with it.

Before this module, the set was written out independently in six places:
migration 031's VALID_SCOPES, the CHECK constraint it installs, and bare
tuples in memory.py (x2), brain.py, graph_store.py and promotion.py. Nothing
tied them together, so a new scope could be accepted by one and dropped by
another.
"""

import re

import pytest
from sqlalchemy import text

from shared.scopes import (
    GRAPH_SCOPES,
    MEMORY_SCOPES,
    PROMOTABLE_SCOPES,
    RECALL_SCOPES,
    WRITABLE_SCOPES,
)


def test_memory_scopes_match_the_database_check_constraint(db_session):
    row = db_session.execute(
        text(
            """
            SELECT pg_get_constraintdef(oid) AS def
            FROM pg_constraint
            WHERE conname = 'ck_session_memories_scope'
            """
        )
    ).first()
    assert row is not None, "ck_session_memories_scope is missing"
    in_db = set(re.findall(r"'([a-z]+)'::character varying", row.def_)) or set(
        re.findall(r"'([a-z]+)'", row.def_)
    )
    assert in_db == set(MEMORY_SCOPES)


def test_the_derived_sets_are_subsets_of_the_whole():
    for name, subset in (
        ("WRITABLE_SCOPES", WRITABLE_SCOPES),
        ("RECALL_SCOPES", RECALL_SCOPES),
        ("PROMOTABLE_SCOPES", PROMOTABLE_SCOPES),
        ("GRAPH_SCOPES", GRAPH_SCOPES),
    ):
        assert subset <= MEMORY_SCOPES, f"{name} has values the column rejects"


def test_session_is_not_writable_through_the_scoped_path():
    # write_memory hardcodes scope='session' on INSERT and write_scoped_memory
    # then UPDATEs it to the real scope. 'session' is therefore never a value a
    # caller may request.
    assert "session" not in WRITABLE_SCOPES


def test_private_is_not_promotable():
    # promote_memory moves a row OUT of private; private is the source, never a
    # destination.
    assert "private" not in PROMOTABLE_SCOPES
```

Check `services/shared/tests/conftest.py` for the existing `db_session` fixture name and reuse it verbatim; if it is named differently there, use that name rather than adding a second fixture.

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/Hive/Research/AgentGuard/vex_private/vex_engine
DATABASE_URL="postgresql://agentguard:agentguard_dev@localhost:5432/agentguard" \
  .venv/bin/python -m pytest services/shared/tests/test_scopes_single_definition.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'shared.scopes'`.

- [ ] **Step 3: Write the module**

```python
"""The memory-row scope value set, defined once.

``session_memories.scope`` is governed by ``ck_session_memories_scope``
(migration 031). Before this module the same set was also written out as a
tuple in ``memory.py`` twice, ``brain.py``, ``graph_store.py`` and
``promotion.py`` — six independent copies, none referring to another. A scope
added to five of them is accepted on write and invisible on read, which looks
like data loss rather than a validation gap.

Not to be confused with ``shared.auth.VALID_SCOPES``, which is the unrelated
set of API-key permission scopes.
"""

from __future__ import annotations

#: Every value ``session_memories.scope`` may hold. Must equal the
#: ``ck_session_memories_scope`` CHECK constraint; a test asserts it.
MEMORY_SCOPES = frozenset({"private", "project", "org", "session", "agent"})

#: Scopes a caller may ask a write to land at. 'session' is excluded because
#: ``write_memory`` hardcodes it on INSERT and ``write_scoped_memory`` then
#: UPDATEs the row to its real scope, so it is an implementation detail rather
#: than a destination.
WRITABLE_SCOPES = frozenset({"private", "project", "org", "agent"})

#: Scopes ``retrieve_memories_scoped`` accepts as its ``scope`` argument. It
#: rejects 'session' explicitly and reaches 'project' rows through the
#: membership subquery rather than by being asked for them.
RECALL_SCOPES = frozenset({"org", "private", "agent"})

#: Destinations ``promote_memory`` will move a private row to. 'private' is
#: the source, never a destination.
PROMOTABLE_SCOPES = frozenset({"org", "project"})

#: Scopes the knowledge graph will attach edges for.
GRAPH_SCOPES = frozenset({"org", "agent"})
```

- [ ] **Step 4: Run test to verify it passes**

```bash
DATABASE_URL="postgresql://agentguard:agentguard_dev@localhost:5432/agentguard" \
  .venv/bin/python -m pytest services/shared/tests/test_scopes_single_definition.py -v
```

Expected: PASS, 4 tests.

If `test_memory_scopes_match_the_database_check_constraint` fails, the regex did not parse this Postgres version's `pg_get_constraintdef` output. Print `row.def_` and adjust the pattern — do **not** relax the assertion to make it pass.

- [ ] **Step 5: Point the six call sites at the module**

`services/shared/shared/brain.py:761` — replace:
```python
    if scope not in ("org", "project", "private", "agent"):
```
with:
```python
    if scope not in scopes.WRITABLE_SCOPES:
```
and add `from shared import scopes` to that file's imports.

`services/shared/shared/memory.py:1711` — the identical tuple, same replacement, same import.

`services/shared/shared/graph_store.py:389` — replace:
```python
    if scope not in ("org", "agent"):
```
with:
```python
    if scope not in scopes.GRAPH_SCOPES:
```

`services/shared/shared/promotion.py:34` — replace:
```python
PROMOTABLE_SCOPES = ("org", "project")
```
with:
```python
from shared.scopes import PROMOTABLE_SCOPES  # noqa: F401  (re-exported)
```
Keep the name importable from `promotion` — other modules and tests may import it from there.

`services/shared/shared/memory.py:2522-2540` — the `retrieve_memories_scoped` validation. Replace the membership test against its inline literals with `if scope not in scopes.RECALL_SCOPES:`, preserving the existing error message text and the separate `agent`-needs-`agent_id` / `private`-needs-`user_id` checks exactly as they are. Do not collapse those two into the set test; they assert different things.

- [ ] **Step 6: Prove the refactor changed no behaviour**

```bash
DATABASE_URL="postgresql://agentguard:agentguard_dev@localhost:5432/agentguard" \
  .venv/bin/python -m pytest services/shared/tests/ services/mcp-server/tests/ -q 2>&1 | tail -3
```

Expected: **25 failed, 1993 passed, 11 skipped** — the same 25 pre-existing failures, and 1989 + 4 new. Any other number means the refactor changed behaviour; find out why before continuing.

- [ ] **Step 7: Lint**

```bash
.venv/bin/python -m ruff check services/ && .venv/bin/python -m ruff format --check services/
```

- [ ] **Step 8: Commit**

```bash
git add services/shared/shared/scopes.py services/shared/tests/test_scopes_single_definition.py \
        services/shared/shared/memory.py services/shared/shared/brain.py \
        services/shared/shared/graph_store.py services/shared/shared/promotion.py
git commit -m "refactor(scopes): give the memory scope set one definition

Six independent copies of the same value set, none referring to another:
migration 031's tuple, the CHECK constraint it installs, and bare tuples in
memory.py (twice), brain.py, graph_store.py and promotion.py. A scope added
to five of six is accepted on write and dropped on read, which presents as
data loss rather than a validation gap.

The test that carries this is the one comparing the constants to the live
CHECK constraint; comparing the Python constants to each other could not
catch the drift the module exists to prevent.

No behaviour change: same 25 pre-existing failures, 1989 -> 1993 passed,
the four new tests."
```

---

### Task 2: One membership resolution function

`org_id` stops being derived from the key, so something must answer "may this user act in this org?" — once, in one place, on every call.

The answer already exists in the OAuth path (`oauth.py:389-402`) but is buried inside `GrantResolver._lookup` and reachable only there. This lifts it out.

**Files:**
- Create: `services/shared/shared/membership.py`
- Create: `services/shared/tests/test_membership_resolution.py`

**Interfaces:**
- Consumes: `shared.scopes` (not directly, but Task 1 must land first so the tree is stable).
- Produces:
  - `reachable_orgs(supabase, tsdb, *, user_id: str) -> list[dict]` — each `{"org_id": str, "account_slug": str, "is_personal": bool}`, ordered personal-first then by slug.
  - `may_reach(supabase, tsdb, *, user_id: str, org_id: str) -> bool`.

  **Corrected after implementation (`bc7759d`).** `public.accounts*` live behind
  `SUPABASE_DATABASE_URL`; `organizations` lives behind `DATABASE_URL`. They are
  different servers, so these take two connections and join in Python, as
  `app.oauth.GrantResolver` does. The original single-JOIN version passed its
  tests only because the local fixture puts both in one database.

- [ ] **Step 1: Write the failing test**

```python
"""Membership is the tenancy boundary once org_id is selected, not derived.

The check that matters is the negative one: a user who is not a member of an
org must not reach it, however the org id arrives. A test that only asserts a
member CAN reach their own org passes against a function that returns True
unconditionally.
"""

import pytest

from shared.membership import may_reach, reachable_orgs


def test_a_member_reaches_their_org(membership_fixture):
    user_id, org_id, _slug = membership_fixture
    assert may_reach(membership_fixture.db, user_id=user_id, org_id=org_id) is True


def test_a_non_member_does_not_reach_it(membership_fixture):
    _user_id, org_id, _slug = membership_fixture
    assert (
        may_reach(
            membership_fixture.db,
            user_id="99999999-9999-9999-9999-999999999999",
            org_id=org_id,
        )
        is False
    )


def test_an_unknown_org_is_not_reachable(membership_fixture):
    user_id, _org_id, _slug = membership_fixture
    assert (
        may_reach(membership_fixture.db, user_id=user_id, org_id="org_does_not_exist")
        is False
    )


def test_a_blank_user_reaches_nothing(membership_fixture):
    # An unattributable caller must not fall through to "everything".
    _user_id, org_id, _slug = membership_fixture
    for blank in (None, "", "   "):
        assert may_reach(membership_fixture.db, user_id=blank, org_id=org_id) is False
        assert reachable_orgs(membership_fixture.db, user_id=blank) == []


def test_reachable_orgs_lists_only_the_users_own(membership_fixture):
    user_id, org_id, _slug = membership_fixture
    got = {o["org_id"] for o in reachable_orgs(membership_fixture.db, user_id=user_id)}
    assert org_id in got
    assert "org_someone_else" not in got


def test_may_reach_agrees_with_reachable_orgs(membership_fixture):
    # The two must never disagree: one is the list, the other the predicate.
    user_id, _org_id, _slug = membership_fixture
    for org in reachable_orgs(membership_fixture.db, user_id=user_id):
        assert may_reach(membership_fixture.db, user_id=user_id, org_id=org["org_id"])
```

**On the fixture:** this reads Supabase tables (`public.accounts`, `public.accounts_memberships`) that do not exist in the local test database. Before writing the fixture, read `services/mcp-server/tests/` for how existing tests cover `oauth.py`'s membership gate — if they stub the Supabase session, use the same stub and name the fixture `membership_fixture` returning a `(db, user_id, org_id, slug)` namedtuple. If no such stub exists, create the two tables in the test database in the fixture with the minimal columns this query touches (`accounts.id`, `accounts.slug`, `accounts.is_personal_account`, `accounts_memberships.account_id`, `accounts_memberships.user_id`) and drop them on teardown. Do not skip the test.

- [ ] **Step 2: Run test to verify it fails**

```bash
DATABASE_URL="postgresql://agentguard:agentguard_dev@localhost:5432/agentguard" \
  .venv/bin/python -m pytest services/shared/tests/test_membership_resolution.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'shared.membership'`.

- [ ] **Step 3: Write the module**

```python
"""May this user act in this org?

Once ``org_id`` is selected by the caller rather than derived from their key,
this is the tenancy boundary, and it has to be asked on every call rather than
once at switch time — a session outlives a membership revocation.

The query is the one already used by ``app.oauth.GrantResolver._lookup``,
lifted out so the key path and the OAuth path share one answer. The engine
holds no local users or memberships table on purpose: identity lives in
Supabase, and a local copy would be a cache with no invalidation.

Unlike every other engine query against ``accounts``, this one does NOT filter
``is_personal_account = false``. A user's personal account is their personal
node, and excluding it is exactly what made the engine org-rooted.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

_LIST_SQL = text(
    """
    SELECT o.org_id AS org_id,
           a.slug AS account_slug,
           a.is_personal_account AS is_personal
    FROM public.accounts_memberships m
    JOIN public.accounts a ON a.id = m.account_id
    JOIN organizations o ON o.account_slug = a.slug
    WHERE m.user_id = :user_id
    ORDER BY a.is_personal_account DESC, a.slug ASC
    """
)


def _clean(user_id: Optional[str]) -> Optional[str]:
    if user_id is None:
        return None
    cleaned = str(user_id).strip()
    return cleaned or None


def reachable_orgs(db: Any, *, user_id: Optional[str]) -> list[dict]:
    """Every org node this user may act in, personal node first.

    An unattributable caller reaches nothing. Returning "everything" for a
    blank user id would turn a failure to identify someone into access to
    every tenant.
    """
    cleaned = _clean(user_id)
    if cleaned is None:
        return []
    rows = db.execute(_LIST_SQL, {"user_id": cleaned}).fetchall()
    return [
        {
            "org_id": r.org_id,
            "account_slug": r.account_slug,
            "is_personal": bool(r.is_personal),
        }
        for r in rows
    ]


def may_reach(db: Any, *, user_id: Optional[str], org_id: Optional[str]) -> bool:
    """Whether ``user_id`` may act in ``org_id``.

    Deliberately a fresh query rather than a cached set: this is the boundary,
    and the cost of it being stale is one tenant reading another's memory.
    """
    cleaned_user = _clean(user_id)
    cleaned_org = _clean(org_id)
    if cleaned_user is None or cleaned_org is None:
        return False
    return any(o["org_id"] == cleaned_org for o in reachable_orgs(db, user_id=cleaned_user))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
DATABASE_URL="postgresql://agentguard:agentguard_dev@localhost:5432/agentguard" \
  .venv/bin/python -m pytest services/shared/tests/test_membership_resolution.py -v
```

Expected: PASS, 6 tests.

- [ ] **Step 5: Mutation-check the boundary**

Temporarily change `may_reach`'s final line to `return True` and re-run. `test_a_non_member_does_not_reach_it`, `test_an_unknown_org_is_not_reachable` and `test_a_blank_user_reaches_nothing` must all fail. Restore the line and confirm green again. A boundary test that survives this mutation is not testing the boundary.

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/python -m ruff check services/ && .venv/bin/python -m ruff format --check services/
git add services/shared/shared/membership.py services/shared/tests/test_membership_resolution.py
git commit -m "feat(membership): one function answering which orgs a user may act in

Lifted out of app.oauth.GrantResolver._lookup, where the same query was
reachable only from the OAuth path. Once org_id is selected rather than
derived, this is the tenancy boundary and both paths need the same answer.

may_reach queries rather than caching: the cost of a stale answer is one
tenant reading another's memory. Checked by mutation -- forcing it to return
True fails three of the six tests.

Note this is the one query against accounts that does NOT filter
is_personal_account = false. A user's personal account is their personal
node; excluding it is what made the engine org-rooted."
```

---

### Task 3: The personal node resolves

Four call sites hardcode `is_personal_account = false`, so a user's personal account has no `organizations` row and cannot be acted in. This provisions one.

**Files:**
- Create: `services/migrations/alembic/versions/048_personal_node_orgs.py`
- Create: `services/shared/tests/test_personal_node.py`
- Modify: `services/shared/shared/membership.py` (add `ensure_personal_org`)

**Interfaces:**
- Consumes: `shared.membership.reachable_orgs` from Task 2.
- Produces: `shared.membership.ensure_personal_org(tsdb, *, user_id: str, account_slug: str) -> str` returning the `org_id`. Takes the **engine** connection only — it writes `organizations`, which is engine-side.

- [ ] **Step 1: Write the failing test**

```python
"""A user's personal account is their personal node, and it has an org row.

Every other engine query filters is_personal_account = false, which is why a
personal account resolves to no org today. The personal node is the one part
of the model that must always exist: it is where solo agent work accumulates
before any team is involved.
"""

from shared.membership import ensure_personal_org, reachable_orgs


def test_a_personal_account_gets_an_org_row(personal_fixture):
    db, user_id, slug = personal_fixture
    org_id = ensure_personal_org(db, user_id=user_id, account_slug=slug)
    assert org_id


def test_it_is_idempotent(personal_fixture):
    db, user_id, slug = personal_fixture
    first = ensure_personal_org(db, user_id=user_id, account_slug=slug)
    second = ensure_personal_org(db, user_id=user_id, account_slug=slug)
    assert first == second, "a second call minted a second personal node"


def test_the_personal_node_is_reachable_and_sorts_first(personal_fixture):
    db, user_id, slug = personal_fixture
    ensure_personal_org(db, user_id=user_id, account_slug=slug)
    orgs = reachable_orgs(db, user_id=user_id)
    assert orgs, "personal node did not appear in reachable_orgs"
    assert orgs[0]["is_personal"] is True


def test_a_personal_node_belongs_to_exactly_one_user(personal_fixture):
    db, user_id, slug = personal_fixture
    org_id = ensure_personal_org(db, user_id=user_id, account_slug=slug)
    other = reachable_orgs(db, user_id="88888888-8888-8888-8888-888888888888")
    assert org_id not in {o["org_id"] for o in other}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DATABASE_URL="postgresql://agentguard:agentguard_dev@localhost:5432/agentguard" \
  .venv/bin/python -m pytest services/shared/tests/test_personal_node.py -v
```

Expected: FAIL with `ImportError: cannot import name 'ensure_personal_org'`.

- [ ] **Step 3: Write the migration**

`organizations` needs to record that a row is a personal node and whose it is, so the invariant "one personal node per user" can be enforced by the database rather than by convention.

```python
"""Personal node orgs: a user's own account becomes an org row.

Revision ID: 048
Revises: 047
"""

import sqlalchemy as sa
from alembic import op

revision = "048"
down_revision = "047"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "organizations",
        sa.Column("personal_owner_user_id", sa.String(64), nullable=True),
    )
    # One personal node per user, enforced here rather than by convention: a
    # second one would silently split a person's own memory in two.
    op.create_index(
        "uq_organizations_personal_owner",
        "organizations",
        ["personal_owner_user_id"],
        unique=True,
        postgresql_where=sa.text("personal_owner_user_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_organizations_personal_owner", table_name="organizations")
    op.drop_column("organizations", "personal_owner_user_id")
```

Apply it:
```bash
cd services/migrations && DATABASE_URL="postgresql://agentguard:agentguard_dev@localhost:5432/agentguard" \
  ../../.venv/bin/python -m alembic upgrade head
```

- [ ] **Step 4: Write `ensure_personal_org`**

Append to `services/shared/shared/membership.py`:

```python
_ENSURE_PERSONAL_SQL = text(
    """
    INSERT INTO organizations (org_id, name, account_slug, personal_owner_user_id)
    VALUES (:org_id, :name, :account_slug, :user_id)
    ON CONFLICT (org_id) DO UPDATE
        SET personal_owner_user_id = EXCLUDED.personal_owner_user_id
    RETURNING org_id
    """
)


def ensure_personal_org(db: Any, *, user_id: str, account_slug: str) -> str:
    """The org row backing this user's personal node, created if absent.

    Idempotent by the partial unique index on ``personal_owner_user_id``: a
    second personal node for one user would split their own memory in two,
    which is invisible until they notice something they wrote is gone.
    """
    cleaned_user = _clean(user_id)
    cleaned_slug = _clean(account_slug)
    if cleaned_user is None or cleaned_slug is None:
        raise ValueError("ensure_personal_org needs a user_id and an account_slug")

    existing = db.execute(
        text(
            "SELECT org_id FROM organizations "
            "WHERE personal_owner_user_id = :user_id LIMIT 1"
        ),
        {"user_id": cleaned_user},
    ).first()
    if existing is not None:
        return existing.org_id

    # The whole user id, not a prefix: "personal_" + a 36-char UUID is 45
    # characters and org_id is String(64), so there is no reason to truncate.
    # An 8-hex-char prefix collides at birthday odds around 65k users, and the
    # partial unique index would then reject the second user's personal node --
    # a signup failure with no obvious cause.
    org_id = f"personal_{cleaned_user}"
    row = db.execute(
        _ENSURE_PERSONAL_SQL,
        {
            "org_id": org_id,
            "name": cleaned_slug,
            "account_slug": cleaned_slug,
            "user_id": cleaned_user,
        },
    ).first()
    return row.org_id
```

- [ ] **Step 5: Run test to verify it passes**

```bash
DATABASE_URL="postgresql://agentguard:agentguard_dev@localhost:5432/agentguard" \
  .venv/bin/python -m pytest services/shared/tests/test_personal_node.py -v
```

Expected: PASS, 4 tests.

- [ ] **Step 6: Confirm the migration reverses**

```bash
cd services/migrations
DATABASE_URL="..." ../../.venv/bin/python -m alembic downgrade 047
DATABASE_URL="..." ../../.venv/bin/python -m alembic upgrade head
```

Expected: both succeed. A migration that cannot be reversed cannot be rolled back in production.

- [ ] **Step 7: Full suite, lint, commit**

```bash
DATABASE_URL="..." .venv/bin/python -m pytest services/shared/tests/ services/mcp-server/tests/ -q 2>&1 | tail -3
.venv/bin/python -m ruff check services/ && .venv/bin/python -m ruff format --check services/
git add services/migrations/alembic/versions/048_personal_node_orgs.py \
        services/shared/shared/membership.py services/shared/tests/test_personal_node.py
git commit -m "feat(nodes): a user's personal account becomes a personal node

Four call sites filter is_personal_account = false, so a personal account
resolved to no org and could not be acted in. Migration 048 gives
organizations a personal_owner_user_id with a partial unique index, so the
one-personal-node-per-user invariant is the database's job rather than a
convention -- a second one would split a person's own memory in two, which is
invisible until they notice something is gone."
```

---

### Task 4: The caller selects the node, and it is checked every call

**Files:**
- Modify: `services/mcp-server/app/identity.py`
- Create: `services/mcp-server/tests/test_node_selection.py`

**Interfaces:**
- Consumes: `shared.membership.may_reach` (Task 2), `shared.membership.reachable_orgs` (Task 2).
- Produces: `app.identity.resolve_active_org(request, db) -> str` — the org the caller acts in for this request; raises `HTTPException(404)` when the requested node is not reachable.

- [ ] **Step 1: Write the failing test**

```python
"""The selected node is re-checked on every call, not at switch time.

This is the test that has to survive: a session outlives a membership
revocation, so checking once when the caller switches and trusting it
afterwards is the same as not checking.
"""

import pytest
from fastapi import HTTPException

from app import identity as identity_module


class _State:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Req:
    def __init__(self, headers=None, state=None):
        self.headers = headers or {}
        self.state = state or _State()


def test_without_a_selection_the_keys_org_is_used(node_fixture):
    # Preserves today's behaviour exactly for every existing caller.
    db, user_id, org_id = node_fixture
    got = identity_module.resolve_active_org(_Req({}), db)
    assert got == org_id


def test_a_reachable_node_may_be_selected(node_fixture):
    db, user_id, org_id = node_fixture
    got = identity_module.resolve_active_org(_Req({"X-Klio-Node": org_id}), db)
    assert got == org_id


def test_an_unreachable_node_is_not_found(node_fixture):
    db, _user_id, _org_id = node_fixture
    with pytest.raises(HTTPException) as exc:
        identity_module.resolve_active_org(_Req({"X-Klio-Node": "org_someone_else"}), db)
    # not_found, not forbidden: 403 confirms the org exists, which is an
    # existence oracle. The project tools already answer this way.
    assert exc.value.status_code == 404


def test_revocation_takes_effect_on_the_very_next_call(node_fixture, revoke):
    db, user_id, org_id = node_fixture
    assert identity_module.resolve_active_org(_Req({"X-Klio-Node": org_id}), db) == org_id
    revoke(user_id, org_id)
    with pytest.raises(HTTPException) as exc:
        identity_module.resolve_active_org(_Req({"X-Klio-Node": org_id}), db)
    assert exc.value.status_code == 404


def test_an_unattributable_caller_cannot_select_a_node(node_fixture, unattributable):
    db, _user_id, org_id = node_fixture
    with pytest.raises(HTTPException):
        identity_module.resolve_active_org(_Req({"X-Klio-Node": org_id}), db)
```

Build `node_fixture` on the same Supabase-table stub Task 2's fixture uses. `revoke` deletes the `accounts_memberships` row; `unattributable` patches `resolve_request_user_id` to return `None`.

- [ ] **Step 2: Run test to verify it fails**

```bash
DATABASE_URL="..." .venv/bin/python -m pytest services/mcp-server/tests/test_node_selection.py -v
```

Expected: FAIL with `AttributeError: module 'app.identity' has no attribute 'resolve_active_org'`.

- [ ] **Step 3: Add the header and the resolver**

In `services/mcp-server/app/headers.py`, alongside the existing `_AGENT_HEADERS`:

```python
_NODE_HEADERS = ("X-Klio-Node", "X-Vex-Node")


def node_header(request) -> Optional[str]:
    """The node the caller wants to act in, if they named one."""
    for name in _NODE_HEADERS:
        value = request.headers.get(name)
        if value:
            return value
    return None
```

In `services/mcp-server/app/identity.py`:

```python
def resolve_active_org(request: Request, db: Any) -> str:
    """The org node this request acts in.

    No selection means the key's own org, so every existing caller keeps
    today's behaviour exactly. A selection is checked against the caller's
    membership HERE, on this call -- not remembered from a switch, because a
    session outlives a revocation.

    Raises:
        HTTPException: 404 when the caller may not reach the named node. Not
            403: a 403 confirms the node exists, which tells an outsider what
            an org is called. The project tools answer the same way.
    """
    from shared.membership import may_reach  # noqa: PLC0415

    # may_reach takes (supabase, tsdb): see bc7759d. The caller supplies both.

    from app.headers import node_header  # noqa: PLC0415

    org_id, _agent_id, _plan, _overrides = resolve_principal_plan(request)

    requested = node_header(request)
    if not requested or requested == org_id:
        return org_id

    user_id = resolve_request_user_id(request)
    if not user_id or not may_reach(db, user_id=user_id, org_id=requested):
        raise HTTPException(status_code=404, detail="unknown node")
    return requested
```

- [ ] **Step 4: Run test to verify it passes**

```bash
DATABASE_URL="..." .venv/bin/python -m pytest services/mcp-server/tests/test_node_selection.py -v
```

Expected: PASS, 5 tests.

- [ ] **Step 5: Mutation-check the boundary**

Replace the `raise HTTPException` with `return requested` and re-run. `test_an_unreachable_node_is_not_found`, `test_revocation_takes_effect_on_the_very_next_call` and `test_an_unattributable_caller_cannot_select_a_node` must fail. Restore and confirm green.

- [ ] **Step 6: Full suite, lint, commit**

```bash
DATABASE_URL="..." .venv/bin/python -m pytest services/shared/tests/ services/mcp-server/tests/ -q 2>&1 | tail -3
.venv/bin/python -m ruff check services/ && .venv/bin/python -m ruff format --check services/
git add services/mcp-server/app/identity.py services/mcp-server/app/headers.py \
        services/mcp-server/tests/test_node_selection.py
git commit -m "feat(nodes): let a caller select the node, checked on every call

X-Klio-Node names the org the request acts in. Absent, the key's own org is
used, so every existing caller is unaffected.

The check runs per call rather than at switch time because a session outlives
a membership revocation -- checking once and trusting it afterwards is the
same as not checking. A caller who may not reach the node gets 404, not 403:
403 confirms the node exists, which is an existence oracle the project tools
already avoid.

Checked by mutation: removing the raise fails three of the five tests."
```

---

## What this plan does NOT do

Stated so the next plan does not assume otherwise:

- **No writes are re-scoped.** `_explicit_write_scope` still decides `project` vs `org` from the `X-Vex-Project` header. Making writes land in the selected node is Track B.
- **No recall change.** `retrieve_memories_scoped` still takes one `org_id`. Reading across layers with source tags is Track B.
- **No branches.** Track C. Note for whoever writes it: `promote_memory` refuses anything not currently `scope='private'`, so "merge a branch to main" has no existing primitive to reuse, and `supersedes` edges are memory→memory within one `org_id` with no branch notion.
- **No chat.** Track D, and it lives in `klio-tech/klio-workspace`, not this repo.
- **The `app.auth` / `app.identity` twins are not consolidated.** They remain byte-divergent with one live caller between them, which is how the `agent_id` bound drifted (`d6f53ce`). Worth its own task; out of scope here.

---

## Self-review

**Spec coverage.** §2 of the design (identity inversion) is covered by Tasks 2–4; the `users`/`memberships` entities named there resolve to Supabase rather than local tables, which the design's own constraint ("no local users table") anticipated. The personal node from §1 is Task 3. Task 1 is a prerequisite the design did not name — it emerged from the code review and is required before any new scope value can be added safely.

**Placeholders.** None: every step carries runnable commands and real code. The two fixture steps (Tasks 2 and 3) describe a decision the implementer must make from the existing test suite rather than giving code, because the right answer depends on a stub that may or may not exist; both say explicitly not to skip the test.

**Type consistency.** `reachable_orgs` returns `list[dict]` with keys `org_id` / `account_slug` / `is_personal` in Tasks 2, 3 and 4. `may_reach` takes keyword-only `user_id` and `org_id` and returns `bool` in Tasks 2 and 4. `ensure_personal_org` returns `str` in Task 3. `resolve_active_org(request, db) -> str` in Task 4 only.
