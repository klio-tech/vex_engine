# Klio Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an agent store a file, document, or record in Klio and have every other agent in the org discover it through the `recall` they already use.

**Architecture:** Bytes live in a new `artifacts` table (text ≤ 64KB inline in Postgres) or in a new `klio-artifacts` S3 bucket. Every active artifact also gets one short "card" row in `session_memories` with `memory_type='artifact'`, which is embedded and recalled like any other memory. `recall` needs no code change — it surfaces the card, and the agent calls `artifact_get` for the text or a signed URL.

**Tech Stack:** Python 3.11, FastAPI, FastMCP (`mcp` 1.27.x), SQLAlchemy Core with psycopg2, Alembic, boto3, pytest.

**Spec:** `docs/superpowers/specs/2026-07-30-klio-artifacts-design.md`

## Global Constraints

- **Tenancy:** `org_id` is a predicate inside *every* artifact query. A cross-org id must read as not-found, never as a permission error.
- **Never trust client-declared size.** A client's `size_bytes` selects an upload strategy only. `size_bytes` and `checksum` columns are written from server-side measurement at confirm time, and are NULL while `pending`.
- **Exactly one of `inline_text` / `storage_uri`** is non-null on an `active` artifact row. Enforced by a DB CHECK constraint.
- **Inline text cap: 65_536 bytes (64KB).** Above that, the presigned path.
- **Multipart threshold: 104_857_600 bytes (100MB)** declared size.
- **Presigned URL TTL: 900 seconds (15 minutes).** Never log a presigned URL.
- **`artifact_get(mode='text')` page cap: 51_200 bytes (50KB)** per call, with `offset` and `has_more`.
- **Abandoned `pending` artifacts are collected after 24 hours.**
- **Artifacts are exempt from the memory retention sweep, and so are their cards** while the artifact is `active`.
- **Card content format** is exactly: `[artifact] {title} — {summary} ({kind}, {human_size})` using an em-dash, and `memory_type='artifact'`, `metadata={"artifact_id": <str>, "source": "mcp"}`.
- **Async note:** `shared/` DB helpers use `async def` performing inline synchronous psycopg2 I/O. This is the house pattern (see `shared.brain.write_scoped_memory`). Do NOT wrap DB calls in threads.
- **Migrations are additive.** Never edit a shipped migration; add a new revision.
- Run all commands from the repo root unless stated otherwise. Tests for a service run from that service's directory (pytest cwd matters — see `services/mcp-server/tests/conftest.py`).

---

## File Structure

**Create:**

| Path | Responsibility |
| --- | --- |
| `services/migrations/alembic/versions/027_artifacts.py` | The `artifacts` table, its indexes and CHECK constraint. |
| `services/migrations/alembic/versions/028_retention_excludes_artifact_cards.py` | Replaces `enforce_memory_retention` so live artifact cards survive the sweep. |
| `services/shared/shared/artifact_storage.py` | S3 object I/O and presigning. The only module that imports boto3 for artifacts. No SQL. |
| `services/shared/shared/artifacts.py` | Artifact lifecycle over the DB: create pending, confirm, fetch, retract, storage usage, dedup. No boto3. |
| `services/shared/tests/test_artifact_storage.py` | Unit tests for the storage helpers with a fake S3 client. |
| `services/shared/tests/test_artifacts.py` | Unit tests for the DB lifecycle with a fake session. |
| `services/shared/tests/test_plan_limits_artifacts.py` | Plan-limit values and the `max` plan key. |
| `services/mcp-server/tests/test_artifact_tools.py` | Tool-body tests via the FastMCP registry, patched seams. |

**Modify:**

| Path | Change |
| --- | --- |
| `services/shared/shared/plan_limits.py` | Two `PlanConfig` fields; values on every plan; new `max` key. |
| `services/mcp-server/app/mcp_app.py` | `_perform_artifact_put` / `_perform_artifact_get` helpers, two new tools, `forget` extension, one line on `recall`'s description. |
| `services/mcp-server/app/scheduler.py` | A second background loop that collects abandoned uploads and retracted bytes. |

The split between `artifact_storage.py` (bytes) and `artifacts.py` (rows) is deliberate: it keeps boto3 out of the lifecycle tests and mirrors how `storage-worker` already separates `s3_client.py` from `worker.py`.

---

## Task 1: The `artifacts` table

**Files:**
- Create: `services/migrations/alembic/versions/027_artifacts.py`

**Interfaces:**
- Consumes: nothing.
- Produces: table `artifacts` with columns `id, org_id, space_id, project_id, agent_id, title, summary, kind, mime_type, size_bytes, checksum, inline_text, storage_uri, version, supersedes, status, created_at`. Every later task depends on these exact names.

- [ ] **Step 1: Write the migration**

Create `services/migrations/alembic/versions/027_artifacts.py`:

```python
"""Klio artifacts: durable files, documents and records.

ADDITIVE ONLY. Introduces ``artifacts``. Bytes live either inline
(``inline_text``, text <= 64KB) or in the ``klio-artifacts`` bucket
(``storage_uri``) — never both, never neither, once the row is active.

``size_bytes`` and ``checksum`` are NULL while ``status='pending'``: the
bytes do not exist yet, and a client's declared size is never stored. Both
are written from server-side measurement at confirm time.

Nothing existing is touched. Migration 028 amends the retention function so
cards for active artifacts survive the sweep.

Revision ID: 027
Revises: 026
Create Date: 2026-07-30
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "027"
down_revision: Union[str, None] = "026"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "artifacts",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("space_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("agent_id", sa.String(256), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("mime_type", sa.String(128), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("checksum", sa.String(64), nullable=True),
        sa.Column("inline_text", sa.Text(), nullable=True),
        sa.Column("storage_uri", sa.Text(), nullable=True),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("supersedes", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("status", sa.String(16), server_default="pending", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    # An active artifact has exactly one home for its bytes. A pending row has
    # neither yet. Violating this means a row that reads as present but has no
    # content, so it is a constraint rather than application discipline.
    op.execute(
        """
        ALTER TABLE artifacts ADD CONSTRAINT ck_artifacts_one_body CHECK (
            status <> 'active'
            OR (inline_text IS NULL) <> (storage_uri IS NULL)
        )
        """
    )
    op.execute(
        """
        ALTER TABLE artifacts ADD CONSTRAINT ck_artifacts_active_measured CHECK (
            status <> 'active'
            OR (size_bytes IS NOT NULL AND checksum IS NOT NULL)
        )
        """
    )
    op.execute(
        "ALTER TABLE artifacts ADD CONSTRAINT ck_artifacts_kind "
        "CHECK (kind IN ('document', 'record', 'file'))"
    )
    op.execute(
        "ALTER TABLE artifacts ADD CONSTRAINT ck_artifacts_status "
        "CHECK (status IN ('pending', 'active', 'superseded', 'retracted'))"
    )

    op.create_index("ix_artifacts_org_status", "artifacts", ["org_id", "status"])
    op.create_index(
        "ix_artifacts_org_space_status", "artifacts", ["org_id", "space_id", "status"]
    )
    # Deduplication: identical bytes in one org resolve to one artifact.
    op.execute(
        """
        CREATE UNIQUE INDEX ux_artifacts_org_checksum_active
        ON artifacts (org_id, checksum)
        WHERE status = 'active'
        """
    )
    # The GC pass sweeps pending rows by age.
    op.create_index("ix_artifacts_status_created", "artifacts", ["status", "created_at"])


def downgrade() -> None:
    # Cards would otherwise point at nothing.
    op.execute("DELETE FROM session_memories WHERE memory_type = 'artifact'")
    op.drop_table("artifacts")
```

- [ ] **Step 2: Start the dev database**

Run: `docker compose up -d postgres`
Expected: the `postgres` service reports healthy. If the compose service has a different name, run `docker compose config --services` and use the Postgres one.

- [ ] **Step 3: Apply the migration and verify it fails nothing**

Run from `services/migrations`: `alembic upgrade head`
Expected: `Running upgrade 026 -> 027, Klio artifacts`.

- [ ] **Step 4: Verify the CHECK constraint actually rejects a bad row**

Run: `psql "$AGENTGUARD_DATABASE_URL" -c "INSERT INTO artifacts (org_id, agent_id, title, summary, kind, mime_type, status, size_bytes, checksum) VALUES ('t','a','t','s','file','text/plain','active',1,'x')"`

Expected: `ERROR: new row for relation "artifacts" violates check constraint "ck_artifacts_one_body"` — because neither `inline_text` nor `storage_uri` was set. If this INSERT succeeds, the constraint is wrong and the task is not done.

- [ ] **Step 5: Verify downgrade is clean**

Run from `services/migrations`: `alembic downgrade -1` then `alembic upgrade head`
Expected: both succeed with no error.

- [ ] **Step 6: Commit**

```bash
git add services/migrations/alembic/versions/027_artifacts.py
git commit -m "feat(db): artifacts table"
```

---

## Task 2: Retention must not orphan artifact cards

**Files:**
- Create: `services/migrations/alembic/versions/028_retention_excludes_artifact_cards.py`

**Interfaces:**
- Consumes: the `artifacts` table from Task 1.
- Produces: a replaced `enforce_memory_retention(TEXT, INT)` SQL function. No Python signature changes.

**Why this comes before any card is written:** migration 025 defines `enforce_memory_retention` as a flat `DELETE FROM session_memories WHERE created_at < window` with no `memory_type` filter. On Free (1-day retention) an artifact card would be deleted the morning after it was written while its artifact survived — bytes the customer paid for, listed in the dashboard, unreachable by `recall`. Ship this first.

- [ ] **Step 1: Write the migration**

Create `services/migrations/alembic/versions/028_retention_excludes_artifact_cards.py`:

```python
"""Klio artifacts: retention must not orphan artifact cards.

Replaces ``enforce_memory_retention`` (migration 025), which deleted every
``session_memories`` row past the plan window with no ``memory_type`` filter.
An artifact card swept while its artifact is still active leaves bytes that
are billed, listed, and unreachable by recall.

A card now outlives the window exactly as long as its artifact does, and
becomes collectable the moment the artifact is retracted or superseded.

Revision ID: 028
Revises: 027
Create Date: 2026-07-30
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

revision: str = "028"
down_revision: Union[str, None] = "027"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION enforce_memory_retention(
            p_org_id TEXT, p_retention_days INT
        ) RETURNS void AS $$
        BEGIN
            DELETE FROM session_memories sm
            WHERE sm.org_id = p_org_id
              AND sm.created_at < NOW() - (p_retention_days || ' days')::interval
              AND NOT (
                sm.memory_type = 'artifact'
                AND EXISTS (
                    SELECT 1 FROM artifacts a
                    WHERE a.id = (sm.metadata->>'artifact_id')::uuid
                      AND a.status = 'active'
                )
              );
        END;
        $$ LANGUAGE plpgsql;
        """
    )


def downgrade() -> None:
    # Restore the unfiltered 025 body verbatim.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION enforce_memory_retention(
            p_org_id TEXT, p_retention_days INT
        ) RETURNS void AS $$
        BEGIN
            DELETE FROM session_memories
            WHERE org_id = p_org_id
              AND created_at < NOW() - (p_retention_days || ' days')::interval;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
```

- [ ] **Step 2: Apply**

Run from `services/migrations`: `alembic upgrade head`
Expected: `Running upgrade 027 -> 028`.

- [ ] **Step 3: Prove the behaviour with real rows**

Save as `/tmp/retention_check.sql` and run with `psql "$AGENTGUARD_DATABASE_URL" -f /tmp/retention_check.sql`:

```sql
BEGIN;
INSERT INTO artifacts (id, org_id, agent_id, title, summary, kind, mime_type,
                       status, size_bytes, checksum, inline_text)
VALUES ('11111111-1111-1111-1111-111111111111', 'ret-test', 'a', 't', 's',
        'document', 'text/plain', 'active', 3, 'abc', 'hey');

INSERT INTO session_memories (org_id, agent_id, session_id, memory_type, content,
                              metadata, created_at, scope, status)
VALUES ('ret-test', 'a', 's1', 'artifact', '[artifact] t — s (document, 3 B)',
        '{"artifact_id":"11111111-1111-1111-1111-111111111111"}'::jsonb,
        NOW() - INTERVAL '10 days', 'org', 'active'),
       ('ret-test', 'a', 's1', 'memory', 'an ordinary old memory',
        '{}'::jsonb, NOW() - INTERVAL '10 days', 'org', 'active');

SELECT enforce_memory_retention('ret-test', 1);

-- Expect exactly 1: the card survives, the ordinary memory is gone.
SELECT count(*) AS survivors FROM session_memories WHERE org_id = 'ret-test';

UPDATE artifacts SET status = 'retracted'
WHERE id = '11111111-1111-1111-1111-111111111111';
SELECT enforce_memory_retention('ret-test', 1);

-- Expect 0: once the artifact is retracted the card is collectable.
SELECT count(*) AS after_retract FROM session_memories WHERE org_id = 'ret-test';
ROLLBACK;
```

Expected: `survivors = 1`, then `after_retract = 0`. Any other numbers mean the function is wrong.

- [ ] **Step 4: Commit**

```bash
git add services/migrations/alembic/versions/028_retention_excludes_artifact_cards.py
git commit -m "fix(db): retention must not orphan artifact cards"
```

---

## Task 3: Plan limits and the `max` plan

**Files:**
- Modify: `services/shared/shared/plan_limits.py`
- Test: `services/shared/tests/test_plan_limits_artifacts.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PlanConfig.artifact_max_file_bytes: int` and `PlanConfig.artifact_storage_bytes: int`; a `"max"` key in `PLAN_LIMITS`. Tasks 6–8 read these via the existing `get_plan_config(plan, overrides)`.

- [ ] **Step 1: Write the failing test**

Create `services/shared/tests/test_plan_limits_artifacts.py`:

```python
"""Artifact capacity per plan.

Capacity is the paid axis; the feature itself is on every tier. A plan whose
``artifact_max_file_bytes`` is 0 has artifacts disabled entirely, which no
shipped plan does.
"""

from shared.plan_limits import PLAN_LIMITS, get_plan_config

MB = 1024 * 1024
GB = 1024 * MB

EXPECTED = {
    "free": (10 * MB, 100 * MB),
    "starter": (50 * MB, 5 * GB),
    "max": (1 * GB, 1024 * GB),
}


def test_artifact_limits_per_plan():
    for plan, (per_file, pool) in EXPECTED.items():
        cfg = get_plan_config(plan)
        assert cfg.artifact_max_file_bytes == per_file, plan
        assert cfg.artifact_storage_bytes == pool, plan


def test_max_plan_is_registered():
    # get_plan_config falls back to free for an unknown plan, so a missing
    # "max" key would silently give a $499 customer 10MB files.
    assert "max" in PLAN_LIMITS
    assert get_plan_config("max").artifact_max_file_bytes > (
        get_plan_config("free").artifact_max_file_bytes
    )


def test_every_plan_has_artifacts_enabled():
    for plan in PLAN_LIMITS:
        cfg = get_plan_config(plan)
        assert cfg.artifact_max_file_bytes > 0, plan
        assert cfg.artifact_storage_bytes > 0, plan


def test_overrides_apply_to_artifact_limits():
    cfg = get_plan_config("free", {"artifact_storage_bytes": 999})
    assert cfg.artifact_storage_bytes == 999
```

- [ ] **Step 2: Run it to see it fail**

Run from `services/shared`: `pytest tests/test_plan_limits_artifacts.py -v`
Expected: FAIL — `AttributeError: 'PlanConfig' object has no attribute 'artifact_max_file_bytes'`.

- [ ] **Step 3: Add the fields**

In `services/shared/shared/plan_limits.py`, directly below the existing `memory_sync_enabled: bool = False` field on `PlanConfig`, add:

```python
    # Artifacts (Klio) — durable files, documents and records. Capacity is the
    # paid axis; the feature is available on every plan, because handing work
    # between agents is the wedge (same reason max_agents is -1 everywhere).
    # 0 disables artifacts entirely. Mirror at
    # Dashboard/apps/landing/lib/pricing.ts (keep in sync).
    artifact_max_file_bytes: int = 10 * 1024 * 1024
    artifact_storage_bytes: int = 100 * 1024 * 1024
```

- [ ] **Step 4: Set the values on each plan**

In the `PLAN_LIMITS` dict, add these two lines to each existing `PlanConfig(...)` call:

- `"free"`: `artifact_max_file_bytes=10 * 1024 * 1024,` and `artifact_storage_bytes=100 * 1024 * 1024,`
- `"starter"`: `artifact_max_file_bytes=50 * 1024 * 1024,` and `artifact_storage_bytes=5 * 1024 * 1024 * 1024,`
- `"pro"`: same two values as `starter` (pending the pricing restructure)
- `"team"`: same two values as `starter` (pending the pricing restructure)
- `"enterprise"`: `artifact_max_file_bytes=1024 * 1024 * 1024,` and `artifact_storage_bytes=1024 * 1024 * 1024 * 1024,`

- [ ] **Step 5: Add the `max` plan**

Immediately after the `"team"` entry in `PLAN_LIMITS`, add a `"max"` entry. Copy every field from `"team"` verbatim, then override these three:

```python
    "max": PlanConfig(
        # ... copy all non-artifact fields from "team" unchanged ...
        artifact_max_file_bytes=1024 * 1024 * 1024,          # 1 GB per file
        artifact_storage_bytes=1024 * 1024 * 1024 * 1024,    # 1 TB pool
        memory_sync_enabled=True,
    ),
```

- [ ] **Step 6: Run the test to verify it passes**

Run from `services/shared`: `pytest tests/test_plan_limits_artifacts.py -v`
Expected: 4 passed.

- [ ] **Step 7: Run the whole shared suite for regressions**

Run from `services/shared`: `pytest -q`
Expected: no new failures versus `main`. If a pre-existing failure appears, note it and continue — do not fix unrelated tests here.

- [ ] **Step 8: Commit**

```bash
git add services/shared/shared/plan_limits.py services/shared/tests/test_plan_limits_artifacts.py
git commit -m "feat(plans): artifact capacity limits and the max plan"
```

---

## Task 4: Object storage helpers

**Files:**
- Create: `services/shared/shared/artifact_storage.py`
- Test: `services/shared/tests/test_artifact_storage.py`

**Interfaces:**
- Consumes: nothing.
- Produces, all importable from `shared.artifact_storage`:
  - `ARTIFACT_BUCKET: str` (`"klio-artifacts"`)
  - `PRESIGN_TTL_SECONDS: int` (`900`)
  - `object_key(org_id: str, artifact_id: str, version: int) -> str`
  - `storage_uri(org_id: str, artifact_id: str, version: int) -> str`
  - `presign_put(s3, key: str) -> str`
  - `presign_get(s3, key: str) -> str`
  - `head(s3, key: str) -> dict | None` → `{"size_bytes": int, "checksum": str}` or `None` when absent
  - `put_text(s3, key: str, text: str, mime_type: str) -> dict` → same shape as `head`
  - `get_text(s3, key: str) -> str`
  - `delete(s3, key: str) -> None`

- [ ] **Step 1: Write the failing test**

Create `services/shared/tests/test_artifact_storage.py`:

```python
"""Object-storage helpers for artifacts, exercised against a fake S3 client.

The fake records calls and returns canned payloads — no network, no moto. This
matches how storage-worker tests fake their client.
"""

import hashlib
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from shared import artifact_storage as st


def _fake_s3():
    return MagicMock()


def test_object_key_is_org_scoped_and_versioned():
    key = st.object_key("acme", "abc-123", 2)
    assert key == "acme/abc-123/v2"


def test_storage_uri_points_at_the_artifact_bucket():
    uri = st.storage_uri("acme", "abc-123", 1)
    assert uri == "s3://klio-artifacts/acme/abc-123/v1"


def test_presign_put_uses_the_short_ttl():
    s3 = _fake_s3()
    s3.generate_presigned_url.return_value = "https://example/put"

    url = st.presign_put(s3, "acme/a/v1")

    assert url == "https://example/put"
    kwargs = s3.generate_presigned_url.call_args.kwargs
    assert kwargs["ExpiresIn"] == 900
    assert kwargs["Params"]["Bucket"] == "klio-artifacts"
    assert kwargs["Params"]["Key"] == "acme/a/v1"


def test_presign_get_uses_the_short_ttl():
    s3 = _fake_s3()
    s3.generate_presigned_url.return_value = "https://example/get"

    assert st.presign_get(s3, "acme/a/v1") == "https://example/get"
    assert s3.generate_presigned_url.call_args.kwargs["ExpiresIn"] == 900


def test_head_returns_measured_size_and_checksum():
    body = b"hello world"
    s3 = _fake_s3()
    s3.head_object.return_value = {"ContentLength": len(body)}
    s3.get_object.return_value = {"Body": MagicMock(read=lambda: body)}

    meta = st.head(s3, "acme/a/v1")

    assert meta == {
        "size_bytes": len(body),
        "checksum": hashlib.sha256(body).hexdigest(),
    }


def test_head_returns_none_when_the_object_is_absent():
    s3 = _fake_s3()
    s3.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
    )

    assert st.head(s3, "acme/a/v1") is None


def test_put_text_stores_and_returns_measured_metadata():
    s3 = _fake_s3()
    meta = st.put_text(s3, "acme/a/v1", "hey", "text/plain")

    assert meta == {
        "size_bytes": 3,
        "checksum": hashlib.sha256(b"hey").hexdigest(),
    }
    kwargs = s3.put_object.call_args.kwargs
    assert kwargs["Bucket"] == "klio-artifacts"
    assert kwargs["Key"] == "acme/a/v1"
    assert kwargs["Body"] == b"hey"
    assert kwargs["ContentType"] == "text/plain"


def test_get_text_decodes_utf8():
    s3 = _fake_s3()
    s3.get_object.return_value = {"Body": MagicMock(read=lambda: b"caf\xc3\xa9")}

    assert st.get_text(s3, "acme/a/v1") == "café"


def test_delete_is_scoped_to_the_bucket():
    s3 = _fake_s3()
    st.delete(s3, "acme/a/v1")
    s3.delete_object.assert_called_once_with(
        Bucket="klio-artifacts", Key="acme/a/v1"
    )
```

- [ ] **Step 2: Run it to see it fail**

Run from `services/shared`: `pytest tests/test_artifact_storage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.artifact_storage'`.

- [ ] **Step 3: Implement the module**

Create `services/shared/shared/artifact_storage.py`:

```python
"""Object storage for Klio artifacts.

The only artifact module that imports boto3. Kept separate from
:mod:`shared.artifacts` (which owns the rows) so the lifecycle tests never
need an S3 fake, mirroring how ``storage-worker`` separates ``s3_client`` from
``worker``.

The artifact bucket is deliberately NOT ``agentguard-traces``: traces are
internal telemetry, artifacts are customer data with different retention and a
different blast radius.
"""

import hashlib
import os
from typing import Any, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ARTIFACT_BUCKET = os.environ.get("KLIO_ARTIFACT_BUCKET", "klio-artifacts")

# Presigned URLs are bearer credentials: anyone holding one can read or write
# that single object. Short TTL is the mitigation. Never log them.
PRESIGN_TTL_SECONDS = 900

# Above this declared size a single PUT is unreliable, so the caller is handed
# multipart part URLs instead.
MULTIPART_THRESHOLD_BYTES = 100 * 1024 * 1024


def get_artifact_s3_client() -> Any:
    """Return a boto3 S3 client configured for the artifact bucket."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY", "agentguard"),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY", "agentguard_dev"),
        config=Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=5,
            read_timeout=30,
        ),
    )


def object_key(org_id: str, artifact_id: str, version: int) -> str:
    """Bytes live under the owning org, so a key is never ambiguous."""
    return f"{org_id}/{artifact_id}/v{version}"


def storage_uri(org_id: str, artifact_id: str, version: int) -> str:
    return f"s3://{ARTIFACT_BUCKET}/{object_key(org_id, artifact_id, version)}"


def presign_put(s3: Any, key: str) -> str:
    return s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": ARTIFACT_BUCKET, "Key": key},
        ExpiresIn=PRESIGN_TTL_SECONDS,
    )


def presign_get(s3: Any, key: str) -> str:
    """Presigned download.

    The response headers are forced on the URL itself: Klio never executes an
    artifact, and a browser following this link must not be talked into
    rendering someone's uploaded HTML as a page on our origin.
    """
    return s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": ARTIFACT_BUCKET,
            "Key": key,
            "ResponseContentDisposition": "attachment",
            "ResponseContentType": "application/octet-stream",
        },
        ExpiresIn=PRESIGN_TTL_SECONDS,
    )


def _measure(body: bytes) -> dict:
    return {"size_bytes": len(body), "checksum": hashlib.sha256(body).hexdigest()}


def head(s3: Any, key: str) -> Optional[dict]:
    """Server-side measurement of a stored object.

    Returns ``None`` when the object does not exist, which is how an
    unconfirmed upload is detected. The body is read to compute the checksum:
    a client-supplied one could not be trusted, and dedup depends on it.
    """
    try:
        s3.head_object(Bucket=ARTIFACT_BUCKET, Key=key)
    except ClientError:
        return None

    body = s3.get_object(Bucket=ARTIFACT_BUCKET, Key=key)["Body"].read()
    return _measure(body)


def put_text(s3: Any, key: str, text: str, mime_type: str) -> dict:
    body = text.encode("utf-8")
    s3.put_object(
        Bucket=ARTIFACT_BUCKET,
        Key=key,
        Body=body,
        ContentType=mime_type,
        ContentDisposition="attachment",
    )
    return _measure(body)


def get_text(s3: Any, key: str) -> str:
    return s3.get_object(Bucket=ARTIFACT_BUCKET, Key=key)["Body"].read().decode("utf-8")


def delete(s3: Any, key: str) -> None:
    s3.delete_object(Bucket=ARTIFACT_BUCKET, Key=key)
```

- [ ] **Step 4: Run the test to verify it passes**

Run from `services/shared`: `pytest tests/test_artifact_storage.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add services/shared/shared/artifact_storage.py services/shared/tests/test_artifact_storage.py
git commit -m "feat(artifacts): object storage helpers"
```

---

## Task 5: Artifact lifecycle over the DB

**Files:**
- Create: `services/shared/shared/artifacts.py`
- Test: `services/shared/tests/test_artifacts.py`

**Interfaces:**
- Consumes: `shared.artifact_storage.storage_uri` from Task 4.
- Produces, all importable from `shared.artifacts`:
  - `INLINE_MAX_BYTES: int` (`65_536`), `TEXT_PAGE_BYTES: int` (`51_200`), `PENDING_TTL_HOURS: int` (`24`)
  - `human_size(n: int) -> str`
  - `card_content(title: str, summary: str, kind: str, size_bytes: int) -> str`
  - `org_storage_bytes(db, *, org_id: str) -> int`
  - `find_active_by_checksum(db, *, org_id: str, checksum: str) -> dict | None`
  - `create_pending(db, *, org_id, agent_id, title, summary, kind, mime_type, space_id=None, project_id=None, supersedes=None, version=1) -> str` (returns the new artifact id)
  - `activate(db, *, org_id, artifact_id, size_bytes, checksum, inline_text=None, storage_uri=None) -> dict`
  - `get_artifact(db, *, org_id, artifact_id) -> dict | None`
  - `retract(db, *, org_id, artifact_id) -> bool`
  - `mark_superseded(db, *, org_id, artifact_id) -> None`
  - `collect_abandoned(db, *, older_than_hours: int = 24) -> list[dict]`

- [ ] **Step 1: Write the failing test**

Create `services/shared/tests/test_artifacts.py`:

```python
"""Artifact lifecycle over a fake DB session.

The fake session records executed SQL and its parameters and returns queued
row payloads, so these tests assert on tenancy predicates and on the TS-side
transformation without a live database.
"""

from unittest.mock import MagicMock

import pytest

from shared import artifacts as af


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    def scalar(self):
        return self._rows[0][0] if self._rows else None


class FakeSession:
    """Records (sql, params) per execute and returns queued results FIFO."""

    def __init__(self, *results):
        self.calls = []
        self._queue = list(results)

    def execute(self, sql, params=None):
        self.calls.append((str(sql), params or {}))
        return self._queue.pop(0) if self._queue else FakeResult([])

    def commit(self):
        pass


def test_human_size_is_readable():
    assert af.human_size(900) == "900 B"
    assert af.human_size(2048) == "2.0 KB"
    assert af.human_size(5 * 1024 * 1024) == "5.0 MB"
    assert af.human_size(3 * 1024 * 1024 * 1024) == "3.0 GB"


def test_card_content_matches_the_specified_format():
    card = af.card_content("Q3 pricing", "The model finance signed off", "file", 2048)
    assert card == "[artifact] Q3 pricing — The model finance signed off (file, 2.0 KB)"


def test_org_storage_bytes_counts_only_active_rows_for_the_org():
    db = FakeSession(FakeResult([(4096,)]))

    total = af.org_storage_bytes(db, org_id="acme")

    assert total == 4096
    sql, params = db.calls[0]
    assert "org_id = :org_id" in sql
    assert "status = 'active'" in sql
    assert params["org_id"] == "acme"


def test_org_storage_bytes_is_zero_when_nothing_stored():
    db = FakeSession(FakeResult([(None,)]))
    assert af.org_storage_bytes(db, org_id="acme") == 0


def test_get_artifact_is_org_scoped():
    db = FakeSession(FakeResult([]))

    assert af.get_artifact(db, org_id="other-org", artifact_id="a1") is None

    sql, params = db.calls[0]
    assert "org_id = :org_id" in sql
    assert params["org_id"] == "other-org"


def test_find_active_by_checksum_is_org_scoped():
    db = FakeSession(FakeResult([]))
    af.find_active_by_checksum(db, org_id="acme", checksum="deadbeef")
    sql, params = db.calls[0]
    assert "org_id = :org_id" in sql
    assert "status = 'active'" in sql
    assert params["checksum"] == "deadbeef"


def test_create_pending_inserts_without_size_or_checksum():
    db = FakeSession(FakeResult([("new-id",)]))

    artifact_id = af.create_pending(
        db,
        org_id="acme",
        agent_id="agent-1",
        title="t",
        summary="s",
        kind="document",
        mime_type="text/markdown",
    )

    assert artifact_id == "new-id"
    sql, params = db.calls[0]
    assert "'pending'" in sql
    # Measured values are never written at create time.
    assert "size_bytes" not in sql
    assert "checksum" not in sql
    assert params["org_id"] == "acme"


def test_activate_writes_measured_values_and_is_org_scoped():
    db = FakeSession(FakeResult([("a1", 3, "abc")]))

    af.activate(
        db,
        org_id="acme",
        artifact_id="a1",
        size_bytes=3,
        checksum="abc",
        inline_text="hey",
    )

    sql, params = db.calls[0]
    assert "org_id = :org_id" in sql
    assert "'active'" in sql
    assert params["size_bytes"] == 3
    assert params["checksum"] == "abc"
    assert params["inline_text"] == "hey"


def test_retract_is_org_scoped_and_reports_whether_it_hit():
    db = FakeSession(FakeResult([]))
    assert af.retract(db, org_id="other-org", artifact_id="a1") is False
    sql, params = db.calls[0]
    assert "org_id = :org_id" in sql
    assert "'retracted'" in sql


def test_collect_abandoned_selects_only_stale_pending_rows():
    db = FakeSession(FakeResult([("a1", "acme", 1)]))

    rows = af.collect_abandoned(db, older_than_hours=24)

    assert rows == [{"id": "a1", "org_id": "acme", "version": 1}]
    sql, params = db.calls[0]
    assert "'pending'" in sql
    assert params["hours"] == 24
```

- [ ] **Step 2: Run it to see it fail**

Run from `services/shared`: `pytest tests/test_artifacts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.artifacts'`.

- [ ] **Step 3: Implement the module**

Create `services/shared/shared/artifacts.py`:

```python
"""Artifact lifecycle: the rows, not the bytes.

Every statement carries an ``org_id`` predicate. A cross-org id therefore
reads as not-found rather than as a permission error, matching the memory and
session paths.

``size_bytes`` and ``checksum`` are written only by :func:`activate`, from
values the server measured. A client's declared size selects an upload
strategy and is never persisted.

Async note: like :mod:`shared.brain`, the DB calls here are synchronous
psycopg2 I/O. Callers are sync; do not wrap in threads.
"""

from typing import Any, Optional

from sqlalchemy import text

# Text at or below this size lives in Postgres, so the common read path never
# touches object storage. Covers essentially every spec, record and code file.
INLINE_MAX_BYTES = 65_536

# One artifact_get(mode='text') call returns at most this much, with an offset
# for paging. A context window is the real constraint here, not the transport.
TEXT_PAGE_BYTES = 51_200

# An upload that is requested and never confirmed is collected after this long.
PENDING_TTL_HOURS = 24


def human_size(n: int) -> str:
    """Format bytes for the card a human will read."""
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.1f} GB"


def card_content(title: str, summary: str, kind: str, size_bytes: int) -> str:
    """The memory row that makes an artifact discoverable through recall."""
    return f"[artifact] {title} — {summary} ({kind}, {human_size(size_bytes)})"


def org_storage_bytes(db: Any, *, org_id: str) -> int:
    """Live storage for an org.

    A summed gauge rather than a running counter: a counter is faster but can
    drift, and drift here means either over-charging a customer or letting a
    pool run unbounded. Artifact counts are in the thousands; revisit if that
    stops being true.
    """
    row = db.execute(
        text(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM artifacts "
            "WHERE org_id = :org_id AND status = 'active'"
        ),
        {"org_id": org_id},
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def find_active_by_checksum(db: Any, *, org_id: str, checksum: str) -> Optional[dict]:
    """Deduplication: identical bytes in one org resolve to one artifact."""
    row = db.execute(
        text(
            "SELECT id, version FROM artifacts "
            "WHERE org_id = :org_id AND checksum = :checksum AND status = 'active'"
        ),
        {"org_id": org_id, "checksum": checksum},
    ).fetchone()
    return {"id": str(row[0]), "version": int(row[1])} if row else None


def create_pending(
    db: Any,
    *,
    org_id: str,
    agent_id: str,
    title: str,
    summary: str,
    kind: str,
    mime_type: str,
    space_id: Optional[str] = None,
    project_id: Optional[str] = None,
    supersedes: Optional[str] = None,
    version: int = 1,
) -> str:
    """Reserve an artifact id. No bytes yet, so no size and no checksum."""
    row = db.execute(
        text(
            """
            INSERT INTO artifacts (
                org_id, agent_id, title, summary, kind, mime_type,
                space_id, project_id, supersedes, version, status
            ) VALUES (
                :org_id, :agent_id, :title, :summary, :kind, :mime_type,
                :space_id, :project_id, :supersedes, :version, 'pending'
            )
            RETURNING id
            """
        ),
        {
            "org_id": org_id,
            "agent_id": agent_id,
            "title": title,
            "summary": summary,
            "kind": kind,
            "mime_type": mime_type,
            "space_id": space_id,
            "project_id": project_id,
            "supersedes": supersedes,
            "version": version,
        },
    ).fetchone()
    db.commit()
    return str(row[0])


def activate(
    db: Any,
    *,
    org_id: str,
    artifact_id: str,
    size_bytes: int,
    checksum: str,
    inline_text: Optional[str] = None,
    storage_uri: Optional[str] = None,
) -> Optional[dict]:
    """Promote a pending artifact using server-measured values."""
    row = db.execute(
        text(
            """
            UPDATE artifacts SET
                status = 'active',
                size_bytes = :size_bytes,
                checksum = :checksum,
                inline_text = :inline_text,
                storage_uri = :storage_uri
            WHERE org_id = :org_id AND id = :artifact_id AND status = 'pending'
            RETURNING id, size_bytes, checksum
            """
        ),
        {
            "org_id": org_id,
            "artifact_id": artifact_id,
            "size_bytes": size_bytes,
            "checksum": checksum,
            "inline_text": inline_text,
            "storage_uri": storage_uri,
        },
    ).fetchone()
    db.commit()
    if not row:
        return None
    return {"id": str(row[0]), "size_bytes": int(row[1]), "checksum": row[2]}


def get_artifact(db: Any, *, org_id: str, artifact_id: str) -> Optional[dict]:
    row = db.execute(
        text(
            """
            SELECT id, org_id, agent_id, title, summary, kind, mime_type,
                   size_bytes, checksum, inline_text, storage_uri,
                   version, status
            FROM artifacts
            WHERE org_id = :org_id AND id = :artifact_id
            """
        ),
        {"org_id": org_id, "artifact_id": artifact_id},
    ).fetchone()
    if not row:
        return None
    return {
        "id": str(row[0]),
        "org_id": row[1],
        "agent_id": row[2],
        "title": row[3],
        "summary": row[4],
        "kind": row[5],
        "mime_type": row[6],
        "size_bytes": row[7],
        "checksum": row[8],
        "inline_text": row[9],
        "storage_uri": row[10],
        "version": int(row[11]),
        "status": row[12],
    }


def retract(db: Any, *, org_id: str, artifact_id: str) -> bool:
    """Retract an artifact. Returns False when the org does not own it."""
    row = db.execute(
        text(
            "UPDATE artifacts SET status = 'retracted' "
            "WHERE org_id = :org_id AND id = :artifact_id AND status = 'active' "
            "RETURNING id"
        ),
        {"org_id": org_id, "artifact_id": artifact_id},
    ).fetchone()
    db.commit()
    return row is not None


def mark_superseded(db: Any, *, org_id: str, artifact_id: str) -> None:
    db.execute(
        text(
            "UPDATE artifacts SET status = 'superseded' "
            "WHERE org_id = :org_id AND id = :artifact_id AND status = 'active'"
        ),
        {"org_id": org_id, "artifact_id": artifact_id},
    )
    db.commit()


def collect_abandoned(db: Any, *, older_than_hours: int = PENDING_TTL_HOURS) -> list:
    """Pending artifacts whose upload never arrived. Caller deletes the bytes."""
    rows = db.execute(
        text(
            """
            SELECT id, org_id, version FROM artifacts
            WHERE status = 'pending'
              AND created_at < NOW() - (:hours || ' hours')::interval
            """
        ),
        {"hours": older_than_hours},
    ).fetchall()
    return [
        {"id": str(r[0]), "org_id": r[1], "version": int(r[2])} for r in rows
    ]
```

- [ ] **Step 4: Run the test to verify it passes**

Run from `services/shared`: `pytest tests/test_artifacts.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add services/shared/shared/artifacts.py services/shared/tests/test_artifacts.py
git commit -m "feat(artifacts): DB lifecycle with org-scoped queries"
```

---

## Task 6: `artifact_put` — the inline path

**Files:**
- Modify: `services/mcp-server/app/mcp_app.py`
- Test: `services/mcp-server/tests/test_artifact_tools.py`

**Interfaces:**
- Consumes: everything produced by Tasks 3–5, plus the existing `resolve_principal_plan`, `get_plan_config`, `brain.ensure_space`, `brain.write_scoped_memory`, `SessionLocal` already imported in `mcp_app.py`.
- Produces: `async def _perform_artifact_put(ctx, *, title, summary, kind, content, mime_type, size_bytes, space, project, supersedes, artifact_id) -> dict[str, Any]` and a registered `artifact_put` tool. Task 7 registers `artifact_get`; Task 8 extends `forget`.

Return shape on success: `{"stored": True, "artifact_id": str, "version": int, "status": "active"}`.

- [ ] **Step 1: Write the failing test**

Create `services/mcp-server/tests/test_artifact_tools.py`:

```python
"""Artifact tool bodies, exercised through the FastMCP registry.

Same seams as test_write_tools.py: resolve_principal_plan, SessionLocal, and
brain.write_scoped_memory are patched at the app.mcp_app boundary, so no DB,
no embeddings and no S3 are touched.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.mcp_app as mcp_app
from app.mcp_app import build_mcp_server

MB = 1024 * 1024


def _ctx():
    request = SimpleNamespace(headers={"x-vex-agent": "agent-1"})
    return SimpleNamespace(request_context=SimpleNamespace(request=request))


def _tool(name):
    return build_mcp_server()._tool_manager.get_tool(name).fn


@pytest.fixture
def seams():
    """Patch every external dependency of the artifact tools."""
    plan = SimpleNamespace(
        artifact_max_file_bytes=10 * MB,
        artifact_storage_bytes=100 * MB,
    )
    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("acme", "agent-1", "free", None),
        ),
        patch("app.mcp_app.get_plan_config", return_value=plan),
        patch("app.mcp_app.SessionLocal", MagicMock()),
        patch("app.mcp_app.artifacts") as af,
        patch("app.mcp_app.brain") as brain,
        patch("app.mcp_app.resolve_project_from_request", return_value=None),
    ):
        af.INLINE_MAX_BYTES = 65_536
        af.human_size.return_value = "3 B"
        af.card_content.return_value = "[artifact] t — s (document, 3 B)"
        af.org_storage_bytes.return_value = 0
        af.find_active_by_checksum.return_value = None
        af.create_pending.return_value = "new-artifact-id"
        af.activate.return_value = {
            "id": "new-artifact-id",
            "size_bytes": 3,
            "checksum": "abc",
        }
        brain.write_scoped_memory = AsyncMock(return_value={"stored": True})
        brain.ensure_space.return_value = None
        yield SimpleNamespace(artifacts=af, brain=brain)


@pytest.mark.asyncio
async def test_inline_put_activates_and_writes_a_card(seams):
    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="document", content="hey"
    )

    assert result["stored"] is True
    assert result["artifact_id"] == "new-artifact-id"
    assert result["status"] == "active"

    # The card is the discovery mechanism; without it the artifact is invisible.
    seams.brain.write_scoped_memory.assert_awaited_once()
    kwargs = seams.brain.write_scoped_memory.await_args.kwargs
    assert kwargs["memory_type"] == "artifact"
    assert kwargs["metadata"]["artifact_id"] == "new-artifact-id"
    assert kwargs["content"].startswith("[artifact] ")


@pytest.mark.asyncio
async def test_inline_put_stores_text_inline_not_in_object_storage(seams):
    await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="document", content="hey"
    )

    kwargs = seams.artifacts.activate.call_args.kwargs
    assert kwargs["inline_text"] == "hey"
    assert kwargs["storage_uri"] is None
    # Measured server-side, never taken from the caller.
    assert kwargs["size_bytes"] == 3


@pytest.mark.asyncio
async def test_content_over_the_inline_cap_is_rejected_with_a_pointer(seams):
    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="document", content="x" * 65_537
    )

    assert result["stored"] is False
    assert result["error"] == "too_large_for_inline"
    seams.brain.write_scoped_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_over_pool_write_is_rejected_before_any_work(seams):
    seams.artifacts.org_storage_bytes.return_value = 100 * MB

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="document", content="hey"
    )

    assert result["stored"] is False
    assert result["error"] == "storage_quota_exceeded"
    seams.artifacts.create_pending.assert_not_called()
    seams.brain.write_scoped_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_bytes_return_the_existing_artifact(seams):
    seams.artifacts.find_active_by_checksum.return_value = {"id": "old", "version": 1}

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="document", content="hey"
    )

    assert result["artifact_id"] == "old"
    assert result["deduplicated"] is True
    seams.artifacts.create_pending.assert_not_called()


@pytest.mark.asyncio
async def test_auth_failure_returns_a_clean_error_not_an_exception():
    from fastapi import HTTPException

    with patch(
        "app.mcp_app.resolve_principal_plan",
        side_effect=HTTPException(status_code=401, detail="bad key"),
    ):
        result = await _tool("artifact_put")(
            _ctx(), title="t", summary="s", kind="document", content="hey"
        )

    assert result["stored"] is False
    assert result["error"] == "unauthorized"
```

- [ ] **Step 2: Run it to see it fail**

Run from `services/mcp-server`: `pytest tests/test_artifact_tools.py -v`
Expected: FAIL — `KeyError` / `ToolError` because no tool named `artifact_put` is registered.

- [ ] **Step 3: Add the imports**

At the top of `services/mcp-server/app/mcp_app.py`, beside the existing `shared` imports, add:

```python
from shared import artifact_storage, artifacts
```

- [ ] **Step 4: Implement the helper**

Add `_perform_artifact_put` to `services/mcp-server/app/mcp_app.py`, directly after `_perform_write`. This step implements only the inline and dedup paths; Task 7 adds the presigned and confirm branches at the marked point.

```python
async def _perform_artifact_put(
    ctx: Context,
    *,
    title: str,
    summary: str,
    kind: str,
    content: Optional[str] = None,
    mime_type: Optional[str] = None,
    size_bytes: Optional[int] = None,
    space: Optional[str] = None,
    project: Optional[str] = None,
    supersedes: Optional[str] = None,
    artifact_id: Optional[str] = None,
) -> dict[str, Any]:
    """Shared body for ``artifact_put``.

    Mirrors :func:`_perform_write`'s auth and session handling exactly. Three
    behaviours select on the arguments present: inline (``content`` given),
    presigned request (neither ``content`` nor ``artifact_id``), and confirm
    (``artifact_id`` given). Only inline is implemented here.

    The pool check runs before any write work so an over-quota org never
    reserves an id or uploads bytes.
    """
    request = ctx.request_context.request
    if request is None:
        logger.error("artifact_put got no transport request; cannot authenticate")
        return {"stored": False, "error": "unauthorized", "detail": "no request context"}

    _attach_client_name(ctx, request)

    try:
        org_id, agent_id, plan, plan_overrides = resolve_principal_plan(request)
    except HTTPException as exc:
        return {"stored": False, "error": "unauthorized", "detail": str(exc.detail)}

    cfg = get_plan_config(plan, plan_overrides)
    session_id = brain_session_id(request, org_id, agent_id)

    db = SessionLocal()
    try:
        if content is not None:
            body = content.encode("utf-8")

            if len(body) > artifacts.INLINE_MAX_BYTES:
                return {
                    "stored": False,
                    "error": "too_large_for_inline",
                    "detail": (
                        f"{len(body)} bytes exceeds the "
                        f"{artifacts.INLINE_MAX_BYTES}-byte inline limit. Call "
                        "artifact_put without `content` to get an upload URL."
                    ),
                }

            if len(body) > cfg.artifact_max_file_bytes:
                return {
                    "stored": False,
                    "error": "file_too_large",
                    "detail": (
                        f"{len(body)} bytes exceeds the {cfg.artifact_max_file_bytes}-byte "
                        f"per-file limit on the {plan} plan."
                    ),
                    "suggested_plan": "max",
                }

            used = artifacts.org_storage_bytes(db, org_id=org_id)
            if used + len(body) > cfg.artifact_storage_bytes:
                return {
                    "stored": False,
                    "error": "storage_quota_exceeded",
                    "detail": (
                        f"{used} of {cfg.artifact_storage_bytes} bytes used on the "
                        f"{plan} plan. Reads still work; forget an artifact or upgrade."
                    ),
                    "suggested_plan": "max",
                }

            checksum = hashlib.sha256(body).hexdigest()
            existing = artifacts.find_active_by_checksum(
                db, org_id=org_id, checksum=checksum
            )
            if existing:
                return {
                    "stored": True,
                    "artifact_id": existing["id"],
                    "version": existing["version"],
                    "status": "active",
                    "deduplicated": True,
                }

            stripped_space = space.strip() if space is not None else ""
            space_id = (
                brain.ensure_space(
                    db, org_id=org_id, name=stripped_space, slug=stripped_space
                )
                if stripped_space
                else None
            )
            project_id = resolve_project_from_request(request, db, org_id=org_id)

            new_id = artifacts.create_pending(
                db,
                org_id=org_id,
                agent_id=agent_id,
                title=title,
                summary=summary,
                kind=kind,
                mime_type=mime_type or "text/plain",
                space_id=space_id,
                project_id=project_id,
                supersedes=supersedes,
            )
            artifacts.activate(
                db,
                org_id=org_id,
                artifact_id=new_id,
                size_bytes=len(body),
                checksum=checksum,
                inline_text=content,
                storage_uri=None,
            )

            await _write_artifact_card(
                db,
                org_id=org_id,
                agent_id=agent_id,
                session_id=session_id,
                artifact_id=new_id,
                title=title,
                summary=summary,
                kind=kind,
                size_bytes=len(body),
                space_id=space_id,
                project_id=project_id,
            )

            if supersedes:
                artifacts.mark_superseded(db, org_id=org_id, artifact_id=supersedes)
                _hide_card_for_artifact(db, org_id=org_id, artifact_id=supersedes)

            return {
                "stored": True,
                "artifact_id": new_id,
                "version": 1,
                "status": "active",
            }

        # Presigned request and confirm branches land here in Task 7.
        return {
            "stored": False,
            "error": "not_implemented",
            "detail": "Provide `content` for an inline artifact.",
        }
    except Exception:
        logger.exception("artifact_put failed")
        return {"stored": False, "error": "internal_error"}
    finally:
        db.close()


async def _write_artifact_card(
    db: Any,
    *,
    org_id: str,
    agent_id: str,
    session_id: str,
    artifact_id: str,
    title: str,
    summary: str,
    kind: str,
    size_bytes: int,
    space_id: Optional[str],
    project_id: Optional[str],
) -> None:
    """Write the one memory row that makes an artifact discoverable.

    Dedup is disabled: two artifacts may legitimately carry near-identical
    summaries, and silently collapsing their cards would make one of them
    permanently unfindable.
    """
    await brain.write_scoped_memory(
        db,
        org_id=org_id,
        agent_id=agent_id,
        content=artifacts.card_content(title, summary, kind, size_bytes),
        memory_type="artifact",
        session_id=session_id,
        space_id=space_id,
        project_id=project_id,
        metadata={"artifact_id": artifact_id, "source": "mcp"},
        dedup=False,
        scope="org",
    )


def _hide_card_for_artifact(db: Any, *, org_id: str, artifact_id: str) -> None:
    """Hide an artifact's card from recall without deleting the row."""
    db.execute(
        sa_text(
            "UPDATE session_memories SET recall_hidden = TRUE "
            "WHERE org_id = :org_id AND memory_type = 'artifact' "
            "AND metadata->>'artifact_id' = :artifact_id"
        ),
        {"org_id": org_id, "artifact_id": artifact_id},
    )
    db.commit()
```

Add these two imports at the top of the file if they are not already present:

```python
import hashlib
from sqlalchemy import text as sa_text
```

- [ ] **Step 5: Register the tool**

Inside `_register_tools`, after the existing `note` tool registration, add:

```python
    @mcp.tool(
        name="artifact_put",
        description=(
            "Store a durable artifact — a document, a JSON record, or a file — "
            "and make it discoverable to every agent in the org. Pass `content` "
            "for text up to 64KB (one call). Omit `content` to get an upload URL "
            "for larger or binary data, then call again with the returned "
            "`artifact_id` to confirm. Args: title, summary, kind "
            "('document'|'record'|'file')."
        ),
    )
    async def artifact_put(
        ctx: Context,
        title: Annotated[str, Field(description="Short name for the artifact.", max_length=200)],
        summary: Annotated[
            str,
            Field(
                description="One line describing what this is; other agents find it by this.",
                max_length=500,
            ),
        ],
        kind: Annotated[
            Literal["document", "record", "file"],
            Field(description="'document' (prose), 'record' (JSON), or 'file' (binary)."),
        ],
        content: Annotated[
            Optional[str],
            Field(description="Inline text, up to 64KB. Omit for the upload-URL path."),
        ] = None,
        mime_type: Annotated[
            Optional[str], Field(description="MIME type. Required when content is omitted.")
        ] = None,
        size_bytes: Annotated[
            Optional[int], Field(description="Declared size; selects the upload strategy.")
        ] = None,
        space: SpaceParam = None,
        project: Annotated[Optional[str], Field(description="Optional project id.")] = None,
        supersedes: Annotated[
            Optional[str], Field(description="Artifact id this replaces.")
        ] = None,
        artifact_id: Annotated[
            Optional[str], Field(description="Confirm an upload started earlier.")
        ] = None,
    ) -> dict[str, Any]:
        return await _perform_artifact_put(
            ctx,
            title=title,
            summary=summary,
            kind=kind,
            content=content,
            mime_type=mime_type,
            size_bytes=size_bytes,
            space=space,
            project=project,
            supersedes=supersedes,
            artifact_id=artifact_id,
        )
```

- [ ] **Step 6: Run the test to verify it passes**

Run from `services/mcp-server`: `pytest tests/test_artifact_tools.py -v`
Expected: 6 passed.

- [ ] **Step 7: Run the whole mcp-server suite for regressions**

Run from `services/mcp-server`: `pytest -q`
Expected: no new failures versus `main`.

- [ ] **Step 8: Commit**

```bash
git add services/mcp-server/app/mcp_app.py services/mcp-server/tests/test_artifact_tools.py
git commit -m "feat(mcp): artifact_put inline path with card write"
```

---

## Task 7: `artifact_put` — presigned upload and confirm

**Files:**
- Modify: `services/mcp-server/app/mcp_app.py`
- Test: `services/mcp-server/tests/test_artifact_tools.py`

**Interfaces:**
- Consumes: `_perform_artifact_put` and `_write_artifact_card` from Task 6; `artifact_storage.presign_put`, `.head`, `.object_key`, `.storage_uri`, `.MULTIPART_THRESHOLD_BYTES` from Task 4.
- Produces: two additional return shapes from `artifact_put` — `{"stored": False, "artifact_id": str, "upload_url": str, "expires_in": 900, "method": "PUT"}` for a request, and the Task 6 success shape for a confirm.

- [ ] **Step 1: Write the failing tests**

Append to `services/mcp-server/tests/test_artifact_tools.py`:

```python
@pytest.fixture
def storage(seams):
    """Add a patched artifact_storage to the existing seams."""
    with patch("app.mcp_app.artifact_storage") as st:
        st.MULTIPART_THRESHOLD_BYTES = 100 * MB
        st.PRESIGN_TTL_SECONDS = 900
        st.object_key.return_value = "acme/new-artifact-id/v1"
        st.storage_uri.return_value = "s3://klio-artifacts/acme/new-artifact-id/v1"
        st.presign_put.return_value = "https://example/upload"
        st.head.return_value = {"size_bytes": 2048, "checksum": "abc"}
        st.get_artifact_s3_client.return_value = MagicMock()
        yield SimpleNamespace(storage=st, artifacts=seams.artifacts, brain=seams.brain)


@pytest.mark.asyncio
async def test_request_returns_an_upload_url_and_writes_no_card(storage):
    result = await _tool("artifact_put")(
        _ctx(),
        title="t",
        summary="s",
        kind="file",
        mime_type="application/pdf",
        size_bytes=2048,
    )

    assert result["artifact_id"] == "new-artifact-id"
    assert result["upload_url"] == "https://example/upload"
    assert result["expires_in"] == 900
    # No card until the bytes actually exist — an abandoned upload must never
    # appear in recall.
    storage.brain.write_scoped_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirm_measures_the_object_and_writes_the_card(storage):
    storage.artifacts.get_artifact.return_value = {
        "id": "new-artifact-id",
        "status": "pending",
        "title": "t",
        "summary": "s",
        "kind": "file",
        "version": 1,
    }

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="file", artifact_id="new-artifact-id"
    )

    assert result["stored"] is True
    assert result["status"] == "active"
    # The measured size wins, not anything the client declared.
    assert storage.artifacts.activate.call_args.kwargs["size_bytes"] == 2048
    storage.brain.write_scoped_memory.assert_awaited_once()


@pytest.mark.asyncio
async def test_confirm_rejects_bytes_that_exceed_the_per_file_limit(storage):
    storage.storage.head.return_value = {"size_bytes": 50 * MB, "checksum": "abc"}
    storage.artifacts.get_artifact.return_value = {
        "id": "new-artifact-id",
        "status": "pending",
        "title": "t",
        "summary": "s",
        "kind": "file",
        "version": 1,
    }

    # Declared 1KB, actually uploaded 50MB against a 10MB plan limit.
    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="file", artifact_id="new-artifact-id"
    )

    assert result["stored"] is False
    assert result["error"] == "file_too_large"
    storage.artifacts.activate.assert_not_called()
    storage.brain.write_scoped_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirm_without_uploaded_bytes_is_rejected(storage):
    storage.storage.head.return_value = None
    storage.artifacts.get_artifact.return_value = {
        "id": "new-artifact-id",
        "status": "pending",
        "title": "t",
        "summary": "s",
        "kind": "file",
        "version": 1,
    }

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="file", artifact_id="new-artifact-id"
    )

    assert result["stored"] is False
    assert result["error"] == "no_bytes_uploaded"


@pytest.mark.asyncio
async def test_confirming_another_orgs_artifact_reads_as_not_found(storage):
    storage.artifacts.get_artifact.return_value = None

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="file", artifact_id="someone-elses"
    )

    assert result["stored"] is False
    assert result["error"] == "not_found"
```

- [ ] **Step 2: Run to see them fail**

Run from `services/mcp-server`: `pytest tests/test_artifact_tools.py -v`
Expected: the five new tests FAIL with `assert result["error"] == "not_implemented"` mismatches.

- [ ] **Step 3: Replace the not-implemented branch**

In `_perform_artifact_put`, replace the `# Presigned request and confirm branches land here in Task 7.` block with:

```python
        # ── Confirm: the client says the bytes are uploaded ──────────────
        if artifact_id is not None:
            row = artifacts.get_artifact(db, org_id=org_id, artifact_id=artifact_id)
            if row is None or row["status"] != "pending":
                # A cross-org id reads as not-found, never as a permission error.
                return {"stored": False, "error": "not_found"}

            s3 = artifact_storage.get_artifact_s3_client()
            key = artifact_storage.object_key(org_id, artifact_id, row["version"])
            measured = artifact_storage.head(s3, key)
            if measured is None:
                return {
                    "stored": False,
                    "error": "no_bytes_uploaded",
                    "detail": "No object found at the upload URL. Upload, then confirm.",
                }

            # The declared size only chose a strategy; the quota is charged on
            # what was actually stored.
            if measured["size_bytes"] > cfg.artifact_max_file_bytes:
                return {
                    "stored": False,
                    "error": "file_too_large",
                    "detail": (
                        f"{measured['size_bytes']} bytes exceeds the "
                        f"{cfg.artifact_max_file_bytes}-byte per-file limit on the "
                        f"{plan} plan."
                    ),
                    "suggested_plan": "max",
                }

            used = artifacts.org_storage_bytes(db, org_id=org_id)
            if used + measured["size_bytes"] > cfg.artifact_storage_bytes:
                return {
                    "stored": False,
                    "error": "storage_quota_exceeded",
                    "detail": (
                        f"{used} of {cfg.artifact_storage_bytes} bytes used on the "
                        f"{plan} plan."
                    ),
                    "suggested_plan": "max",
                }

            artifacts.activate(
                db,
                org_id=org_id,
                artifact_id=artifact_id,
                size_bytes=measured["size_bytes"],
                checksum=measured["checksum"],
                inline_text=None,
                storage_uri=artifact_storage.storage_uri(
                    org_id, artifact_id, row["version"]
                ),
            )

            await _write_artifact_card(
                db,
                org_id=org_id,
                agent_id=agent_id,
                session_id=session_id,
                artifact_id=artifact_id,
                title=row["title"],
                summary=row["summary"],
                kind=row["kind"],
                size_bytes=measured["size_bytes"],
                space_id=None,
                project_id=None,
            )

            if supersedes:
                artifacts.mark_superseded(db, org_id=org_id, artifact_id=supersedes)
                _hide_card_for_artifact(db, org_id=org_id, artifact_id=supersedes)

            return {
                "stored": True,
                "artifact_id": artifact_id,
                "version": row["version"],
                "status": "active",
            }

        # ── Request: reserve an id and hand back an upload URL ───────────
        if not mime_type:
            return {
                "stored": False,
                "error": "mime_type_required",
                "detail": "Provide `mime_type` when omitting `content`.",
            }

        declared = size_bytes or 0
        if declared > cfg.artifact_max_file_bytes:
            return {
                "stored": False,
                "error": "file_too_large",
                "detail": (
                    f"{declared} bytes exceeds the {cfg.artifact_max_file_bytes}-byte "
                    f"per-file limit on the {plan} plan."
                ),
                "suggested_plan": "max",
            }

        stripped_space = space.strip() if space is not None else ""
        space_id = (
            brain.ensure_space(
                db, org_id=org_id, name=stripped_space, slug=stripped_space
            )
            if stripped_space
            else None
        )

        new_id = artifacts.create_pending(
            db,
            org_id=org_id,
            agent_id=agent_id,
            title=title,
            summary=summary,
            kind=kind,
            mime_type=mime_type,
            space_id=space_id,
            project_id=resolve_project_from_request(request, db, org_id=org_id),
            supersedes=supersedes,
        )

        s3 = artifact_storage.get_artifact_s3_client()
        key = artifact_storage.object_key(org_id, new_id, 1)

        if declared > artifact_storage.MULTIPART_THRESHOLD_BYTES:
            return {
                "stored": False,
                "artifact_id": new_id,
                "multipart": True,
                "detail": (
                    "Over 100MB. Use the Klio CLI, which performs the multipart "
                    "upload, then confirm with this artifact_id."
                ),
                "expires_in": artifact_storage.PRESIGN_TTL_SECONDS,
            }

        return {
            "stored": False,
            "artifact_id": new_id,
            "upload_url": artifact_storage.presign_put(s3, key),
            "method": "PUT",
            "expires_in": artifact_storage.PRESIGN_TTL_SECONDS,
            "detail": "Upload the bytes, then call artifact_put again with this artifact_id.",
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run from `services/mcp-server`: `pytest tests/test_artifact_tools.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add services/mcp-server/app/mcp_app.py services/mcp-server/tests/test_artifact_tools.py
git commit -m "feat(mcp): presigned upload and confirm for artifact_put"
```

---

## Task 8: `artifact_get`

**Files:**
- Modify: `services/mcp-server/app/mcp_app.py`
- Test: `services/mcp-server/tests/test_artifact_tools.py`

**Interfaces:**
- Consumes: `artifacts.get_artifact`, `artifacts.TEXT_PAGE_BYTES`, `artifact_storage.presign_get`, `.get_text`, `.object_key`.
- Produces: `async def _perform_artifact_get(ctx, *, artifact_id, mode, offset) -> dict[str, Any]` and a registered `artifact_get` tool.
  - `mode='text'` → `{"artifact_id", "title", "kind", "text", "offset", "has_more"}`
  - `mode='url'` → `{"artifact_id", "title", "kind", "url", "expires_in"}`

- [ ] **Step 1: Write the failing tests**

Append to `services/mcp-server/tests/test_artifact_tools.py`:

```python
@pytest.mark.asyncio
async def test_get_returns_inline_text(storage):
    storage.artifacts.TEXT_PAGE_BYTES = 51_200
    storage.artifacts.get_artifact.return_value = {
        "id": "a1", "status": "active", "title": "t", "kind": "document",
        "mime_type": "text/markdown", "inline_text": "hello", "storage_uri": None,
        "version": 1, "size_bytes": 5,
    }

    result = await _tool("artifact_get")(_ctx(), artifact_id="a1")

    assert result["text"] == "hello"
    assert result["has_more"] is False


@pytest.mark.asyncio
async def test_get_pages_long_text_and_reports_more(storage):
    storage.artifacts.TEXT_PAGE_BYTES = 10
    storage.artifacts.get_artifact.return_value = {
        "id": "a1", "status": "active", "title": "t", "kind": "document",
        "mime_type": "text/plain", "inline_text": "abcdefghijKLMNO",
        "storage_uri": None, "version": 1, "size_bytes": 15,
    }

    first = await _tool("artifact_get")(_ctx(), artifact_id="a1")
    assert first["text"] == "abcdefghij"
    assert first["has_more"] is True

    second = await _tool("artifact_get")(_ctx(), artifact_id="a1", offset=10)
    assert second["text"] == "KLMNO"
    assert second["has_more"] is False


@pytest.mark.asyncio
async def test_get_text_on_a_binary_points_at_url_mode(storage):
    storage.artifacts.get_artifact.return_value = {
        "id": "a1", "status": "active", "title": "t", "kind": "file",
        "mime_type": "application/pdf", "inline_text": None,
        "storage_uri": "s3://klio-artifacts/acme/a1/v1", "version": 1,
        "size_bytes": 2048,
    }

    result = await _tool("artifact_get")(_ctx(), artifact_id="a1", mode="text")

    # v1 does no extraction; saying so beats returning mojibake.
    assert result["error"] == "not_text"
    assert "url" in result["detail"]


@pytest.mark.asyncio
async def test_get_url_returns_a_short_lived_signed_url(storage):
    storage.storage.presign_get.return_value = "https://example/download"
    storage.artifacts.get_artifact.return_value = {
        "id": "a1", "status": "active", "title": "t", "kind": "file",
        "mime_type": "application/pdf", "inline_text": None,
        "storage_uri": "s3://klio-artifacts/acme/a1/v1", "version": 1,
        "size_bytes": 2048,
    }

    result = await _tool("artifact_get")(_ctx(), artifact_id="a1", mode="url")

    assert result["url"] == "https://example/download"
    assert result["expires_in"] == 900


@pytest.mark.asyncio
async def test_get_on_another_orgs_artifact_reads_as_not_found(storage):
    storage.artifacts.get_artifact.return_value = None
    result = await _tool("artifact_get")(_ctx(), artifact_id="someone-elses")
    assert result["error"] == "not_found"


@pytest.mark.asyncio
async def test_get_on_a_retracted_artifact_reads_as_not_found(storage):
    storage.artifacts.get_artifact.return_value = {
        "id": "a1", "status": "retracted", "title": "t", "kind": "document",
        "mime_type": "text/plain", "inline_text": "gone", "storage_uri": None,
        "version": 1, "size_bytes": 4,
    }
    result = await _tool("artifact_get")(_ctx(), artifact_id="a1")
    assert result["error"] == "not_found"
```

- [ ] **Step 2: Run to see them fail**

Run from `services/mcp-server`: `pytest tests/test_artifact_tools.py -v`
Expected: the six new tests FAIL — no tool named `artifact_get`.

- [ ] **Step 3: Implement the helper**

Add to `services/mcp-server/app/mcp_app.py`, after `_perform_artifact_put`:

```python
_TEXT_MIME_PREFIXES = ("text/",)
_TEXT_MIME_EXACT = {"application/json", "application/xml", "application/yaml"}


def _is_text_mime(mime_type: str) -> bool:
    return mime_type.startswith(_TEXT_MIME_PREFIXES) or mime_type in _TEXT_MIME_EXACT


async def _perform_artifact_get(
    ctx: Context,
    *,
    artifact_id: str,
    mode: str = "text",
    offset: int = 0,
) -> dict[str, Any]:
    """Shared body for ``artifact_get``.

    Text is returned one bounded page at a time: a recall result can point at a
    multi-megabyte artifact, and a context window is the real constraint.
    Binary artifacts are never guessed at — v1 does no extraction, so the caller
    is told to use ``mode='url'``.
    """
    request = ctx.request_context.request
    if request is None:
        return {"error": "unauthorized", "detail": "no request context"}

    _attach_client_name(ctx, request)

    try:
        org_id, _agent_id, _plan, _overrides = resolve_principal_plan(request)
    except HTTPException as exc:
        return {"error": "unauthorized", "detail": str(exc.detail)}

    db = SessionLocal()
    try:
        row = artifacts.get_artifact(db, org_id=org_id, artifact_id=artifact_id)
        if row is None or row["status"] not in ("active", "superseded"):
            return {"error": "not_found"}

        head = {
            "artifact_id": row["id"],
            "title": row["title"],
            "kind": row["kind"],
        }

        if mode == "url":
            s3 = artifact_storage.get_artifact_s3_client()
            key = artifact_storage.object_key(org_id, row["id"], row["version"])
            if row["inline_text"] is not None:
                # Inline artifacts have no object yet; materialise one so the
                # URL contract holds for every kind.
                artifact_storage.put_text(
                    s3, key, row["inline_text"], row["mime_type"]
                )
            return {
                **head,
                "url": artifact_storage.presign_get(s3, key),
                "expires_in": artifact_storage.PRESIGN_TTL_SECONDS,
            }

        if row["inline_text"] is not None:
            full = row["inline_text"]
        elif _is_text_mime(row["mime_type"]):
            s3 = artifact_storage.get_artifact_s3_client()
            key = artifact_storage.object_key(org_id, row["id"], row["version"])
            full = artifact_storage.get_text(s3, key)
        else:
            return {
                **head,
                "error": "not_text",
                "detail": (
                    f"{row['mime_type']} is not text. Call artifact_get with "
                    "mode='url' to download it."
                ),
            }

        page = full[offset : offset + artifacts.TEXT_PAGE_BYTES]
        return {
            **head,
            "text": page,
            "offset": offset,
            "has_more": (offset + len(page)) < len(full),
        }
    except Exception:
        logger.exception("artifact_get failed")
        return {"error": "internal_error"}
    finally:
        db.close()
```

- [ ] **Step 4: Register the tool**

Inside `_register_tools`, after `artifact_put`, add:

```python
    @mcp.tool(
        name="artifact_get",
        description=(
            "Fetch an artifact surfaced by recall. mode='text' returns its "
            "contents one page at a time (use `offset` to continue); mode='url' "
            "returns a short-lived download link for binary content."
        ),
    )
    async def artifact_get(
        ctx: Context,
        artifact_id: Annotated[str, Field(description="From a recall result's metadata.")],
        mode: Annotated[
            Literal["text", "url"],
            Field(description="'text' (default) or 'url' for a download link."),
        ] = "text",
        offset: Annotated[
            int, Field(description="Character offset for paging text.", ge=0)
        ] = 0,
    ) -> dict[str, Any]:
        return await _perform_artifact_get(
            ctx, artifact_id=artifact_id, mode=mode, offset=offset
        )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run from `services/mcp-server`: `pytest tests/test_artifact_tools.py -v`
Expected: 17 passed.

- [ ] **Step 6: Commit**

```bash
git add services/mcp-server/app/mcp_app.py services/mcp-server/tests/test_artifact_tools.py
git commit -m "feat(mcp): artifact_get with paged text and signed URLs"
```

---

## Task 9: `forget` an artifact, and tell agents the follow-up exists

**Files:**
- Modify: `services/mcp-server/app/mcp_app.py`
- Test: `services/mcp-server/tests/test_artifact_tools.py`

**Interfaces:**
- Consumes: `artifacts.retract`, `_hide_card_for_artifact` from Task 6.
- Produces: no new symbols. `forget` gains an artifact branch; `recall`'s description gains one sentence.

- [ ] **Step 1: Write the failing tests**

Append to `services/mcp-server/tests/test_artifact_tools.py`:

```python
@pytest.mark.asyncio
async def test_forget_retracts_the_artifact_and_hides_its_card(storage):
    storage.artifacts.retract.return_value = True

    result = await _tool("forget")(_ctx(), memory_id="a1")

    assert result["forgotten"] is True
    storage.artifacts.retract.assert_called_once()
    assert storage.artifacts.retract.call_args.kwargs["org_id"] == "acme"


@pytest.mark.asyncio
async def test_recall_description_tells_agents_about_artifact_get():
    tool = build_mcp_server()._tool_manager.get_tool("recall")
    # Without this, an agent receives an artifact card and has no idea a
    # follow-up tool exists.
    assert "artifact_get" in tool.description
```

- [ ] **Step 2: Run to see them fail**

Run from `services/mcp-server`: `pytest tests/test_artifact_tools.py -k "forget or recall_description" -v`
Expected: both FAIL.

- [ ] **Step 3: Extend `forget`**

In the existing `forget` tool body in `services/mcp-server/app/mcp_app.py`, before the memory-retraction logic runs, add the artifact branch:

```python
        # An id may name an artifact or a memory. Try the artifact first: a hit
        # must retract the bytes AND hide the card, or recall keeps surfacing a
        # card pointing at something that is gone.
        if artifacts.retract(db, org_id=org_id, artifact_id=memory_id):
            _hide_card_for_artifact(db, org_id=org_id, artifact_id=memory_id)
            return {"forgotten": True, "kind": "artifact", "id": memory_id}
```

If `retract` raises because the id is not a valid UUID, catch and fall through to the memory path:

```python
        try:
            if artifacts.retract(db, org_id=org_id, artifact_id=memory_id):
                _hide_card_for_artifact(db, org_id=org_id, artifact_id=memory_id)
                return {"forgotten": True, "kind": "artifact", "id": memory_id}
        except Exception:
            logger.debug("forget: id %s is not an artifact, trying memory", memory_id)
```

- [ ] **Step 4: Update `recall`'s description**

In the `@mcp.tool(name="recall", ...)` decorator, append this sentence to the existing `description` string:

```python
            "Results with memory_type 'artifact' are stored files or documents: "
            "call artifact_get with the metadata's artifact_id to read them."
```

- [ ] **Step 5: Run the tests to verify they pass**

Run from `services/mcp-server`: `pytest tests/test_artifact_tools.py -v`
Expected: 19 passed.

- [ ] **Step 6: Commit**

```bash
git add services/mcp-server/app/mcp_app.py services/mcp-server/tests/test_artifact_tools.py
git commit -m "feat(mcp): forget an artifact; point recall at artifact_get"
```

---

## Task 10: Collect abandoned uploads

**Files:**
- Modify: `services/mcp-server/app/scheduler.py`
- Test: `services/mcp-server/tests/test_artifact_gc.py`

**Interfaces:**
- Consumes: `artifacts.collect_abandoned`, `artifact_storage.object_key`, `.delete`, `.get_artifact_s3_client`.
- Produces: `async def collect_artifacts_once(db_factory) -> int` (returns how many were collected) and `start_artifact_gc(app, factory, interval_seconds) -> None`, mirroring the existing `start_curator` shape.

- [ ] **Step 1: Write the failing test**

Create `services/mcp-server/tests/test_artifact_gc.py`:

```python
"""Abandoned-upload collection.

An artifact_put that hands out an upload URL and is never confirmed leaves a
pending row and possibly orphaned bytes. Neither is reachable — no card was
written — so both are collected after 24 hours.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.scheduler import collect_artifacts_once


@pytest.mark.asyncio
async def test_collects_pending_rows_and_deletes_their_objects():
    db = MagicMock()
    with (
        patch("app.scheduler.artifacts") as af,
        patch("app.scheduler.artifact_storage") as st,
    ):
        af.collect_abandoned.return_value = [
            {"id": "a1", "org_id": "acme", "version": 1},
            {"id": "a2", "org_id": "acme", "version": 1},
        ]
        st.object_key.side_effect = lambda o, a, v: f"{o}/{a}/v{v}"
        st.get_artifact_s3_client.return_value = MagicMock()

        collected = await collect_artifacts_once(lambda: db)

    assert collected == 2
    assert st.delete.call_count == 2


@pytest.mark.asyncio
async def test_a_missing_object_does_not_stop_the_sweep():
    db = MagicMock()
    with (
        patch("app.scheduler.artifacts") as af,
        patch("app.scheduler.artifact_storage") as st,
    ):
        af.collect_abandoned.return_value = [
            {"id": "a1", "org_id": "acme", "version": 1},
            {"id": "a2", "org_id": "acme", "version": 1},
        ]
        st.get_artifact_s3_client.return_value = MagicMock()
        st.delete.side_effect = [Exception("gone"), None]

        collected = await collect_artifacts_once(lambda: db)

    # The row is deleted even when its object was never uploaded — that is the
    # common case for an abandoned upload.
    assert collected == 2


@pytest.mark.asyncio
async def test_nothing_to_collect_is_not_an_error():
    db = MagicMock()
    with (
        patch("app.scheduler.artifacts") as af,
        patch("app.scheduler.artifact_storage"),
    ):
        af.collect_abandoned.return_value = []
        assert await collect_artifacts_once(lambda: db) == 0
```

- [ ] **Step 2: Run to see it fail**

Run from `services/mcp-server`: `pytest tests/test_artifact_gc.py -v`
Expected: FAIL — `ImportError: cannot import name 'collect_artifacts_once'`.

- [ ] **Step 3: Implement the sweep**

Add to `services/mcp-server/app/scheduler.py`:

```python
from sqlalchemy import text as sa_text

from shared import artifact_storage, artifacts


async def collect_artifacts_once(db_factory: Any) -> int:
    """Delete pending artifacts whose upload never arrived, and their bytes.

    An abandoned upload has no card, so nothing can reach it — it is invisible
    storage the org would otherwise be charged for. Object deletion is
    best-effort: the common case is that no object was ever uploaded, and a
    missing object must not block the row cleanup.
    """
    db = db_factory()
    try:
        stale = artifacts.collect_abandoned(db)
        if not stale:
            return 0

        s3 = artifact_storage.get_artifact_s3_client()
        for row in stale:
            key = artifact_storage.object_key(row["org_id"], row["id"], row["version"])
            try:
                artifact_storage.delete(s3, key)
            except Exception:
                logger.debug("artifact GC: no object at %s", key)

        db.execute(
            sa_text("DELETE FROM artifacts WHERE id = ANY(:ids)"),
            {"ids": [row["id"] for row in stale]},
        )
        db.commit()
        logger.info("artifact GC: collected %d abandoned uploads", len(stale))
        return len(stale)
    finally:
        db.close()


async def _artifact_gc_loop(db_factory: Any, interval_seconds: int) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await collect_artifacts_once(db_factory)
        except Exception:
            logger.exception("artifact GC tick failed")


def start_artifact_gc(app: Any, db_factory: Any, interval_seconds: int = 3600) -> None:
    """Start the hourly abandoned-upload sweep. Mirrors ``start_curator``."""
    app.state.artifact_gc_task = asyncio.create_task(
        _artifact_gc_loop(db_factory, interval_seconds)
    )
```

- [ ] **Step 4: Wire it into startup**

In `services/mcp-server/app/main.py`, find the existing `start_curator(...)` call in the lifespan/startup handler and add immediately after it:

```python
    start_artifact_gc(app, SessionLocal)
```

Add `start_artifact_gc` to the existing `from app.scheduler import ...` line in that file.

- [ ] **Step 5: Run the tests to verify they pass**

Run from `services/mcp-server`: `pytest tests/test_artifact_gc.py -v`
Expected: 3 passed.

- [ ] **Step 6: Run both suites**

Run from `services/mcp-server`: `pytest -q`
Then run from `services/shared`: `pytest -q`
Expected: no new failures versus `main`.

- [ ] **Step 7: Commit**

```bash
git add services/mcp-server/app/scheduler.py services/mcp-server/app/main.py services/mcp-server/tests/test_artifact_gc.py
git commit -m "feat(mcp): collect abandoned artifact uploads"
```

---

## Task 11: End-to-end proof against a live stack

**Files:**
- Create: `services/mcp-server/tests/test_artifact_e2e.py`

**Interfaces:**
- Consumes: every tool from Tasks 6–9.
- Produces: nothing importable. This task exists because every prior test patched the DB and S3 seams, so nothing so far proves the pieces fit together.

- [ ] **Step 1: Bring up the stack**

Run: `docker compose up -d postgres minio`
Then create the bucket: `docker compose exec minio mc mb local/klio-artifacts || true`

If the compose file names these services differently, run `docker compose config --services` and substitute. If MinIO has no `mc` in the image, create the bucket with:

```bash
python -c "
from shared.artifact_storage import get_artifact_s3_client, ARTIFACT_BUCKET
get_artifact_s3_client().create_bucket(Bucket=ARTIFACT_BUCKET)
"
```

- [ ] **Step 2: Write the end-to-end test**

Create `services/mcp-server/tests/test_artifact_e2e.py`:

```python
"""Live-stack proof: agent A stores, agent B discovers and reads.

Requires postgres and minio from docker-compose plus the klio-artifacts
bucket. Skipped when KLIO_E2E is not set, so CI without a stack stays green.
"""

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("KLIO_E2E"), reason="set KLIO_E2E=1 with a live stack"
)


@pytest.mark.asyncio
async def test_agent_b_recalls_an_artifact_agent_a_wrote():
    from types import SimpleNamespace

    from app.mcp_app import build_mcp_server

    server = build_mcp_server()
    put = server._tool_manager.get_tool("artifact_put").fn
    get = server._tool_manager.get_tool("artifact_get").fn
    recall = server._tool_manager.get_tool("recall").fn

    marker = uuid.uuid4().hex[:8]

    def ctx(agent):
        return SimpleNamespace(
            request_context=SimpleNamespace(
                request=SimpleNamespace(
                    headers={"x-vex-key": os.environ["KLIO_E2E_KEY"], "x-vex-agent": agent}
                )
            )
        )

    stored = await put(
        ctx("agent-a"),
        title=f"Rate limiter design {marker}",
        summary=f"Token bucket at the edge, decision {marker}",
        kind="document",
        content="# Rate limiting\n\nToken bucket, 100 rps per key.",
    )
    assert stored["stored"] is True

    # Agent B has never seen this artifact and searches semantically.
    found = await recall(ctx("agent-b"), query="how do we rate limit", limit=25)
    cards = [
        m for m in found.get("memories", [])
        if m.get("memory_type") == "artifact" and marker in m.get("content", "")
    ]
    assert cards, "agent B could not discover agent A's artifact through recall"

    artifact_id = cards[0]["metadata"]["artifact_id"]
    body = await get(ctx("agent-b"), artifact_id=artifact_id)
    assert "Token bucket" in body["text"]

    link = await get(ctx("agent-b"), artifact_id=artifact_id, mode="url")
    assert link["url"].startswith("http")
    assert link["expires_in"] == 900
```

- [ ] **Step 3: Run it**

Run from `services/mcp-server`:
`KLIO_E2E=1 KLIO_E2E_KEY=<a dev API key for a test org> pytest tests/test_artifact_e2e.py -v`

Expected: 1 passed. A failure here means the pieces do not fit even though every unit test passes — debug before shipping.

- [ ] **Step 4: Verify the card survives retention on a 1-day plan**

Run: `psql "$AGENTGUARD_DATABASE_URL" -c "SELECT enforce_memory_retention('<test-org>', 1)"`
Then re-run the recall assertion from Step 3.
Expected: the artifact card is still discoverable. If it vanished, migration 028 did not take effect.

- [ ] **Step 5: Commit**

```bash
git add services/mcp-server/tests/test_artifact_e2e.py
git commit -m "test(artifacts): end-to-end discovery across two agents"
```

---

## Follow-ups (deliberately not in this plan)

Each needs its own spec, per the design doc's non-goals:

- **Text extraction from binaries** — `artifact_get(mode='text')` on a PDF returns `not_text` today.
- **Dashboard Artifacts page** — list, preview, download, forget.
- **CLI `klio push` / `klio pull`** — including the real multipart upload that Task 7 currently defers to it.
- **`pricing.ts` mirror** — `PlanConfig` carries a "keep in sync" comment; the landing-site plan data and the Max tier's public copy are a separate change in the Dashboard repo.
- **`pro` / `team` storage values** — left at `starter` levels; must be set deliberately before Max ships publicly.
