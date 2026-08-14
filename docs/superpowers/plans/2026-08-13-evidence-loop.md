# Evidence Loop (Builds 1+2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retain redacted session transcripts + tool traces (Build 1) and join Vex verdicts to the memories served in each recall (Build 2), so dreaming has fuel and "prove the memory worked" becomes a query.

**Architecture:** A new `session_traces` table + S3 object per session, written synchronously at the end of `capture_transcript` (reusing the storage-worker's boto3 client factory). `brain_recall_events` gains a `memory_ids` array + `session_id`; a new `recall_outcomes` table joins recall events to verification verdicts, graded by a new async-worker job that reads the retained trace and runs the existing `engine.pipeline.verify`. Retroactive stale-serve marking rides the existing supersession path.

**Tech Stack:** FastAPI (mcp-server), SQLAlchemy Core + Alembic (migrations service), boto3 (S3), pytest + the repo's mocked-pool test policy, verification-engine as a library.

## Global Constraints

- **ADDITIVE ONLY migrations.** Next revision is `041` (latest is `040_project_access_roles.py`). No column drops, no rewrites of existing rows.
- **Redaction before persistence, always.** The transcript must pass `shared.redaction.redact` BEFORE the S3 put — the same chokepoint contract documented in `capture.py:469-499`. Record `redaction_version` so a redactor bug can be scoped.
- **Best-effort logging never breaks the request.** Trace persistence and recall logging follow `recall_log.py`'s pattern: catch, rollback, `logger.warning`, continue. A failed trace write must not fail `capture_transcript`.
- **Tenant boundary is absolute.** Every new table carries `org_id`; every query predicates on it. The grader job partitions strictly per org.
- **Mocked-pool test policy:** the pool mock never evaluates SQL, so tests assert SQL shape + Python assembly. DB-gated tests use `KLIO_REQUIRE_DB=1` and a migrated database.
- **Free-tier retention applies to traces:** traces are deleted by the same sweep that deletes memories (`run_retention.py` guards: non-positive retention = skip).
- **No new services.** Build 1 rides mcp-server + the existing S3 client factory pattern; Build 2 rides async-worker.
- Engine freeze is lifted for this work by explicit decision (spec approved 2026-08-13).

---

## File Structure

| File | Responsibility |
| --- | --- |
| `services/migrations/alembic/versions/041_session_traces.py` | `session_traces` table |
| `services/migrations/alembic/versions/042_recall_evidence.py` | `memory_ids[]` + `session_id` on `brain_recall_events`; `recall_outcomes` table |
| `services/shared/shared/traces.py` | Write/read API for traces: S3 put/get + row insert (server-side of Build 1) |
| `services/mcp-server/app/capture.py` | Call `traces.persist_trace` at the end of `capture_transcript` |
| `services/mcp-server/app/recall_log.py` | Log served `memory_ids` + `session_id` |
| `services/async-worker/app/outcome_grader.py` | Grade session outputs, write `recall_outcomes` |
| `services/shared/shared/memory.py` | On supersede: mark later serves of the old row `served_stale` |

---

### Task 1: Migration 041 — `session_traces`

**Files:**
- Create: `services/migrations/alembic/versions/041_session_traces.py`
- Test: `services/migrations/tests/test_041_session_traces.py`

**Interfaces:**
- Produces table `session_traces`: `id UUID pk`, `org_id String(64) NOT NULL`, `project_id UUID NULL`, `user_id UUID NULL`, `agent_id Text NULL`, `session_id Text NOT NULL`, `started_at timestamptz NULL`, `ended_at timestamptz NULL`, `transcript_ref Text NOT NULL`, `tool_calls_ref Text NULL`, `redaction_version Text NOT NULL`, `retention_class Text NOT NULL DEFAULT 'plan'`, `created_at timestamptz NOT NULL DEFAULT now()`. Indexes: `(org_id, created_at)`, unique `(org_id, session_id)`.

- [ ] **Step 1: Write the failing test**

```python
# services/migrations/tests/test_041_session_traces.py
"""Migration 041 shape test. DB-gated: requires a migrated database."""
import os

import pytest
import sqlalchemy as sa

pytestmark = pytest.mark.skipif(
    not os.environ.get("KLIO_REQUIRE_DB"), reason="needs migrated DB"
)


def _inspector():
    engine = sa.create_engine(os.environ["DATABASE_URL"])
    return sa.inspect(engine)


def test_session_traces_exists_with_required_columns():
    cols = {c["name"]: c for c in _inspector().get_columns("session_traces")}
    for name in (
        "id", "org_id", "project_id", "user_id", "agent_id", "session_id",
        "started_at", "ended_at", "transcript_ref", "tool_calls_ref",
        "redaction_version", "retention_class", "created_at",
    ):
        assert name in cols, f"missing column {name}"
    assert cols["org_id"]["nullable"] is False
    assert cols["session_id"]["nullable"] is False
    assert cols["transcript_ref"]["nullable"] is False
    assert cols["redaction_version"]["nullable"] is False


def test_session_traces_indexes():
    idx = _inspector().get_indexes("session_traces")
    names = {i["name"] for i in idx}
    assert "ix_session_traces_org_created" in names
    unique = [i for i in idx if i["name"] == "uq_session_traces_org_session"]
    assert unique and unique[0]["unique"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/migrations && KLIO_REQUIRE_DB=1 DATABASE_URL=$DATABASE_URL pytest tests/test_041_session_traces.py -v`
Expected: FAIL — `session_traces` table does not exist.

- [ ] **Step 3: Write the migration**

```python
# services/migrations/alembic/versions/041_session_traces.py
"""session_traces: retained, redacted session evidence.

ADDITIVE ONLY. Build 1 of the evidence loop (spec
2026-08-13-context-coordination-design.md). Rows point at S3 objects holding
the REDACTED transcript and tool-call log; raw text never lands in Postgres.
``redaction_version`` scopes any future redactor bug to affected traces.
``retention_class='plan'`` means the org's plan retention governs deletion.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "041_session_traces"
down_revision = "040_project_access_roles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "session_traces",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("agent_id", sa.Text(), nullable=True),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("transcript_ref", sa.Text(), nullable=False),
        sa.Column("tool_calls_ref", sa.Text(), nullable=True),
        sa.Column("redaction_version", sa.Text(), nullable=False),
        sa.Column(
            "retention_class", sa.Text(), server_default="plan", nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_session_traces_org_created", "session_traces", ["org_id", "created_at"]
    )
    op.create_index(
        "uq_session_traces_org_session",
        "session_traces",
        ["org_id", "session_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_session_traces_org_session", table_name="session_traces")
    op.drop_index("ix_session_traces_org_created", table_name="session_traces")
    op.drop_table("session_traces")
```

- [ ] **Step 4: Apply and run the test**

Run: `cd services/migrations && alembic upgrade head && KLIO_REQUIRE_DB=1 pytest tests/test_041_session_traces.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add services/migrations/alembic/versions/041_session_traces.py services/migrations/tests/test_041_session_traces.py
git commit -m "feat(migrations): 041 session_traces — retained redacted evidence"
```

---

### Task 2: `shared.traces` — persist and fetch

**Files:**
- Create: `services/shared/shared/traces.py`
- Test: `services/shared/tests/test_traces.py`

**Interfaces:**
- Produces: `persist_trace(db, s3, *, org_id, session_id, transcript_text, tool_calls, project_id=None, user_id=None, agent_id=None, started_at=None, ended_at=None) -> str | None` (returns trace id, or `None` on failure — never raises). `fetch_trace(db, s3, *, org_id, session_id) -> dict | None` returning `{"transcript": str, "tool_calls": list, "row": dict}`.
- Consumes: `shared.redaction.redact` (existing); an S3 client with `put_object`/`get_object` (the storage-worker factory's boto3 shape); env `KLIO_TRACE_BUCKET` (default `klio-traces`), `REDACTION_VERSION` (default `"1"`).

- [ ] **Step 1: Write the failing test**

```python
# services/shared/tests/test_traces.py
"""Unit tests: mocked S3 + mocked db (pool policy: assert SQL shape + assembly)."""
import json
from unittest.mock import MagicMock

from shared import traces


def _db():
    db = MagicMock()
    return db


def _s3():
    s3 = MagicMock()
    return s3


def test_persist_redacts_before_put(monkeypatch):
    """The S3 object must contain redacted text, never the raw transcript."""
    calls = {}

    class FakeRedacted:
        clean_text = "REDACTED-BODY"

    monkeypatch.setattr(traces, "redact", lambda text: FakeRedacted())
    s3 = _s3()
    trace_id = traces.persist_trace(
        _db(), s3, org_id="org1", session_id="sess1",
        transcript_text="SECRET sk-live-123", tool_calls=[{"tool": "Edit"}],
    )
    assert trace_id is not None
    body = s3.put_object.call_args_list[0].kwargs["Body"]
    assert b"REDACTED-BODY" in body
    assert b"sk-live-123" not in body


def test_persist_inserts_row_with_org_and_refs():
    db = _db()
    traces.persist_trace(
        db, _s3(), org_id="org1", session_id="sess1",
        transcript_text="hello", tool_calls=[],
    )
    sql = str(db.execute.call_args_list[0].args[0])
    params = db.execute.call_args_list[0].args[1]
    assert "INSERT INTO session_traces" in sql
    assert params["org_id"] == "org1"
    assert params["session_id"] == "sess1"
    assert params["transcript_ref"].startswith("s3://")
    assert params["redaction_version"]


def test_persist_never_raises_on_s3_failure():
    s3 = _s3()
    s3.put_object.side_effect = RuntimeError("bucket gone")
    result = traces.persist_trace(
        _db(), s3, org_id="org1", session_id="s", transcript_text="x", tool_calls=[]
    )
    assert result is None  # best-effort: request must not break


def test_persist_refuses_when_redaction_fails(monkeypatch):
    """Unscreened text is NEVER persisted — same contract as recall_log."""
    def boom(text):
        raise RuntimeError("redactor down")

    monkeypatch.setattr(traces, "redact", boom)
    s3 = _s3()
    result = traces.persist_trace(
        _db(), s3, org_id="org1", session_id="s", transcript_text="raw", tool_calls=[]
    )
    assert result is None
    s3.put_object.assert_not_called()


def test_fetch_predicates_on_org():
    db = _db()
    db.execute.return_value.mappings.return_value.first.return_value = None
    traces.fetch_trace(db, _s3(), org_id="org1", session_id="sess1")
    sql = str(db.execute.call_args_list[0].args[0])
    assert "org_id = :org_id" in sql
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/shared && pytest tests/test_traces.py -v`
Expected: FAIL — `cannot import name 'traces'`.

- [ ] **Step 3: Write the implementation**

```python
# services/shared/shared/traces.py
"""Retained session evidence (Build 1 of the evidence loop).

One S3 object per session (redacted transcript + tool calls, JSON), one
``session_traces`` row pointing at it. Persistence is BEST-EFFORT: a failure
logs and returns ``None`` — it never breaks the capture request that
triggered it. Redaction is mandatory: if ``redact`` fails, nothing is
persisted (unscreened text never leaves the process — the recall_log
contract).
"""
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import text

from shared.redaction import redact

logger = logging.getLogger("klio.traces")


def _bucket() -> str:
    return os.environ.get("KLIO_TRACE_BUCKET", "klio-traces")


def _redaction_version() -> str:
    return os.environ.get("REDACTION_VERSION", "1")


def persist_trace(
    db: Any,
    s3: Any,
    *,
    org_id: str,
    session_id: str,
    transcript_text: str,
    tool_calls: list[dict[str, Any]],
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    started_at: Optional[datetime] = None,
    ended_at: Optional[datetime] = None,
) -> Optional[str]:
    """Redact, store to S3, insert the row. Returns trace id or None."""
    try:
        clean = redact(transcript_text).clean_text
    except Exception:
        # Same contract as recall_log: the text that failed screening is
        # exactly the text we must not persist (or log).
        logger.warning(
            "traces: redact() failed for org %s session %s; trace NOT "
            "persisted (content withheld — it is unscreened)",
            org_id,
            session_id,
            exc_info=True,
        )
        return None

    trace_id = str(uuid.uuid4())
    key = f"traces/{org_id}/{session_id}/{trace_id}.json"
    ref = f"s3://{_bucket()}/{key}"
    payload = json.dumps(
        {"transcript": clean, "tool_calls": tool_calls},
        ensure_ascii=False,
    ).encode("utf-8")

    try:
        s3.put_object(Bucket=_bucket(), Key=key, Body=payload)
        db.execute(
            text(
                """
                INSERT INTO session_traces
                  (id, org_id, project_id, user_id, agent_id, session_id,
                   started_at, ended_at, transcript_ref, tool_calls_ref,
                   redaction_version)
                VALUES (:id, :org_id, :project_id, :user_id, :agent_id,
                        :session_id, :started_at, :ended_at, :transcript_ref,
                        :tool_calls_ref, :redaction_version)
                ON CONFLICT DO NOTHING
                """
            ),
            {
                "id": trace_id,
                "org_id": org_id,
                "project_id": project_id,
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "started_at": started_at,
                "ended_at": ended_at,
                "transcript_ref": ref,
                "tool_calls_ref": ref,  # single object holds both
                "redaction_version": _redaction_version(),
            },
        )
        db.commit()
        return trace_id
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning(
            "traces: persist failed for org %s session %s; ignoring",
            org_id,
            session_id,
            exc_info=True,
        )
        return None


def fetch_trace(
    db: Any, s3: Any, *, org_id: str, session_id: str
) -> Optional[dict[str, Any]]:
    """Load a trace row + its S3 payload. Org-predicated. None if absent."""
    try:
        row = (
            db.execute(
                text(
                    """
                    SELECT id, transcript_ref, tool_calls_ref, created_at
                    FROM session_traces
                    WHERE org_id = :org_id AND session_id = :session_id
                    """
                ),
                {"org_id": org_id, "session_id": session_id},
            )
            .mappings()
            .first()
        )
        if row is None:
            return None
        ref = row["transcript_ref"]
        key = ref.split("/", 3)[3]  # s3://bucket/<key>
        obj = s3.get_object(Bucket=_bucket(), Key=key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return {
            "transcript": data.get("transcript", ""),
            "tool_calls": data.get("tool_calls", []),
            "row": dict(row),
        }
    except Exception:
        logger.warning(
            "traces: fetch failed for org %s session %s", org_id, session_id,
            exc_info=True,
        )
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/shared && pytest tests/test_traces.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add services/shared/shared/traces.py services/shared/tests/test_traces.py
git commit -m "feat(shared): traces module — redact-then-persist session evidence"
```

---

### Task 3: Wire `capture_transcript` to persist the trace

**Files:**
- Modify: `services/mcp-server/app/capture.py` (the `capture_transcript` handler, after distillation succeeds)
- Test: `services/mcp-server/tests/test_capture_trace_retention.py`

**Interfaces:**
- Consumes: `shared.traces.persist_trace` (Task 2); the handler's existing `org_id`, `req.session_id`, `req.messages`; an S3 client obtained lazily via a new module-level `_trace_s3()` helper (boto3 client, same env-driven config as `services/storage-worker/app/s3_client.py` — copy that factory's shape, do not import across service boundaries).
- Produces: every successful `POST /capture/transcript` also persists a trace. `tool_calls` for this endpoint is `[]` (transcript messages carry the content; tool traces arrive when harness hooks send them — out of scope here).

- [ ] **Step 1: Write the failing test**

```python
# services/mcp-server/tests/test_capture_trace_retention.py
"""capture_transcript must persist a trace (best-effort) after distilling."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(app_with_auth):  # reuse the repo's existing app fixture pattern
    return TestClient(app_with_auth)


def _transcript_body():
    return {
        "session_id": "sess-trace-1",
        "messages": [
            {"role": "user", "content": "we chose postgres for tenant isolation"},
            {"role": "assistant", "content": "noted"},
        ],
    }


def test_transcript_capture_persists_trace(client):
    with patch("app.capture.persist_trace") as persist:
        persist.return_value = "trace-id-1"
        resp = client.post("/capture/transcript", json=_transcript_body())
    assert resp.status_code == 200
    assert persist.called
    kwargs = persist.call_args.kwargs
    assert kwargs["session_id"] == "sess-trace-1"
    assert "postgres" in kwargs["transcript_text"]


def test_trace_failure_does_not_break_capture(client):
    with patch("app.capture.persist_trace", side_effect=RuntimeError("s3 down")):
        resp = client.post("/capture/transcript", json=_transcript_body())
    assert resp.status_code == 200  # capture succeeded regardless
```

Note for the implementer: `app_with_auth` refers to the mcp-server test suite's existing FastAPI app fixture with auth dependencies overridden — find it in `services/mcp-server/tests/conftest.py` and reuse it verbatim (do not invent a new auth bypass).

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/mcp-server && pytest tests/test_capture_trace_retention.py -v`
Expected: FAIL — `cannot import name 'persist_trace' from 'app.capture'`.

- [ ] **Step 3: Wire it in**

In `services/mcp-server/app/capture.py`:

1. Import: `from shared.traces import persist_trace`.
2. Add a lazy S3 client helper near the module's other helpers:

```python
_TRACE_S3 = None


def _trace_s3():
    """Lazy boto3 client for trace storage (same env config as storage-worker)."""
    global _TRACE_S3
    if _TRACE_S3 is None:
        import boto3

        _TRACE_S3 = boto3.client(
            "s3",
            endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    return _TRACE_S3
```

3. In `capture_transcript`, after distillation has succeeded (just before the return), add — wrapped so failure cannot break the response, and run off the event loop like the other blocking calls in this handler:

```python
        # Build 1 (evidence loop): retain the redacted transcript. Best-effort —
        # persist_trace redacts internally and returns None on any failure.
        try:
            flat_text = _flatten_transcript(req.messages)  # existing helper
            await anyio.to_thread.run_sync(
                lambda: persist_trace(
                    db,
                    _trace_s3(),
                    org_id=org_id,
                    session_id=req.session_id or "unknown",
                    transcript_text=flat_text,
                    tool_calls=[],
                    user_id=user_id,
                    agent_id=agent_id,
                )
            )
        except Exception:
            logger.warning("trace retention failed; capture unaffected", exc_info=True)
```

(If the flatten helper has a different name, use the one `capture_transcript` already calls at `capture.py:460` — "Flatten transcript messages into one text blob" — and pass the SAME flattened text it feeds the LLM.)

- [ ] **Step 4: Run tests**

Run: `cd services/mcp-server && pytest tests/test_capture_trace_retention.py tests/ -k "transcript" -v`
Expected: new tests PASS; existing transcript tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add services/mcp-server/app/capture.py services/mcp-server/tests/test_capture_trace_retention.py
git commit -m "feat(capture): retain redacted session traces on transcript capture"
```

---

### Task 4: Migration 042 — recall evidence columns + `recall_outcomes`

**Files:**
- Create: `services/migrations/alembic/versions/042_recall_evidence.py`
- Test: `services/migrations/tests/test_042_recall_evidence.py`

**Interfaces:**
- Produces: `brain_recall_events` gains `memory_ids postgresql.ARRAY(UUID) NULL` and `session_id Text NULL`. New table `recall_outcomes`: `id UUID pk`, `org_id String(64) NOT NULL`, `recall_event_id UUID NOT NULL` (fk `brain_recall_events.id`), `session_id Text NULL`, `verdict Text NOT NULL` (`pass|flag|block|unknown`), `confidence Numeric NULL`, `served_stale Boolean NOT NULL DEFAULT false`, `grader_version Text NOT NULL`, `graded_at timestamptz NOT NULL DEFAULT now()`. Indexes: `(org_id, graded_at)` on outcomes; `(org_id, session_id)` on `brain_recall_events`.

- [ ] **Step 1: Write the failing test**

```python
# services/migrations/tests/test_042_recall_evidence.py
import os

import pytest
import sqlalchemy as sa

pytestmark = pytest.mark.skipif(
    not os.environ.get("KLIO_REQUIRE_DB"), reason="needs migrated DB"
)


def _inspector():
    return sa.inspect(sa.create_engine(os.environ["DATABASE_URL"]))


def test_recall_events_gained_evidence_columns():
    cols = {c["name"] for c in _inspector().get_columns("brain_recall_events")}
    assert "memory_ids" in cols
    assert "session_id" in cols


def test_recall_outcomes_exists():
    cols = {c["name"]: c for c in _inspector().get_columns("recall_outcomes")}
    for name in (
        "id", "org_id", "recall_event_id", "session_id", "verdict",
        "confidence", "served_stale", "grader_version", "graded_at",
    ):
        assert name in cols, f"missing {name}"
    assert cols["verdict"]["nullable"] is False
    fks = _inspector().get_foreign_keys("recall_outcomes")
    assert any(fk["referred_table"] == "brain_recall_events" for fk in fks)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd services/migrations && KLIO_REQUIRE_DB=1 pytest tests/test_042_recall_evidence.py -v`
Expected: FAIL — columns/table missing.

- [ ] **Step 3: Write the migration**

```python
# services/migrations/alembic/versions/042_recall_evidence.py
"""Recall evidence: which memories were served, and how it went.

ADDITIVE ONLY. Build 2 of the evidence loop. ``memory_ids`` on
``brain_recall_events`` records WHICH rows a recall returned (022 only
recorded the count). ``recall_outcomes`` joins a recall to a verification
verdict; ``served_stale`` is set retroactively when a served memory is later
superseded — the silent-failure counter.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "042_recall_evidence"
down_revision = "041_session_traces"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "brain_recall_events",
        sa.Column("memory_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
    )
    op.add_column(
        "brain_recall_events", sa.Column("session_id", sa.Text(), nullable=True)
    )
    op.create_index(
        "ix_brain_recall_events_org_session",
        "brain_recall_events",
        ["org_id", "session_id"],
    )
    op.create_table(
        "recall_outcomes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column(
            "recall_event_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("brain_recall_events.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("session_id", sa.Text(), nullable=True),
        sa.Column("verdict", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Numeric(), nullable=True),
        sa.Column(
            "served_stale", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column("grader_version", sa.Text(), nullable=False),
        sa.Column(
            "graded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_recall_outcomes_org_graded", "recall_outcomes", ["org_id", "graded_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_recall_outcomes_org_graded", table_name="recall_outcomes")
    op.drop_table("recall_outcomes")
    op.drop_index("ix_brain_recall_events_org_session", table_name="brain_recall_events")
    op.drop_column("brain_recall_events", "session_id")
    op.drop_column("brain_recall_events", "memory_ids")
```

- [ ] **Step 4: Apply and test**

Run: `cd services/migrations && alembic upgrade head && KLIO_REQUIRE_DB=1 pytest tests/test_042_recall_evidence.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add services/migrations/alembic/versions/042_recall_evidence.py services/migrations/tests/test_042_recall_evidence.py
git commit -m "feat(migrations): 042 recall evidence — served memory_ids + outcomes"
```

---

### Task 5: Log served memory ids in `recall_log`

**Files:**
- Modify: `services/mcp-server/app/recall_log.py`
- Modify: call sites (`services/mcp-server/app/mcp_app.py:2317` area and `services/mcp-server/app/capture.py` recall handler) to pass `memory_ids` + `session_id`
- Test: `services/mcp-server/tests/test_recall_log_memory_ids.py`

**Interfaces:**
- Produces: `log_recall(db, *, org_id, agent_id, project_id, query_text, result_count, source, memory_ids=None, session_id=None)` — two new OPTIONAL kwargs, so untouched call sites keep working.
- Consumes: recall handlers already hold the serialized memories; pass `[m["id"] for m in memories]`.

- [ ] **Step 1: Write the failing test**

```python
# services/mcp-server/tests/test_recall_log_memory_ids.py
from unittest.mock import MagicMock

from app import recall_log


def test_log_recall_includes_memory_ids_and_session():
    db = MagicMock()
    recall_log.log_recall(
        db,
        org_id="org1", agent_id="a", project_id=None,
        query_text="q", result_count=2, source="mcp",
        memory_ids=["11111111-1111-1111-1111-111111111111"],
        session_id="sess9",
    )
    sql = str(db.execute.call_args.args[0])
    params = db.execute.call_args.args[1]
    assert "memory_ids" in sql and "session_id" in sql
    assert params["memory_ids"] == ["11111111-1111-1111-1111-111111111111"]
    assert params["session_id"] == "sess9"


def test_log_recall_defaults_remain_compatible():
    db = MagicMock()
    recall_log.log_recall(
        db,
        org_id="org1", agent_id=None, project_id=None,
        query_text=None, result_count=0, source="mcp",
    )
    params = db.execute.call_args.args[1]
    assert params["memory_ids"] is None
    assert params["session_id"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd services/mcp-server && pytest tests/test_recall_log_memory_ids.py -v`
Expected: FAIL — unexpected keyword argument.

- [ ] **Step 3: Implement**

In `recall_log.py`, add the two optional params and extend the INSERT:

```python
def log_recall(
    db: Any,
    *,
    org_id: str,
    agent_id: Optional[str],
    project_id: Optional[str],
    query_text: Optional[str],
    result_count: int,
    source: str,
    memory_ids: Optional[list[str]] = None,
    session_id: Optional[str] = None,
) -> None:
```

SQL becomes:

```sql
INSERT INTO brain_recall_events
  (id, org_id, agent_id, project_id, query_text, result_count, source,
   memory_ids, session_id)
VALUES (:id, :org_id, :agent_id, :project_id, :query_text,
        :result_count, :source, :memory_ids, :session_id)
```

with `"memory_ids": memory_ids, "session_id": session_id` in the params dict. Then update the two call sites to pass `memory_ids=[str(m["id"]) for m in memories]` (or the serialized records the handler holds) and the session id the handler already knows (`req.session_id` in capture; the MCP session identifier at the `mcp_app.py:2317` site — use whatever session identifier that scope already has; if none exists there, pass `None` and leave a one-line comment).

- [ ] **Step 4: Run the full mcp-server suite**

Run: `cd services/mcp-server && pytest tests/ -k "recall" -v`
Expected: new tests PASS, all existing recall tests PASS (optional kwargs = no breakage).

- [ ] **Step 5: Commit**

```bash
git add services/mcp-server/app/recall_log.py services/mcp-server/app/mcp_app.py services/mcp-server/app/capture.py services/mcp-server/tests/test_recall_log_memory_ids.py
git commit -m "feat(recall): record served memory ids + session on recall events"
```

---

### Task 6: The outcome grader job

**Files:**
- Create: `services/async-worker/app/outcome_grader.py`
- Test: `services/async-worker/tests/test_outcome_grader.py`

**Interfaces:**
- Produces: `async def grade_session(db, s3, *, org_id: str, session_id: str, grader_version: str = "v1") -> int` — returns number of `recall_outcomes` rows written. Wiring into the worker loop/scheduler is a follow-up; this task delivers the callable + tests (same pattern as `ownership_sweep.py`, which ships as an invocable module).
- Consumes: `shared.traces.fetch_trace` (Task 2); `engine.pipeline.verify` (existing — signature `verify(output, task=None, ..., session_memories=None) -> VerificationResult` with `.confidence` and `.action`); recall events from Task 5.

- [ ] **Step 1: Write the failing test**

```python
# services/async-worker/tests/test_outcome_grader.py
"""Grader: trace in, one recall_outcomes row per recall event."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import outcome_grader


def _db_with_recalls(rows):
    db = MagicMock()
    db.execute.return_value.mappings.return_value.all.return_value = rows
    return db


@pytest.mark.anyio
async def test_grades_each_recall_event():
    db = _db_with_recalls(
        [
            {"id": "re-1", "memory_ids": ["m-1"], "session_id": "s1"},
            {"id": "re-2", "memory_ids": ["m-2"], "session_id": "s1"},
        ]
    )
    trace = {"transcript": "user: x\nassistant: final answer", "tool_calls": [], "row": {}}
    verdict = MagicMock(confidence=0.9, action="pass")
    with patch.object(outcome_grader, "fetch_trace", return_value=trace), \
         patch.object(outcome_grader, "verify", new=AsyncMock(return_value=verdict)):
        written = await outcome_grader.grade_session(
            db, MagicMock(), org_id="org1", session_id="s1"
        )
    assert written == 2
    inserts = [c for c in db.execute.call_args_list if "INSERT INTO recall_outcomes" in str(c.args[0])]
    assert len(inserts) == 2
    params = inserts[0].args[1]
    assert params["org_id"] == "org1"
    assert params["verdict"] == "pass"
    assert params["grader_version"] == "v1"


@pytest.mark.anyio
async def test_no_trace_means_unknown_verdicts_not_errors():
    db = _db_with_recalls([{"id": "re-1", "memory_ids": ["m-1"], "session_id": "s1"}])
    with patch.object(outcome_grader, "fetch_trace", return_value=None):
        written = await outcome_grader.grade_session(
            db, MagicMock(), org_id="org1", session_id="s1"
        )
    assert written == 1
    params = db.execute.call_args_list[-2].args[1] if len(db.execute.call_args_list) > 1 else db.execute.call_args_list[-1].args[1]
    # the INSERT carries verdict='unknown' when there is nothing to grade
    inserts = [c for c in db.execute.call_args_list if "INSERT INTO recall_outcomes" in str(c.args[0])]
    assert inserts[0].args[1]["verdict"] == "unknown"


@pytest.mark.anyio
async def test_org_boundary_in_recall_query():
    db = _db_with_recalls([])
    with patch.object(outcome_grader, "fetch_trace", return_value=None):
        await outcome_grader.grade_session(db, MagicMock(), org_id="org1", session_id="s1")
    sql = str(db.execute.call_args_list[0].args[0])
    assert "org_id = :org_id" in sql
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd services/async-worker && pytest tests/test_outcome_grader.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

```python
# services/async-worker/app/outcome_grader.py
"""Grade a session's outputs and join verdicts to the recalls that fed them.

Build 2 of the evidence loop. For each ``brain_recall_events`` row of the
session, run the verification pipeline over the session's final output (from
the retained trace) and write one ``recall_outcomes`` row. A missing trace
grades as ``unknown`` — the join must exist even when evidence is thin,
otherwise coverage gaps read as "no data" instead of "ungraded".

Tenant boundary: every query predicates on org_id; a grader run never crosses
orgs.
"""
import logging
import uuid
from typing import Any

from sqlalchemy import text

from engine.pipeline import verify
from shared.traces import fetch_trace

logger = logging.getLogger("agentguard.outcome-grader")

GRADER_VERSION = "v1"


def _final_output(transcript: str) -> str:
    """Last assistant utterance = the session's final output (grading target)."""
    lines = [l for l in transcript.splitlines() if l.strip()]
    for line in reversed(lines):
        if line.lower().startswith("assistant:"):
            return line.split(":", 1)[1].strip()
    return transcript[-2000:]  # fallback: tail of the flattened transcript


async def grade_session(
    db: Any,
    s3: Any,
    *,
    org_id: str,
    session_id: str,
    grader_version: str = GRADER_VERSION,
) -> int:
    recalls = (
        db.execute(
            text(
                """
                SELECT id, memory_ids, session_id
                FROM brain_recall_events
                WHERE org_id = :org_id AND session_id = :session_id
                """
            ),
            {"org_id": org_id, "session_id": session_id},
        )
        .mappings()
        .all()
    )
    if not recalls:
        return 0

    trace = fetch_trace(db, s3, org_id=org_id, session_id=session_id)

    verdict_str = "unknown"
    confidence = None
    if trace and trace["transcript"]:
        try:
            result = await verify(_final_output(trace["transcript"]))
            verdict_str = result.action  # pass | flag | block
            confidence = result.confidence
        except Exception:
            logger.warning(
                "grader: verify failed for org %s session %s; verdict=unknown",
                org_id,
                session_id,
                exc_info=True,
            )

    written = 0
    for recall in recalls:
        try:
            db.execute(
                text(
                    """
                    INSERT INTO recall_outcomes
                      (id, org_id, recall_event_id, session_id, verdict,
                       confidence, grader_version)
                    VALUES (:id, :org_id, :recall_event_id, :session_id,
                            :verdict, :confidence, :grader_version)
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "org_id": org_id,
                    "recall_event_id": str(recall["id"]),
                    "session_id": session_id,
                    "verdict": verdict_str,
                    "confidence": confidence,
                    "grader_version": grader_version,
                },
            )
            written += 1
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass
            logger.warning("grader: insert failed; continuing", exc_info=True)
    db.commit()
    return written
```

- [ ] **Step 4: Run tests**

Run: `cd services/async-worker && pytest tests/test_outcome_grader.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add services/async-worker/app/outcome_grader.py services/async-worker/tests/test_outcome_grader.py
git commit -m "feat(async-worker): outcome grader — join verdicts to served recalls"
```

---

### Task 7: Retroactive stale-serve marking on supersession

**Files:**
- Modify: `services/shared/shared/memory.py` (inside `supersede_memory`, after the supersession commit point)
- Test: `services/shared/tests/test_stale_serve_marking.py`

**Interfaces:**
- Consumes: `supersede_memory`'s existing scope — it knows `org_id` and the superseded row id.
- Produces: on supersession, every `recall_outcomes` row whose recall event served the old memory id AFTER the supersession moment... — careful: serves BEFORE the supersession were correct at the time. The rule from the spec: mark rows where the recall happened after the *fact changed*, which is the supersession event itself, so the correct predicate is recalls of the old id **from now on**. Since future recalls will not return a superseded row, the practically meaningful marker is: recalls of the old id in the window where it was already stale. That window cannot be known at supersede time, so the implementable contract is: **at supersede time, mark all `recall_outcomes` joined to recall events that served the old id and occurred within the trailing N days (default 30) as `served_stale = true`** — i.e. "these answers were built on a belief we now know was outdated". `mark_stale_serves(db, *, org_id, memory_id, window_days=30) -> int` returns rows marked.

- [ ] **Step 1: Write the failing test**

```python
# services/shared/tests/test_stale_serve_marking.py
from unittest.mock import MagicMock

from shared import memory as memory_mod


def test_mark_stale_serves_sql_shape():
    db = MagicMock()
    db.execute.return_value.rowcount = 3
    marked = memory_mod.mark_stale_serves(
        db, org_id="org1", memory_id="22222222-2222-2222-2222-222222222222"
    )
    assert marked == 3
    sql = str(db.execute.call_args.args[0])
    params = db.execute.call_args.args[1]
    assert "UPDATE recall_outcomes" in sql
    assert "served_stale" in sql
    assert ":memory_id = ANY(bre.memory_ids)" in sql.replace("\n", " ")
    assert "org_id = :org_id" in sql
    assert params["org_id"] == "org1"


def test_mark_stale_serves_never_raises():
    db = MagicMock()
    db.execute.side_effect = RuntimeError("db down")
    marked = memory_mod.mark_stale_serves(db, org_id="o", memory_id="m")
    assert marked == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd services/shared && pytest tests/test_stale_serve_marking.py -v`
Expected: FAIL — no attribute `mark_stale_serves`.

- [ ] **Step 3: Implement**

Add to `services/shared/shared/memory.py`:

```python
def mark_stale_serves(
    db: Any, *, org_id: str, memory_id: str, window_days: int = 30
) -> int:
    """Flag outcomes built on a belief that has just been superseded.

    Best-effort (never raises): the supersession itself must not fail because
    the evidence ledger is unavailable. Marks recall_outcomes joined to recall
    events that served ``memory_id`` in the trailing window — "answers that
    rested on what we now know was outdated".
    """
    try:
        result = db.execute(
            text(
                """
                UPDATE recall_outcomes ro
                SET served_stale = true
                FROM brain_recall_events bre
                WHERE ro.recall_event_id = bre.id
                  AND ro.org_id = :org_id
                  AND bre.org_id = :org_id
                  AND :memory_id = ANY(bre.memory_ids)
                  AND bre.created_at >= now() - make_interval(days => :window_days)
                """
            ),
            {"org_id": org_id, "memory_id": memory_id, "window_days": window_days},
        )
        return int(result.rowcount or 0)
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning(
            "stale-serve marking failed for org %s memory %s; ignoring",
            org_id,
            memory_id,
            exc_info=True,
        )
        return 0
```

Then call it inside `supersede_memory` immediately after the supersession is applied (same transaction context the function already manages), passing the superseded row's id:

```python
    # Evidence loop: answers built on this belief are now known-stale.
    mark_stale_serves(db_session, org_id=org_id, memory_id=str(old_row_id))
```

(`old_row_id` = whatever local name `supersede_memory` uses for the row being superseded — match the existing variable.)

- [ ] **Step 4: Run the shared suite**

Run: `cd services/shared && pytest tests/test_stale_serve_marking.py tests/ -k "supersede" -v`
Expected: new tests PASS; existing supersession tests PASS.

- [ ] **Step 5: Commit**

```bash
git add services/shared/shared/memory.py services/shared/tests/test_stale_serve_marking.py
git commit -m "feat(memory): mark outcomes served from a now-superseded belief"
```

---

### Task 8: Retention sweep covers traces + end-to-end DB test

**Files:**
- Modify: the retention sweep (`run_retention.py` — locate at `services/` root or scripts; keep its non-positive-retention guard intact)
- Test: `services/shared/tests/test_trace_retention_sweep.py` (DB-gated)

**Interfaces:**
- Consumes: `plan_limits.py` retention (`memory_retention_days`); the sweep's existing per-org loop.
- Produces: orgs with positive retention also delete `session_traces` rows older than the window (S3 objects are left to a lifecycle rule — document this in the sweep's docstring; deleting DB rows removes reachability).

- [ ] **Step 1: Write the failing test**

```python
# services/shared/tests/test_trace_retention_sweep.py
"""DB-gated: the sweep deletes old traces for positive-retention orgs only."""
import os
import uuid

import pytest
import sqlalchemy as sa

pytestmark = pytest.mark.skipif(
    not os.environ.get("KLIO_REQUIRE_DB"), reason="needs migrated DB"
)


@pytest.fixture()
def engine():
    return sa.create_engine(os.environ["DATABASE_URL"])


def _insert_trace(conn, org_id, days_old):
    conn.execute(
        sa.text(
            """
            INSERT INTO session_traces
              (id, org_id, session_id, transcript_ref, redaction_version, created_at)
            VALUES (:id, :org, :sess, 's3://b/k', '1', now() - make_interval(days => :d))
            """
        ),
        {"id": str(uuid.uuid4()), "org": org_id, "sess": str(uuid.uuid4()), "d": days_old},
    )


def test_sweep_deletes_only_expired_traces(engine):
    from shared.retention import sweep_traces_for_org  # new function under test

    org = f"org-{uuid.uuid4().hex[:8]}"
    with engine.begin() as conn:
        _insert_trace(conn, org, days_old=40)
        _insert_trace(conn, org, days_old=5)

    with engine.begin() as conn:
        deleted = sweep_traces_for_org(conn, org_id=org, retention_days=30)
    assert deleted == 1

    with engine.begin() as conn:
        left = conn.execute(
            sa.text("SELECT count(*) FROM session_traces WHERE org_id = :o"), {"o": org}
        ).scalar()
    assert left == 1


def test_sweep_skips_non_positive_retention(engine):
    from shared.retention import sweep_traces_for_org

    org = f"org-{uuid.uuid4().hex[:8]}"
    with engine.begin() as conn:
        _insert_trace(conn, org, days_old=400)
    with engine.begin() as conn:
        deleted = sweep_traces_for_org(conn, org_id=org, retention_days=-1)
    assert deleted == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd services/shared && KLIO_REQUIRE_DB=1 pytest tests/test_trace_retention_sweep.py -v`
Expected: FAIL — no module `shared.retention` / function missing. (If a retention module already exists under a different name, add `sweep_traces_for_org` there instead and adjust the import in the test — the sweep entry point is `run_retention.py`; follow where it imports from.)

- [ ] **Step 3: Implement**

```python
# in the module run_retention.py already imports its per-org helpers from
# (create services/shared/shared/retention.py if none exists):
from typing import Any

from sqlalchemy import text


def sweep_traces_for_org(conn: Any, *, org_id: str, retention_days: int) -> int:
    """Delete expired session_traces rows. Non-positive retention = keep forever.

    S3 payloads are governed by a bucket lifecycle rule; deleting the row
    removes all reachability from the product.
    """
    if retention_days is None or retention_days <= 0:
        return 0
    result = conn.execute(
        text(
            """
            DELETE FROM session_traces
            WHERE org_id = :org_id
              AND created_at < now() - make_interval(days => :days)
            """
        ),
        {"org_id": org_id, "days": retention_days},
    )
    return int(result.rowcount or 0)
```

Wire it into `run_retention.py`'s per-org loop next to the memory sweep call, passing the same resolved retention the org already gets (including `plan_overrides`).

- [ ] **Step 4: Run DB-gated tests**

Run: `cd services/shared && KLIO_REQUIRE_DB=1 pytest tests/test_trace_retention_sweep.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 5: Full-suite sanity + commit**

```bash
cd services/shared && pytest tests/ -x -q && cd ../mcp-server && pytest tests/ -x -q && cd ../async-worker && pytest tests/ -x -q
git add -A services/shared services/migrations
git commit -m "feat(retention): traces expire with the org's plan retention"
```

---

## Deployment notes

- **Migrations are manual in production.** Run 041 + 042 against the Neon **direct** endpoint (strip `-pooler`), not the pooled one.
- New env for mcp-server: `KLIO_TRACE_BUCKET` (create the bucket + a lifecycle rule matching max plan retention), `REDACTION_VERSION=1`, plus the S3 credentials the storage-worker already uses.
- The grader is a callable this phase; scheduling it (cron / stream consumer) is Phase-2 wiring alongside claims.
- Rollout order: migrations → shared → mcp-server → async-worker. Each task is independently deployable; traces simply accumulate until the grader runs.

## Out of scope (later phases of the spec)

- Optimistic concurrency (Build 3), claims (Build 4), dreaming (Build 5).
- Grader scheduling + the holdout experiment.
- Tool-call capture from harness hooks (the `tool_calls` list is plumbed but arrives empty until the hook payload exists).
