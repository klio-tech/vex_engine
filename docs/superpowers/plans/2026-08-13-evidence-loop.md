# Coordination System (Builds 1–5, complete) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the complete coordination system in one release: retained evidence, an honest outcome join, hash-based optimistic concurrency, semantic work claims, and dreaming as evidence-backed proposals.

**Architecture:** mcp-server publishes redacted session payloads to a Redis stream; storage-worker persists them (S3 + Postgres, its existing job) and emits `sessions.stored`; async-worker grades each stored session (verification pipeline + per-memory usage attribution) and, on schedule, dreams over accumulated evidence to produce reviewable proposals. Writes gain revision-based check-and-set; claims ride every recall response.

**Tech Stack:** FastAPI/FastMCP (mcp-server), aioredis streams, boto3 (storage-worker only), SQLAlchemy Core + Alembic, verification-engine as a library, pytest with the repo's mocked-pool policy (`KLIO_REQUIRE_DB=1` for DB-gated tests).

## Design corrections from review (binding on every task)

1. **No session verdict masquerading as a memory verdict.** The session grade lives ONCE in `session_outcomes`. Per-recall rows carry *attribution* (`usage_score`, `used`), never a copied pass/flag/block.
2. **`verify()` receives `session_memories`** — the pipeline's memory-consistency check is the only Vex check that speaks about memories; not passing them wastes the one relevant grader.
3. **Final output comes from structured messages** (`messages[-1]` with `role == "assistant"`), never a string-prefix hunt over a flattened blob. Traces store the structured messages array.
4. **Stale-serve = served after the superseding belief already existed** (`bre.created_at >= new_belief.created_at`). Serves that were correct at the time are never marked. This measures curator lag — the honest silent-failure counter.
5. **mcp-server never touches S3; the hand-off is a Redis stream.** Verified 2026-08-13: mcp-server has no Redis today — this plan ADDS `aioredis` + `REDIS_URL` to it, deliberately. Redis streams are already this repo's production event backbone (`executions.raw` → `stored` → `verified`, consumer groups + acks); publishing `sessions.captured` the same way keeps ONE queueing pattern instead of inventing a second. Publish is best-effort (evidence, not billing): Redis down → warn and drop, capture still succeeds. Deployment: point mcp-server at the same Railway Redis the workers already use.
6. **A later Stop hook wins:** trace upsert keeps the transcript with MORE messages (`ON CONFLICT ... DO UPDATE ... WHERE EXCLUDED.message_count > session_traces.message_count`).
7. **`tool_calls` is a first-class request field now**, not a promised follow-up. Hooks that don't send it yield `[]`, which is recorded as `tool_calls_count = 0`, visibly distinct from "not supported".
8. **The grader is scheduled by construction** — it is a stream consumer, not a callable awaiting Phase 2.
9. **Engine freeze:** lifted for this work by the founder's explicit instruction (2026-08-13, "all phases complete system to be built in one go").

## Global Constraints

- ADDITIVE ONLY migrations; next revision `041`, then 042–045 in this plan's order.
- Redaction before any persistence or stream publish (`shared.redaction.redact`); on redaction failure, nothing leaves the process — the `recall_log.py` contract.
- Best-effort evidence: no evidence write may fail a user-facing request. Catch, rollback, `logger.warning`, continue.
- Every table carries `org_id`; every query predicates on it; the dreamer partitions strictly per org.
- Mocked-pool test policy: unit tests assert SQL shape + Python assembly; DB-gated tests require `KLIO_REQUIRE_DB=1` + migrated DB.
- Backward compatibility: all new tool/request fields optional; existing clients unchanged.
- Production migrations run against the Neon **direct** endpoint (strip `-pooler`).

## File Structure

| File | Responsibility |
| --- | --- |
| `services/migrations/alembic/versions/041_session_traces.py` | Trace rows (message_count for upsert-wins) |
| `services/migrations/alembic/versions/042_recall_evidence.py` | `memory_ids[]`+`session_id` on recalls; `session_outcomes`; `recall_outcomes` |
| `services/migrations/alembic/versions/043_memory_revisions.py` | `revision` on `session_memories` (CAS) |
| `services/migrations/alembic/versions/044_work_claims.py` | Claims with embeddings |
| `services/migrations/alembic/versions/045_memory_proposals.py` | Dreaming proposals |
| `services/shared/shared/evidence_stream.py` | Stream names + payload schema (producer/consumer contract) |
| `services/shared/shared/traces.py` | Row upsert + S3 object I/O (called by storage-worker) |
| `services/shared/shared/stale.py` | Corrected stale-serve marking |
| `services/shared/shared/claims.py` | Claim create/release/live-lookup with semantic collision |
| `services/mcp-server/app/capture.py` | Accept `tool_calls`; publish to `sessions.captured` |
| `services/mcp-server/app/recall_log.py` | Record served `memory_ids` + `session_id` |
| `services/mcp-server/app/mcp_app.py` | `base_revision` on write tools; `revision` + claims in recall; `claim`/`release` tools; proposals REST |
| `services/storage-worker/app/trace_consumer.py` | `sessions.captured` → S3+row → `sessions.stored` |
| `services/async-worker/app/outcome_grader.py` | `sessions.stored` → session verdict + per-recall attribution |
| `services/async-worker/app/dreamer.py` | Scheduled: evidence → `memory_proposals` |

---

### Task 1: Migration 041 — `session_traces`

**Files:**
- Create: `services/migrations/alembic/versions/041_session_traces.py`
- Test: `services/migrations/tests/test_041_session_traces.py`

**Interfaces:** table `session_traces`: `id UUID pk`, `org_id String(64) NOT NULL`, `project_id UUID NULL`, `user_id UUID NULL`, `agent_id Text NULL`, `session_id Text NOT NULL`, `transcript_ref Text NOT NULL` (set by storage-worker at insert — the row exists only once the object does), `message_count Integer NOT NULL DEFAULT 0`, `tool_calls_count Integer NOT NULL DEFAULT 0`, `redaction_version Text NOT NULL`, `created_at/updated_at timestamptz DEFAULT now()`. Unique `(org_id, session_id)`; index `(org_id, created_at)`.

- [ ] **Step 1: Write the failing test**

```python
# services/migrations/tests/test_041_session_traces.py
import os
import pytest
import sqlalchemy as sa

pytestmark = pytest.mark.skipif(
    not os.environ.get("KLIO_REQUIRE_DB"), reason="needs migrated DB"
)


def _insp():
    return sa.inspect(sa.create_engine(os.environ["DATABASE_URL"]))


def test_session_traces_columns():
    cols = {c["name"]: c for c in _insp().get_columns("session_traces")}
    for n in ("id", "org_id", "session_id", "transcript_ref", "message_count",
              "tool_calls_count", "redaction_version", "created_at", "updated_at"):
        assert n in cols, n
    assert cols["org_id"]["nullable"] is False
    assert cols["message_count"]["nullable"] is False


def test_session_traces_unique_org_session():
    idx = _insp().get_indexes("session_traces")
    u = [i for i in idx if i["name"] == "uq_session_traces_org_session"]
    assert u and u[0]["unique"] is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd services/migrations && KLIO_REQUIRE_DB=1 pytest tests/test_041_session_traces.py -v` → FAIL (no table).

- [ ] **Step 3: Write the migration**

```python
# services/migrations/alembic/versions/041_session_traces.py
"""session_traces: retained, redacted session evidence (Build 1).

ADDITIVE ONLY. Rows point at one S3 object holding the REDACTED structured
payload {messages, tool_calls}. message_count enables richer-transcript-wins
upserts (a second Stop hook with more messages replaces the first).
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
        sa.Column("transcript_ref", sa.Text(), nullable=False),
        sa.Column("message_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("tool_calls_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("redaction_version", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_session_traces_org_created", "session_traces",
                    ["org_id", "created_at"])
    op.create_index("uq_session_traces_org_session", "session_traces",
                    ["org_id", "session_id"], unique=True)


def downgrade() -> None:
    op.drop_index("uq_session_traces_org_session", table_name="session_traces")
    op.drop_index("ix_session_traces_org_created", table_name="session_traces")
    op.drop_table("session_traces")
```

- [ ] **Step 4: Apply + test** — `alembic upgrade head && KLIO_REQUIRE_DB=1 pytest tests/test_041_session_traces.py -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(migrations): 041 session_traces"`

---

### Task 2: Evidence stream contract + capture publish

**Files:**
- Create: `services/shared/shared/evidence_stream.py`
- Modify: `services/mcp-server/app/capture.py` (`CaptureTranscriptRequest` + handler)
- Test: `services/shared/tests/test_evidence_stream.py`, `services/mcp-server/tests/test_capture_publishes_trace.py`

**Interfaces:**
- Produces: `SESSIONS_CAPTURED = "sessions.captured"`, `SESSIONS_STORED = "sessions.stored"`; `build_captured_payload(*, org_id, session_id, messages, tool_calls, project_id=None, user_id=None, agent_id=None) -> dict | None` — redacts every message content and tool-call args; returns `None` if redaction fails (nothing leaves the process). Payload capped at 256 KB post-redaction (tail-trim messages to fit, counts preserved). `publish_captured(*, payload) -> None` — XADD to `SESSIONS_CAPTURED`; raises on failure (the caller wraps: evidence is best-effort, capture is not). `CaptureTranscriptRequest` gains `tool_calls: list[dict] = []` (optional).
- Consumes: `shared.redaction.redact`; a NEW lazy aioredis client in mcp-server from `REDIS_URL` (verified 2026-08-13: mcp-server has no Redis today — this task adds `aioredis` to its `pyproject.toml` and `REDIS_URL` to its env, matching the pattern async-worker already uses; point it at the same Railway Redis as the workers).

- [ ] **Step 1: Write the failing tests**

```python
# services/shared/tests/test_evidence_stream.py
from unittest.mock import patch

from shared import evidence_stream as es


def test_payload_redacts_messages_and_tool_args():
    class Clean:
        clean_text = "CLEAN"
    with patch.object(es, "redact", return_value=Clean()):
        p = es.build_captured_payload(
            org_id="o", session_id="s",
            messages=[{"role": "user", "content": "sk-live-9"}],
            tool_calls=[{"tool": "Bash", "args": {"command": "curl -H sk-live-9"}}],
        )
    assert p["messages"][0]["content"] == "CLEAN"
    assert "sk-live-9" not in str(p["tool_calls"])
    assert p["org_id"] == "o" and p["session_id"] == "s"


def test_payload_none_when_redaction_fails():
    with patch.object(es, "redact", side_effect=RuntimeError("down")):
        p = es.build_captured_payload(
            org_id="o", session_id="s", messages=[{"role": "user", "content": "x"}],
            tool_calls=[],
        )
    assert p is None
```

```python
# services/mcp-server/tests/test_capture_publishes_trace.py
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import pytest


@pytest.fixture()
def client(app_with_auth):  # existing conftest fixture; do not invent auth bypass
    return TestClient(app_with_auth)


def _body():
    return {
        "session_id": "s1",
        "messages": [{"role": "user", "content": "hello"},
                     {"role": "assistant", "content": "done"}],
        "tool_calls": [{"tool": "Edit", "args": {"file_path": "a.py"}}],
    }


def test_transcript_capture_enqueues_outbox(client):
    with patch("app.capture.enqueue_trace_outbox") as enq:
        r = client.post("/capture/transcript", json=_body())
    assert r.status_code == 200
    assert enq.called
    payload = enq.call_args.kwargs["payload"]
    assert payload["session_id"] == "s1"
    assert len(payload["tool_calls"]) == 1


def test_outbox_failure_does_not_break_capture(client):
    with patch("app.capture.enqueue_trace_outbox", side_effect=RuntimeError):
        r = client.post("/capture/transcript", json=_body())
    assert r.status_code == 200
```

- [ ] **Step 2: Run to verify they fail** — both suites FAIL on missing imports.

- [ ] **Step 3: Implement**

```python
# services/shared/shared/evidence_stream.py
"""Producer/consumer contract for the evidence pipeline.

mcp-server publishes REDACTED payloads to ``sessions.captured``;
storage-worker persists and emits ``sessions.stored``; the grader consumes
that. Redaction failure means NO publish — unscreened text never leaves the
process (recall_log contract).
"""
import json
import logging
from typing import Any, Optional

from shared.redaction import redact

logger = logging.getLogger("klio.evidence")

SESSIONS_CAPTURED = "sessions.captured"
SESSIONS_STORED = "sessions.stored"


def build_captured_payload(
    *,
    org_id: str,
    session_id: str,
    messages: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    try:
        clean_messages = [
            {"role": m.get("role", "user"),
             "content": redact(str(m.get("content", ""))).clean_text}
            for m in messages
        ]
        clean_tools = [
            {"tool": t.get("tool", ""),
             "args": redact(json.dumps(t.get("args", {}), ensure_ascii=False)).clean_text,
             "status": t.get("status")}
            for t in tool_calls
        ]
    except Exception:
        logger.warning(
            "evidence: redaction failed for org %s session %s; payload dropped "
            "(content withheld — it is unscreened)", org_id, session_id,
            exc_info=True,
        )
        return None
    return {
        "org_id": org_id, "session_id": session_id,
        "project_id": project_id, "user_id": user_id, "agent_id": agent_id,
        "messages": clean_messages, "tool_calls": clean_tools,
    }
```

In `capture.py`: add `tool_calls: list[dict[str, Any]] = Field(default_factory=list)` to `CaptureTranscriptRequest`; add

```python
async def publish_captured(*, payload: dict[str, Any]) -> None:
    """XADD the redacted payload to sessions.captured. Raises on failure —
    the caller wraps it (evidence is best-effort, capture is not)."""
    redis = await _get_redis()  # reuse existing helper or lazy aioredis.from_url(os.environ["REDIS_URL"])
    await redis.xadd(SESSIONS_CAPTURED, {"payload": json.dumps(payload)})
```

and at the end of `capture_transcript` (after distillation, before return):

```python
        # Evidence loop: publish the redacted session for retention. Best-effort.
        try:
            payload = build_captured_payload(
                org_id=org_id, session_id=req.session_id or "unknown",
                messages=[m.model_dump() for m in req.messages],
                tool_calls=req.tool_calls, user_id=user_id, agent_id=agent_id,
            )
            if payload is not None:
                await publish_captured(payload=payload)
        except Exception:
            logger.warning("evidence publish failed; capture unaffected", exc_info=True)
```

- [ ] **Step 4: Run** both test files + existing transcript tests → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(capture): tool_calls field + redacted publish to sessions.captured"`

---

### Task 3: storage-worker trace consumer

**Files:**
- Create: `services/storage-worker/app/trace_consumer.py`
- Create: `services/shared/shared/traces.py`
- Modify: `services/storage-worker/app/main.py` (start the consumer loop alongside `_consume_raw`)
- Test: `services/storage-worker/tests/test_trace_consumer.py`, `services/shared/tests/test_traces.py`

**Interfaces:**
- `shared.traces.upsert_trace(db, *, org_id, session_id, transcript_ref, message_count, tool_calls_count, redaction_version, project_id=None, user_id=None, agent_id=None) -> str | None` — richer-wins upsert, returns trace id or None (best-effort).
- `shared.traces.fetch_trace(db, s3, *, org_id, session_id) -> dict | None` → `{"messages": [...], "tool_calls": [...], "row": {...}}` (org-predicated).
- `trace_consumer.handle_captured(db, s3, redis, entry: dict) -> bool` — persists one stream entry, emits `SESSIONS_STORED` with `{org_id, session_id}`; returns success. Env: `KLIO_TRACE_BUCKET` (**required** — the consumer refuses to start without it; no default bucket footgun), `REDACTION_VERSION` (default "1").

- [ ] **Step 1: Write the failing tests**

```python
# services/shared/tests/test_traces.py
from unittest.mock import MagicMock

from shared import traces


def test_upsert_richer_wins_sql():
    db = MagicMock()
    traces.upsert_trace(
        db, org_id="o", session_id="s", transcript_ref="s3://b/k",
        message_count=5, tool_calls_count=2, redaction_version="1",
    )
    sql = " ".join(str(db.execute.call_args.args[0]).split())
    assert "INSERT INTO session_traces" in sql
    assert "ON CONFLICT" in sql and "DO UPDATE" in sql
    assert "EXCLUDED.message_count > session_traces.message_count" in sql


def test_upsert_never_raises():
    db = MagicMock()
    db.execute.side_effect = RuntimeError("down")
    assert traces.upsert_trace(
        db, org_id="o", session_id="s", transcript_ref="r",
        message_count=1, tool_calls_count=0, redaction_version="1",
    ) is None


def test_fetch_predicates_on_org():
    db = MagicMock()
    db.execute.return_value.mappings.return_value.first.return_value = None
    traces.fetch_trace(db, MagicMock(), org_id="o", session_id="s")
    assert "org_id = :org_id" in str(db.execute.call_args.args[0])
```

```python
# services/storage-worker/tests/test_trace_consumer.py
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import trace_consumer


@pytest.mark.anyio
async def test_handle_captured_puts_s3_then_row_then_emits():
    db, s3 = MagicMock(), MagicMock()
    redis = MagicMock()
    redis.xadd = AsyncMock()
    entry = {"payload": json.dumps({
        "org_id": "o", "session_id": "s", "project_id": None, "user_id": None,
        "agent_id": None,
        "messages": [{"role": "assistant", "content": "done"}],
        "tool_calls": [],
    })}
    with patch.object(trace_consumer, "_bucket", return_value="klio-traces"), \
         patch.object(trace_consumer.traces, "upsert_trace", return_value="tid") as up:
        ok = await trace_consumer.handle_captured(db, s3, redis, entry)
    assert ok
    body = s3.put_object.call_args.kwargs["Body"]
    assert b"done" in body
    assert up.call_args.kwargs["message_count"] == 1
    stored = redis.xadd.call_args.args
    assert stored[0] == trace_consumer.SESSIONS_STORED
    assert json.loads(redis.xadd.call_args.args[1]["payload"])["session_id"] == "s"


@pytest.mark.anyio
async def test_bad_payload_is_skipped_not_fatal():
    ok = await trace_consumer.handle_captured(
        MagicMock(), MagicMock(), MagicMock(), {"payload": "not json"}
    )
    assert ok is False
```

- [ ] **Step 2: Run to verify they fail** — missing modules.

- [ ] **Step 3: Implement**

```python
# services/shared/shared/traces.py
"""Trace persistence primitives (row upsert + fetch). Called by storage-worker."""
import json
import logging
import uuid
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger("klio.traces")


def upsert_trace(
    db: Any, *, org_id: str, session_id: str, transcript_ref: str,
    message_count: int, tool_calls_count: int, redaction_version: str,
    project_id: Optional[str] = None, user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Optional[str]:
    """Insert or replace-if-richer. A second Stop hook with MORE messages wins;
    a shorter duplicate never clobbers a fuller trace. Best-effort."""
    trace_id = str(uuid.uuid4())
    try:
        db.execute(
            text(
                """
                INSERT INTO session_traces
                  (id, org_id, project_id, user_id, agent_id, session_id,
                   transcript_ref, message_count, tool_calls_count,
                   redaction_version)
                VALUES (:id, :org_id, :project_id, :user_id, :agent_id,
                        :session_id, :transcript_ref, :message_count,
                        :tool_calls_count, :redaction_version)
                ON CONFLICT (org_id, session_id) DO UPDATE SET
                  transcript_ref = EXCLUDED.transcript_ref,
                  message_count = EXCLUDED.message_count,
                  tool_calls_count = EXCLUDED.tool_calls_count,
                  redaction_version = EXCLUDED.redaction_version,
                  updated_at = now()
                WHERE EXCLUDED.message_count > session_traces.message_count
                """
            ),
            {
                "id": trace_id, "org_id": org_id, "project_id": project_id,
                "user_id": user_id, "agent_id": agent_id,
                "session_id": session_id, "transcript_ref": transcript_ref,
                "message_count": message_count,
                "tool_calls_count": tool_calls_count,
                "redaction_version": redaction_version,
            },
        )
        db.commit()
        return trace_id
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning("traces: upsert failed org=%s session=%s", org_id,
                       session_id, exc_info=True)
        return None


def fetch_trace(
    db: Any, s3: Any, *, org_id: str, session_id: str
) -> Optional[dict[str, Any]]:
    try:
        row = db.execute(
            text(
                """
                SELECT id, transcript_ref, message_count, created_at
                FROM session_traces
                WHERE org_id = :org_id AND session_id = :session_id
                """
            ),
            {"org_id": org_id, "session_id": session_id},
        ).mappings().first()
        if row is None:
            return None
        bucket, key = row["transcript_ref"][len("s3://"):].split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return {"messages": data.get("messages", []),
                "tool_calls": data.get("tool_calls", []), "row": dict(row)}
    except Exception:
        logger.warning("traces: fetch failed org=%s session=%s", org_id,
                       session_id, exc_info=True)
        return None
```

```python
# services/storage-worker/app/trace_consumer.py
"""sessions.captured -> S3 + session_traces row -> sessions.stored.

Payloads arrive ALREADY REDACTED (evidence_stream contract). KLIO_TRACE_BUCKET
is required — refusing to start beats silently writing to a default bucket.
"""
import json
import logging
import os
import uuid
from typing import Any

from shared import traces
from shared.evidence_stream import SESSIONS_STORED

logger = logging.getLogger("agentguard.trace-consumer")


def _bucket() -> str:
    bucket = os.environ.get("KLIO_TRACE_BUCKET")
    if not bucket:
        raise RuntimeError("KLIO_TRACE_BUCKET is required for trace retention")
    return bucket


async def handle_captured(db: Any, s3: Any, redis: Any, entry: dict) -> bool:
    try:
        payload = json.loads(entry["payload"])
        org_id, session_id = payload["org_id"], payload["session_id"]
    except Exception:
        logger.warning("trace-consumer: malformed entry skipped", exc_info=True)
        return False
    try:
        key = f"traces/{org_id}/{session_id}/{uuid.uuid4()}.json"
        body = json.dumps(
            {"messages": payload["messages"], "tool_calls": payload["tool_calls"]},
            ensure_ascii=False,
        ).encode("utf-8")
        s3.put_object(Bucket=_bucket(), Key=key, Body=body)
        traces.upsert_trace(
            db, org_id=org_id, session_id=session_id,
            transcript_ref=f"s3://{_bucket()}/{key}",
            message_count=len(payload["messages"]),
            tool_calls_count=len(payload["tool_calls"]),
            redaction_version=os.environ.get("REDACTION_VERSION", "1"),
            project_id=payload.get("project_id"), user_id=payload.get("user_id"),
            agent_id=payload.get("agent_id"),
        )
        await redis.xadd(
            SESSIONS_STORED,
            {"payload": json.dumps({"org_id": org_id, "session_id": session_id})},
        )
        return True
    except Exception:
        logger.warning("trace-consumer: persist failed org=%s session=%s",
                       org_id, session_id, exc_info=True)
        return False
```

Wire a consumer loop in `main.py` following `_consume_raw`'s exact pattern (consumer group, XREADGROUP, ack on success), reading `SESSIONS_CAPTURED`.

- [ ] **Step 4: Run** both suites → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(storage-worker): persist session traces, emit sessions.stored"`

---

### Task 4: Migration 042 — recall evidence (+ honest outcome tables)

**Files:**
- Create: `services/migrations/alembic/versions/042_recall_evidence.py`
- Test: `services/migrations/tests/test_042_recall_evidence.py`

**Interfaces:** `brain_recall_events` += `memory_ids ARRAY(UUID) NULL`, `session_id Text NULL` (+ index `(org_id, session_id)`). New `session_outcomes`: `id UUID pk, org_id String(64) NOT NULL, session_id Text NOT NULL, verdict Text NOT NULL, confidence Numeric NULL, memory_consistency Numeric NULL, grader_version Text NOT NULL, graded_at timestamptz DEFAULT now()`, unique `(org_id, session_id, grader_version)`. New `recall_outcomes`: `id UUID pk, org_id NOT NULL, recall_event_id UUID fk→brain_recall_events ON DELETE CASCADE, memory_id UUID NOT NULL, session_outcome_id UUID fk→session_outcomes ON DELETE CASCADE, used Boolean NOT NULL DEFAULT false, usage_score Numeric NULL, served_stale Boolean NOT NULL DEFAULT false, created_at timestamptz DEFAULT now()`, index `(org_id, memory_id)`.

- [ ] **Step 1: Write the failing test**

```python
# services/migrations/tests/test_042_recall_evidence.py
import os
import pytest
import sqlalchemy as sa

pytestmark = pytest.mark.skipif(
    not os.environ.get("KLIO_REQUIRE_DB"), reason="needs migrated DB"
)


def _insp():
    return sa.inspect(sa.create_engine(os.environ["DATABASE_URL"]))


def test_recall_events_columns():
    cols = {c["name"] for c in _insp().get_columns("brain_recall_events")}
    assert {"memory_ids", "session_id"} <= cols


def test_outcome_tables():
    so = {c["name"] for c in _insp().get_columns("session_outcomes")}
    assert {"org_id", "session_id", "verdict", "confidence",
            "memory_consistency", "grader_version"} <= so
    ro = {c["name"] for c in _insp().get_columns("recall_outcomes")}
    assert {"org_id", "recall_event_id", "memory_id", "session_outcome_id",
            "used", "usage_score", "served_stale"} <= ro
    fks = {fk["referred_table"] for fk in _insp().get_foreign_keys("recall_outcomes")}
    assert {"brain_recall_events", "session_outcomes"} <= fks
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Write the migration** (same style as 041; the docstring must carry design-correction #1):

```python
# services/migrations/alembic/versions/042_recall_evidence.py
"""Recall evidence with an HONEST join (Build 2).

ADDITIVE ONLY. The session grade lives ONCE in session_outcomes. Per-memory
rows in recall_outcomes carry ATTRIBUTION (used / usage_score) and staleness —
never a copied session verdict. Querying "how did this memory do" joins
attribution to the session outcome explicitly, so nobody mistakes session
weather for memory quality.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "042_recall_evidence"
down_revision = "041_session_traces"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("brain_recall_events",
                  sa.Column("memory_ids",
                            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
                            nullable=True))
    op.add_column("brain_recall_events",
                  sa.Column("session_id", sa.Text(), nullable=True))
    op.create_index("ix_brain_recall_events_org_session",
                    "brain_recall_events", ["org_id", "session_id"])
    op.create_table(
        "session_outcomes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("verdict", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Numeric(), nullable=True),
        sa.Column("memory_consistency", sa.Numeric(), nullable=True),
        sa.Column("grader_version", sa.Text(), nullable=False),
        sa.Column("graded_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("uq_session_outcomes_org_session_grader",
                    "session_outcomes",
                    ["org_id", "session_id", "grader_version"], unique=True)
    op.create_table(
        "recall_outcomes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("recall_event_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("brain_recall_events.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("memory_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_outcome_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("session_outcomes.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("used", sa.Boolean(), server_default=sa.text("false"),
                  nullable=False),
        sa.Column("usage_score", sa.Numeric(), nullable=True),
        sa.Column("served_stale", sa.Boolean(), server_default=sa.text("false"),
                  nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_recall_outcomes_org_memory", "recall_outcomes",
                    ["org_id", "memory_id"])


def downgrade() -> None:
    op.drop_index("ix_recall_outcomes_org_memory", table_name="recall_outcomes")
    op.drop_table("recall_outcomes")
    op.drop_index("uq_session_outcomes_org_session_grader",
                  table_name="session_outcomes")
    op.drop_table("session_outcomes")
    op.drop_index("ix_brain_recall_events_org_session",
                  table_name="brain_recall_events")
    op.drop_column("brain_recall_events", "session_id")
    op.drop_column("brain_recall_events", "memory_ids")
```

- [ ] **Step 4: Apply + test** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(migrations): 042 honest recall evidence"`

---

### Task 5: `recall_log` records served memory ids + session

**Files:**
- Modify: `services/mcp-server/app/recall_log.py`; call sites (`mcp_app.py:~2317`, `capture.py` recall handler)
- Test: `services/mcp-server/tests/test_recall_log_memory_ids.py`

**Interfaces:** `log_recall(..., memory_ids: Optional[list[str]] = None, session_id: Optional[str] = None)` — optional kwargs; INSERT gains both columns; call sites pass `[str(m["id"]) for m in memories]` and the session id in scope (`req.session_id` in capture; `None` + one-line comment at the MCP site if no session identifier exists there).

- [ ] **Step 1: Write the failing test**

```python
# services/mcp-server/tests/test_recall_log_memory_ids.py
from unittest.mock import MagicMock

from app import recall_log


def test_log_recall_includes_new_fields():
    db = MagicMock()
    recall_log.log_recall(
        db, org_id="o", agent_id="a", project_id=None, query_text="q",
        result_count=1, source="mcp",
        memory_ids=["11111111-1111-1111-1111-111111111111"], session_id="s9",
    )
    sql = str(db.execute.call_args.args[0])
    params = db.execute.call_args.args[1]
    assert "memory_ids" in sql and "session_id" in sql
    assert params["session_id"] == "s9"


def test_log_recall_backward_compatible():
    db = MagicMock()
    recall_log.log_recall(db, org_id="o", agent_id=None, project_id=None,
                          query_text=None, result_count=0, source="mcp")
    assert db.execute.call_args.args[1]["memory_ids"] is None
```

- [ ] **Steps 2–4:** FAIL on unexpected kwarg → extend signature + INSERT (columns `memory_ids, :session_id` appended) → update both call sites → `pytest tests/ -k recall -v` all PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(recall): record served memory ids + session"`

---

### Task 6: Outcome grader — honest join, scheduled by construction

**Files:**
- Create: `services/async-worker/app/outcome_grader.py`
- Modify: `services/async-worker/app/main.py` (consume `SESSIONS_STORED` alongside the existing consumer)
- Test: `services/async-worker/tests/test_outcome_grader.py`

**Interfaces:**
- `grade_stored_session(db, s3, *, org_id, session_id, grader_version="v1") -> dict` returning `{"session_outcome_id": str | None, "attributions": int}`.
- Consumes: `shared.traces.fetch_trace`; `engine.pipeline.verify(output, session_memories=...)`; `shared.memory.embed_text` (existing) for usage attribution; recall events from Task 5; served memory contents via one org-predicated `SELECT id, content FROM session_memories WHERE id = ANY(:ids) AND org_id = :org_id`.
- Semantics (design corrections #1–#3): final output = last `role=="assistant"` message from structured trace; ONE `verify()` per session with `session_memories` = the served rows; per-memory `usage_score` = cosine(memory embedding, output embedding), `used = usage_score >= 0.5`; verdict/confidence stored once in `session_outcomes`.

- [ ] **Step 1: Write the failing test**

```python
# services/async-worker/tests/test_outcome_grader.py
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import outcome_grader as og


def _db(recalls, memories):
    db = MagicMock()
    def execute(stmt, params=None):
        m = MagicMock()
        s = str(stmt)
        if "FROM brain_recall_events" in s:
            m.mappings.return_value.all.return_value = recalls
        elif "FROM session_memories" in s:
            m.mappings.return_value.all.return_value = memories
        return m
    db.execute.side_effect = execute
    return db


TRACE = {"messages": [{"role": "user", "content": "q"},
                      {"role": "assistant", "content": "final answer about postgres"}],
         "tool_calls": [], "row": {}}


@pytest.mark.anyio
async def test_one_session_verdict_and_per_memory_attribution():
    recalls = [{"id": "re1", "memory_ids": ["m1", "m2"], "session_id": "s"}]
    memories = [{"id": "m1", "content": "postgres decision"},
                {"id": "m2", "content": "unrelated fact"}]
    verdict = MagicMock(confidence=0.9, action="pass")
    checks = {"memory_consistency": MagicMock(score=0.8)}
    verdict.checks = checks
    with patch.object(og, "fetch_trace", return_value=TRACE), \
         patch.object(og, "verify", new=AsyncMock(return_value=verdict)) as v, \
         patch.object(og, "embed_text", new=AsyncMock(side_effect=[
             [1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])):  # output, m1, m2
        result = await og.grade_stored_session(
            _db(recalls, memories), MagicMock(), org_id="o", session_id="s")
    # verify called ONCE, with the served memories
    assert v.await_count == 1
    assert v.await_args.kwargs["session_memories"] is not None
    assert result["attributions"] == 2


@pytest.mark.anyio
async def test_session_verdict_written_once_not_per_recall():
    recalls = [{"id": f"re{i}", "memory_ids": ["m1"], "session_id": "s"}
               for i in range(8)]
    memories = [{"id": "m1", "content": "x"}]
    verdict = MagicMock(confidence=0.5, action="flag", checks={})
    db = _db(recalls, memories)
    with patch.object(og, "fetch_trace", return_value=TRACE), \
         patch.object(og, "verify", new=AsyncMock(return_value=verdict)), \
         patch.object(og, "embed_text", new=AsyncMock(return_value=[1.0, 0.0])):
        await og.grade_stored_session(db, MagicMock(), org_id="o", session_id="s")
    session_inserts = [c for c in db.execute.call_args_list
                       if "INSERT INTO session_outcomes" in str(c.args[0])]
    assert len(session_inserts) == 1  # eight recalls, ONE session verdict


@pytest.mark.anyio
async def test_missing_trace_writes_unknown_session_outcome():
    recalls = [{"id": "re1", "memory_ids": ["m1"], "session_id": "s"}]
    db = _db(recalls, [{"id": "m1", "content": "x"}])
    with patch.object(og, "fetch_trace", return_value=None):
        result = await og.grade_stored_session(db, MagicMock(), org_id="o",
                                               session_id="s")
    inserts = [c for c in db.execute.call_args_list
               if "INSERT INTO session_outcomes" in str(c.args[0])]
    assert inserts and inserts[0].args[1]["verdict"] == "unknown"
```

- [ ] **Step 2: Run to verify it fails** — module missing.

- [ ] **Step 3: Implement**

```python
# services/async-worker/app/outcome_grader.py
"""Grade a stored session and attribute served memories — honestly.

Design corrections (binding):
- ONE verify() per session; verdict lives ONCE in session_outcomes.
- recall_outcomes rows carry attribution (used/usage_score), never a verdict.
- Final output = last assistant message from the STRUCTURED trace.
- verify() receives session_memories so memory_consistency actually runs.
Missing evidence grades as verdict='unknown' — coverage gaps must be visible.
"""
import logging
import math
import uuid
from typing import Any, Optional

from sqlalchemy import text

from engine.pipeline import verify
from shared.memory import embed_text
from shared.traces import fetch_trace

logger = logging.getLogger("agentguard.outcome-grader")

GRADER_VERSION = "v1"
USED_THRESHOLD = 0.5


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _final_output(messages: list[dict[str, Any]]) -> Optional[str]:
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            return str(m["content"])
    return None


async def grade_stored_session(
    db: Any, s3: Any, *, org_id: str, session_id: str,
    grader_version: str = GRADER_VERSION,
) -> dict[str, Any]:
    recalls = db.execute(
        text(
            """
            SELECT id, memory_ids, session_id FROM brain_recall_events
            WHERE org_id = :org_id AND session_id = :session_id
              AND memory_ids IS NOT NULL
            """
        ),
        {"org_id": org_id, "session_id": session_id},
    ).mappings().all()
    if not recalls:
        return {"session_outcome_id": None, "attributions": 0}

    served_ids = sorted({str(mid) for r in recalls for mid in (r["memory_ids"] or [])})
    memories = db.execute(
        text(
            """
            SELECT id, content FROM session_memories
            WHERE org_id = :org_id AND id = ANY(:ids)
            """
        ),
        {"org_id": org_id, "ids": served_ids},
    ).mappings().all()

    trace = fetch_trace(db, s3, org_id=org_id, session_id=session_id)
    output = _final_output(trace["messages"]) if trace else None

    verdict_str, confidence, mem_consistency = "unknown", None, None
    scores: dict[str, Optional[float]] = {str(m["id"]): None for m in memories}
    if output:
        try:
            result = await verify(
                output,
                session_memories=[dict(m) for m in memories],
            )
            verdict_str = result.action
            confidence = result.confidence
            mc = getattr(result, "checks", {}) or {}
            if "memory_consistency" in mc:
                mem_consistency = getattr(mc["memory_consistency"], "score", None)
        except Exception:
            logger.warning("grader: verify failed org=%s session=%s", org_id,
                           session_id, exc_info=True)
        try:
            out_emb = await embed_text(output)
            if out_emb:
                for m in memories:
                    m_emb = await embed_text(str(m["content"]))
                    if m_emb:
                        scores[str(m["id"])] = _cosine(out_emb, m_emb)
        except Exception:
            logger.warning("grader: attribution embedding failed", exc_info=True)

    session_outcome_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO session_outcomes
              (id, org_id, session_id, verdict, confidence, memory_consistency,
               grader_version)
            VALUES (:id, :org_id, :session_id, :verdict, :confidence,
                    :memory_consistency, :grader_version)
            ON CONFLICT (org_id, session_id, grader_version) DO NOTHING
            """
        ),
        {"id": session_outcome_id, "org_id": org_id, "session_id": session_id,
         "verdict": verdict_str, "confidence": confidence,
         "memory_consistency": mem_consistency, "grader_version": grader_version},
    )

    attributions = 0
    for recall in recalls:
        for mid in recall["memory_ids"] or []:
            score = scores.get(str(mid))
            db.execute(
                text(
                    """
                    INSERT INTO recall_outcomes
                      (id, org_id, recall_event_id, memory_id,
                       session_outcome_id, used, usage_score)
                    VALUES (:id, :org_id, :recall_event_id, :memory_id,
                            :session_outcome_id, :used, :usage_score)
                    """
                ),
                {"id": str(uuid.uuid4()), "org_id": org_id,
                 "recall_event_id": str(recall["id"]), "memory_id": str(mid),
                 "session_outcome_id": session_outcome_id,
                 "used": bool(score is not None and score >= USED_THRESHOLD),
                 "usage_score": score},
            )
            attributions += 1
    db.commit()
    return {"session_outcome_id": session_outcome_id, "attributions": attributions}
```

Wire in `main.py`: a consumer-group loop over `SESSIONS_STORED` (same pattern as the existing consumer) calling `grade_stored_session`.

- [ ] **Step 4: Run** → PASS, 3 tests.
- [ ] **Step 5: Commit** — `git commit -m "feat(async-worker): honest outcome grader as a stream consumer"`

---

### Task 7: Corrected stale-serve marking

**Files:**
- Create: `services/shared/shared/stale.py`
- Modify: `services/shared/shared/memory.py` — call inside `supersede_memory` with the NEW belief's created_at
- Test: `services/shared/tests/test_stale_serve_marking.py`

**Interfaces:** `mark_stale_serves(db, *, org_id, old_memory_id, superseding_created_at) -> int`. Predicate (design correction #4): recall events that served `old_memory_id` **at or after `superseding_created_at`** — the window where the newer belief already existed but the old one was still being served (curator lag). Never raises.

- [ ] **Step 1: Write the failing test**

```python
# services/shared/tests/test_stale_serve_marking.py
from datetime import datetime, timezone
from unittest.mock import MagicMock

from shared import stale


def test_predicate_is_after_superseding_belief_existed():
    db = MagicMock()
    db.execute.return_value.rowcount = 2
    ts = datetime(2026, 8, 1, tzinfo=timezone.utc)
    marked = stale.mark_stale_serves(
        db, org_id="o", old_memory_id="m-old", superseding_created_at=ts)
    assert marked == 2
    sql = " ".join(str(db.execute.call_args.args[0]).split())
    params = db.execute.call_args.args[1]
    assert "bre.created_at >= :superseding_created_at" in sql
    assert ":old_memory_id = ANY(bre.memory_ids)" in sql
    assert params["superseding_created_at"] == ts
    # the naive trailing-window predicate must NOT be present
    assert "make_interval" not in sql


def test_never_raises():
    db = MagicMock()
    db.execute.side_effect = RuntimeError("down")
    assert stale.mark_stale_serves(
        db, org_id="o", old_memory_id="m",
        superseding_created_at=datetime.now(timezone.utc)) == 0
```

- [ ] **Step 2: Run to verify it fails** → module missing.

- [ ] **Step 3: Implement**

```python
# services/shared/shared/stale.py
"""Retroactive stale-serve marking — the honest version.

A serve is stale ONLY if the superseding belief already existed when the old
one was served (curator lag). Serves before the new belief existed were
correct at the time and are never marked. This is the silent-failure counter;
its integrity is the point.
"""
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import text

logger = logging.getLogger("klio.stale")


def mark_stale_serves(
    db: Any, *, org_id: str, old_memory_id: str,
    superseding_created_at: datetime,
) -> int:
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
                  AND ro.memory_id = :old_memory_id
                  AND :old_memory_id = ANY(bre.memory_ids)
                  AND bre.created_at >= :superseding_created_at
                """
            ),
            {"org_id": org_id, "old_memory_id": old_memory_id,
             "superseding_created_at": superseding_created_at},
        )
        return int(result.rowcount or 0)
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning("stale marking failed org=%s memory=%s", org_id,
                       old_memory_id, exc_info=True)
        return 0
```

In `supersede_memory`, after the new row is written, call with the **new belief's** `created_at` (the moment the fact changed) and the old row's id — match the local variable names the function already uses.

- [ ] **Step 4: Run** shared suite `-k "stale or supersede"` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(memory): honest stale-serve marking (curator-lag window)"`

---

### Task 8: Migration 043 + CAS on write tools (Build 3)

**Files:**
- Create: `services/migrations/alembic/versions/043_memory_revisions.py`
- Modify: `services/shared/shared/memory.py` (CAS supersede path), `services/mcp-server/app/mcp_app.py` (write tools accept `base_revision`; recall returns `revision`)
- Test: `services/migrations/tests/test_043_revisions.py`, `services/shared/tests/test_cas_write.py`

**Interfaces:**
- Migration: `session_memories` += `revision UUID NOT NULL server_default gen_random_uuid()`.
- `shared.memory.supersede_with_revision(db, *, org_id, memory_id, base_revision, new_content, **write_kwargs) -> dict` — atomic guard `UPDATE ... WHERE id=:id AND org_id=:org_id AND revision=:base_revision`; on 0 rows returns `{"conflict": True, "current": {id, content, revision, updated_by}}` (the 409 payload — agent re-reads and retries, Mukta's loop); on success proceeds through the EXISTING supersession machinery and returns `{"conflict": False, "new_id": ..., "new_revision": ...}`.
- MCP: `remember`/`decide`/`note`/`observe` gain optional `base_revision: str | None`; absent = today's behaviour exactly. Recall serialization includes `revision`.
- Metric: `klio_write_conflicts_total` counter (follow the metrics pattern in `curator.py`).

- [ ] **Step 1: Write the failing tests**

```python
# services/shared/tests/test_cas_write.py
from unittest.mock import MagicMock, patch

from shared import memory as memory_mod


def test_stale_base_revision_returns_conflict_payload():
    db = MagicMock()
    guard = MagicMock(rowcount=0)  # CAS guard missed
    current = MagicMock()
    current.mappings.return_value.first.return_value = {
        "id": "m1", "content": "newer text", "revision": "r2", "user_id": "u2"}
    db.execute.side_effect = [guard, current]
    result = memory_mod.supersede_with_revision(
        db, org_id="o", memory_id="m1", base_revision="r1", new_content="mine")
    assert result["conflict"] is True
    assert result["current"]["revision"] == "r2"
    sql = " ".join(str(db.execute.call_args_list[0].args[0]).split())
    assert "revision = :base_revision" in sql and "org_id = :org_id" in sql


def test_current_base_revision_proceeds_to_supersede():
    db = MagicMock()
    db.execute.return_value.rowcount = 1
    with patch.object(memory_mod, "_apply_supersession_for_cas",
                      return_value={"new_id": "m2", "new_revision": "r9"}) as ap:
        result = memory_mod.supersede_with_revision(
            db, org_id="o", memory_id="m1", base_revision="r1",
            new_content="mine")
    assert result == {"conflict": False, "new_id": "m2", "new_revision": "r9"}
    assert ap.called
```

(Migration test mirrors 041/042: `revision` column exists, NOT NULL.)

- [ ] **Step 2: Run to verify failure** → missing function/column.

- [ ] **Step 3: Implement**

Migration 043:

```python
def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.add_column(
        "session_memories",
        sa.Column("revision", postgresql.UUID(as_uuid=True),
                  server_default=sa.text("gen_random_uuid()"), nullable=False),
    )
```

`supersede_with_revision` in `memory.py`:

```python
def supersede_with_revision(
    db: Any, *, org_id: str, memory_id: str, base_revision: str,
    new_content: str, **write_kwargs: Any,
) -> dict[str, Any]:
    """Mukta's optimistic-concurrency loop, server side.

    The guard is one atomic UPDATE: it bumps the revision ONLY if the caller's
    base_revision is still current. Zero rows means someone moved the belief
    while this agent worked — return the current state so the agent re-reads
    and re-asserts, instead of silently last-writer-winning.
    """
    guard = db.execute(
        text(
            """
            UPDATE session_memories
            SET revision = gen_random_uuid()
            WHERE id = :memory_id AND org_id = :org_id
              AND revision = :base_revision
            """
        ),
        {"memory_id": memory_id, "org_id": org_id,
         "base_revision": base_revision},
    )
    if not guard.rowcount:
        record_write_conflict()  # klio_write_conflicts_total
        current = db.execute(
            text(
                """
                SELECT id, content, revision, user_id
                FROM session_memories
                WHERE id = :memory_id AND org_id = :org_id
                """
            ),
            {"memory_id": memory_id, "org_id": org_id},
        ).mappings().first()
        return {"conflict": True,
                "current": dict(current) if current else None}
    return {"conflict": False,
            **_apply_supersession_for_cas(
                db, org_id=org_id, old_id=memory_id,
                new_content=new_content, **write_kwargs)}
```

`_apply_supersession_for_cas` delegates to the existing supersession write path (`supersede_memory` internals) — the implementer wires it to whatever `supersede_memory` already calls, changing none of its semantics. In `mcp_app.py`: thread optional `base_revision` through the four write tools (absent → current behaviour, no CAS); on `conflict: True` return the tool-error payload `{"error": "revision_conflict", "current": {...}, "hint": "belief changed while you worked — re-read and re-assert"}`. Add `revision` to `_serialize_memory`.

- [ ] **Step 4: Run** shared + mcp-server suites → new PASS, zero regressions (absent field = old path).
- [ ] **Step 5: Commit** — `git commit -m "feat(memory): revision CAS on writes; conflict returns current state"`

---

### Task 9: Migration 044 + claims (Build 4)

**Files:**
- Create: `services/migrations/alembic/versions/044_work_claims.py`, `services/shared/shared/claims.py`
- Modify: `services/mcp-server/app/mcp_app.py` (tools `claim`, `release`; claims injected into recall responses)
- Test: `services/shared/tests/test_claims.py`, `services/mcp-server/tests/test_claim_tools.py`

**Interfaces:**
- Migration: `work_claims`: `id UUID pk, org_id String(64) NOT NULL, project_id UUID NULL, agent_id Text NULL, user_id UUID NULL, description Text NOT NULL, embedding vector(1024) NULL, created_at timestamptz DEFAULT now(), expires_at timestamptz NOT NULL, released_at timestamptz NULL`; index `(org_id, expires_at)`.
- `shared.claims.create_claim(db, *, org_id, description, agent_id, user_id=None, project_id=None, ttl_minutes=15) -> dict` — embeds description (`embed_text`), inserts, then returns `{"claim_id", "collisions": [{id, description, agent_id, similarity}]}` where collisions are live claims (`released_at IS NULL AND expires_at > now()`) with cosine similarity ≥ 0.85, computed via pgvector `1 - (embedding <=> :emb)`.
- `shared.claims.release_claim(db, *, org_id, claim_id, agent_id) -> bool` (owner-only).
- `shared.claims.live_claims(db, *, org_id, project_id=None, limit=10) -> list[dict]` — injected into every recall response under `"active_claims"` (design: agents already recall before working; they cannot not see claims). Advisory on cloud; local `PreToolUse` blocking is the client's job.

- [ ] **Step 1: Write the failing tests**

```python
# services/shared/tests/test_claims.py
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared import claims


@pytest.mark.anyio
async def test_create_claim_returns_semantic_collisions():
    db = MagicMock()
    db.execute.return_value.mappings.return_value.all.return_value = [
        {"id": "c-old", "description": "refactor auth", "agent_id": "B",
         "similarity": 0.91}]
    with patch.object(claims, "embed_text", new=AsyncMock(return_value=[0.1] * 4)):
        result = await claims.create_claim(
            db, org_id="o", description="fix login token expiry", agent_id="A")
    assert result["collisions"][0]["agent_id"] == "B"
    sql = " ".join(str(db.execute.call_args_list[-1].args[0]).split())
    assert "released_at IS NULL" in sql and "expires_at > now()" in sql
    assert "org_id = :org_id" in sql


@pytest.mark.anyio
async def test_claim_without_embedding_still_registers():
    db = MagicMock()
    db.execute.return_value.mappings.return_value.all.return_value = []
    with patch.object(claims, "embed_text", new=AsyncMock(return_value=None)):
        result = await claims.create_claim(
            db, org_id="o", description="x", agent_id="A")
    assert result["claim_id"]
    assert result["collisions"] == []


def test_release_is_owner_scoped():
    db = MagicMock()
    db.execute.return_value.rowcount = 1
    assert claims.release_claim(db, org_id="o", claim_id="c1", agent_id="A")
    sql = " ".join(str(db.execute.call_args.args[0]).split())
    assert "agent_id = :agent_id" in sql and "org_id = :org_id" in sql
```

- [ ] **Step 2: Run to verify failure** → module missing.

- [ ] **Step 3: Implement** migration 044 (vector column matches the 1024-dim e5 embeddings used by `session_memories`), then:

```python
# services/shared/shared/claims.py
"""Work claims: advisory, semantic, TTL'd awareness (Build 4).

NOT locks. A claim records intent; collisions are detected semantically
("auth refactor" ≈ "fix login token expiry") because exact-key locking is
useless against natural language. TTL + released_at handle crashed agents
with no cleanup job. Enforcement is the client's business: cloud agents get
awareness, local PreToolUse hooks may block.
"""
import logging
import uuid
from typing import Any, Optional

from sqlalchemy import text

from shared.memory import embed_text

logger = logging.getLogger("klio.claims")

COLLISION_THRESHOLD = 0.85


async def create_claim(
    db: Any, *, org_id: str, description: str, agent_id: str,
    user_id: Optional[str] = None, project_id: Optional[str] = None,
    ttl_minutes: int = 15,
) -> dict[str, Any]:
    claim_id = str(uuid.uuid4())
    emb = None
    try:
        emb = await embed_text(description)
    except Exception:
        logger.warning("claims: embed failed; registering without collision "
                       "detection", exc_info=True)
    db.execute(
        text(
            """
            INSERT INTO work_claims
              (id, org_id, project_id, agent_id, user_id, description,
               embedding, expires_at)
            VALUES (:id, :org_id, :project_id, :agent_id, :user_id,
                    :description, :embedding,
                    now() + make_interval(mins => :ttl))
            """
        ),
        {"id": claim_id, "org_id": org_id, "project_id": project_id,
         "agent_id": agent_id, "user_id": user_id, "description": description,
         "embedding": str(emb) if emb else None, "ttl": ttl_minutes},
    )
    collisions: list[dict[str, Any]] = []
    if emb:
        rows = db.execute(
            text(
                """
                SELECT id, description, agent_id,
                       1 - (embedding <=> :emb) AS similarity
                FROM work_claims
                WHERE org_id = :org_id AND id != :id
                  AND released_at IS NULL AND expires_at > now()
                  AND embedding IS NOT NULL
                  AND 1 - (embedding <=> :emb) >= :threshold
                ORDER BY similarity DESC LIMIT 5
                """
            ),
            {"emb": str(emb), "org_id": org_id, "id": claim_id,
             "threshold": COLLISION_THRESHOLD},
        ).mappings().all()
        collisions = [dict(r) for r in rows]
    db.commit()
    return {"claim_id": claim_id, "collisions": collisions}


def release_claim(db: Any, *, org_id: str, claim_id: str, agent_id: str) -> bool:
    result = db.execute(
        text(
            """
            UPDATE work_claims SET released_at = now()
            WHERE id = :claim_id AND org_id = :org_id
              AND agent_id = :agent_id AND released_at IS NULL
            """
        ),
        {"claim_id": claim_id, "org_id": org_id, "agent_id": agent_id},
    )
    db.commit()
    return bool(result.rowcount)


def live_claims(
    db: Any, *, org_id: str, project_id: Optional[str] = None, limit: int = 10
) -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            """
            SELECT id, description, agent_id, user_id, expires_at
            FROM work_claims
            WHERE org_id = :org_id AND released_at IS NULL
              AND expires_at > now()
              AND (:project_id::uuid IS NULL OR project_id = :project_id)
            ORDER BY created_at DESC LIMIT :limit
            """
        ),
        {"org_id": org_id, "project_id": project_id, "limit": limit},
    ).mappings().all()
    return [dict(r) for r in rows]
```

MCP wiring: `claim(description, ttl_minutes=15)` and `release(claim_id)` tools (thin wrappers, org/agent from the principal); recall handlers append `"active_claims": live_claims(...)` to their response dict.

- [ ] **Step 4: Run** shared + mcp-server suites → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(claims): semantic advisory claims, injected into recall"`

---

### Task 10: Migration 045 + dreamer (Build 5)

**Files:**
- Create: `services/migrations/alembic/versions/045_memory_proposals.py`, `services/async-worker/app/dreamer.py`
- Modify: `services/async-worker/app/main.py` (interval schedule: run per org after `DREAM_MIN_GRADED_SESSIONS` new graded sessions, default 25, or every `DREAM_INTERVAL_HOURS`, default 24)
- Test: `services/async-worker/tests/test_dreamer.py`

**Interfaces:**
- Migration: `memory_proposals`: `id UUID pk, org_id String(64) NOT NULL, scope Text NOT NULL, kind Text NOT NULL` (`add|revise|retire`), `target_memory_id UUID NULL, diff Text NOT NULL, evidence JSONB NOT NULL` (trace refs, session ids, prevalence, outcome stats), `confidence Numeric NULL, status Text NOT NULL DEFAULT 'open'` (`open|approved|rejected|auto`), `created_at/decided_at timestamptz, decided_by UUID NULL`; index `(org_id, status)`.
- `dreamer.dream_for_org(db, s3, *, org_id, batch_limit=50) -> int` — returns proposals written. Inputs: recent `session_outcomes` + `recall_outcomes` + traces + current memories. Deterministic detectors FIRST (no LLM): (a) **repeated tool failure** — same `tool` with `status=="error"` in ≥3 sessions' tool_calls; (b) **bad-outcome memory** — memory with ≥5 attributions, `used=true` rate ≥ 0.4, and joined session verdict `pass` rate < 0.5; (c) **stale-serve cluster** — memory with ≥3 `served_stale` rows. Each detector emits a proposal with the evidence rows it counted. An LLM drafting pass (via the same `acompletion` config the curator uses) turns detector output into the `diff` text — the LLM WRITES PROSE ONLY; detection is arithmetic, so dreaming cannot hallucinate a pattern.
- Proposal application: approved proposals apply through `supersede_with_revision` (Task 8) — the dreamer obeys CAS like every writer. `status='auto'` allowed only for `scope='agent'`.
- REST (mcp-server): `GET /proposals?status=open`, `POST /proposals/{id}/approve`, `POST /proposals/{id}/reject` — org-scoped via the existing auth dependency; approve requires an attributable user (`resolve_user_id`), recorded in `decided_by`. (Dashboard UI consumes these; that surface is a Dashboard-repo task, out of this plan.)

- [ ] **Step 1: Write the failing tests**

```python
# services/async-worker/tests/test_dreamer.py
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import dreamer


def _sessions_with_tool_failures(n):
    return [{"session_id": f"s{i}",
             "tool_calls": [{"tool": "Bash", "status": "error",
                             "args": "npm test"}]} for i in range(n)]


@pytest.mark.anyio
async def test_repeated_tool_failure_becomes_proposal_with_prevalence():
    db = MagicMock()
    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": _sessions_with_tool_failures(3),
            "bad_memories": [], "stale_clusters": []}), \
         patch.object(dreamer, "_draft_diff", new=AsyncMock(return_value="Add a note: npm test fails without X")):
        written = await dreamer.dream_for_org(db, MagicMock(), org_id="o")
    assert written == 1
    ins = [c for c in db.execute.call_args_list
           if "INSERT INTO memory_proposals" in str(c.args[0])][0]
    evidence = json.loads(ins.args[1]["evidence"])
    assert evidence["prevalence"] == 3
    assert ins.args[1]["status"] == "open"


@pytest.mark.anyio
async def test_two_failures_is_below_threshold_no_proposal():
    db = MagicMock()
    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": _sessions_with_tool_failures(2),
            "bad_memories": [], "stale_clusters": []}):
        written = await dreamer.dream_for_org(db, MagicMock(), org_id="o")
    assert written == 0


@pytest.mark.anyio
async def test_llm_failure_still_files_proposal_with_raw_evidence():
    db = MagicMock()
    with patch.object(dreamer, "_load_recent_evidence", return_value={
            "sessions": _sessions_with_tool_failures(4),
            "bad_memories": [], "stale_clusters": []}), \
         patch.object(dreamer, "_draft_diff", new=AsyncMock(side_effect=RuntimeError)):
        written = await dreamer.dream_for_org(db, MagicMock(), org_id="o")
    assert written == 1  # detection is arithmetic; drafting is optional prose
```

- [ ] **Step 2: Run to verify failure** → module missing.

- [ ] **Step 3: Implement** — `dreamer.py` with: `_load_recent_evidence` (three org-predicated SQL queries + trace fetches for tool_calls), the three deterministic detectors as pure functions over that dict, `_draft_diff` (LLM prose via the curator's `acompletion` config; on failure the proposal's diff is the deterministic summary line, e.g. `"[auto] Bash failed identically in 4 sessions: npm test"`), and one `INSERT INTO memory_proposals` per detection with `evidence` JSON `{"prevalence": n, "sessions": [...], "detector": "tool_failure"}`. Migration 045 in the established style. REST endpoints in `mcp_app.py` following the existing FastAPI route patterns with the standard auth dependencies.

- [ ] **Step 4: Run** → PASS, 3 tests; mcp-server suite green.
- [ ] **Step 5: Commit** — `git commit -m "feat(dreamer): deterministic detection, LLM prose, reviewable proposals"`

---

### Task 11: Retention + wiring sanity (whole-system)

**Files:**
- Modify: retention sweep (locate via `run_retention.py` imports) — traces AND outcome tables expire with plan retention; claims are self-expiring; proposals persist (they are decisions, not evidence)
- Test: `services/shared/tests/test_evidence_retention_sweep.py` (DB-gated, mirrors the Task 8 pattern from the reviewed draft: expired deleted, fresh kept, non-positive retention skipped — for `session_traces`, `session_outcomes`, `recall_outcomes`)
- Full-suite run: every service's pytest green.

- [ ] **Step 1–4:** failing DB-gated test → `sweep_evidence_for_org(conn, *, org_id, retention_days) -> dict[str, int]` deleting from the three tables with the same guard (`<= 0` → skip) → wire into the per-org loop → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(retention): evidence expires with plan retention"`

---

## Deployment notes

- Migrations 041–045 run manually against the Neon **direct** endpoint, **one revision at a time** (`alembic upgrade 041`, then `042`, `043`, `044`, `045` — not a single `upgrade head`), against the Neon **direct** endpoint (never the pooled one — see the pooler-constraints note below). Running one revision at a time keeps 043's `autocommit_block()` backfill (which commits mid-migration, outside Alembic's normal one-transaction-per-migration wrapping) isolated and observable per step, rather than folded into a single multi-migration run where a failure's blast radius is harder to reason about.
- New env — storage-worker: `KLIO_TRACE_BUCKET` **before deploy**. As of the final whole-branch review this requirement is scoped tighter than "the process refuses to start": only the traces consumer (`sessions.captured`) refuses to start without it — the legacy `executions.raw`/`executions.verified` consumers start and run regardless, unaffected by a missing `KLIO_TRACE_BUCKET`. Still set it before deploying storage-worker: without it, session-evidence retention (the point of this build) silently does not happen, even though the rest of the service is healthy. Also: `REDACTION_VERSION=1`.
- New env — async-worker: `DREAM_MIN_GRADED_SESSIONS=25`, `DREAM_INTERVAL_HOURS=24`, and **`S3_ENDPOINT`/`S3_ACCESS_KEY`/`S3_SECRET_KEY`** — the outcome grader reads trace objects through these same generic `S3_*` vars (see `app.s3_client`). If they're missing or left at the localhost/MinIO dev defaults in a real deployment, `fetch_trace` cannot reach the bucket traces were written to and **every session grades `verdict='unknown'`** — there is no crash, no error surfaced anywhere else, just silent all-unknown grading. `app.main` now logs an error-level line at startup whenever this looks misconfigured (no fail-fast — grading degrading to unknown must not be worse than not starting), so check the async-worker startup log on deploy.
- New env — mcp-server: `REDIS_URL` (verify it isn't already set).
- Rollout order: migrations → shared → storage-worker → mcp-server → async-worker.
- The S3 bucket's **lifecycle expiry rule is load-bearing**, not a nice-to-have: it is the only mechanism that actually deletes trace objects. Application code only does best-effort deletes on the redacted-trace write/replace path (`app.trace_consumer`; see that module's docstring) — a failed best-effort delete, or any other orphaned object, is cleaned up ONLY by the bucket's own lifecycle rule. If that rule isn't configured to match the max plan retention window, retained session evidence outlives its retention policy with nothing else to catch it.
- The Dashboard proposals UI (list/diff/approve) consumes the REST endpoints and is a separate Dashboard-repo task.
- The 5% holdout experiment (prove-it pitch) consumes `session_outcomes` and ships after this plan; the schema already supports it (`grader_version` distinguishes arms).

### Follow-ups filed, not in this branch

- **M4** — `revise` proposal approval has no surface for a human to supply replacement content when the detector left `proposed_content` NULL (the `bad_outcome_memory` detector always does — it has evidence a belief is hurting outcomes, not a drafted replacement). Needs a content-supplying step in the review flow before this path is usable end-to-end. Auto-apply is unreachable in the meantime for anything routed through that gap.
- **M6** — the dreamer's per-detector evidence queries (`_load_recent_evidence`) have no fairness guarantee across detectors when aggregate `LIMIT`s are hit — a busy detector's evidence can crowd out a quieter one's within the same lookback window. Needs a fair-interleave pass across detections before filing, not just per-query limits.
- **L1** — the proposal payload trim in the review surface is O(n²) in the number of messages, with no message cap; fine at current volume, but will need a linear trim + explicit cap before proposal payloads grow much larger.
- **L2** — no reaper exists for proposals that are `decided` (approved/rejected) but never actually applied (e.g. a client that approved and then never followed through, or a partial failure that wasn't retried). Needs a periodic sweep to surface or re-open these rather than letting them sit invisibly stale.
- **L5** — agent impersonation in the auto-apply path (a caller claiming an `agent_id` it doesn't own, then relying on auto-apply's no-user-principal path to write) is moot while M4 holds, because auto-apply is unreachable without a content-supplying surface — but the underlying gap should be closed for real once M4 ships, not left load-bearing on M4 staying unfinished.

## Out of scope

- `klio init` / hooks / local daemon (client-side repo).
- Dashboard proposals UI (Dashboard repo).
- Cross-org learning (excluded until legal/consent design exists).
