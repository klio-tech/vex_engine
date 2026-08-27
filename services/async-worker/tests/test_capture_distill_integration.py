"""The queue end to end, against real Postgres.

The unit tests patch the write primitive, so they prove the distiller passes
the right kwargs -- not that a memory actually lands, readable, at the scope
the request authorised. This closes that gap: a real ``capture_jobs`` row, the
real ``write_scoped_memory``, and a real ``session_memories`` row read back.

Only the two LLM calls are faked. Everything between the job row and the
stored memory is the production code path.
"""

import random
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

from app import capture_distill
from shared import memory as memory_mod
from shared.capture_jobs import create_job
from shared.taxonomy import ExtractedMemory
from shared.testing import database_url, requires_db

DB_URL = database_url()
pytestmark = requires_db("the capture distillation integration test")

ORG = "org-distil-int"
AGENT = "agent-distil-int"
USER = "user-distil-int"
EMBED_DIM = 1024


async def _fake_embed(content, **_kwargs):
    """A deterministic vector whose DIRECTION varies with the content.

    A constant vector (``[base] * dim``) is the obvious fake and is wrong
    here: any two constant vectors have cosine similarity 1.0 no matter what
    the constant is, so every memory would dedup against every other one and
    this test would silently assert nothing.
    """
    rng = random.Random(content)
    return [rng.uniform(-1.0, 1.0) for _ in range(EMBED_DIM)]


@pytest.fixture(autouse=True)
def _patch_embedding(monkeypatch):
    monkeypatch.setattr(memory_mod, "embed_text", _fake_embed)


@pytest.fixture()
def db():
    conn = create_engine(DB_URL).connect()
    for tbl in ("session_memories", "capture_jobs"):
        conn.execute(text(f"DELETE FROM {tbl} WHERE org_id = :o"), {"o": ORG})
    conn.commit()
    yield conn
    for tbl in ("session_memories", "capture_jobs"):
        conn.execute(text(f"DELETE FROM {tbl} WHERE org_id = :o"), {"o": ORG})
    conn.commit()
    conn.close()


async def test_a_queued_transcript_becomes_readable_memory_at_its_own_scope(db):
    job_id = create_job(
        db,
        org_id=ORG,
        agent_id=AGENT,
        session_id="sess-int-1",
        transcript="user: we moved distillation off the request path",
        scope="private",
        user_id=USER,
        metadata={"source": "hook-transcript"},
    )

    with (
        patch.object(
            capture_distill,
            "generate_summary",
            AsyncMock(return_value="Distillation moved off the request path."),
        ),
        patch.object(
            capture_distill,
            "extract_typed_memories",
            AsyncMock(
                return_value=[
                    ExtractedMemory(
                        content="Transcript distillation runs in async-worker.",
                        memory_type="decision",
                    )
                ]
            ),
        ),
    ):
        assert await capture_distill.distill_job(db, job_id) == "done"

    rows = db.execute(
        text(
            "SELECT memory_type, content, scope, user_id, agent_id "
            "  FROM session_memories WHERE org_id = :o ORDER BY memory_type"
        ),
        {"o": ORG},
    ).all()
    assert [r.memory_type for r in rows] == ["decision", "summary"]
    # The boundary the request authorised must be the boundary on disk. A row
    # that came back 'org' here would be readable by the whole tenant.
    for r in rows:
        assert r.scope == "private"
        assert r.user_id == USER
        assert r.agent_id == AGENT

    job = db.execute(
        text(
            "SELECT status, summary_written, memories_written, attempts "
            "  FROM capture_jobs WHERE id = :i"
        ),
        {"i": job_id},
    ).one()
    assert (job.status, job.summary_written, job.memories_written, job.attempts) == (
        "done",
        True,
        1,
        1,
    )


async def test_redelivery_of_a_finished_job_writes_nothing_twice(db):
    """The stream is at-least-once; the memory table must stay exactly-once."""
    job_id = create_job(
        db,
        org_id=ORG,
        agent_id=AGENT,
        session_id="sess-int-2",
        transcript="user: idempotence matters",
        scope="org",
    )
    patches = (
        patch.object(capture_distill, "generate_summary", AsyncMock(return_value="S.")),
        patch.object(
            capture_distill,
            "extract_typed_memories",
            AsyncMock(return_value=[ExtractedMemory(content="A.", memory_type="fact")]),
        ),
    )
    for p in patches:
        p.start()
    try:
        assert await capture_distill.distill_job(db, job_id) == "done"
        assert await capture_distill.distill_job(db, job_id) == "skipped"
    finally:
        for p in patches:
            p.stop()

    count = db.execute(
        text("SELECT count(*) FROM session_memories WHERE org_id = :o"), {"o": ORG}
    ).scalar()
    assert count == 2


async def test_a_model_failure_leaves_a_readable_reason_on_the_job(db):
    """The defect this change exists to remove: a failure that looked exactly
    like a turn with nothing worth keeping."""
    job_id = create_job(
        db,
        org_id=ORG,
        agent_id=AGENT,
        session_id="sess-int-3",
        transcript="user: something durable",
        scope="org",
    )
    with patch.object(
        capture_distill,
        "generate_summary",
        AsyncMock(side_effect=RuntimeError("gateway 503")),
    ):
        assert await capture_distill.distill_job(db, job_id) == "failed"

    job = db.execute(
        text("SELECT status, error FROM capture_jobs WHERE id = :i"), {"i": job_id}
    ).one()
    assert job.status == "failed"
    assert "gateway 503" in job.error
    assert (
        db.execute(
            text("SELECT count(*) FROM session_memories WHERE org_id = :o"), {"o": ORG}
        ).scalar()
        == 0
    )
