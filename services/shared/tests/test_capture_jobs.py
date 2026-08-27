"""The capture-job queue: durability and the redelivery guard. Real Postgres.

These are the two properties the distiller's correctness rests on, and both
are SQL, so a mock would only echo the string back:

* ``claim_job`` is an atomic compare-and-set. A Redis consumer group redelivers
  on worker death, and distillation costs two LLM calls and writes memories --
  running one twice is money and duplicate rows, so the SECOND claim of a
  running or finished job must come back ``None``.
* a failed job is re-claimable, because a transient model failure should not
  bury a transcript forever.
"""

import pytest
from sqlalchemy import create_engine, text

from shared.capture_jobs import (
    claim_job,
    create_job,
    fail_job,
    finish_job,
)
from shared.testing import database_url, requires_db

DB_URL = database_url()

pytestmark = requires_db("the capture job queue tests")

ORG = "org-capture-jobs"
AGENT = "agent-capture-jobs"


@pytest.fixture()
def db():
    engine = create_engine(DB_URL)
    conn = engine.connect()
    conn.execute(text("DELETE FROM capture_jobs WHERE org_id = :o"), {"o": ORG})
    conn.commit()
    yield conn
    conn.execute(text("DELETE FROM capture_jobs WHERE org_id = :o"), {"o": ORG})
    conn.commit()
    conn.close()


def _new(db, **over):
    kwargs = dict(
        org_id=ORG,
        agent_id=AGENT,
        session_id="sess-1",
        transcript="user: hello\nassistant: hi",
        scope="private",
    )
    kwargs.update(over)
    return create_job(db, **kwargs)


def test_create_job_persists_the_transcript_and_starts_pending(db):
    job_id = _new(db)
    row = db.execute(
        text(
            "SELECT status, attempts, transcript, scope FROM capture_jobs WHERE id = :i"
        ),
        {"i": job_id},
    ).one()
    assert row.status == "pending"
    assert row.attempts == 0
    assert row.transcript == "user: hello\nassistant: hi"
    assert row.scope == "private"


def test_claim_job_returns_the_work_and_marks_it_running(db):
    job_id = _new(db, project_id=None, user_id="u-1")
    claimed = claim_job(db, job_id)
    assert claimed is not None
    assert claimed["org_id"] == ORG
    assert claimed["user_id"] == "u-1"
    assert claimed["transcript"].startswith("user: hello")
    row = db.execute(
        text("SELECT status, attempts, started_at FROM capture_jobs WHERE id = :i"),
        {"i": job_id},
    ).one()
    assert row.status == "running"
    assert row.attempts == 1
    assert row.started_at is not None


def test_second_claim_of_a_running_job_is_refused(db):
    """Redis redelivers; distillation must not run twice for one transcript."""
    job_id = _new(db)
    assert claim_job(db, job_id) is not None
    assert claim_job(db, job_id) is None


def test_a_finished_job_is_never_reclaimed(db):
    job_id = _new(db)
    claim_job(db, job_id)
    finish_job(db, job_id, summary_written=True, memories_written=3)
    assert claim_job(db, job_id) is None
    row = db.execute(
        text(
            "SELECT status, summary_written, memories_written, finished_at "
            "FROM capture_jobs WHERE id = :i"
        ),
        {"i": job_id},
    ).one()
    assert row.status == "done"
    assert row.summary_written is True
    assert row.memories_written == 3
    assert row.finished_at is not None


def test_a_failed_job_is_reclaimable_and_keeps_its_error(db):
    """A transient model failure must not bury the transcript forever."""
    job_id = _new(db)
    claim_job(db, job_id)
    fail_job(db, job_id, error="model timed out")
    row = db.execute(
        text("SELECT status, error FROM capture_jobs WHERE id = :i"), {"i": job_id}
    ).one()
    assert row.status == "failed"
    assert row.error == "model timed out"

    reclaimed = claim_job(db, job_id)
    assert reclaimed is not None
    assert (
        db.execute(
            text("SELECT attempts FROM capture_jobs WHERE id = :i"), {"i": job_id}
        ).scalar()
        == 2
    )


def test_claiming_an_unknown_job_is_none_not_an_error(db):
    """A stream entry can outlive its row (retention, manual cleanup)."""
    assert claim_job(db, "00000000-0000-0000-0000-000000000000") is None


def test_failure_reason_is_truncated_rather_than_overflowing_the_row(db):
    job_id = _new(db)
    claim_job(db, job_id)
    fail_job(db, job_id, error="x" * 10_000)
    stored = db.execute(
        text("SELECT error FROM capture_jobs WHERE id = :i"), {"i": job_id}
    ).scalar()
    assert len(stored) <= 2_000
