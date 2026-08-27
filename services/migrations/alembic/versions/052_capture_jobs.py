"""Capture jobs: move transcript distillation off the request path.

Revision ID: 052
Revises: 051

``POST /capture/transcript`` used to call the LLM twice — ``generate_summary``
then ``extract_typed_memories`` — INSIDE the request handler, so the Stop hook
that posted the transcript blocked until both finished. The 10s
``VERIFICATION_TIMEOUT_S`` existed to bound that stall, and it was doing real
damage: measured against 40 real captured transcripts, only ~67% of turns
produced BOTH a summary and its distilled memories inside 10s; the rest
returned ``None`` / ``[]``, which is indistinguishable from "this turn had
nothing durable in it". Roughly a third of every captured session was being
dropped, silently, by a timeout whose only job was to protect a latency budget
that distillation does not actually need.

This table is the durable half of the fix. The request handler resolves
identity, scope, project and branch as before — all the things that must fail
LOUDLY and synchronously — then writes the redacted transcript here and
returns. ``async-worker`` picks the job up and distils it with a timeout sized
for the work rather than for a hook.

Why a table and not just the Redis payload: the stream contract for
``agents.turns`` is IDS ONLY, never content, because a worker can always
re-read content from the database. A transcript has no such home — it exists
only in the request — so it needs one before the stream can stay honest. The
table also buys the thing the old path most conspicuously lacked: a distiller
failure is now a ``status='failed'`` row with an ``error`` on it, not a
warning line in a log nobody reads.

``transcript`` holds text that has ALREADY been through
``shared.redaction.redact`` at ``_combine_messages``. Nothing unredacted is
written here.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "052"
down_revision = "051"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "capture_jobs",
        sa.Column(
            "id",
            sa.dialects.postgresql.UUID(as_uuid=False),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        # Identity and filing, all resolved synchronously in the request so a
        # bad key or an unreachable node still fails at the caller.
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("agent_id", sa.String(128), nullable=False),
        sa.Column("session_id", sa.String(256), nullable=False),
        sa.Column("user_id", sa.String(64), nullable=True),
        sa.Column("scope", sa.String(32), nullable=False),
        sa.Column(
            "project_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=True
        ),
        sa.Column(
            "branch_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=True
        ),
        sa.Column(
            "work_item_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=True
        ),
        # Already redacted. See the module docstring.
        sa.Column("transcript", sa.Text(), nullable=False),
        sa.Column("metadata", JSONB, nullable=True),
        # ``over_cap`` is decided at REQUEST time, not distil time: the quota
        # answer belongs to the moment the caller asked, and re-asking it in
        # the worker would charge a different month's bucket for a transcript
        # that queued at a month boundary.
        sa.Column(
            "over_cap", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column(
            "attempts", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("summary_written", sa.Boolean(), nullable=True),
        sa.Column("memories_written", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending','running','done','failed')",
            name="ck_capture_jobs_status",
        ),
    )
    # The operator query this table exists to make possible: what is stuck,
    # what failed, and for whom. Partial because 'done' is the overwhelming
    # majority within a day of any given job and nobody pages on those.
    op.create_index(
        "ix_capture_jobs_unfinished",
        "capture_jobs",
        ["status", "created_at"],
        postgresql_where=sa.text("status <> 'done'"),
    )
    op.create_index(
        "ix_capture_jobs_org_created", "capture_jobs", ["org_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_capture_jobs_org_created", table_name="capture_jobs")
    op.drop_index("ix_capture_jobs_unfinished", table_name="capture_jobs")
    op.drop_table("capture_jobs")
