"""Dream state: the dreamer's trigger cursor, persisted (Build 5 follow-up).

WHY THIS TABLE EXISTS. ``app.main._maybe_dream_for_org`` tracked both dream
triggers — "N new graded sessions" and "M hours since the last pass" — in a
process-local dict (``dream_state: dict[str, dict[str, Any]]``, keyed by
``time.monotonic()``). ``async-worker`` redeploys routinely, and every
redeploy reset BOTH the session counter and the interval clock to zero. In
production that meant neither trigger was ever reached: ``memory_proposals``
held zero rows since the feature shipped. The counter is not an
optimisation, it is the trigger — so it belongs in the only store that
survives a deploy.

This is deliberately NOT modelled on the ownership sweep's
no-persisted-cursor contract (``app.main``'s ``next_sweep_at``). That
contract is correct for the sweep because the sweep is idempotent and
purely time-driven: a reset clock costs at most one delayed pass. It is
wrong for the dreamer, whose count trigger accumulates evidence over days
and whose interval trigger is only ever EVALUATED on a grading event — a
reset clock there costs every pass, forever.

Columns:

* ``org_id`` — the primary key AND the tenancy fence. One row per org, not
  one row per pass: this is a cursor, not an audit log. ``String(64)``,
  matching ``session_traces.org_id`` / ``memory_proposals.org_id``.
* ``sessions_since`` — graded sessions accumulated since the last pass.
  "Graded" here means exactly what ``app.main._consume_sessions_stored``
  counts: a ``grade_stored_session`` call that returned a non-NULL
  ``session_outcome_id``. It is NOT a verdict — a ``verdict='unknown'``
  session still counts, because the trigger is "new evidence arrived", not
  "new evidence was conclusive". Reset to 0 by the claim below.
* ``last_run_at`` — the interval clock, NOT NULL. Reset to ``now()`` by
  every CLAIMED pass whether that pass wrote proposals or failed, so a
  persistently-failing org retries once per interval rather than on every
  graded session. Seeded on row creation from the org's OLDEST
  ``session_traces`` row, not from ``now()`` — see below.
* ``last_dreamed_at`` — nullable, set only when a pass actually completed.
  ``NULL`` means "this org has never been dreamed for", which is the exact
  question that could not be answered while the state lived in process
  memory. Purely observability; no trigger reads it.
* ``last_proposals_written`` — nullable, how many rows the last completed
  pass filed. ``0`` is a meaningful, different answer from ``NULL``: the
  pass ran and found nothing, versus never ran.
* ``dream_passes`` — cumulative claimed passes, NOT NULL. Distinguishes
  "the trigger never fired" from "the trigger fires and every pass finds
  nothing", which are very different bugs.

WHY ``last_run_at`` IS SEEDED FROM THE OLDEST TRACE, NOT ``now()``. The
in-memory version started an org's clock the first time that PROCESS saw a
graded session for it, which is why a redeploy pushed the first pass a full
interval into the future every time. Persisting the clock fixes the reset,
but seeding a NEW row from ``now()`` would still mean "an org with three
days of accumulated evidence waits another interval before its first ever
pass". Seeding from ``MIN(session_traces.created_at)`` instead makes the
clock measure "how long this org has had evidence nobody has dreamed over",
which is what the interval trigger is actually trying to express. A young
org with a real corpus becomes due immediately; an org that produced its
first trace a minute ago does not. See
``app.dream_state.record_graded_session``.

CONCURRENCY. Two ``async-worker`` replicas may grade sessions for the same
org at the same time. The pass is claimed by a single conditional ``UPDATE``
whose ``WHERE`` clause IS the threshold test (``app.dream_state.claim_dream_pass``)
— the same statement that decides a pass is due also resets the counters, so
exactly one replica can win. There is no read-then-write window to lose, and
therefore no advisory lock and no ``work_claims`` row is needed here.

NO ``autocommit_block``. This migration is one ``CREATE TABLE`` plus one
plain (non-concurrent) ``CREATE INDEX``, both of which are fine inside
``env.py``'s single wrapping transaction. Nothing here is a
``CREATE INDEX CONCURRENTLY`` / ``ALTER TYPE ... ADD VALUE`` that would
require breaking out of it.

Revision ID: 046
Revises: 045
Create Date: 2026-08-17
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "046"
down_revision: Union[str, None] = "045"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "dream_state",
        sa.Column("org_id", sa.String(64), primary_key=True),
        sa.Column(
            "sessions_since",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "last_run_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("last_dreamed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_proposals_written", sa.Integer(), nullable=True),
        sa.Column(
            "dream_passes",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    # Serves the interval-timer scan (``app.dream_state.orgs_due_for_dream``),
    # which is the ONLY query here that does not already know its org_id — it
    # asks "which orgs are past their interval" across the whole table.
    op.create_index("ix_dream_state_last_run_at", "dream_state", ["last_run_at"])


def downgrade() -> None:
    op.drop_index("ix_dream_state_last_run_at", table_name="dream_state")
    op.drop_table("dream_state")
