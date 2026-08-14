"""Memory proposals: the dreamer's reviewable output (Build 5).

``memory_proposals`` is a NEW table, not an extension of ``session_memories``
— a proposal is not yet a belief. It is the dreamer's claim that the
evidence in ``session_outcomes``/``recall_outcomes``/``session_traces``
supports a specific ``add``/``revise``/``retire`` action, filed for a human
(or, narrowly, the same agent, and only for ``revise``) to decide on.
Nothing here writes to ``session_memories`` directly; approval routes
through the same primitives every other writer uses —
:func:`shared.memory.supersede_with_revision` for ``revise``,
:func:`shared.memory.retract_memory` for ``retire``, and
:func:`shared.brain.write_scoped_memory` for ``add``.

Columns:

* ``org_id`` — the tenancy fence, always enforced, matching every other
  table in this schema (``String(64)``, same as ``session_memories.org_id``).
* ``scope`` — the scope the target memory (or new memory, for ``kind='add'``)
  would live at: ``private`` / ``project`` / ``agent`` / ``org`` /
  ``session``. Free text, matching ``session_memories.scope``'s own typing —
  no enum exists for that column either, so this one does not invent one.
* ``kind`` — ``add`` / ``revise`` / ``retire``. ``add`` proposes a brand-new
  memory (no target); ``revise``/``retire`` name an existing one via
  ``target_memory_id``.
* ``detector`` — which of the three deterministic detectors
  (``app.dreamer``) filed this proposal: ``tool_failure`` /
  ``bad_outcome_memory`` / ``stale_serve_cluster``. A first-class column
  (not just a key inside ``evidence``) because the partial unique index
  below needs to index on it directly.
* ``target_memory_id`` — nullable (``add`` has none), no foreign key. Same
  rationale as ``memory_edges`` (migration 035) and ``work_claims``
  (migration 044): a proposal must never be the reason a memory row becomes
  undeletable, and the target may itself be retired/superseded by the time
  a human reviews the proposal — the approval path re-validates via CAS
  (or re-checks existence, for ``retire``) regardless.
* ``diff`` — REVIEWER-FACING RATIONALE ONLY, written by an LLM pass over
  the ALREADY-computed detector evidence (:func:`app.dreamer._draft_diff`);
  on LLM failure this is instead the detector's own deterministic summary
  sentence. This column is NEVER written to ``session_memories`` — it
  exists to help a human decide, not to become a belief. See
  ``proposed_content`` for the column that actually can become memory
  content.
* ``proposed_content`` — nullable ``Text``. The literal content that would
  be written on approval, for ``kind='add'``/``kind='revise'`` only
  (``retire`` needs no content — it retracts, it does not replace).
  Deliberately populated by DETECTOR ARITHMETIC only where the detector has
  one to offer (the tool-failure detector composes a deterministic note),
  and left ``NULL`` where it does not (the bad-outcome-memory detector has
  evidence a belief is hurting outcomes but no drafted replacement) — an
  LLM never writes this column, only ``diff``. ``NULL`` here means the
  proposal cannot be applied until a human supplies content (``app.proposals``
  refuses with ``needs_content``); the dreamer never auto-applies a
  proposal with ``proposed_content IS NULL`` either.
* ``evidence`` — JSONB, the detector's raw counted evidence:
  ``{"prevalence": n, "detector": "...", ...session/memory refs}``. This is
  what a reviewer (or an audit) checks ``diff``/``proposed_content``
  against.
* ``confidence`` — nullable ``Numeric``, reserved for a future score; no
  detector in Build 5 populates it.
* ``status`` — ``open`` (default) / ``approved`` / ``rejected`` / ``auto``.
  ``auto`` marks a proposal the dreamer applied itself at detection time —
  allowed ONLY for ``kind='revise'`` at ``scope='agent'``: ``scope='agent'``
  is the only scope ``shared.memory._authorize_scoped_action`` lets a
  caller with no user principal (the dreamer has none) act on without
  either an owning ``user_id`` (private/project) or human review, and
  ``retire`` never auto-applies regardless of scope — a destructive action
  always waits for an accountable human (see ``app.dreamer``'s module
  docstring).
* ``decided_by`` — nullable UUID, no foreign key (same non-FK convention as
  every other loosely-typed identity column here); set to the resolving
  user's id on human approve/reject, left ``NULL`` for ``auto``.

Indexes:

* ``ix_memory_proposals_org_status`` serves the REST list endpoint's
  ``GET /proposals?status=open`` — org-scoped, filtered by status, on every
  request.
* ``ux_memory_proposals_org_detector_target_open`` — a PARTIAL unique index
  on ``(org_id, detector, target_memory_id)`` WHERE ``status = 'open'``.
  Every dream pass re-detects the same ongoing pattern (a tool still
  failing, a memory still under-performing) on every run; without this,
  each pass would file a fresh duplicate proposal for evidence a human
  hasn't even looked at yet. Scoped to ``status = 'open'`` specifically so
  a DECIDED proposal (approved/rejected/auto) never blocks a genuinely NEW
  occurrence of the same pattern from being filed later.
  ``app.dreamer._insert_proposal`` inserts with
  ``ON CONFLICT (org_id, detector, target_memory_id) WHERE status = 'open'
  DO NOTHING``, whose inference clause must match this index's columns AND
  predicate exactly.

Revision ID: 045
Revises: 044
Create Date: 2026-08-13
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "045"
down_revision: Union[str, None] = "044"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_proposals",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("scope", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("detector", sa.Text(), nullable=False),
        sa.Column("target_memory_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("diff", sa.Text(), nullable=False),
        sa.Column("proposed_content", sa.Text(), nullable=True),
        sa.Column("evidence", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("confidence", sa.Numeric(), nullable=True),
        sa.Column(
            "status", sa.Text(), nullable=False, server_default=sa.text("'open'")
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decided_by", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index(
        "ix_memory_proposals_org_status",
        "memory_proposals",
        ["org_id", "status"],
    )
    # Partial unique index — Alembic's op.create_index supports postgresql_where
    # for exactly this. The ON CONFLICT inference clause in
    # app.dreamer._insert_proposal must match these columns AND this predicate
    # verbatim, or Postgres refuses to infer the index (42P10).
    op.create_index(
        "ux_memory_proposals_org_detector_target_open",
        "memory_proposals",
        ["org_id", "detector", "target_memory_id"],
        unique=True,
        postgresql_where=sa.text("status = 'open'"),
    )


def downgrade() -> None:
    op.drop_index(
        "ux_memory_proposals_org_detector_target_open",
        table_name="memory_proposals",
    )
    op.drop_index("ix_memory_proposals_org_status", table_name="memory_proposals")
    op.drop_table("memory_proposals")
