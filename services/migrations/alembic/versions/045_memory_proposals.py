"""Memory proposals: the dreamer's reviewable output (Build 5).

``memory_proposals`` is a NEW table, not an extension of ``session_memories``
— a proposal is not yet a belief. It is the dreamer's claim that the
evidence in ``session_outcomes``/``recall_outcomes``/``session_traces``
supports a specific ``add``/``revise``/``retire`` action, filed for a human
(or, narrowly, the same agent) to decide on. Nothing here writes to
``session_memories`` directly; approval routes through the existing CAS path
(:func:`shared.memory.supersede_with_revision`) exactly like every other
writer.

Columns:

* ``org_id`` — the tenancy fence, always enforced, matching every other
  table in this schema (``String(64)``, same as ``session_memories.org_id``).
* ``scope`` — the scope the target memory (or new memory, for ``kind='add'``)
  would live at: ``private`` / ``project`` / ``agent`` / ``org`` /
  ``session``. Free text, matching ``session_memories.scope``'s own typing —
  no enum exists for that column either, so this one does not invent one.
* ``kind`` — ``add`` / ``revise`` / ``retire``. ``add`` proposes a brand-new
  memory (no target); ``revise``/``retire`` name an existing one via
  ``target_memory_id`` and apply through the CAS path on approval.
* ``target_memory_id`` — nullable (``add`` has none), no foreign key. Same
  rationale as ``memory_edges`` (migration 035) and ``work_claims``
  (migration 044): a proposal must never be the reason a memory row becomes
  undeletable, and the target may itself be retired/superseded by the time
  a human reviews the proposal — the approval path re-validates via CAS
  regardless.
* ``diff`` — human-readable prose describing the proposed change. Written by
  an LLM pass over the ALREADY-computed detector evidence
  (:func:`app.dreamer._draft_diff`); on LLM failure this is instead the
  detector's own deterministic summary sentence. Detection is arithmetic;
  this column is the only place an LLM's output appears at all, and it can
  only ever restate what the detector already found — never invent a
  pattern.
* ``evidence`` — JSONB, the detector's raw counted evidence:
  ``{"prevalence": n, "detector": "...", ...session/memory refs}``. This is
  what a reviewer (or an audit) checks the ``diff`` prose against.
* ``confidence`` — nullable ``Numeric``, reserved for a future score; no
  detector in Build 5 populates it.
* ``status`` — ``open`` (default) / ``approved`` / ``rejected`` / ``auto``.
  ``auto`` marks a proposal the dreamer applied itself at detection time —
  allowed ONLY for ``scope='agent'`` targets, since that is the only scope
  ``shared.memory._authorize_scoped_action`` lets a caller with no user
  principal (the dreamer has none) act on without either an owning
  ``user_id`` (private/project) or human review.
* ``decided_by`` — nullable UUID, no foreign key (same non-FK convention as
  every other loosely-typed identity column here); set to the resolving
  user's id on human approve/reject, left ``NULL`` for ``auto``.

``ix_memory_proposals_org_status`` serves the REST list endpoint's
``GET /proposals?status=open`` — org-scoped, filtered by status, on every
request.

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
        sa.Column("target_memory_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("diff", sa.Text(), nullable=False),
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


def downgrade() -> None:
    op.drop_index("ix_memory_proposals_org_status", table_name="memory_proposals")
    op.drop_table("memory_proposals")
