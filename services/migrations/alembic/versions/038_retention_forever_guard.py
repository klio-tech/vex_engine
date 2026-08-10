"""Retention: a non-positive day count means "keep forever", never "delete all".

Klio's published Free and Team tiers promise ``Memory retention: Forever``. The
natural way to express that in ``PlanConfig`` is ``memory_retention_days = -1``,
matching the ``-1 = unlimited`` convention every quota field already uses.

Passed to these functions as written, ``-1`` is catastrophic::

    NOW() - ('-1 days')::interval  =  NOW() + 1 day

The cutoff lands in the FUTURE, so ``created_at < cutoff`` matches every row and
the sweep deletes the org's entire history. Zero behaves the same way: the cutoff
becomes ``NOW()`` and everything written before this instant is eligible.

That is the identical shape as the ``RAW_OBSERVATION_TTL_DAYS`` defect fixed
earlier: an operator or a config author reaching for the obvious value for
"disabled" gets the most destructive behaviour available.

``run_retention.py`` now skips the call for non-positive values, so this is the
second of two guards rather than the only one. It is here as well because the
functions are callable directly from SQL, and a guard that only exists in the
caller protects only that caller.

Both functions are guarded, not just the memory one: ``enforce_plan_retention``
has the same expression over ``agent_executions`` and the same exposure.
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

revision: str = "038"
down_revision: Union[str, None] = "037"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Memory retention: migration 029's body, with the guard prepended. The
    # DELETE is otherwise byte-for-byte 029 so this migration changes exactly
    # one thing.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION enforce_memory_retention(
            p_org_id TEXT, p_retention_days INT
        ) RETURNS void AS $$
        BEGIN
            -- Keep forever. A non-positive interval would move the cutoff to
            -- NOW() or later and delete the whole org.
            IF p_retention_days IS NULL OR p_retention_days <= 0 THEN
                RETURN;
            END IF;

            DELETE FROM session_memories sm
            WHERE sm.org_id = p_org_id
              AND sm.created_at < NOW() - (p_retention_days || ' days')::interval
              AND NOT (
                sm.memory_type = 'artifact'
                AND (sm.metadata->>'artifact_id') ~*
                    '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
                AND EXISTS (
                    SELECT 1 FROM artifacts a
                    WHERE a.id = (sm.metadata->>'artifact_id')::uuid
                      AND a.org_id = p_org_id
                      AND a.status = 'active'
                )
              );
        END;
        $$ LANGUAGE plpgsql;
        """
    )

    # Vex's execution retention: migration 008's body, guard prepended and
    # nothing else touched. This one deletes from check_results (via executions)
    # and executions, keyed on `timestamp` — NOT the columns or tables the
    # memory sweep uses. Reproduced verbatim on purpose: paraphrasing it would
    # silently retire Vex retention while looking like a guard.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION enforce_plan_retention(
            p_org_id TEXT,
            p_retention_days INT
        )
        RETURNS void AS $$
        BEGIN
            IF p_retention_days IS NULL OR p_retention_days <= 0 THEN
                RETURN;
            END IF;

            DELETE FROM check_results cr
            USING executions e
            WHERE cr.execution_id = e.execution_id
              AND e.org_id = p_org_id
              AND cr.timestamp < NOW() - (p_retention_days || ' days')::INTERVAL;

            DELETE FROM executions
            WHERE org_id = p_org_id
              AND timestamp < NOW() - (p_retention_days || ' days')::INTERVAL;
        END;
        $$ LANGUAGE plpgsql;
        """
    )


def downgrade() -> None:
    # Restore 029's memory body verbatim: unguarded, so -1 deletes everything.
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
                AND (sm.metadata->>'artifact_id') ~*
                    '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
                AND EXISTS (
                    SELECT 1 FROM artifacts a
                    WHERE a.id = (sm.metadata->>'artifact_id')::uuid
                      AND a.org_id = p_org_id
                      AND a.status = 'active'
                )
              );
        END;
        $$ LANGUAGE plpgsql;
        """
    )

    # Restore 008's execution body verbatim: unguarded.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION enforce_plan_retention(
            p_org_id TEXT,
            p_retention_days INT
        )
        RETURNS void AS $$
        BEGIN
            DELETE FROM check_results cr
            USING executions e
            WHERE cr.execution_id = e.execution_id
              AND e.org_id = p_org_id
              AND cr.timestamp < NOW() - (p_retention_days || ' days')::INTERVAL;

            DELETE FROM executions
            WHERE org_id = p_org_id
              AND timestamp < NOW() - (p_retention_days || ' days')::INTERVAL;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
