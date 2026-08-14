"""Best-effort recall-event logging for the Klio brain.

Appends a row to brain_recall_events (migration 022) for each recall, so the
dashboard can show per-agent recall activity. SYNCHRONOUS psycopg2 I/O — call via
anyio.to_thread from async handlers (see app.capture). Every failure is swallowed
and logged: logging a recall must NEVER break the recall itself.

Migration 042 (evidence loop) added two further columns, both optional so
every pre-existing call site keeps working unchanged:

* ``memory_ids`` (``UUID[]``) — the ids of the memories actually served, so
  the grader can later join back "was this memory used" without re-deriving
  the served set from the response payload.
* ``session_id`` (``TEXT``) — the SAME derived session identity
  :func:`app.capture._capture_session_id` computes for a transcript capture,
  so the grader can join a recall event to the ``session_traces`` row for the
  session it was served into. A caller with no session identifier in scope
  (the MCP ``recall``/``explore`` tools) passes ``None``.
"""

import logging
import uuid
from typing import Any, Optional

from shared.redaction import redact
from sqlalchemy import text

logger = logging.getLogger("agentguard.mcp.recall_log")


def log_recall_event(
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
    """Append one brain_recall_events row. Best-effort; never raises.

    ``query_text`` is redacted via :func:`shared.redaction.redact` before it
    is bound into the INSERT — secrets and PII in a recall query never reach
    the audit table.  On redaction error ``query_text`` is stored as ``None``
    (best-effort log; unscreened text is never persisted).

    ``memory_ids`` and ``session_id`` are optional and additive (migration
    042): every existing call site that omits them keeps inserting ``NULL``
    for both, matching the columns' nullable definition.
    """
    if query_text:
        try:
            query_text = redact(query_text).clean_text
        except Exception:
            # Control flow is unchanged and deliberate: unscreened text is
            # never persisted, so the column becomes NULL and the insert goes
            # ahead. What was missing was any trace of it — a redactor that
            # starts failing made every row's query_text NULL, which reads as
            # "nobody searched anything" rather than "the redactor is broken".
            # The failing text is NOT logged: it is precisely the string that
            # could not be screened.
            logger.warning(
                "redact() failed for a recall query in org %s; storing NULL "
                "query_text (content withheld — it is unscreened)",
                org_id,
                exc_info=True,
            )
            query_text = None  # best-effort log; never store unscreened
    try:
        # ``CAST(:memory_ids AS uuid[])``, not a plain bind: psycopg2 adapts a
        # Python list to an untyped Postgres array literal (effectively
        # ``text[]``), and Postgres refuses to insert that into a ``uuid[]``
        # column without an explicit cast (DatatypeMismatch). The explicit
        # ``CAST(... AS uuid[])`` also handles ``memory_ids=None`` cleanly —
        # it casts to a NULL array, matching the column's nullability.
        db.execute(
            text(
                """
                INSERT INTO brain_recall_events
                  (id, org_id, agent_id, project_id, query_text, result_count, source,
                   memory_ids, session_id)
                VALUES (:id, :org_id, :agent_id, :project_id, :query_text,
                        :result_count, :source, CAST(:memory_ids AS uuid[]),
                        :session_id)
                """
            ),
            {
                "id": str(uuid.uuid4()),
                "org_id": org_id,
                "agent_id": agent_id,
                "project_id": project_id,
                "query_text": query_text,
                "result_count": result_count,
                "source": source,
                "memory_ids": memory_ids,
                "session_id": session_id,
            },
        )
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning("recall log failed for org %s; ignoring", org_id, exc_info=True)
