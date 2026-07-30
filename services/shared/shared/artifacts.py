"""Artifact lifecycle: the rows, not the bytes.

Every statement carries an ``org_id`` predicate. A cross-org id therefore
reads as not-found rather than as a permission error, matching the memory and
session paths.

``size_bytes`` and ``checksum`` are written only by :func:`activate`, from
values the server measured. A client's declared size selects an upload
strategy and is never persisted.

Async note: like :mod:`shared.brain`, the DB calls here are synchronous
psycopg2 I/O. Callers are sync; do not wrap in threads.
"""

from typing import Any, Optional

from sqlalchemy import text

# Text at or below this size lives in Postgres, so the common read path never
# touches object storage. Covers essentially every spec, record and code file.
INLINE_MAX_BYTES = 65_536

# One artifact_get(mode='text') call returns at most this much, with an offset
# for paging. A context window is the real constraint here, not the transport.
TEXT_PAGE_BYTES = 51_200

# An upload that is requested and never confirmed is collected after this long.
PENDING_TTL_HOURS = 24


def human_size(n: int) -> str:
    """Format bytes for the card a human will read."""
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.1f} GB"


def card_content(title: str, summary: str, kind: str, size_bytes: int) -> str:
    """The memory row that makes an artifact discoverable through recall."""
    return f"[artifact] {title} — {summary} ({kind}, {human_size(size_bytes)})"


def org_storage_bytes(db: Any, *, org_id: str) -> int:
    """Live storage for an org.

    A summed gauge rather than a running counter: a counter is faster but can
    drift, and drift here means either over-charging a customer or letting a
    pool run unbounded. Artifact counts are in the thousands; revisit if that
    stops being true.
    """
    row = db.execute(
        text(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM artifacts "
            "WHERE org_id = :org_id AND status = 'active'"
        ),
        {"org_id": org_id},
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def find_active_by_checksum(db: Any, *, org_id: str, checksum: str) -> Optional[dict]:
    """Deduplication: identical bytes in one org resolve to one artifact."""
    row = db.execute(
        text(
            "SELECT id, version FROM artifacts "
            "WHERE org_id = :org_id AND checksum = :checksum AND status = 'active'"
        ),
        {"org_id": org_id, "checksum": checksum},
    ).fetchone()
    return {"id": str(row[0]), "version": int(row[1])} if row else None


def create_pending(
    db: Any,
    *,
    org_id: str,
    agent_id: str,
    title: str,
    summary: str,
    kind: str,
    mime_type: str,
    space_id: Optional[str] = None,
    project_id: Optional[str] = None,
    supersedes: Optional[str] = None,
    version: int = 1,
) -> str:
    """Reserve an artifact id. No bytes yet, so no size and no checksum."""
    row = db.execute(
        text(
            """
            INSERT INTO artifacts (
                org_id, agent_id, title, summary, kind, mime_type,
                space_id, project_id, supersedes, version, status
            ) VALUES (
                :org_id, :agent_id, :title, :summary, :kind, :mime_type,
                :space_id, :project_id, :supersedes, :version, 'pending'
            )
            RETURNING id
            """
        ),
        {
            "org_id": org_id,
            "agent_id": agent_id,
            "title": title,
            "summary": summary,
            "kind": kind,
            "mime_type": mime_type,
            "space_id": space_id,
            "project_id": project_id,
            "supersedes": supersedes,
            "version": version,
        },
    ).fetchone()
    db.commit()
    return str(row[0])


def activate(
    db: Any,
    *,
    org_id: str,
    artifact_id: str,
    size_bytes: int,
    checksum: str,
    inline_text: Optional[str] = None,
    storage_uri: Optional[str] = None,
) -> Optional[dict]:
    """Promote a pending artifact using server-measured values."""
    row = db.execute(
        text(
            """
            UPDATE artifacts SET
                status = 'active',
                size_bytes = :size_bytes,
                checksum = :checksum,
                inline_text = :inline_text,
                storage_uri = :storage_uri
            WHERE org_id = :org_id AND id = :artifact_id AND status = 'pending'
            RETURNING id, size_bytes, checksum
            """
        ),
        {
            "org_id": org_id,
            "artifact_id": artifact_id,
            "size_bytes": size_bytes,
            "checksum": checksum,
            "inline_text": inline_text,
            "storage_uri": storage_uri,
        },
    ).fetchone()
    db.commit()
    if not row:
        return None
    return {"id": str(row[0]), "size_bytes": int(row[1]), "checksum": row[2]}


def get_artifact(db: Any, *, org_id: str, artifact_id: str) -> Optional[dict]:
    row = db.execute(
        text(
            """
            SELECT id, org_id, agent_id, title, summary, kind, mime_type,
                   size_bytes, checksum, inline_text, storage_uri,
                   version, status, space_id, project_id
            FROM artifacts
            WHERE org_id = :org_id AND id = :artifact_id
            """
        ),
        {"org_id": org_id, "artifact_id": artifact_id},
    ).fetchone()
    if not row:
        return None
    return {
        "id": str(row[0]),
        "org_id": row[1],
        "agent_id": row[2],
        "title": row[3],
        "summary": row[4],
        "kind": row[5],
        "mime_type": row[6],
        "size_bytes": row[7],
        "checksum": row[8],
        "inline_text": row[9],
        "storage_uri": row[10],
        "version": int(row[11]),
        "status": row[12],
        # Carried so the confirm path can file the card into the same space
        # the pending row was created with — otherwise a presigned artifact's
        # card silently lands unfiled.
        "space_id": row[13],
        "project_id": row[14],
    }


def retract(db: Any, *, org_id: str, artifact_id: str) -> bool:
    """Retract an artifact. Returns False when the org does not own it."""
    row = db.execute(
        text(
            "UPDATE artifacts SET status = 'retracted' "
            "WHERE org_id = :org_id AND id = :artifact_id AND status = 'active' "
            "RETURNING id"
        ),
        {"org_id": org_id, "artifact_id": artifact_id},
    ).fetchone()
    db.commit()
    return row is not None


def mark_superseded(db: Any, *, org_id: str, artifact_id: str) -> None:
    db.execute(
        text(
            "UPDATE artifacts SET status = 'superseded' "
            "WHERE org_id = :org_id AND id = :artifact_id AND status = 'active'"
        ),
        {"org_id": org_id, "artifact_id": artifact_id},
    )
    db.commit()


def collect_abandoned(db: Any, *, older_than_hours: int = PENDING_TTL_HOURS) -> list:
    """Pending artifacts whose upload never arrived. Caller deletes the bytes."""
    rows = db.execute(
        text(
            """
            SELECT id, org_id, version FROM artifacts
            WHERE status = 'pending'
              AND created_at < NOW() - (:hours || ' hours')::interval
            """
        ),
        {"hours": older_than_hours},
    ).fetchall()
    return [
        {"id": str(r[0]), "org_id": r[1], "version": int(r[2])} for r in rows
    ]
