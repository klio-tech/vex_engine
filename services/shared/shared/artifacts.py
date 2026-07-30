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

# One artifact_get(mode='text') call returns at most this many *characters*
# (not bytes) of a Python str, with an offset for paging. Characters, not
# bytes, are the right unit: the thing being protected is an agent's context
# window, which is measured in tokens, and characters track token count far
# more closely than UTF-8 byte length does — a single CJK or emoji character
# can be 3-4 bytes but is still roughly one token. Slicing by byte count would
# also risk splitting a multi-byte character mid-codepoint and would make
# ``offset`` meaningless across pages.
TEXT_PAGE_CHARS = 51_200

# An upload that is requested and never confirmed is collected after this long.
PENDING_TTL_HOURS = 24

# A retracted or superseded artifact keeps its bytes this long before the sweep
# destroys them. The window exists so an undelete remains POSSIBLE: `forget` is
# a single tool call an agent can make by mistake on customer document content,
# and once the object is gone there is nothing to restore from. Seven days is
# the same order as a consumer trash bin — long enough that a human notices a
# wrong `forget` and asks, short enough that a genuine deletion request is
# still honoured promptly. Nothing today exposes an undelete; this constant is
# what keeps building one a product decision rather than an impossible one.
PURGE_GRACE_HOURS = 24 * 7

# The most `pending` rows one org may hold at once. Each pending row is a live
# presigned PUT that nothing counts against the storage pool until it is
# confirmed, so uncapped they are a way to park unbilled bytes: mint a thousand
# reservations, upload through every one, wait for GC, repeat. The cap bounds
# that exposure to MAX_PENDING_PER_ORG x the plan's per-file limit (320MB on
# Free) instead of leaving it unbounded. 32 is well above any legitimate
# concurrency — a client uploading a batch in parallel opens a handful of
# connections, not dozens — and a caller that hits it only has to confirm or
# abandon what it already started.
MAX_PENDING_PER_ORG = 32


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


def pending_count(db: Any, *, org_id: str) -> int:
    """How many upload reservations this org currently holds.

    Read before handing out another presigned PUT: see
    :data:`MAX_PENDING_PER_ORG` for why an uncapped count is a quota bypass.
    """
    row = db.execute(
        text(
            "SELECT COUNT(*) FROM artifacts "
            "WHERE org_id = :org_id AND status = 'pending'"
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


def delete_pending(db: Any, *, org_id: str, artifact_id: str) -> bool:
    """Drop a reservation that will never be activated. Returns True on a hit.

    Used when a confirm turns out to duplicate bytes the org already has: the
    existing artifact is returned instead, so this pending row has no future.
    Leaving it would park a GC candidate for 24h and consume one of the org's
    :data:`MAX_PENDING_PER_ORG` slots for no reason.

    Guarded on ``status = 'pending'`` as well as ``org_id`` so it can never
    remove a row that has meanwhile been activated.
    """
    row = db.execute(
        text(
            "DELETE FROM artifacts "
            "WHERE org_id = :org_id AND id = :artifact_id AND status = 'pending' "
            "RETURNING id"
        ),
        {"org_id": org_id, "artifact_id": artifact_id},
    ).fetchone()
    db.commit()
    return row is not None


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
    """Pending artifacts whose upload never arrived. Caller deletes the bytes.

    Intentionally cross-org: this is the background maintenance sweep, not a
    request-scoped query, so it has no ``org_id`` predicate and returns stale
    rows across every tenant. That is correct here, but it is the one place in
    this module that breaks the "every artifact query is org-scoped" invariant
    the rest of the feature relies on. Do not "fix" this into a tenant-scoped
    function, and do not copy this as a template for a request-handling query.
    Any caller doing follow-up work per row (e.g. deleting its S3 object) MUST
    scope that work using the ``org_id`` each row returns.
    """
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


def collect_purgeable(db: Any, *, older_than_hours: int = PURGE_GRACE_HOURS) -> list:
    """Dead artifacts whose bytes are still on disk. Caller deletes them.

    ``retract`` (``forget``) and ``mark_superseded`` only change ``status``.
    Nothing else in this module ever removes an object or clears
    ``inline_text``, so without this selector a forgotten 900MB PDF is
    unreachable from every API, uncounted by :func:`org_storage_bytes` (which
    filters ``status = 'active'``), and stored forever. That is both an
    unhonourable deletion request over customer document content and a pool
    that under-counts what the bucket actually holds.

    ``created_at`` — not a retraction timestamp — is the age used, because the
    table has no ``retracted_at`` column; combined with
    :data:`PURGE_GRACE_HOURS` this means a row must be both dead AND old.
    An artifact retracted the day it was created therefore keeps its bytes for
    the remainder of the window, which is the conservative direction to err in.

    Same cross-org contract and the same warning as
    :func:`collect_abandoned`: this is the background maintenance sweep, so it
    has no ``org_id`` predicate deliberately. Any caller doing per-row work
    (deleting an object) MUST scope it with the ``org_id`` each row returns,
    and must not copy this shape into a request-handling query.
    """
    rows = db.execute(
        text(
            """
            SELECT id, org_id, version, status FROM artifacts
            WHERE status IN ('retracted', 'superseded')
              AND created_at < NOW() - (:hours || ' hours')::interval
            """
        ),
        {"hours": older_than_hours},
    ).fetchall()
    return [
        {
            "id": str(r[0]),
            "org_id": r[1],
            "version": int(r[2]),
            "status": r[3],
        }
        for r in rows
    ]
