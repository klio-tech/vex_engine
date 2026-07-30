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

# A retracted or superseded artifact keeps its bytes this long AFTER IT DIES,
# before the sweep destroys them. The window exists so an undelete remains
# POSSIBLE: `forget` is a single tool call an agent can make by mistake on
# customer document content, and once the object is gone there is nothing to
# restore from. Seven days is the same order as a consumer trash bin — long
# enough that a human notices a wrong `forget` and asks, short enough that a
# genuine deletion request is still honoured promptly. Nothing today exposes an
# undelete; this constant is what keeps building one a product decision rather
# than an impossible one.
#
# The age this is measured against is ``COALESCE(retracted_at, created_at)``
# (migration 030), i.e. when the row actually became dead. It used to be
# ``created_at`` alone, which gave the full window only to content younger than
# the window and ZERO grace to anything older — a document created eight days
# ago and retracted one minute ago was purgeable on the next tick, so for most
# real content the protection this constant advertises did not exist. The one
# residual gap is rows that were already dead when 030 ran: they have no
# ``retracted_at`` and still fall back to ``created_at``.
PURGE_GRACE_HOURS = 24 * 7

# The most `pending` rows one org may hold at once. Each pending row is a live
# presigned PUT that nothing counts against the storage pool until it is
# confirmed, so uncapped they are a way to park unbilled bytes: mint a thousand
# reservations, upload through every one, wait for GC, repeat.
#
# The cap bounds that exposure to MAX_PENDING_PER_ORG x the DECLARED size of
# each reservation, because ``artifact_storage.presign_put`` signs
# ``ContentLength`` — a PUT of any other size fails signature verification at
# the storage layer, so the URL is genuinely capped and not merely advertised
# as capped. The declared size is separately checked against the plan's
# per-file limit before the URL is issued, so the true ceiling is
# MAX_PENDING_PER_ORG x that limit (320MB on Free), recycled every
# PENDING_TTL_HOURS.
#
# Before ``ContentLength`` was signed, this bound was FALSE as written: each
# presigned PUT accepted an object of unbounded size, so the cap bounded the
# number of concurrent objects and not one byte of storage. Both halves are
# load-bearing — remove the signed length and the number below stops meaning
# anything.
#
# 32 is well above any legitimate concurrency — a client uploading a batch in
# parallel opens a handful of connections, not dozens — and a caller that hits
# it only has to confirm or abandon what it already started.
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
    """Retract an artifact. Returns False when the org does not own it.

    Stamps ``retracted_at`` in the same statement that flips the status, so the
    grace window in :func:`collect_purgeable` starts now rather than at
    creation. Both must move together: a row marked dead without a death time
    silently falls back to ``created_at`` and gets no grace at all.
    """
    row = db.execute(
        text(
            "UPDATE artifacts SET status = 'retracted', retracted_at = NOW() "
            "WHERE org_id = :org_id AND id = :artifact_id AND status = 'active' "
            "RETURNING id"
        ),
        {"org_id": org_id, "artifact_id": artifact_id},
    ).fetchone()
    db.commit()
    return row is not None


def mark_superseded(db: Any, *, org_id: str, artifact_id: str) -> None:
    """Retire a version that a newer artifact replaces.

    Stamps ``retracted_at`` for the same reason :func:`retract` does: a
    superseded version is dead, and its grace window has to run from the moment
    it was replaced.
    """
    db.execute(
        text(
            "UPDATE artifacts SET status = 'superseded', retracted_at = NOW() "
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

    The age is measured from ``COALESCE(retracted_at, created_at)`` — when the
    row DIED, not when it was born (migration 030). Keying off ``created_at``
    alone gave the full :data:`PURGE_GRACE_HOURS` only to content younger than
    the window and no grace at all to anything older: a document created eight
    days ago and retracted one minute ago was purgeable on the very next sweep
    tick. The ``COALESCE`` fallback applies only to rows that were already dead
    when migration 030 ran, which is a finite, shrinking set; every row
    retracted since carries a real ``retracted_at``.

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
              AND COALESCE(retracted_at, created_at)
                  < NOW() - (:hours || ' hours')::interval
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
