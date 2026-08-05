"""Reconcile the artifact object bucket against the ``artifacts`` table.

Why this script exists
----------------------
Nobody currently knows whether orphans exist. ``retract`` (``forget``) and
``mark_superseded`` only flip a status; the GC sweep is what actually deletes
the bytes. If that sweep has ever failed quietly — a throttled DELETE, a
crashed tick, a row removed by retention before its object was collected —
then the deletion promise has not held, and nothing anywhere emits a signal
saying so. Running this is the only way to find out.

Because nobody knows, the classification is careful in both directions.
Calling a live object an orphan invites deleting a customer's document.
Calling an orphan healthy reports a clean bill of health that is false. So the
bucket is sorted into populations that name their own cause:

``orphan``
    An object past the grace window with no row claiming it at all. Unclaimed
    bytes. This is the only population ``--execute`` would delete.

``unpurged``
    An object whose row is ``retracted``/``superseded`` and already past
    ``PURGE_GRACE_HOURS``. The GC should have deleted these and did not — a
    *named failing mechanism*, which is why it is not merged into ``orphan``.
    ``--execute`` deliberately does NOT delete them: that is the GC's job, and
    doing it as a side effect of an orphan sweep would destroy customer
    document content nobody separately approved.

``too_recent``
    An unclaimed object inside the grace window. A presigned PUT can land
    between this script reading the bucket and reading the table. Without the
    window, a healthy in-flight upload gets classified as garbage — the same
    class of mistake as the curator advancing past a row it never read.

``unrecognised``
    A key that does not parse as ``<org_id>/<artifact_id>/v<version>``. It
    cannot be matched to a row, so it cannot be *shown* to be unclaimed.
    Reported and never touched: understanding a key is the precondition for
    deleting it.

``missing_object``
    The other direction — a row claiming a ``storage_uri`` whose bytes are not
    there. Nothing can be executed about it (bytes cannot be recreated), but
    it is a read that will fail for a customer and this is the only place that
    would ever notice. Half a reconciliation is not a reconciliation.

Credentials
-----------
This script builds its client through
:func:`shared.artifact_storage.get_artifact_s3_client`, which reads the
``KLIO_ARTIFACT_S3_*`` / ``S3_*`` environment. It never reads, logs or accepts
a credential itself. It was developed and tested against a fake client because
this environment has no object-store credentials; **the live run is the
user's to make**.

Usage:
    python -m scripts.reconcile_orphan_artifacts                    # dry run, bucket
    python -m scripts.reconcile_orphan_artifacts --org <org_id>     # dry run, one org
    python -m scripts.reconcile_orphan_artifacts --org <org_id> \\
        --execute --confirm <token-from-the-dry-run>

Environment:
    DATABASE_URL: TimescaleDB connection string.
    KLIO_ARTIFACT_BUCKET / KLIO_ARTIFACT_S3_*: see shared.artifact_storage.
    KLIO_ORPHAN_GRACE_HOURS: how old an unclaimed object must be before it
        counts as an orphan (default 24, matching artifacts.PENDING_TTL_HOURS).
"""

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))

import reconcile_common as common  # noqa: E402
from shared.artifact_storage import (  # noqa: E402
    ARTIFACT_BUCKET,
    get_artifact_s3_client,
)
from shared.artifacts import PENDING_TTL_HOURS, PURGE_GRACE_HOURS  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("reconcile.orphan_artifacts")

SCRIPT_NAME = "reconcile_orphan_artifacts"

#: ``object_key`` builds ``<org_id>/<artifact_id>/v<version>``. The artifact id
#: is a UUID and the version an integer; anything else is a key this script
#: does not understand, and it says so rather than guessing.
_KEY_RE = re.compile(
    r"\A(?P<org_id>.+)/"
    r"(?P<artifact_id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})/"
    r"v(?P<version>\d+)\Z"
)

#: Rows are looked up in chunks so a large bucket does not build one enormous
#: IN-list.
LOOKUP_CHUNK = 500


@dataclass
class Reconciliation:
    """What the bucket and the table say about each other. Keys and counts."""

    scanned: int = 0
    claimed: int = 0
    too_recent: int = 0
    orphan_keys: list = field(default_factory=list)
    unpurged_keys: list = field(default_factory=list)
    unrecognised_keys: list = field(default_factory=list)
    missing_object_ids: list = field(default_factory=list)


def grace_hours() -> int:
    """How old an unclaimed object must be before it counts as an orphan.

    Defaults to ``PENDING_TTL_HOURS``: an upload that has not confirmed within
    that window is already the GC's definition of abandoned, so it is the
    window the rest of the system already reasons in. An unparseable or
    non-positive value falls back to the default rather than raising — a
    shorter window is the dangerous direction, and this script's whole
    purpose is to produce a measurement.
    """
    raw = os.environ.get("KLIO_ORPHAN_GRACE_HOURS")
    if raw is None:
        return PENDING_TTL_HOURS
    try:
        hours = int(raw.strip())
    except (AttributeError, ValueError):
        logger.warning(
            "KLIO_ORPHAN_GRACE_HOURS=%r is unparseable; using the %d-hour default",
            raw,
            PENDING_TTL_HOURS,
        )
        return PENDING_TTL_HOURS
    if hours < 1:
        logger.warning(
            "KLIO_ORPHAN_GRACE_HOURS=%d would classify in-flight uploads as "
            "orphans; using the %d-hour default",
            hours,
            PENDING_TTL_HOURS,
        )
        return PENDING_TTL_HOURS
    return hours


def parse_key(key: str) -> Optional[tuple]:
    """``<org_id>/<artifact_id>/v<version>`` -> triple, or ``None``."""
    match = _KEY_RE.match(key)
    if match is None:
        return None
    return (
        match.group("org_id"),
        match.group("artifact_id").lower(),
        int(match.group("version")),
    )


def _list_objects(s3: Any, *, prefix: str) -> list:
    """Every object under ``prefix``, across every page of the listing.

    A paginator read only to its first page under-reports silently, and a
    clean bill of health from a partially-read bucket is worse than not
    running at all.
    """
    objects = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=ARTIFACT_BUCKET, Prefix=prefix):
        objects.extend(page.get("Contents") or [])
    return objects


def _claims(db: Any, org_id: str, artifact_ids: list) -> dict:
    """Map ``(artifact_id, version) -> (status, dead_and_past_grace)``.

    Org-scoped in SQL. The id came out of an object key, which is a string
    from the object store; matching on it alone would let a crafted or
    corrupted key address another tenant's row.
    """
    claims: dict = {}
    for start in range(0, len(artifact_ids), LOOKUP_CHUNK):
        chunk = artifact_ids[start : start + LOOKUP_CHUNK]
        rows = db.execute(
            text(
                """
                SELECT id, version, status,
                       (COALESCE(retracted_at, created_at)
                            < NOW() - (:hours || ' hours')::interval) AS past_grace
                FROM artifacts
                WHERE org_id = :org_id
                  AND id = ANY(CAST(:ids AS uuid[]))
                """
            ),
            {"org_id": org_id, "ids": chunk, "hours": PURGE_GRACE_HOURS},
        ).fetchall()
        for row in rows:
            claims[(str(row.id).lower(), int(row.version))] = (
                row.status,
                bool(row.past_grace),
            )
    return claims


def _missing_objects(db: Any, org_id: Optional[str], present_keys: set) -> list:
    """Artifact rows whose ``storage_uri`` names bytes that are not there.

    Only rows that actually claim an object: a small artifact lives in
    ``inline_text`` and has none, and flagging every one of those would drown
    the signal this is trying to surface.
    """
    where = ["storage_uri IS NOT NULL", "status = 'active'"]
    params: dict[str, Any] = {}
    if org_id:
        where.append("org_id = :org_id")
        params["org_id"] = org_id
    rows = db.execute(
        text(
            f"""
            SELECT id, storage_uri FROM artifacts
            WHERE {" AND ".join(where)}
            ORDER BY id ASC
            """
        ),
        params,
    ).fetchall()

    missing = []
    for row in rows:
        # ``storage_uri`` is ``s3://<bucket>/<key>``; the key is everything
        # after the bucket.
        uri = row.storage_uri or ""
        _, _, tail = uri.partition("://")
        _, _, key = tail.partition("/")
        if key and key not in present_keys:
            missing.append(str(row.id))
    return missing


def reconcile(db: Any, s3: Any, *, org_id: Optional[str] = None) -> Reconciliation:
    """Sort the bucket against the table. Reads only; writes nothing, ever."""
    from datetime import datetime, timedelta, timezone

    prefix = f"{org_id}/" if org_id else ""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=grace_hours())
    result = Reconciliation()

    parsed: list = []
    present_keys: set = set()
    for obj in _list_objects(s3, prefix=prefix):
        key = obj.get("Key")
        if not key:
            continue
        present_keys.add(key)
        fields = parse_key(key)
        if fields is None:
            result.unrecognised_keys.append(key)
            continue
        key_org, artifact_id, version = fields
        if org_id and key_org != org_id:
            # Belt and braces on top of the Prefix: an object store is not a
            # tenancy boundary, and a key that does not start with the org we
            # asked for is not this org's business.
            continue
        result.scanned += 1
        parsed.append((key, key_org, artifact_id, version, obj.get("LastModified")))

    by_org: dict = {}
    for entry in parsed:
        by_org.setdefault(entry[1], []).append(entry)

    for key_org, entries in by_org.items():
        claims = _claims(db, key_org, [e[2] for e in entries])
        for key, _, artifact_id, version, last_modified in entries:
            claim = claims.get((artifact_id, version))
            if claim is None:
                if last_modified is not None and last_modified > cutoff:
                    result.too_recent += 1
                else:
                    result.orphan_keys.append(key)
                continue
            status, past_grace = claim
            if status in ("retracted", "superseded") and past_grace:
                result.unpurged_keys.append(key)
            else:
                result.claimed += 1

    result.missing_object_ids = _missing_objects(db, org_id, present_keys)
    return result


def candidate_ids(result: Reconciliation) -> list:
    """The set the confirmation token covers: the orphans, and only those.

    ``unpurged`` is excluded because ``--execute`` will not delete it, and a
    token that moved for rows the operator's approval would not act on trains
    them to re-copy tokens without reading.
    """
    return list(result.orphan_keys)


def _still_unclaimed(db: Any, key: str) -> bool:
    """Re-check one key against the table immediately before deleting it.

    The confirmation token cannot cover this: it describes the set the
    operator approved, and a presigned upload that confirms between the scan
    and the delete produces a row that was never in that set. So the last word
    belongs to the database, not to the scan.
    """
    fields = parse_key(key)
    if fields is None:
        return False
    key_org, artifact_id, version = fields
    row = db.execute(
        text(
            """
            SELECT 1 FROM artifacts
            WHERE org_id = :org_id AND id = CAST(:id AS uuid) AND version = :version
            """
        ),
        {"org_id": key_org, "id": artifact_id, "version": version},
    ).fetchone()
    return row is None


def report(result: Reconciliation, *, limit: int) -> None:
    """Log the reconciliation. Keys and counts; never a credential or endpoint."""
    logger.info(
        "scanned=%d claimed=%d orphan=%d unpurged=%d too_recent=%d "
        "unrecognised=%d missing_object=%d",
        result.scanned,
        result.claimed,
        len(result.orphan_keys),
        len(result.unpurged_keys),
        result.too_recent,
        len(result.unrecognised_keys),
        len(result.missing_object_ids),
    )
    logger.info(
        "  orphan keys:       %s", common.format_sample(result.orphan_keys, limit)
    )
    logger.info(
        "  unpurged keys:     %s", common.format_sample(result.unpurged_keys, limit)
    )
    logger.info(
        "  unrecognised keys: %s",
        common.format_sample(result.unrecognised_keys, limit),
    )
    logger.info(
        "  rows missing bytes: %s",
        common.format_sample(result.missing_object_ids, limit),
    )
    logger.info(
        "WOULD DO: delete %d unclaimed object(s), each re-checked against the "
        "artifacts table immediately before its DELETE.",
        len(result.orphan_keys),
    )
    logger.info(
        "WOULD NOT DO: delete the %d unpurged object(s) — that is the GC "
        "sweep's job and this is not it; the %d unrecognised key(s) — a key "
        "this script cannot parse is one it does not understand; or anything "
        "at all about the %d row(s) whose bytes are already gone.",
        len(result.unpurged_keys),
        len(result.unrecognised_keys),
        len(result.missing_object_ids),
    )


def run(db: Any, s3: Any, args: Any) -> Reconciliation:
    """Reconcile, report, and — only when fully confirmed — delete orphans."""
    result = reconcile(db, s3, org_id=args.org)
    report(result, limit=args.sample_limit)

    ids = candidate_ids(result)
    execute = common.resolve_execute(args, script=SCRIPT_NAME, candidate_ids=ids)
    if not execute:
        logger.info(
            "DRY RUN — nothing was deleted. To delete the unclaimed objects, "
            "review the counts above and re-run with: --org <org_id> "
            "--execute --confirm %s",
            common.confirmation_token(SCRIPT_NAME, args.org, ids),
        )
        return result

    deleted = 0
    for key in ids:
        if not _still_unclaimed(db, key):
            logger.info(
                "skipping %s: a row now claims it (it was uploaded between the "
                "scan and this delete)",
                key,
            )
            continue
        s3.delete_object(Bucket=ARTIFACT_BUCKET, Key=key)
        deleted += 1
    logger.info("[EXECUTED] deleted %d unclaimed object(s)", deleted)
    return result


def main() -> None:
    parser = common.build_parser(__doc__)
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable is required")
        sys.exit(1)

    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        s3 = get_artifact_s3_client()
        with engine.connect() as db:
            run(db, s3, args)
    except common.ConfirmationError as exc:
        logger.error("REFUSED: %s", exc)
        sys.exit(2)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
