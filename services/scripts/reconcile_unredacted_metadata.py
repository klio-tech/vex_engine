"""Count the secrets sitting unmasked in ``session_memories.metadata``.

Why there is anything to count
------------------------------
:func:`shared.brain.write_scoped_memory` redacts ``content``, and only
``content``. The ``decide`` MCP tool takes a caller-supplied ``rationale`` and
files it under ``metadata``, where nothing screens it. A rationale is free
prose an agent wrote about *why* it did something — precisely the shape of
text a credential ends up inside — and no backfill in this repo has ever
looked at it.

WP-B fixed the adjacent defect: ``sync-gateway`` and ``async-worker`` wrote
LLM summaries and extracted facts through ``write_memory`` with no redactor at
all. That was a ``content`` problem, and ``run_redaction_backfill.py`` already
sweeps ``content`` (re-embedding as it goes, which is why it must own that
column alone). Metadata is the part nothing covers, so it is what this script
measures.

The output rule, which is the whole point
-----------------------------------------
This script selects rows *because* they contain an unmasked secret. Printing
the value it found — or the words around it — would move that secret from the
database into a log with worse access control, and would defeat the reason for
running it. So the report is **pattern class, key path, row id and count**. No
value. No context. No excerpt. Not even on a near miss.

Detection uses the current production redactor, so what it counts is exactly
what the write path would mask today. Idempotent: placeholders never re-match,
so a second run over this script's own output reports a clean database, which
is the only way anyone can confirm the remediation worked.

What ``--execute`` does
-----------------------
Rewrites the offending metadata leaves in place with the redactor's
placeholders, preserving every other key. Metadata is not embedded, so unlike
the content backfill this rewrite cannot desynchronise a row from its vector,
and it never touches ``content`` — that column belongs to
``run_redaction_backfill.py``, which can re-embed and this cannot.

Usage:
    python -m scripts.reconcile_unredacted_metadata                 # dry run, fleet
    python -m scripts.reconcile_unredacted_metadata --org <org_id>  # dry run, one org
    python -m scripts.reconcile_unredacted_metadata --org <org_id> \\
        --execute --confirm <token-from-the-dry-run>

Environment:
    DATABASE_URL: TimescaleDB connection string.
"""

import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence

from sqlalchemy import create_engine, text

# Both the sibling module and the shared package must resolve whether this is
# run as ``python -m scripts.<name>`` from ``services/`` (which puts
# ``services/`` on the path, not ``services/scripts/``) or executed directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))

import reconcile_common as common  # noqa: E402
from shared.redaction import redact  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("reconcile.unredacted_metadata")

SCRIPT_NAME = "reconcile_unredacted_metadata"

#: Rows are read in batches so a fleet-wide count does not materialise every
#: metadata blob in the database at once.
BATCH_SIZE = 500


@dataclass(frozen=True)
class MetadataFinding:
    """One metadata leaf that the current redactor would mask.

    Carries the *kinds* and never the value. There is deliberately no field
    on this object that could hold the secret, so no reporting mistake
    downstream can print one.
    """

    org_id: str
    row_id: str
    key_path: str
    kinds: tuple


def _walk(value: Any, path: str = "") -> Iterator[tuple]:
    """Yield ``(key_path, string)`` for every string leaf in a JSON value.

    Metadata is free-form JSONB. A scanner that only checked known top-level
    keys would go blind the moment a tool nests something, and the keys are
    not a fixed set — ``decide`` writes ``rationale``, the curator writes
    ``input_count``, the backfill writes ``redactions``.
    """
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            yield from _walk(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")


def _scrub(value: Any) -> tuple:
    """Return ``(new_value, changed)`` with every string leaf redacted.

    Builds a new structure rather than mutating in place: the caller still
    holds the original and compares against it, and an in-place rewrite of a
    row's metadata that half-succeeded would be unreviewable.
    """
    if isinstance(value, str):
        result = redact(value)
        return (result.clean_text, True) if result.redacted else (value, False)
    if isinstance(value, dict):
        changed = False
        out = {}
        for key, item in value.items():
            out[key], item_changed = _scrub(item)
            changed = changed or item_changed
        return out, changed
    if isinstance(value, list):
        changed = False
        out = []
        for item in value:
            new_item, item_changed = _scrub(item)
            out.append(new_item)
            changed = changed or item_changed
        return out, changed
    return value, False


def _rows(db: Any, org_id: Optional[str]) -> Iterator[Any]:
    """Stream rows that have metadata at all, org-scoped in SQL.

    Keyset-paginated on ``id`` rather than OFFSET: the population is whatever
    is in the table now, and an OFFSET scan re-reads or skips rows when
    anything is inserted mid-sweep.
    """
    where = ["metadata IS NOT NULL", "id > :cursor"]
    params: dict[str, Any] = {"batch": BATCH_SIZE}
    if org_id:
        where.append("org_id = :org_id")
        params["org_id"] = org_id

    cursor = "00000000-0000-0000-0000-000000000000"
    while True:
        rows = db.execute(
            text(
                f"""
                SELECT id, org_id, metadata
                FROM session_memories
                WHERE {" AND ".join(where)}
                ORDER BY id ASC
                LIMIT :batch
                """
            ),
            {**params, "cursor": cursor},
        ).fetchall()
        if not rows:
            return
        yield from rows
        cursor = str(rows[-1].id)


def scan_metadata(db: Any, *, org_id: Optional[str] = None) -> list[MetadataFinding]:
    """Every metadata leaf the current redactor would mask.

    Deliberately runs the *production* redactor rather than a copy of its
    patterns, so the count is exactly what the write path would catch today
    and cannot drift away from it.
    """
    findings: list[MetadataFinding] = []
    for row in _rows(db, org_id):
        metadata = row.metadata
        if not isinstance(metadata, (dict, list)):
            continue
        for key_path, value in _walk(metadata):
            result = redact(value)
            if not result.redacted:
                continue
            findings.append(
                MetadataFinding(
                    org_id=row.org_id,
                    row_id=str(row.id),
                    key_path=key_path,
                    kinds=tuple(sorted({f.kind for f in result.findings})),
                )
            )
    return findings


def counts_by_kind(findings: Sequence[MetadataFinding]) -> dict:
    """Aggregate by pattern class. The only shape this data is reported in."""
    counter: Counter = Counter()
    for finding in findings:
        for kind in finding.kinds:
            counter[kind] += 1
    return dict(counter)


def candidate_ids(findings: Sequence[MetadataFinding]) -> list[str]:
    """The row ids the confirmation token is computed over, de-duplicated.

    One row can hold several offending leaves; the token describes the set of
    rows an execute would write to.
    """
    seen: list[str] = []
    for finding in findings:
        if finding.row_id not in seen:
            seen.append(finding.row_id)
    return seen


def scrub_row(db: Any, *, org_id: str, row_id: str) -> bool:
    """Mask every offending leaf of one row's metadata. Returns whether it changed.

    Reads the row back under its own ``org_id`` predicate rather than trusting
    the id alone, so a scan result that has gone stale — or been tampered with
    between scan and write — cannot address another tenant's row.

    ``content`` is untouched on purpose: this script cannot re-embed, and a
    row whose text no longer matches its vector is a worse bug than the one
    being fixed. ``run_redaction_backfill.py`` owns that column.
    """
    row = db.execute(
        text(
            """
            SELECT metadata FROM session_memories
            WHERE id = :id AND org_id = :org_id
            """
        ),
        {"id": row_id, "org_id": org_id},
    ).fetchone()
    if row is None or not isinstance(row[0], (dict, list)):
        return False

    scrubbed, changed = _scrub(row[0])
    if not changed:
        return False

    db.execute(
        text(
            """
            UPDATE session_memories
            SET metadata = CAST(:meta AS jsonb),
                updated_at = now()
            WHERE id = :id AND org_id = :org_id
            """
        ),
        {"meta": json.dumps(scrubbed), "id": row_id, "org_id": org_id},
    )
    db.commit()
    return True


def report(findings: Sequence[MetadataFinding], *, limit: int) -> None:
    """Log the measurement: pattern class, key path, row id, count. Never a value."""
    if not findings:
        logger.info("[DRY-RUN] no unredacted secrets found in metadata")
        return

    counts = counts_by_kind(findings)
    rows = candidate_ids(findings)
    logger.info(
        "%d unredacted metadata value(s) across %d row(s)", len(findings), len(rows)
    )
    for kind, count in sorted(counts.items()):
        logger.info("  kind=%s count=%d", kind, count)

    key_paths: Counter = Counter(f.key_path for f in findings)
    for key_path, count in sorted(key_paths.items()):
        logger.info("  key=%s count=%d", key_path, count)

    org_counts: Counter = Counter(f.org_id for f in findings)
    for org, count in sorted(org_counts.items()):
        logger.info("  org=%s count=%d", org, count)

    logger.info("  affected row ids: %s", common.format_sample(rows, limit))
    logger.info(
        "WOULD DO: rewrite %d metadata value(s) on %d row(s), replacing each "
        "detected secret with its [REDACTED:<kind>:<fingerprint>] placeholder "
        "and leaving every other key intact. The content column and the "
        "embeddings are not touched.",
        len(findings),
        len(rows),
    )


def run(db: Any, args: Any) -> list[MetadataFinding]:
    """Measure, report, and — only when fully confirmed — scrub."""
    findings = scan_metadata(db, org_id=args.org)
    report(findings, limit=args.sample_limit)

    ids = candidate_ids(findings)
    execute = common.resolve_execute(args, script=SCRIPT_NAME, candidate_ids=ids)
    if not execute:
        logger.info(
            "DRY RUN — nothing was written. To scrub, review the counts above "
            "and re-run with: --org <org_id> --execute --confirm %s",
            common.confirmation_token(SCRIPT_NAME, args.org, ids),
        )
        return findings

    scrubbed = 0
    for row_id in ids:
        if scrub_row(db, org_id=args.org, row_id=row_id):
            scrubbed += 1
    logger.info("[EXECUTED] scrubbed metadata on %d row(s)", scrubbed)
    return findings


def main() -> None:
    parser = common.build_parser(__doc__)
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable is required")
        sys.exit(1)

    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as db:
            run(db, args)
    except common.ConfirmationError as exc:
        logger.error("REFUSED: %s", exc)
        sys.exit(2)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
