"""Backfill ``session_memories.embedding_v2`` — the E5-prefixed vector space.

Migration 046 adds the column and its HNSW index; this script fills it. Until
it reports 100 % complete, nothing should set ``KLIO_EMBEDDING_SPACE=v2``:
recall still reads ``embedding``, ranking is exactly as good (or bad) as it was
yesterday, and a half-finished run costs nothing but the rows it has not
reached yet. That property — a partial backfill leaves recall *working* — is
the whole reason this is a second column rather than an in-place re-embed.
See the migration's docstring for the measurement behind that choice.

Resumable and idempotent, by construction rather than by bookkeeping:

* the work list is ``WHERE embedding_v2 IS NULL``, so a row that has been done
  is never selected again and there is no cursor file to lose;
* paging is by primary key (``WHERE id > :cursor``), which is index-ordered and
  stable under concurrent inserts — a row written while the script runs
  already has its ``embedding_v2`` from the dual-write path, and one written
  behind the cursor is picked up by simply running the script again;
* each batch commits on its own. Killing the process loses at most one batch's
  work and corrupts nothing.

Usage::

    python -m scripts.run_embedding_v2_backfill --status
    python -m scripts.run_embedding_v2_backfill --apply
    python -m scripts.run_embedding_v2_backfill --apply --org <org_id>

Dry-run (the default) embeds nothing and only reports what is outstanding, so
it is safe to point at production.

Environment:
    DATABASE_URL: Postgres connection string. Strip ``-pooler`` when running
        against Neon — this is a long-lived connection doing many round trips,
        which is exactly what a transaction-mode pooler is worst at.
    LITELLM_API_URL / LITELLM_API_KEY / MEMORY_EMBEDDING_MODEL: as the services
        use.
    KLIO_BACKFILL_CONCURRENCY: parallel embedding calls (default 16). Measured
        against production: 4 → 4.7/s, 8 → 8.4/s, 16 → 15.7/s. Above 16 the
        gateway's own concurrency cap starts queueing.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from typing import Any, Optional

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from shared import embedding_space  # noqa: E402
from shared.memory import embed_text  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("embedding_v2_backfill")

#: Rows fetched, embedded and written per transaction. Sized so one batch is a
#: few seconds of gateway time — small enough that a kill loses little, large
#: enough that per-batch round trips are not the bottleneck.
BATCH_SIZE = 200

DEFAULT_CONCURRENCY = 16


def _concurrency() -> int:
    raw = os.environ.get("KLIO_BACKFILL_CONCURRENCY")
    try:
        value = int(raw) if raw else DEFAULT_CONCURRENCY
    except ValueError:
        logger.warning(
            "KLIO_BACKFILL_CONCURRENCY=%r is not a number; using %d",
            raw,
            DEFAULT_CONCURRENCY,
        )
        return DEFAULT_CONCURRENCY
    return max(1, min(value, 64))


def _org_filter(org_id: Optional[str]) -> tuple[str, dict[str, Any]]:
    if not org_id:
        return "", {}
    return " AND org_id = :org_id", {"org_id": org_id}


def report_status(db: Any, org_id: Optional[str]) -> tuple[int, int]:
    """``(done, total)`` over rows that have any content to embed.

    ``embedding IS NOT NULL`` is the denominator rather than every row: a row
    with no v1 vector was never embeddable in the first place (the write path
    refuses to store one), so counting it as outstanding would mean this script
    could never report complete.
    """
    clause, params = _org_filter(org_id)
    row = db.execute(
        text(
            f"""
            SELECT count(*) FILTER (WHERE {embedding_space.V2_COLUMN} IS NOT NULL),
                   count(*)
            FROM session_memories
            WHERE {embedding_space.V1_COLUMN} IS NOT NULL{clause}
            """
        ),
        params,
    ).first()
    return int(row[0]), int(row[1])


async def _embed_batch(rows: list[Any], concurrency: int) -> dict[str, list[float]]:
    """``{id: vector}`` for *rows*, embedded in parallel as passages.

    A row whose embedding fails is simply absent from the result: it keeps its
    ``NULL``, and the next run picks it up. Raising instead would let one bad
    row (content the gateway rejects at every truncation) stall a 63,000-row
    backfill forever.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def one(row: Any) -> tuple[str, Optional[list[float]]]:
        async with semaphore:
            vector = await embed_text(
                row.content or "",
                role=embedding_space.ROLE_PASSAGE,
                space=embedding_space.SPACE_V2,
            )
            return str(row.id), vector

    results = await asyncio.gather(*(one(row) for row in rows))
    return {row_id: vector for row_id, vector in results if vector is not None}


async def run(
    database_url: str, *, apply: bool, org_id: Optional[str], status_only: bool
) -> int:
    """Backfill until nothing is outstanding. Returns the number of rows written."""
    engine = create_engine(database_url, pool_pre_ping=True)
    concurrency = _concurrency()
    written = 0
    failed = 0
    started = time.monotonic()

    with engine.connect() as db:
        done, total = report_status(db, org_id)
        logger.info(
            "%s: %d/%d rows have %s (%.1f%%)",
            "status" if status_only else ("apply" if apply else "dry-run"),
            done,
            total,
            embedding_space.V2_COLUMN,
            100.0 * done / total if total else 100.0,
        )
        if status_only:
            return 0
        if not apply:
            logger.info(
                "dry-run: %d rows would be embedded. Re-run with --apply.",
                total - done,
            )
            return 0

        clause, org_params = _org_filter(org_id)
        cursor: Optional[str] = None
        while True:
            params: dict[str, Any] = {"batch": BATCH_SIZE, "cursor": cursor}
            params.update(org_params)
            rows = db.execute(
                text(
                    f"""
                    SELECT id, content
                    FROM session_memories
                    WHERE {embedding_space.V2_COLUMN} IS NULL
                      AND {embedding_space.V1_COLUMN} IS NOT NULL
                      AND (CAST(:cursor AS uuid) IS NULL OR id > CAST(:cursor AS uuid))
                      {clause}
                    ORDER BY id
                    LIMIT :batch
                    """
                ),
                params,
            ).fetchall()
            if not rows:
                break

            cursor = str(rows[-1].id)
            vectors = await _embed_batch(list(rows), concurrency)
            failed += len(rows) - len(vectors)

            for row_id, vector in vectors.items():
                db.execute(
                    text(
                        f"""
                        UPDATE session_memories
                        SET {embedding_space.V2_COLUMN} = :vector
                        WHERE id = CAST(:id AS uuid)
                          AND {embedding_space.V2_COLUMN} IS NULL
                        """
                    ),
                    {"vector": str(vector), "id": row_id},
                )
            db.commit()
            written += len(vectors)

            elapsed = time.monotonic() - started
            logger.info(
                "%d written, %d failed, %.1f rows/s",
                written,
                failed,
                written / elapsed if elapsed else 0.0,
            )

        done, total = report_status(db, org_id)
        logger.info(
            "complete: %d/%d (%.2f%%), %d written this run, %d failed",
            done,
            total,
            100.0 * done / total if total else 100.0,
            written,
            failed,
        )
        if done < total:
            logger.warning(
                "%d rows still have no %s — re-run to retry them; do NOT switch "
                "%s until this reports 100%%",
                total - done,
                embedding_space.V2_COLUMN,
                embedding_space.SPACE_ENV,
            )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write embeddings")
    parser.add_argument("--org", default=None, help="restrict to one org_id")
    parser.add_argument(
        "--status", action="store_true", help="report progress and exit"
    )
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable is required")
        sys.exit(1)
    if "-pooler" in database_url:
        logger.warning(
            "DATABASE_URL points at a pooled endpoint; this backfill holds one "
            "connection across many round trips and should use the direct one"
        )
    asyncio.run(
        run(
            database_url,
            apply=args.apply,
            org_id=args.org,
            status_only=args.status,
        )
    )


if __name__ == "__main__":
    main()
