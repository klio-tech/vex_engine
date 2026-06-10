"""One-off backfill: mask secrets already stored in memory, all orgs.

Sweeps ``session_memories.content`` and ``brain_recall_events.query_text``
with the same detector the write path now uses, rewrites hits with
placeholders, and re-embeds rewritten memories. Dry-run by default.

Usage:
    python -m scripts.run_redaction_backfill                # dry-run, all orgs
    python -m scripts.run_redaction_backfill --apply        # write changes
    python -m scripts.run_redaction_backfill --org <org_id> # restrict to one org

Environment:
    DATABASE_URL: TimescaleDB connection string
    EMBED_BATCH_SLEEP_S: pause between applied batches (default 0.5)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import Counter
from typing import Any, Callable, Optional

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from shared.memory import embed_text  # noqa: E402
from shared.redaction import redact  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("redaction_backfill")

BATCH_SIZE = 200


async def process_memory_batch(
    db: Any,
    rows: list[Any],
    *,
    apply: bool,
    embed_fn: Callable = embed_text,
) -> dict:
    """Scan one batch of memory rows; mask + re-embed hits when ``apply``.

    Returns a report Counter keyed by ``(org_id, kind)``. A row whose
    re-embedding fails is reported but left UNMODIFIED (content and embedding
    must stay consistent). Idempotent: placeholders never re-match.
    """
    report: Counter = Counter()
    for row in rows:
        result = redact(row.content or "")
        if not result.redacted:
            continue
        for f in result.findings:
            report[(row.org_id, f.kind)] += 1
        if not apply:
            continue
        embedding = await embed_fn(result.clean_text)
        if embedding is None:
            logger.warning("re-embed failed for memory %s; leaving unmodified", row.id)
            continue
        db.execute(
            text(
                """
                UPDATE session_memories
                SET content = :content,
                    embedding = :embedding,
                    metadata = COALESCE(metadata, '{}'::jsonb)
                               || CAST(:extra_meta AS jsonb),
                    updated_at = NOW()
                WHERE id = :id
                """
            ),
            {
                "content": result.clean_text,
                "embedding": str(embedding),
                "extra_meta": json.dumps(
                    {
                        "redaction_backfill": time.strftime("%Y-%m-%d"),
                        "redactions": sorted({f.kind for f in result.findings}),
                    }
                ),
                "id": str(row.id),
            },
        )
        db.commit()
    return dict(report)


def process_recall_events(db: Any, *, apply: bool, org_id: Optional[str]) -> int:
    """Mask secrets in brain_recall_events.query_text. Returns hit count."""
    where = "query_text IS NOT NULL"
    params: dict[str, Any] = {}
    if org_id:
        where += " AND org_id = :org_id"
        params["org_id"] = org_id
    rows = db.execute(
        text(f"SELECT id, query_text FROM brain_recall_events WHERE {where}"),
        params,
    ).fetchall()
    hits = 0
    for row in rows:
        result = redact(row.query_text)
        if not result.redacted:
            continue
        hits += 1
        if apply:
            db.execute(
                text("UPDATE brain_recall_events SET query_text = :q WHERE id = :id"),
                {"q": result.clean_text, "id": str(row.id)},
            )
    if apply:
        db.commit()
    return hits


async def run(database_url: str, *, apply: bool, org_id: Optional[str]) -> None:
    engine = create_engine(database_url, pool_pre_ping=True)
    sleep_s = float(os.environ.get("EMBED_BATCH_SLEEP_S", "0.5"))
    total: Counter = Counter()
    cursor_at, cursor_id = None, None
    with engine.connect() as db:
        while True:
            where = [
                "(:cursor_ts IS NULL OR created_at > :cursor_ts"
                " OR (created_at = :cursor_ts AND id > :cursor_id))"
            ]
            params: dict[str, Any] = {
                "cursor_ts": cursor_at,
                "cursor_id": cursor_id,
                "batch": BATCH_SIZE,
            }
            if org_id:
                where.append("org_id = :org_id")
                params["org_id"] = org_id
            rows = db.execute(
                text(
                    f"""
                    SELECT id, org_id, content, created_at
                    FROM session_memories
                    WHERE {" AND ".join(where)}
                    ORDER BY created_at ASC, id ASC
                    LIMIT :batch
                    """
                ),
                params,
            ).fetchall()
            if not rows:
                break
            batch_report = await process_memory_batch(db, rows, apply=apply)
            total.update(batch_report)
            cursor_at, cursor_id = rows[-1].created_at, rows[-1].id
            if apply and batch_report:
                await asyncio.sleep(sleep_s)  # rate-limit embedding provider

        recall_hits = process_recall_events(db, apply=apply, org_id=org_id)

    mode = "APPLIED" if apply else "DRY-RUN"
    logger.info("[%s] recall_events with secrets: %d", mode, recall_hits)
    for (org, kind), count in sorted(total.items()):
        logger.info("[%s] org=%s kind=%s count=%d", mode, org, kind, count)
    if not total and not recall_hits:
        logger.info("[%s] no secrets found", mode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes")
    parser.add_argument("--org", default=None, help="restrict to one org_id")
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable is required")
        sys.exit(1)
    asyncio.run(run(database_url, apply=args.apply, org_id=args.org))


if __name__ == "__main__":
    main()
