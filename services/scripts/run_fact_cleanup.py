"""One-off cleanup: retract tool-call/build-artifact 'facts' from the brain.

The curator historically distilled passive hook tool-call captures into
``fact``/``summary`` rows like "Tool Bash was used with command: …". Those are
build artifacts, not durable memory, and pollute recall. This script retracts
them (reversible: status='retracted' + audit metadata) across all orgs.

Usage:
    python -m scripts.run_fact_cleanup                 # dry-run, all orgs
    python -m scripts.run_fact_cleanup --apply         # write changes
    python -m scripts.run_fact_cleanup --org <org_id>  # restrict to one org

Environment:
    DATABASE_URL: TimescaleDB connection string
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from typing import Any, Optional

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from shared.fact_noise import is_tool_call_noise  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("fact_cleanup")

BATCH_SIZE = 200


def process_fact_batch(db: Any, rows: list[Any], *, apply: bool) -> dict:
    """Scan one batch of fact/summary rows; retract noise when ``apply``.

    Returns a report dict ``{org_id: noise_count}``. Retraction is reversible
    (status='retracted'); recall already excludes non-active rows so cleanup
    is immediate. Idempotent: the batch query only selects active rows, and a
    retracted row won't be re-selected.
    """
    report: Counter = Counter()
    for row in rows:
        if not is_tool_call_noise(row.content):
            continue
        report[row.org_id] += 1
        if not apply:
            continue
        db.execute(
            text(
                """
                UPDATE session_memories
                SET status = 'retracted',
                    metadata = COALESCE(metadata, '{}'::jsonb)
                               || CAST(:extra AS jsonb),
                    updated_at = NOW()
                WHERE id = :id
                """
            ),
            {
                "id": str(row.id),
                "extra": json.dumps(
                    {
                        "cleanup": time.strftime("%Y-%m-%d"),
                        "cleanup_reason": "tool_call_noise",
                    }
                ),
            },
        )
        db.commit()
    return dict(report)


def run(database_url: str, *, apply: bool, org_id: Optional[str]) -> None:
    engine = create_engine(database_url, pool_pre_ping=True)
    total: Counter = Counter()
    cursor_at, cursor_id = None, None
    samples: list[str] = []
    with engine.connect() as db:
        while True:
            where = [
                "status = 'active'",
                "memory_type IN ('fact', 'summary')",
                "(:cursor_ts IS NULL OR created_at > :cursor_ts"
                " OR (created_at = :cursor_ts AND id > :cursor_id))",
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
            batch_report = process_fact_batch(db, rows, apply=apply)
            total.update(batch_report)
            # Collect a few dry-run samples for human review.
            if not apply and len(samples) < 15:
                for r in rows:
                    if len(samples) >= 15:
                        break
                    if is_tool_call_noise(r.content):
                        samples.append(r.content[:120])
            cursor_at, cursor_id = rows[-1].created_at, rows[-1].id

    mode = "APPLIED" if apply else "DRY-RUN"
    for org, count in sorted(total.items()):
        logger.info("[%s] org=%s noise_facts=%d", mode, org, count)
    if not total:
        logger.info("[%s] no tool-call noise facts found", mode)
    if samples:
        logger.info("[%s] sample matches (review before --apply):", mode)
        for s in samples:
            logger.info("    %r", s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes")
    parser.add_argument("--org", default=None, help="restrict to one org_id")
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable is required")
        sys.exit(1)
    run(database_url, apply=args.apply, org_id=args.org)


if __name__ == "__main__":
    main()
