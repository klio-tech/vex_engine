"""Recompute every stored embedding with the current model.

Vectors from two embedding models are not comparable: a query embedded by the
new model and a row embedded by the old one produce a cosine that measures
nothing. So an embedding-model change is not a config change, it is a data
migration, and this is it.

It was written for the 2026-09-18 move from ``intfloat/multilingual-e5-large-
instruct`` to ``alibaba/qwen3-embedding-0.6b`` (see
``shared.memory.DEFAULT_EMBEDDING_MODEL``), but reads the model and endpoint
from the same environment the engine does, so it re-runs unchanged for the
next one.

**Same input as a live write.** Rows are embedded as passages (no prefix — the
``shared.embedding_prompt`` convention) and head-truncated to
``shared.memory.MAX_EMBED_CHARS``, exactly what ``embed_text`` does on its first
attempt, so a re-embedded row and a fresh write of the same content get the
same vector. A batch the gateway rejects is retried one row at a time through
``embed_text`` itself, which adds its progressive-truncation fallback.

**In place, resumable, idempotent.** Rows are walked in ``id`` order and each
page is committed on its own; the last committed id is logged, and
``--after <id>`` resumes from it. Re-running a finished table recomputes the
same vectors. Only rows that already HAVE an embedding are touched: a NULL
means "never embedded", which is the write path's business, not this job's.

**Order of operations for a model change** — the job must run against the NEW
model while the engine still points at the OLD one, then the engine switches:

    1. back up the engine database
    2. MEMORY_EMBEDDING_MODEL=<new> python reembed_memories.py --dry-run
    3. MEMORY_EMBEDDING_MODEL=<new> python reembed_memories.py
    4. switch the engine's MEMORY_EMBEDDING_MODEL and restart it

Between 3 and 4 the engine reads new vectors with old queries, so run it when
the old model is already unusable (as on 2026-09-18) or accept that window.

Usage::

    python reembed_memories.py --dry-run
    python reembed_memories.py [--table session_memories] [--after <uuid>]

Environment: ``DATABASE_URL``, ``MEMORY_EMBEDDING_MODEL`` and the embedding
gateway variables ``shared.memory`` reads (``MEMORY_EMBEDDING_API_URL`` /
``_KEY``, falling back to ``LITELLM_API_URL`` / ``_KEY``).
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from typing import Any, Awaitable, Callable, Optional, Sequence

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from shared import embedding_prompt  # noqa: E402
from shared.memory import (  # noqa: E402
    MAX_EMBED_CHARS,
    _embed_endpoint,
    _embed_model,
    embed_text,
)

logger = logging.getLogger("reembed_memories")

#: Tables whose ``vector(1024)`` column this job recomputes, and the text column
#: each vector was computed from.
#:
#: ``entities`` also has a ``vector(1024)`` column but is deliberately absent:
#: ``graph_store.upsert_entity`` accepts an optional embedding that no caller
#: supplies, so there is no defined source text to reproduce (all 12,317 rows
#: were NULL on 2026-09-18). Guessing ``name`` would write vectors nothing else
#: in the engine would ever compute the same way.
TABLES: dict[str, str] = {
    "session_memories": "content",
    "work_claims": "description",
}

PAGE_SIZE = 512
BATCH_SIZE = 64

EmbedBatch = Callable[[Sequence[str]], Awaitable[list[list[float]]]]
EmbedOne = Callable[[str], Awaitable[Optional[list[float]]]]


def model_input(value: Optional[str]) -> str:
    """The exact string a live write would embed for *value*."""
    return (value or "")[:MAX_EMBED_CHARS]


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


async def embed_page(
    texts: Sequence[str],
    *,
    embed_batch: EmbedBatch,
    embed_one: EmbedOne,
    batch_size: int = BATCH_SIZE,
    concurrency: int = 4,
) -> list[Optional[list[float]]]:
    """Vectors for *texts*, in order. ``None`` only where even ``embed_one`` failed.

    Batches go out concurrently. A batch the backend rejects is not retried as
    a batch — one bad row would sink the other 63 every time — but row by row
    through *embed_one*, which is ``embed_text`` in production and so carries
    its truncation fallback.
    """
    out: list[Optional[list[float]]] = [None] * len(texts)
    sem = asyncio.Semaphore(concurrency)

    async def run(start: int) -> None:
        chunk = list(texts[start : start + batch_size])
        async with sem:
            try:
                vectors = await embed_batch(chunk)
                if len(vectors) != len(chunk):
                    raise ValueError(
                        f"backend returned {len(vectors)} vectors for {len(chunk)} inputs"
                    )
                out[start : start + len(chunk)] = vectors
                return
            except Exception as exc:  # noqa: BLE001 -- fall back per row, below
                logger.warning(
                    "batch at %d failed (%s); retrying row by row", start, exc
                )
            for offset, item in enumerate(chunk):
                out[start + offset] = await embed_one(item)

    await asyncio.gather(*(run(i) for i in range(0, len(texts), batch_size)))
    return out


def write_vectors(
    db: Any, table: str, ids: Sequence[str], vectors: Sequence[Optional[list[float]]]
) -> tuple[int, int]:
    """Store *vectors*; returns ``(written, skipped)``. A ``None`` keeps the old row."""
    written = skipped = 0
    for row_id, vector in zip(ids, vectors):
        if vector is None:
            skipped += 1
            continue
        db.execute(
            text(
                f"UPDATE {table} SET embedding = CAST(:v AS vector) WHERE id = CAST(:id AS uuid)"
            ),
            {"v": vector_literal(vector), "id": row_id},
        )
        written += 1
    return written, skipped


async def reembed_table(
    engine: Any,
    table: str,
    *,
    embed_batch: EmbedBatch,
    embed_one: EmbedOne,
    after: Optional[str] = None,
    limit: Optional[int] = None,
    dry_run: bool = False,
    concurrency: int = 4,
) -> dict:
    column = TABLES[table]
    with engine.connect() as conn:
        total = conn.execute(
            text(f"SELECT count(*) FROM {table} WHERE embedding IS NOT NULL")
        ).scalar_one()
    logger.info("%s: %d rows with an embedding", table, total)
    if dry_run:
        return {"table": table, "rows": total, "written": 0, "skipped": 0}

    written = skipped = 0
    cursor = after
    started = time.monotonic()
    while True:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT id::text AS id, {column} AS body FROM {table} "
                    "WHERE embedding IS NOT NULL AND (CAST(:after AS uuid) IS NULL OR id > CAST(:after AS uuid)) "
                    "ORDER BY id LIMIT :page"
                ),
                {"after": cursor, "page": PAGE_SIZE},
            ).fetchall()
        if not rows:
            break
        ids = [r.id for r in rows]
        vectors = await embed_page(
            [model_input(r.body) for r in rows],
            embed_batch=embed_batch,
            embed_one=embed_one,
            concurrency=concurrency,
        )
        with engine.begin() as tx:
            w, s = write_vectors(tx, table, ids, vectors)
        written += w
        skipped += s
        cursor = ids[-1]
        rate = written / max(time.monotonic() - started, 1e-6)
        logger.info(
            "%s: %d/%d written, %d skipped, %.0f rows/s — resume with --after %s",
            table,
            written,
            total,
            skipped,
            rate,
            cursor,
        )
        if limit is not None and written + skipped >= limit:
            break
    return {
        "table": table,
        "rows": total,
        "written": written,
        "skipped": skipped,
        "last_id": cursor,
    }


def gateway_embedder() -> tuple[EmbedBatch, EmbedOne]:
    """The production embedding call, batched. Same model, endpoint and wiring as ``embed_text``."""
    from litellm import aembedding

    model = _embed_model()
    api_base, api_key = _embed_endpoint()

    async def embed_batch(texts: Sequence[str]) -> list[list[float]]:
        kwargs: dict = {"model": model, "input": list(texts)}
        if api_base:
            kwargs["api_base"] = api_base
            kwargs["custom_llm_provider"] = "openai"
        if api_key:
            kwargs["api_key"] = api_key
        response = await aembedding(**kwargs)
        data = sorted(
            response.data, key=lambda d: d["index"] if isinstance(d, dict) else d.index
        )
        return [d["embedding"] if isinstance(d, dict) else d.embedding for d in data]

    async def embed_one(value: str) -> Optional[list[float]]:
        return await embed_text(value, role=embedding_prompt.ROLE_PASSAGE)

    return embed_batch, embed_one


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table", choices=[*TABLES, "all"], default="all")
    parser.add_argument(
        "--after", help="resume after this id (from the last progress line)"
    )
    parser.add_argument(
        "--limit", type=int, help="stop after about this many rows (for a trial run)"
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count rows, embed nothing, write nothing",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    engine = create_engine(os.environ["DATABASE_URL"])
    embed_batch, embed_one = gateway_embedder()
    logger.info(
        "model=%s endpoint=%s",
        _embed_model(),
        _embed_endpoint()[0] or "(litellm default)",
    )
    tables = list(TABLES) if args.table == "all" else [args.table]
    failed = 0
    for table in tables:
        result = asyncio.run(
            reembed_table(
                engine,
                table,
                embed_batch=embed_batch,
                embed_one=embed_one,
                after=args.after if len(tables) == 1 else None,
                limit=args.limit,
                dry_run=args.dry_run,
                concurrency=args.concurrency,
            )
        )
        logger.info("done: %s", result)
        failed += result["skipped"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
