"""Measure recall ranking: the old behaviour, the new one, side by side.

The 2026-08-16 ranking work is a set of claims about ordering, and ordering
claims are worthless unasserted. This script is the assertion. It runs the same
query set through three configurations against the same corpus and reports what
each returns, so "recall got better" is a number somebody else can reproduce
rather than an opinion.

**The three configurations**

``before``
    The shipped behaviour as of the fix, replayed verbatim from
    :data:`_BASELINE_SQL`: ``0.85 * cosine + 0.15 * recency`` un-normalised, a
    candidate pool of ``limit * 4``, no relevance floor, and
    ``project_id = NULL`` — which is what 94.8 % of production hook recalls
    actually resolved.
``after-unfenced``
    The new path with the project filter deliberately switched OFF. This
    isolates what the embedding prefixes and the re-weighted blend did on
    their own, with no help from scoping. It is the honest comparison: fencing
    a query to one project will obviously stop cross-project results, and
    quoting that as evidence the ranking improved would be circular.
``after``
    The new path as shipped, with the project resolved exactly the way
    ``capture.py`` now resolves it.

**Two metrics, both objective**

*Contamination* — of the rows returned, how many are filed under a project
OTHER than the one the session is in. This is the reported symptom stated as a
number, and it needs no human judgement: the ground truth is the ``project_id``
the writer stored.

*Separation* — the cosine gap between the best on-project row and the best
off-project row, measured over the WHOLE corpus rather than the returned page.
The diagnosis measured this at ~0.085 for the entire corpus against any query
— a band so narrow that the best match in 18,367 rows barely outscored a
random one. This is the number that has to move for anything else to matter,
and it is measured unfenced, because fencing does not change a similarity.

**Where the corpus comes from.** ``seed`` copies a sample of a source database
(read-only against it) into a target one, so the evaluation runs against real
content and real vectors without touching production. Only the columns ranking
depends on are copied. Nothing is re-embedded: the fix deliberately does not
change stored vectors (see :mod:`shared.embedding_prompt`).

Usage::

    # read-only against source, writes only to target
    python -m scripts.eval_recall_ranking seed \\
        --source-url "$PROD_URL" --target-url "$LOCAL_URL" --org vex

    python -m scripts.eval_recall_ranking run --url "$LOCAL_URL"
    python -m scripts.eval_recall_ranking run --url "$LOCAL_URL" --markdown

Environment: ``LITELLM_API_URL`` / ``LITELLM_API_KEY`` /
``MEMORY_EMBEDDING_MODEL`` for the embedding gateway.
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
from typing import Any, Optional

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from shared import embedding_prompt  # noqa: E402
from shared.memory import embed_text, retrieve_memories_scoped  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("eval_recall_ranking")

QUERY_FILE = os.path.join(os.path.dirname(__file__), "eval_queries.json")

#: How many rows a query returns in this evaluation. Matches the Node proxy's
#: ``RECALL_LIMIT`` of 8 rounded up to a round page, so the ranking is judged
#: over roughly what an agent actually receives.
LIMIT = 10

#: Projects seeded in full. Everything the query set names, plus the two that
#: produced the reported leak, plus the largest projects in the corpus — the
#: distractors have to be present or the evaluation is measuring an empty room.
SEED_PROJECTS = (
    "Klio",
    "vex_engine",
    "Agent Memory",
    "moonforge-prod",
    "Astrology",
    "hirly",
    "hireloop",
    "loopreel",
    "relio",
    ".buzz",
)

#: Additional rows drawn at random from every OTHER project, so the corpus has
#: the long tail that the diagnosis identified as the real problem ("118
#: projects all pouring into one undifferentiated pool").
SEED_OTHER_ROWS = 3000

#: The pre-fix ranking statement, kept verbatim so "before" is a measurement
#: and not a memory. Every constant in it is the one that shipped:
#: ``limit * 4`` pool, ``0.85``/``0.15`` weights on an UN-normalised cosine,
#: a 30-day recency window, and no floor of any kind.
_BASELINE_SQL = """
    SELECT * FROM (
        SELECT id, content, created_at, project_id,
               (1 - (embedding <=> :embedding)) AS relevance
        FROM session_memories
        WHERE org_id = :org_id
          AND status = 'active'
          AND superseded_by IS NULL
          AND recall_hidden = FALSE
          AND (scope = 'org' OR scope = 'private')
        ORDER BY embedding <=> :embedding
        LIMIT :candidate_pool
    ) candidates
    ORDER BY (0.85 * relevance
              + 0.15 * LEAST(1, GREATEST(
                  0, 1 - EXTRACT(EPOCH FROM (NOW() - created_at)) / (30 * 86400)
                ))) DESC
    LIMIT :limit
"""

#: Corpus-wide separation, unfenced. ``project_id`` is compared against the
#: session's project so the two halves can be scored apart.
_SEPARATION_SQL = """
    SELECT max(1 - (embedding <=> :embedding))
             FILTER (WHERE project_id = :project_id) AS best_on,
           max(1 - (embedding <=> :embedding))
             FILTER (WHERE project_id IS NOT NULL AND project_id <> :project_id)
             AS best_off,
           max(1 - (embedding <=> :embedding)) AS best_any,
           percentile_cont(0.5) WITHIN GROUP (
               ORDER BY 1 - (embedding <=> :embedding)) AS median_any
    FROM session_memories
    WHERE org_id = :org_id
      AND status = 'active'
      AND superseded_by IS NULL
      AND recall_hidden = FALSE
      AND embedding IS NOT NULL
"""


def _load_queries() -> dict[str, Any]:
    with open(QUERY_FILE, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------


def _seed_rows(source: Any, org_id: str) -> list[Any]:
    named = source.execute(
        text(
            """
            SELECT m.id, m.content, m.created_at, m.memory_type, m.scope,
                   m.user_id, m.project_id, m.embedding, m.session_id,
                   m.org_id, m.agent_id
            FROM session_memories m
            JOIN projects p ON p.id = m.project_id
            WHERE m.org_id = :org_id AND m.status = 'active'
              AND m.recall_hidden = FALSE AND m.embedding IS NOT NULL
              AND p.display_name = ANY(:names)
            """
        ),
        {"org_id": org_id, "names": list(SEED_PROJECTS)},
    ).fetchall()
    others = source.execute(
        text(
            """
            SELECT m.id, m.content, m.created_at, m.memory_type, m.scope,
                   m.user_id, m.project_id, m.embedding, m.session_id,
                   m.org_id, m.agent_id
            FROM session_memories m
            LEFT JOIN projects p ON p.id = m.project_id
            WHERE m.org_id = :org_id AND m.status = 'active'
              AND m.recall_hidden = FALSE AND m.embedding IS NOT NULL
              AND (p.display_name IS NULL OR NOT (p.display_name = ANY(:names)))
            ORDER BY md5(m.id::text)
            LIMIT :n
            """
        ),
        {"org_id": org_id, "names": list(SEED_PROJECTS), "n": SEED_OTHER_ROWS},
    ).fetchall()
    return list(named) + list(others)


def _copy_projects(source: Any, target: Any, org_id: str) -> None:
    rows = source.execute(
        text(
            "SELECT id, org_id, display_name, git_remote, repo_root_path "
            "FROM projects WHERE org_id = :org_id"
        ),
        {"org_id": org_id},
    ).fetchall()
    for row in rows:
        target.execute(
            text(
                """
                INSERT INTO projects (id, org_id, display_name, git_remote,
                                      repo_root_path)
                VALUES (:id, :org_id, :name, :remote, :path)
                ON CONFLICT (id) DO NOTHING
                """
            ),
            {
                "id": str(row[0]),
                "org_id": row[1],
                "name": row[2],
                "remote": row[3],
                "path": row[4],
            },
        )
    target.commit()
    logger.info("copied %d projects", len(rows))


async def seed(source_url: str, target_url: str, org_id: str) -> None:
    source_engine = create_engine(source_url, pool_pre_ping=True)
    target_engine = create_engine(target_url, pool_pre_ping=True)

    with source_engine.connect() as source, target_engine.connect() as target:
        # Belt and braces: the source is production. Nothing here writes to it,
        # and the transaction refuses to even if something tried.
        source.execute(text("SET TRANSACTION READ ONLY"))
        _copy_projects(source, target, org_id)
        rows = _seed_rows(source, org_id)
        logger.info("copied %d memories; embedding them into v2 now", len(rows))

        for row in rows:
            target.execute(
                text(
                    """
                    INSERT INTO session_memories (
                        id, session_id, org_id, agent_id, memory_type, content,
                        embedding, scope, status, created_at, updated_at,
                        recall_hidden, user_id, project_id
                    ) VALUES (
                        :id, :session_id, :org_id, :agent_id, :memory_type,
                        :content, :embedding, :scope, 'active', :created_at,
                        :created_at, FALSE, :user_id, :project_id
                    )
                    ON CONFLICT (id) DO NOTHING
                    """
                ),
                {
                    "id": str(row.id),
                    "session_id": row.session_id,
                    "org_id": row.org_id,
                    "agent_id": row.agent_id,
                    "memory_type": row.memory_type,
                    "content": row.content,
                    # Already the pgvector text form (``[0.1,0.2,...]``) —
                    # this connection registers no pgvector adapter, so the
                    # driver hands back the literal, which is exactly what the
                    # target's bind parameter wants. Parsing it into floats and
                    # re-rendering would be a lossy round trip for no reason.
                    "embedding": row.embedding,
                    "scope": row.scope,
                    "created_at": row.created_at,
                    "user_id": str(row.user_id) if row.user_id else None,
                    "project_id": str(row.project_id) if row.project_id else None,
                },
            )
        target.commit()

    logger.info("run the v2 backfill against the target to finish seeding:")
    logger.info(
        "  DATABASE_URL=<target> python -m scripts.run_embedding_v2_backfill --apply"
    )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _project_id(db: Any, org_id: str, name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    row = db.execute(
        text("SELECT id FROM projects WHERE org_id = :org_id AND display_name = :name"),
        {"org_id": org_id, "name": name},
    ).first()
    return str(row[0]) if row else None


def _project_names(db: Any) -> dict[str, str]:
    return {
        str(row[0]): row[1]
        for row in db.execute(text("SELECT id, display_name FROM projects")).fetchall()
    }


async def _baseline(db: Any, org_id: str, query: str) -> list[dict[str, Any]]:
    """The pre-fix path: bare query, old blend, no project, no floor."""
    vector = await embed_text(query, role=embedding_prompt.ROLE_PASSAGE)
    if vector is None:
        return []
    rows = db.execute(
        text(_BASELINE_SQL),
        {
            "org_id": org_id,
            "embedding": str(vector),
            "candidate_pool": LIMIT * 4,
            "limit": LIMIT,
        },
    ).fetchall()
    return [
        {
            "id": str(row.id),
            "relevance": round(float(row.relevance), 4),
            "project_id": str(row.project_id) if row.project_id else None,
            "content": row.content,
        }
        for row in rows
    ]


async def _current(
    db: Any, org_id: str, query: str, project_id: Optional[str]
) -> list[dict[str, Any]]:
    records = await retrieve_memories_scoped(
        db, org_id, query, project_id=project_id, limit=LIMIT
    )
    return [
        {
            "id": str(record.id),
            "relevance": record.relevance,
            "project_id": str(record.project_id) if record.project_id else None,
            "content": record.content,
        }
        for record in records
    ]


def _contamination(rows: list[dict[str, Any]], project_id: Optional[str]) -> int:
    """Rows filed under some OTHER project. Unfiled rows are not contamination."""
    return sum(
        1
        for row in rows
        if row["project_id"] and project_id and row["project_id"] != project_id
    )


async def _separation(
    db: Any, org_id: str, query: str, project_id: Optional[str], template: str
) -> dict[str, Optional[float]]:
    """Corpus-wide on-project vs off-project cosine, under one query template.

    This is the number the whole embedding argument turns on, and it is
    measured over every row rather than the returned page: fencing changes
    which rows are RETURNED, it cannot change a similarity, so this isolates
    what the model can and cannot separate on its own.
    """
    os.environ[embedding_prompt.TEMPLATE_ENV] = template
    vector = await embed_text(query, role=embedding_prompt.ROLE_QUERY)
    if vector is None or project_id is None:
        return {}
    row = db.execute(
        text(_SEPARATION_SQL),
        {"org_id": org_id, "embedding": str(vector), "project_id": project_id},
    ).first()
    if row is None or row[0] is None or row[1] is None:
        return {}
    return {
        "best_on": float(row[0]),
        "best_off": float(row[1]),
        "margin": float(row[0]) - float(row[1]),
        "band": float(row[2]) - float(row[3]),
    }


def _print_rows(
    label: str,
    rows: list[dict[str, Any]],
    names: dict[str, str],
    project_id: Optional[str],
    markdown: bool,
) -> None:
    if not rows:
        print(f"    {label}: (empty)")
        return
    print(f"    {label}:")
    for index, row in enumerate(rows, start=1):
        project = names.get(row["project_id"] or "", "(unfiled)")
        flag = (
            "  <-- OFF-PROJECT"
            if row["project_id"] and project_id and row["project_id"] != project_id
            else ""
        )
        score = "  n/a" if row["relevance"] is None else f"{row['relevance']:.3f}"
        content = " ".join((row["content"] or "").split())[:78]
        print(f"      {index:2d}. {score}  {project[:22]:24s} {content}{flag}")


async def run(url: str, markdown: bool) -> int:
    """Report both configurations for every query. Returns a process exit code."""
    spec = _load_queries()
    org_id = spec["org_id"]
    engine = create_engine(url, pool_pre_ping=True)
    failures = 0

    with engine.connect() as db:
        names = _project_names(db)
        summary: list[dict[str, Any]] = []

        for entry in spec["queries"]:
            query = entry["text"]
            project_id = _project_id(db, org_id, entry.get("project"))
            print(f"\n=== {query!r}  [project={entry.get('project')}]")

            os.environ[embedding_prompt.TEMPLATE_ENV] = embedding_prompt.TEMPLATE_BARE
            before = await _baseline(db, org_id, query)
            sep_bare = await _separation(
                db, org_id, query, project_id, embedding_prompt.TEMPLATE_BARE
            )
            sep_instruct = await _separation(
                db, org_id, query, project_id, embedding_prompt.TEMPLATE_INSTRUCT
            )

            # The shipped default is the bare template; the instruct template
            # is reported alongside so the flip is a decision with numbers
            # rather than a hunch.
            os.environ[embedding_prompt.TEMPLATE_ENV] = embedding_prompt.TEMPLATE_BARE
            after_unfenced = await _current(db, org_id, query, None)
            after = await _current(db, org_id, query, project_id)

            _print_rows("before", before, names, project_id, markdown)
            _print_rows("after-unfenced", after_unfenced, names, project_id, markdown)
            _print_rows("after", after, names, project_id, markdown)

            row = {
                "query": query,
                "project": entry.get("project"),
                "expect_empty": bool(entry.get("expect_empty")),
                "before_contam": _contamination(before, project_id),
                "before_n": len(before),
                "unfenced_contam": _contamination(after_unfenced, project_id),
                "unfenced_n": len(after_unfenced),
                "after_contam": _contamination(after, project_id),
                "after_n": len(after),
                "margin_before": sep_bare.get("margin"),
                "margin_after": sep_instruct.get("margin"),
                "band_before": sep_bare.get("band"),
                "band_after": sep_instruct.get("band"),
            }
            summary.append(row)

            # Off-project rows surviving the fence is a REGRESSION and fails
            # the run: it is the reported bug, and the fence is what fixes it.
            if row["after_contam"]:
                failures += 1
                print("    !! off-project rows survived the fence")
            # The control query is reported, not enforced. It is a standing
            # measurement rather than a bar, because the measurement says the
            # bar cannot currently be met: a query about a subject the corpus
            # contains nothing on scores 0.78 against its best row, and real
            # queries score 0.78-0.87. The distributions overlap, so no
            # absolute floor empties the control without also emptying real
            # recalls. Failing the run over it would mean a permanently red
            # evaluation that everyone learns to ignore. Watch this number:
            # the day it separates is the day an absolute floor becomes
            # settable, and that day arrives by fixing the CORPUS (99.5%
            # auto-mined session exhaust), not the ranking.
            if entry.get("expect_empty"):
                best = after[0]["relevance"] if after else None
                print(
                    "    ~~ control query: "
                    + (f"{len(after)} rows, best {best}" if after else "empty")
                    + " (reported, not enforced — see the summary)"
                )

        _print_summary(summary)
    return 1 if failures else 0


def _print_summary(summary: list[dict[str, Any]]) -> None:
    print("\n\n=== summary ===\n")
    header = (
        f"{'query':46s} {'contam before':>13s} {'unfenced':>9s} {'after':>6s} "
        f"{'margin bare':>13s} {'margin instr':>12s}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        margin_before = (
            f"{row['margin_before']:+.4f}"
            if row["margin_before"] is not None
            else "  n/a"
        )
        margin_after = (
            f"{row['margin_after']:+.4f}"
            if row["margin_after"] is not None
            else "  n/a"
        )
        print(
            f"{row['query'][:46]:46s} "
            f"{row['before_contam']:>5d}/{row['before_n']:<7d} "
            f"{row['unfenced_contam']:>3d}/{row['unfenced_n']:<5d} "
            f"{row['after_contam']:>2d}/{row['after_n']:<3d} "
            f"{margin_before:>13s} {margin_after:>12s}"
        )

    def _mean(key: str) -> Optional[float]:
        values = [r[key] for r in summary if r[key] is not None]
        return statistics.fmean(values) if values else None

    total_before = sum(r["before_contam"] for r in summary)
    total_rows_before = sum(r["before_n"] for r in summary)
    total_unfenced = sum(r["unfenced_contam"] for r in summary)
    total_rows_unfenced = sum(r["unfenced_n"] for r in summary)
    total_after = sum(r["after_contam"] for r in summary)
    total_rows_after = sum(r["after_n"] for r in summary)

    print(
        f"\noff-project rows:  before {total_before}/{total_rows_before}"
        f"   after-unfenced {total_unfenced}/{total_rows_unfenced}"
        f"   after {total_after}/{total_rows_after}"
    )
    mb, ma = _mean("margin_before"), _mean("margin_after")
    bb, ba = _mean("band_before"), _mean("band_after")
    if mb is not None and ma is not None:
        print(
            f"mean on-vs-off cosine margin:  bare {mb:+.4f}   instruct {ma:+.4f}"
            "   (negative = the best row in the corpus is off-project)"
        )
    if bb is not None and ba is not None:
        print(f"mean corpus band (max-median): bare {bb:.4f}    instruct {ba:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    seed_parser = sub.add_parser("seed", help="copy a corpus sample into a target DB")
    seed_parser.add_argument("--source-url", required=True)
    seed_parser.add_argument("--target-url", required=True)
    seed_parser.add_argument("--org", default="vex")

    run_parser = sub.add_parser("run", help="report before/after for every query")
    run_parser.add_argument("--url", default=os.environ.get("DATABASE_URL"))
    run_parser.add_argument("--markdown", action="store_true")

    args = parser.parse_args()
    if args.command == "seed":
        asyncio.run(seed(args.source_url, args.target_url, args.org))
        return
    if not args.url:
        logger.error("--url or DATABASE_URL is required")
        sys.exit(1)
    sys.exit(asyncio.run(run(args.url, args.markdown)))


if __name__ == "__main__":
    main()
