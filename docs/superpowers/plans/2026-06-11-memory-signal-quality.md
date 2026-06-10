# Memory Signal Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved spec at `docs/superpowers/specs/2026-06-11-memory-signal-quality-design.md`: stop the curator from manufacturing tool-call "facts" (exclude `recall_hidden` rows), truncate embedding input so oversized content can't loop the embedder, and provide a reversible all-org script that retracts the existing junk-fact backlog.

**Architecture:** Three independent changes. (1) One WHERE-clause addition in the curator's raw-row read. (2) One head-truncation guard at the top of `embed_text` — a single chokepoint every caller already routes through. (3) A new keyset-paginated cleanup script modeled byte-for-byte on `run_redaction_backfill.py`, retracting `fact`/`summary` rows whose content matches a narrow tool-call noise-pattern set.

**Tech Stack:** Python 3.11, SQLAlchemy raw SQL via `text()`, pytest. Engine repo `vex_private/vex_engine`. Test commands: shared suite `cd services/shared && python3 -m pytest tests/ -q`; mcp-server suite from repo root `.venv/bin/python -m pytest services/mcp-server/tests -q`. Engine deploys via `git push origin main` → Railway auto-build.

**Commit rule:** conventional commits, no co-author lines.

---

## File structure

| File | Responsibility |
|---|---|
| Modify `services/mcp-server/app/curator.py` | Add `recall_hidden = FALSE` to `_read_new_raw_rows` |
| Modify `services/shared/shared/memory.py` | `MAX_EMBED_CHARS` constant + head-truncate in `embed_text` |
| Create `services/shared/shared/fact_noise.py` | Pure noise-pattern matcher (`is_tool_call_noise`) shared by the cleanup script + tests |
| Create `services/scripts/run_fact_cleanup.py` | All-org reversible noise-retraction sweep (dry-run default) |
| Create `services/shared/tests/test_fact_noise.py` | True-positive + false-positive corpus for the matcher |
| Create `services/shared/tests/test_fact_cleanup.py` | Cleanup batch logic (dry-run / apply / idempotent) |
| Create `services/shared/tests/test_embed_truncation.py` | Embed-input truncation unit tests |
| Modify `services/mcp-server/tests/test_curator.py` | Assert curator read SQL excludes hidden rows |

---

### Task 1: Embedding-input truncation

**Files:**
- Modify: `services/shared/shared/memory.py` (`embed_text`, lines 86-104 region; constants block near line 27-37)
- Create: `services/shared/tests/test_embed_truncation.py`

- [ ] **Step 1: Write the failing tests** — create `services/shared/tests/test_embed_truncation.py`:

```python
"""embed_text must truncate input below the model's token limit so oversized
content cannot return HTTP 400 and loop the embedder."""

import shared.memory as memory_mod
from shared.memory import MAX_EMBED_CHARS


async def _run_embed(monkeypatch, text_input):
    captured = {}

    async def _fake_aembedding(**kwargs):
        captured["input"] = kwargs["input"]

        class _Item(dict):
            pass

        return type("R", (), {"data": [{"embedding": [0.1, 0.2]}]})()

    monkeypatch.setattr(memory_mod, "aembedding", _fake_aembedding)
    result = await memory_mod.embed_text(text_input)
    return result, captured


async def test_oversized_input_is_truncated(monkeypatch):
    big = "x" * (MAX_EMBED_CHARS + 5000)
    result, captured = await _run_embed(monkeypatch, big)
    assert result == [0.1, 0.2]
    sent = captured["input"][0]
    assert len(sent) == MAX_EMBED_CHARS


async def test_short_input_passes_through_unchanged(monkeypatch):
    small = "a durable fact about the user"
    result, captured = await _run_embed(monkeypatch, small)
    assert captured["input"][0] == small


async def test_max_embed_chars_is_conservative():
    # ~4 chars/token, model limit 512 tokens; stay well under.
    assert MAX_EMBED_CHARS <= 2000
```

(Match the suite's async convention — `asyncio_mode=auto`, plain `async def` collects. If `_fake_aembedding`'s return shape doesn't satisfy `embed_text`'s response parsing, mirror the real shape: `embed_text` reads `response.data[0]` then `item["embedding"]` if dict else `item.embedding`. The fixture above returns `data=[{"embedding": [...]}]`, which hits the dict branch.)

- [ ] **Step 2: Run to verify failure**

Run: `cd services/shared && python3 -m pytest tests/test_embed_truncation.py -v`
Expected: FAIL — `ImportError: cannot import name 'MAX_EMBED_CHARS'`.

- [ ] **Step 3: Implement.** In `services/shared/shared/memory.py`, add the constant in the Constants block (after `DEFAULT_LLM_TIMEOUT_S`, ~line 37):

```python
MAX_EMBED_CHARS: int = 1800
"""Head-truncation budget for embedding input.

The default embedding model caps input at 512 tokens; at ~4 chars/token a
1800-char head stays safely under that (≈450 tokens) even for multibyte text.
Truncation applies ONLY to the vector input — stored ``content`` is never
modified. This prevents oversized content from returning HTTP 400 and looping
the embedder (which previously flooded logs at the host's rate limit)."""
```

Then in `embed_text`, truncate before building `kwargs` — change the body start (after the docstring, before `model = ...`):

```python
    if text_input and len(text_input) > MAX_EMBED_CHARS:
        logger.debug(
            "embed_text: truncating input from %d to %d chars",
            len(text_input),
            MAX_EMBED_CHARS,
        )
        text_input = text_input[:MAX_EMBED_CHARS]

    model = os.environ.get("MEMORY_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
```

(The module already has `logger = logging.getLogger("agentguard.shared.memory")`.)

- [ ] **Step 4: Run** `cd services/shared && python3 -m pytest tests/test_embed_truncation.py tests/test_memory.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add services/shared/shared/memory.py services/shared/tests/test_embed_truncation.py
git commit -m "fix(memory): truncate embedding input below model token limit"
```

---

### Task 2: Curator excludes recall_hidden rows

**Files:**
- Modify: `services/mcp-server/app/curator.py` (`_read_new_raw_rows`, lines 254-279)
- Modify: `services/mcp-server/tests/test_curator.py` (append)

- [ ] **Step 1: Write the failing test** — append to `services/mcp-server/tests/test_curator.py`:

```python
def test_read_new_raw_rows_excludes_recall_hidden():
    """The curator must not ingest passive hook tool-call observations
    (recall_hidden=TRUE); synthesizing those into 'Tool X was used' facts is
    the noise leak. It synthesizes only from deliberate writes."""
    import app.curator as curator_mod
    from app.curator import CursorState

    captured = {}

    class _DB:
        def execute(self, sql, params=None):
            captured["sql"] = str(sql)

            class R:
                def fetchall(self_inner):
                    return []

            return R()

    curator_mod._read_new_raw_rows(_DB(), "org-1", CursorState(None, None), 50)
    assert "recall_hidden = FALSE" in captured["sql"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest services/mcp-server/tests/test_curator.py -k recall_hidden -v` (from repo root)
Expected: FAIL — `recall_hidden = FALSE` not in SQL.

- [ ] **Step 3: Implement.** In `services/mcp-server/app/curator.py` `_read_new_raw_rows`, add the clause to the WHERE (after `AND status = 'active'`, line 261):

```python
            WHERE org_id = :org_id
              AND scope = 'org'
              AND status = 'active'
              AND recall_hidden = FALSE
              AND memory_type IN ({_RAW_TYPES_SQL})
```

Update the function docstring's scoping sentence to note: "and `recall_hidden=FALSE`, so passive hook tool-call captures (curator fuel that is recall-hidden) are NOT distilled into facts — the curator synthesizes only deliberate writes."

- [ ] **Step 4: Run** the full mcp-server suite: `.venv/bin/python -m pytest services/mcp-server/tests -q` → PASS (existing curator tests unaffected; seeded-row tests still pass because their fixtures don't set recall_hidden, defaulting to FALSE/visible — verify, and if any curator test seeds rows via raw INSERT without the column, it still works since the column defaults FALSE).

- [ ] **Step 5: Commit**

```bash
git add services/mcp-server/app/curator.py services/mcp-server/tests/test_curator.py
git commit -m "fix(curator): exclude recall-hidden rows so tool-call noise is never synthesized"
```

---

### Task 3: Noise-pattern matcher

**Files:**
- Create: `services/shared/shared/fact_noise.py`
- Create: `services/shared/tests/test_fact_noise.py`

- [ ] **Step 1: Write the failing tests** — create `services/shared/tests/test_fact_noise.py`:

```python
"""is_tool_call_noise: matches tool-call/build-artifact shapes, NEVER durable facts."""

import pytest

from shared.fact_noise import is_tool_call_noise

# Real noise observed in the production brain — must be retracted.
NOISE = [
    "Used tool Bash\ninput: {\"command\":\"git show 96519d9\"}",
    "Tool Bash was used with command: grep -n memory file.json",
    "Tool Read was used on file /Users/x/y/env.py",
    "Agent executed Bash command to find JSON files in the sdk directory.",
    'The agent used the Bash tool with command "npx oxlint packages/...".',
    "Used tool mcp__plugin_stripe_stripe__get_stripe_account_info\ninput: {}",
    "response: {\"stdout\":\"\",\"stderr\":\"\",\"interrupted\":false}",
]

# Durable, useful facts — must SURVIVE (false-positive guards).
LEGIT = [
    "The team decided to use Stripe for billing.",
    "The user prefers production-ready changes only, no quick fixes.",
    "Bash scripts for deployment live in services/scripts.",
    "AgentGuard was renamed to Klio.tech in June 2026.",
    "The user's name is Abhishek and they work on Klio.",
    "The engine validates Supabase tokens via JWKS.",
    "MoonForge uses a Stripe metered billing meter named moonforge_events.",
]


@pytest.mark.parametrize("content", NOISE)
def test_matches_tool_call_noise(content):
    assert is_tool_call_noise(content) is True


@pytest.mark.parametrize("content", LEGIT)
def test_spares_durable_facts(content):
    assert is_tool_call_noise(content) is False


def test_empty_and_none_safe():
    assert is_tool_call_noise("") is False
    assert is_tool_call_noise(None) is False
```

- [ ] **Step 2: Run to verify failure**

Run: `cd services/shared && python3 -m pytest tests/test_fact_noise.py -v`
Expected: FAIL — `ModuleNotFoundError: shared.fact_noise`.

- [ ] **Step 3: Implement** `services/shared/shared/fact_noise.py`:

```python
"""Detect tool-call / build-artifact memories that should not be durable facts.

Used by the one-off cleanup script (:mod:`scripts.run_fact_cleanup`) to retract
the backlog of low-signal ``fact``/``summary`` rows the curator synthesized
from passive hook captures (e.g. "Tool Bash was used with command: …").

Deliberately CONSERVATIVE: every pattern describes a tool INVOCATION or a raw
tool I/O dump, never a durable statement. False positives (retracting a real
fact) are worse than false negatives (leaving a borderline one), so the
patterns match only unambiguous tool-call shapes. The retraction is reversible
(status='retracted'), and the cleanup script's dry-run lets a human review
hits before applying.
"""

import re

# Each pattern targets an unambiguous tool-call / raw-IO shape. Case-insensitive.
_NOISE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^\s*used tool\b", re.IGNORECASE),
    re.compile(r"\btool\s+[\w./-]+\s+was used\b", re.IGNORECASE),
    re.compile(r"\bwas used with command\b", re.IGNORECASE),
    re.compile(r"\bused the bash tool\b", re.IGNORECASE),
    re.compile(r"\bexecuted (?:a |the )?bash (?:command|tool)\b", re.IGNORECASE),
    re.compile(r"\bexecuted bash command\b", re.IGNORECASE),
    re.compile(r"\btool\s+\w+\s+was used (?:on|to)\b", re.IGNORECASE),
    # Raw tool I/O dumps captured verbatim.
    re.compile(r'input:\s*\{"command"', re.IGNORECASE),
    re.compile(r'response:\s*\{"stdout"', re.IGNORECASE),
    re.compile(r"\bused tool mcp__", re.IGNORECASE),
)


def is_tool_call_noise(content: str | None) -> bool:
    """Return True if ``content`` is a tool-call / build-artifact memory.

    Pure and side-effect-free. ``None``/empty is never noise (returns False).
    """
    if not content:
        return False
    return any(p.search(content) for p in _NOISE_PATTERNS)
```

- [ ] **Step 4: Run** until green. If a LEGIT case matches (false positive), TIGHTEN the offending pattern — never loosen a LEGIT assertion. (Watch the `executed ... bash` and `tool X was used on/to` patterns against "Bash scripts ... live in" — confirm they don't fire; the corpus covers this.)

- [ ] **Step 5: Commit**

```bash
git add services/shared/shared/fact_noise.py services/shared/tests/test_fact_noise.py
git commit -m "feat(fact-noise): conservative tool-call noise matcher for fact cleanup"
```

---

### Task 4: Cleanup batch logic + script

**Files:**
- Create: `services/scripts/run_fact_cleanup.py`
- Create: `services/shared/tests/test_fact_cleanup.py`

- [ ] **Step 1: Write the failing tests** — create `services/shared/tests/test_fact_cleanup.py`:

```python
"""run_fact_cleanup: dry-run reports, apply retracts + stamps, idempotent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from run_fact_cleanup import process_fact_batch  # noqa: E402


class _Row:
    def __init__(self, id, org_id, content):
        self.id = id
        self.org_id = org_id
        self.content = content


class FakeDB:
    def __init__(self):
        self.updates = []

    def execute(self, sql, params=None):
        self.updates.append((str(sql), params))

        class R:
            def fetchall(self_inner):
                return []

        return R()

    def commit(self):
        pass


def test_dry_run_reports_but_writes_nothing():
    db = FakeDB()
    rows = [
        _Row("f1", "org-a", "Tool Bash was used with command: ls"),
        _Row("f2", "org-a", "The team decided to use Stripe."),
        _Row("f3", "org-a", "Used tool Read on file x.py"),
    ]
    report = process_fact_batch(db, rows, apply=False)
    assert report == {"org-a": 2}
    assert db.updates == []


def test_apply_retracts_noise_only():
    db = FakeDB()
    rows = [
        _Row("f1", "org-a", "Tool Bash was used with command: ls"),
        _Row("f2", "org-a", "The user prefers production-ready changes."),
    ]
    report = process_fact_batch(db, rows, apply=True)
    assert report == {"org-a": 1}
    assert len(db.updates) == 1
    sql, params = db.updates[0]
    assert "status = 'retracted'" in sql
    assert params["id"] == "f1"


def test_idempotent_second_pass_noop_on_clean_rows():
    db = FakeDB()
    rows = [_Row("f2", "org-a", "A durable fact.")]
    report = process_fact_batch(db, rows, apply=True)
    assert report == {}
    assert db.updates == []
```

- [ ] **Step 2: Run to verify failure**

Run: `cd services/shared && python3 -m pytest tests/test_fact_cleanup.py -v`
Expected: FAIL — `ModuleNotFoundError: run_fact_cleanup`.

- [ ] **Step 3: Implement** `services/scripts/run_fact_cleanup.py` (mirror `run_redaction_backfill.py`'s structure — read it first for the engine/keyset idiom):

```python
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
                    WHERE {' AND '.join(where)}
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
```

- [ ] **Step 4: Run** `cd services/shared && python3 -m pytest tests/test_fact_cleanup.py -v` → PASS, then the full shared suite → green.

- [ ] **Step 5: Commit**

```bash
git add services/scripts/run_fact_cleanup.py services/shared/tests/test_fact_cleanup.py
git commit -m "feat(scripts): reversible all-org tool-call-noise fact cleanup (dry-run default)"
```

---

### Task 5: Verification + rollout

**Files:** none — verification only.

- [ ] **Step 1: Both suites** — `cd services/shared && python3 -m pytest tests/ -q` and `.venv/bin/python -m pytest services/mcp-server/tests -q` → ALL PASS.
- [ ] **Step 2: Coverage** on new/changed code — `cd services/shared && python3 -m pytest tests/ --cov=shared.fact_noise --cov=shared.memory --cov-report=term-missing`; target ≥80% on `fact_noise.py` and the new `embed_text` branch. Add targeted tests for any uncovered branch.
- [ ] **Step 3: Lint** — `ruff check services/shared/shared/fact_noise.py services/shared/shared/memory.py services/mcp-server/app/curator.py services/scripts/run_fact_cleanup.py` and `ruff format --check services/` → clean.
- [ ] **Step 4: Bind-cast SQL scan** — `cd services/shared && python3 -m pytest tests/test_memory_sql.py -q` → green (covers the modules; if `run_fact_cleanup.py`/`fact_noise.py` should be in the scan list, add them — they use `CAST(... AS jsonb)`, not `::`, so they pass).
- [ ] **Step 5: Deploy + run cleanup (production rollout, in order):**
  1. `git push origin main` → Railway rebuilds. Tasks 1+2 stop new noise and end the embedding loop/log-flood immediately.
  2. `DATABASE_URL=<prod> python -m scripts.run_fact_cleanup` (dry-run) → review the per-org counts + sample matches → confirm no legit facts in the samples → `--apply`.
  3. From the connected Claude session, run `recall` for "what do you know about me" and confirm the result is durable signal (preferences, decisions, project facts), not build logs.
