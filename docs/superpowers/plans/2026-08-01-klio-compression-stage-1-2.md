# Klio Compression — Stages 1 & 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the structural (lossless) compressor and wire it to `recall`, then add the originals cache and `klio_retrieve` so lossy compressors have a recovery contract to build on.

**Architecture:** A new `shared/compression/` package with a router and pluggable compressors. Stage 1 adds only the structural compressor — pure Python, no model, no network, provably lossless. Stage 2 adds `compression_cache` (migration `034`) and a `klio_retrieve` MCP tool siloed by the same ladder as memory.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy Core (`text()` + bound params), Alembic, pytest, ruff.

## Global Constraints

- **Structural compression is LOSSLESS.** Verbatim from the spec: *"Every byte removed is provably redundant."* Any change that could alter meaning belongs in a later stage.
- **Nothing technical is ever altered.** Verbatim: *"A compressor that mangles a stack trace is worse than no compressor."* Code blocks, inline code, URLs, paths, `CONST_CASE`, `dotted.method()`, function calls, version numbers, error strings and numbers are protected.
- **Fail-open, always.** Any compressor error returns the input unchanged. Compression failing must never fail a request.
- **`recall` output is a public API.** Existing callers must not break. Additive fields only; no field removed without a compatibility path.
- **The cache is siloed exactly like memory.** `org_id` + `user_id` as SQL predicates. Another user's handle resolves to `not_found`, never `forbidden`.
- **The cache is NOT a memory.** Separate table. Never appears in `recall`, never curated, never counts against storage quota.
- SQL uses bound parameters via `text()`. Never f-strings.
- `ruff format` and `ruff check` pass before every commit.
- Migration head is `033`. This adds `034`.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `services/shared/shared/compression/__init__.py` | Public API: `compress()`, `CompressionResult` | Create |
| `services/shared/shared/compression/protect.py` | Protected-token substitution and restore | Create |
| `services/shared/shared/compression/structural.py` | The lossless JSON/payload compressor | Create |
| `services/shared/shared/compression/router.py` | Content-type detection, compressor selection | Create |
| `services/shared/tests/test_compression_protect.py` | Protection round-trip tests | Create |
| `services/shared/tests/test_compression_structural.py` | Losslessness + savings tests | Create |
| `services/migrations/alembic/versions/034_compression_cache.py` | Stage 2 schema | Create |
| `services/migrations/tests/test_migration_034.py` | Schema assertions | Create |
| `services/mcp-server/app/compression_cache.py` | Cache read/write, siloed | Create |
| `services/mcp-server/tests/test_compression_cache_isolation.py` | Adversarial siloing tests | Create |

`compression/` is a package rather than one module because the protection layer,
the compressors and the router are independently testable and will each grow.
`protect.py` in particular is the correctness-critical piece and deserves its own
test file.

---

## STAGE 1 — Structural compressor

### Task 1: Protected-token substitution

**Files:**
- Create: `services/shared/shared/compression/protect.py`
- Test: `services/shared/tests/test_compression_protect.py`

**Interfaces:**
- Produces: `protect(text: str) -> tuple[str, list[str]]` and `restore(text: str, segments: list[str]) -> str`

- [ ] **Step 1: Write the failing test**

```python
"""Protection is the correctness floor of the whole compression system.

If a compressor mangles a stack trace, a file path or a function name, it is
worse than no compressor: the model reads something that looks right and is
not. Every compressor runs inside protect/restore, so these tests guard every
compressor that will ever exist.
"""

import pytest

from shared.compression.protect import protect, restore


def _roundtrip(text: str) -> str:
    protected, segments = protect(text)
    return restore(protected, segments)


@pytest.mark.parametrize(
    "text",
    [
        "See `foo.bar()` for details",
        "```python\ndef f(x):\n    return x\n```",
        "Fetch https://example.com/a/b?c=1 then retry",
        "Open /Users/dev/src/main.py line 42",
        "Set MAX_RETRY_COUNT to 5",
        "Call parse_config(path, strict=True) first",
        "Upgrade to 1.2.3 or later",
        "Error: ECONNREFUSED 127.0.0.1:5432",
    ],
)
def test_protected_content_survives_a_roundtrip_byte_for_byte(text):
    assert _roundtrip(text) == text


def test_protection_survives_an_aggressive_transform():
    """The real risk: a compressor mutates the protected region.

    protect() must hide the content well enough that a transform which
    uppercases and strips everything cannot touch it.
    """
    text = "Run `git rebase -i HEAD~3` in /tmp/repo"
    protected, segments = protect(text)
    mangled = protected.upper().replace("THE", "")
    out = restore(mangled, segments)
    assert "`git rebase -i HEAD~3`" in out
    assert "/tmp/repo" in out


def test_nested_patterns_restore_completely():
    """A path inside a code fence is matched by two patterns.

    A single restore pass leaves the inner sentinel unresolved; the restore
    must iterate until no sentinels remain.
    """
    text = "```\nopen('/etc/hosts')\n```"
    assert _roundtrip(text) == text
    protected, _ = protect(text)
    assert "/etc/hosts" not in protected


def test_text_with_nothing_to_protect_is_unchanged():
    assert _roundtrip("the quick brown fox") == "the quick brown fox"


def test_empty_input():
    assert _roundtrip("") == ""


def test_sentinel_lookalikes_in_source_text_are_not_confused():
    """Input containing something shaped like a sentinel must not corrupt."""
    text = "counts were 1 2 3 and then 4"
    assert _roundtrip(text) == text
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd services/shared && python -m pytest tests/test_compression_protect.py -v`
Expected: `ModuleNotFoundError: No module named 'shared.compression'`

- [ ] **Step 3: Implement**

Create `services/shared/shared/compression/__init__.py` (empty for now) and
`services/shared/shared/compression/protect.py`:

```python
"""Hide content no compressor may alter, then put it back.

Every compressor runs inside protect/restore. A compressor that rewrites a
file path or a function name produces output that looks correct and is not,
which is worse than no compression at all — the reader has no signal that
anything was lost.

Sentinels are a rare unicode private-use character plus an index, chosen so
that input text cannot collide with them. An earlier design using ` N ` (a
bare integer) collided with ordinary prose containing numbers.

Restore iterates: patterns nest (a path inside a code fence is matched twice),
so a single pass leaves inner sentinels unresolved. The pass count is bounded
by nesting depth, never by input size.
"""

from __future__ import annotations

import re

# U+E000 is in the Unicode private use area — it does not appear in source
# code, prose, logs or JSON in practice, so a sentinel cannot collide with
# real input the way a numeric marker can.
_SENTINEL_OPEN = ""
_SENTINEL_CLOSE = ""
_SENTINEL_RE = re.compile(f"{_SENTINEL_OPEN}(\\d+){_SENTINEL_CLOSE}")

MAX_RESTORE_PASSES = 8

PROTECTED_PATTERNS: list[re.Pattern] = [
    re.compile(r"```[\s\S]*?```"),                       # fenced code
    re.compile(r"`[^`\n]+`"),                            # inline code
    re.compile(r"\bhttps?://\S+", re.IGNORECASE),        # URLs
    re.compile(r"(?:^|(?<=\s))[\w.-]*[/\\][\w.\-/\\]+"), # filesystem paths
    re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"),    # CONST_CASE
    re.compile(r"\b\w+(?:\.\w+)+\(?\)?"),                # dotted.method()
    re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*\([^)\n]*\)"), # function calls
    re.compile(r"\b\d+\.\d+\.\d+\b"),                    # versions
    re.compile(r"\b\d[\d,.:]*\b"),                       # any number
]


def protect(text: str) -> tuple[str, list[str]]:
    """Replace every protected span with a sentinel.

    Returns the protected text and the ordered list of original spans.
    """
    segments: list[str] = []
    working = text
    for pattern in PROTECTED_PATTERNS:

        def _stash(match: re.Match) -> str:
            index = len(segments)
            segments.append(match.group(0))
            return f"{_SENTINEL_OPEN}{index}{_SENTINEL_CLOSE}"

        working = pattern.sub(_stash, working)
    return working, segments


def restore(text: str, segments: list[str]) -> str:
    """Splice the original spans back in.

    Iterates because protected patterns nest. Bounded by MAX_RESTORE_PASSES,
    which grows with nesting depth and not with input size.
    """
    out = text
    for _ in range(MAX_RESTORE_PASSES):
        if not _SENTINEL_RE.search(out):
            break
        out = _SENTINEL_RE.sub(
            lambda m: segments[int(m.group(1))]
            if int(m.group(1)) < len(segments)
            else m.group(0),
            out,
        )
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd services/shared && python -m pytest tests/test_compression_protect.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
ruff format services/ && ruff check services/
git add services/shared/shared/compression/ services/shared/tests/test_compression_protect.py
git commit -m "feat(compression): protect content no compressor may alter

Every compressor runs inside protect/restore. A compressor that rewrites a
file path or a function name produces output that looks correct and is not,
which is worse than no compression — the reader gets no signal.

Sentinels use a Unicode private-use character rather than a bare integer:
ordinary prose contains numbers, and a numeric marker collides with real
input. Restore iterates because patterns nest — a path inside a code fence
is matched twice, and one pass leaves the inner sentinel unresolved."
```

---

### Task 2: The structural compressor

**Files:**
- Create: `services/shared/shared/compression/structural.py`
- Test: `services/shared/tests/test_compression_structural.py`

**Interfaces:**
- Consumes: nothing from Task 1 (structural works on parsed JSON, not text)
- Produces: `compress_payload(payload: dict) -> tuple[dict, CompressionStats]`

- [ ] **Step 1: Write the failing test**

```python
"""The structural compressor is lossless by contract.

It removes only redundancy: nulls, values identical across every row, and
verbose representations of the same information. Every test here asserts the
information survives — a test that only asserted "it got smaller" would pass
against a compressor that simply deleted data.
"""

import json

from shared.compression.structural import compress_payload

RECALL = {
    "memories": [
        {
            "id": "80590c00-cb5f-45c8-b813-952054a10b11",
            "memory_type": "fact",
            "content": "Team uses pgvector for recall",
            "scope": "org",
            "confidence": None,
            "provenance": "EXTRACTED",
            "relevance": 0.866,
            "space_id": None,
            "project_id": "93e3f7df-788a-4de0-9c3f-654e1259ce00",
            "project": None,
            "created_at": "2026-07-27T16:02:38.526112+00:00",
            "metadata": {"source": "hook-transcript"},
        },
        {
            "id": "9151f0aa-1b2c-4d5e-8f90-112233445566",
            "memory_type": "fact",
            "content": "Threshold is 0.85",
            "scope": "org",
            "confidence": None,
            "provenance": "EXTRACTED",
            "relevance": 0.81,
            "space_id": None,
            "project_id": "93e3f7df-788a-4de0-9c3f-654e1259ce00",
            "project": None,
            "created_at": "2026-07-27T17:00:00.000000+00:00",
            "metadata": {"source": "hook-transcript"},
        },
    ],
    "count": 2,
}


def test_every_content_string_survives_verbatim():
    """The one thing that must never be lost."""
    out, _ = compress_payload(RECALL)
    blob = json.dumps(out)
    for m in RECALL["memories"]:
        assert m["content"] in blob


def test_every_id_survives_verbatim():
    """Ids are how an agent calls forget/share. Losing one breaks those."""
    out, _ = compress_payload(RECALL)
    blob = json.dumps(out)
    for m in RECALL["memories"]:
        assert m["id"] in blob


def test_null_fields_are_dropped():
    out, _ = compress_payload(RECALL)
    blob = json.dumps(out)
    assert '"confidence": null' not in blob
    assert '"space_id": null' not in blob


def test_values_identical_across_all_rows_are_hoisted_not_repeated():
    """scope, provenance and project_id are the same on every row here."""
    out, _ = compress_payload(RECALL)
    rows = out["memories"]
    assert all("scope" not in r for r in rows), "constant field left on rows"
    assert out.get("common", {}).get("scope") == "org"


def test_a_field_that_differs_stays_on_the_row():
    """Hoisting must only apply to genuinely constant values."""
    payload = json.loads(json.dumps(RECALL))
    payload["memories"][1]["scope"] = "private"
    out, _ = compress_payload(payload)
    rows = out["memories"]
    assert {r.get("scope") for r in rows} == {"org", "private"}
    assert "scope" not in out.get("common", {})


def test_it_actually_saves_something():
    before = len(json.dumps(RECALL))
    out, stats = compress_payload(RECALL)
    after = len(json.dumps(out))
    assert after < before
    assert stats.before_chars == before
    assert stats.after_chars == after


def test_single_row_payload_does_not_hoist_everything():
    """With one row every field is trivially 'constant'. Hoisting them all
    just moves bytes around and makes the shape harder to read."""
    payload = {"memories": [RECALL["memories"][0]], "count": 1}
    out, _ = compress_payload(payload)
    assert out["memories"][0]["content"] == RECALL["memories"][0]["content"]


def test_empty_payload_is_safe():
    out, stats = compress_payload({"memories": [], "count": 0})
    assert out["memories"] == []


def test_malformed_input_returns_unchanged():
    """Fail-open: compression must never raise into a request path."""
    weird = {"not": "a recall payload"}
    out, _ = compress_payload(weird)
    assert out == weird
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd services/shared && python -m pytest tests/test_compression_structural.py -v`
Expected: FAIL — no module `shared.compression.structural`.

- [ ] **Step 3: Implement**

Create `services/shared/shared/compression/structural.py`:

```python
"""Lossless structural compression for JSON payloads.

Removes only redundancy that can be reconstructed: fields that are null, and
fields whose value is identical on every row. Nothing that carries information
is touched, so there is no recovery contract and no cache entry — the caller
gets the same information in fewer bytes.

Measured motivation: a recall(limit=10) payload is 5,768 characters of which
1,029 is memory content. 83% is envelope.

Hoisting requires at least two rows. With a single row every field is trivially
constant, and lifting them all just relocates bytes while making the shape
harder for a model to read.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("klio.compression.structural")

MIN_ROWS_TO_HOIST = 2


@dataclass
class CompressionStats:
    before_chars: int
    after_chars: int

    @property
    def saved_ratio(self) -> float:
        if self.before_chars == 0:
            return 0.0
        return 1.0 - (self.after_chars / self.before_chars)


def compress_payload(payload: Any) -> tuple[Any, CompressionStats]:
    """Compress a recall-shaped payload losslessly.

    Fail-open: anything unexpected returns the input unchanged. Compression is
    an optimisation and must never raise into a request path.
    """
    import json

    before = len(json.dumps(payload, default=str))
    try:
        out = _compress(payload)
    except Exception:
        logger.warning("structural compression failed; passing through", exc_info=True)
        out = payload
    after = len(json.dumps(out, default=str))
    return out, CompressionStats(before_chars=before, after_chars=after)


def _compress(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    rows = payload.get("memories")
    if not isinstance(rows, list) or not rows:
        return payload
    if not all(isinstance(r, dict) for r in rows):
        return payload

    stripped = [{k: v for k, v in row.items() if v is not None} for row in rows]

    common: dict[str, Any] = {}
    if len(stripped) >= MIN_ROWS_TO_HOIST:
        candidate_keys = set(stripped[0])
        for row in stripped[1:]:
            candidate_keys &= set(row)
        for key in sorted(candidate_keys):
            if key in {"id", "content"}:
                # Never hoist identity or payload, even if two rows collide.
                continue
            values = {_hashable(row[key]) for row in stripped}
            if len(values) == 1:
                common[key] = stripped[0][key]
        if common:
            stripped = [
                {k: v for k, v in row.items() if k not in common} for row in stripped
            ]

    out = dict(payload)
    out["memories"] = stripped
    if common:
        out["common"] = common
    return out


def _hashable(value: Any) -> Any:
    """Make a value comparable for the constant check."""
    if isinstance(value, (dict, list)):
        import json

        return json.dumps(value, sort_keys=True, default=str)
    return value
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd services/shared && python -m pytest tests/test_compression_structural.py -v`
Expected: all PASS.

- [ ] **Step 5: Measure the real saving**

```bash
cd services/shared && python -c "
import json
from shared.compression.structural import compress_payload
payload = json.load(open('/tmp/klio-recall-sample.json'))
out, stats = compress_payload(payload)
print(f'{stats.before_chars} -> {stats.after_chars} ({stats.saved_ratio:.1%} saved)')
"
```
If `/tmp/klio-recall-sample.json` is absent, build one from the test fixture.
Record the number in the commit message — a compression change with no measured
saving is not finished.

- [ ] **Step 6: Commit**

```bash
ruff format services/ && ruff check services/
git add services/shared/shared/compression/structural.py services/shared/tests/test_compression_structural.py
git commit -m "feat(compression): lossless structural compression for payloads

Drops nulls and hoists values identical across every row. Nothing carrying
information is touched, so there is no recovery contract and no cache entry.

Motivation is measured: recall(limit=10) is 5,768 characters delivering
1,029 characters of memory — 83% envelope.

id and content are never hoisted even if two rows collide: ids are how an
agent calls forget/share, and content is the entire point. Hoisting needs
two rows — with one, every field is trivially constant and lifting them all
just relocates bytes."
```

---

### Task 3: Wire it to `recall`

**Files:**
- Modify: `services/mcp-server/app/mcp_app.py` (the `recall` response assembly)
- Test: `services/mcp-server/tests/test_recall_compression.py`

**Interfaces:**
- Consumes: `compress_payload` from Task 2

- [ ] **Step 1: Write the failing test**

```python
"""recall returns compressed payloads, without breaking its contract.

recall's output is a public API. The compression is additive: information is
preserved, the shape gains an optional `common` block, and every existing
consumer can still find what it needs.
"""

import json

import pytest


@pytest.mark.asyncio
async def test_recall_payload_is_compressed(storage):
    """The envelope shrinks."""
    # Uses the existing recall test fixtures in this file's neighbours.
    ...


@pytest.mark.asyncio
async def test_recall_still_returns_every_content_and_id(storage):
    """Compression must not cost the caller any information."""
    ...
```

**Note for the implementer:** this file's fixtures depend on the existing
`test_recall_tool.py` setup in this repo. Read that file first and mirror its
fixture usage rather than inventing a new harness. The two assertions above are
the contract; express them in whatever fixture style that file already uses.

- [ ] **Step 2: Run to verify it fails**

Run: `cd services/mcp-server && python -m pytest tests/test_recall_compression.py -v`
Expected: FAIL — recall returns the uncompressed shape.

- [ ] **Step 3: Wire it in**

In `mcp_app.py`, at the point the recall response dict is assembled, pass it
through `compress_payload` before returning. Import at module scope:

```python
from shared.compression.structural import compress_payload
```

and immediately before the return:

```python
    # Structural compression is lossless, so it is unconditional: there is no
    # information to opt into keeping. See the compression design spec.
    payload, _stats = compress_payload(payload)
    return payload
```

- [ ] **Step 4: Run the full mcp-server suite**

Run: `cd services/mcp-server && python -m pytest -q`
Expected: all pass. **If an existing recall test fails, do not weaken it** —
either the compression lost information (a bug) or the test asserts an exact
payload shape that legitimately changed. Read it and decide; note which in the
commit message.

- [ ] **Step 5: Commit**

```bash
ruff format services/ && ruff check services/
git add services/mcp-server/app/mcp_app.py services/mcp-server/tests/test_recall_compression.py
git commit -m "feat(recall): compress payloads structurally

Unconditional because it is lossless — there is nothing to opt into. An
agent receives the same information in materially fewer tokens."
```

---

## STAGE 2 — Originals cache and retrieval

### Task 4: Migration 034

**Files:**
- Create: `services/migrations/alembic/versions/034_compression_cache.py`
- Test: `services/migrations/tests/test_migration_034.py`

**Interfaces:**
- Produces: table `compression_cache`

- [ ] **Step 1: Write the failing test**

```python
"""Migration 034: the originals cache.

Mirrors test_migration_031.py's conventions — real Postgres, skips without it.
"""

import os

import pytest
from sqlalchemy import create_engine, text

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",
)


@pytest.fixture(scope="module")
def conn():
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("Postgres not reachable; migration 034 tests require the live DB")
    with create_engine(DB_URL).connect() as c:
        yield c


def _columns(conn, table):
    rows = conn.execute(
        text(
            "SELECT column_name, is_nullable FROM information_schema.columns "
            "WHERE table_name = :t"
        ),
        {"t": table},
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def test_compression_cache_exists_with_the_siloing_columns(conn):
    cols = _columns(conn, "compression_cache")
    assert {"id", "org_id", "user_id", "content_hash", "expires_at"} <= set(cols)
    assert cols["org_id"] == "NO", "org_id is the tenancy boundary and cannot be null"


def test_user_id_is_nullable(conn):
    """An unattributed caller still gets a cache entry; it is simply
    unreachable by any user-scoped read, which is the correct outcome."""
    assert _columns(conn, "compression_cache")["user_id"] == "YES"


def test_dedup_index_on_org_and_hash(conn):
    idx = conn.execute(
        text("SELECT indexdef FROM pg_indexes WHERE tablename = 'compression_cache'")
    ).fetchall()
    defs = " ".join(r[0] for r in idx)
    assert "content_hash" in defs, "dedup requires an index on the hash"
    assert "org_id" in defs


def test_expiry_index_exists(conn):
    idx = conn.execute(
        text("SELECT indexdef FROM pg_indexes WHERE tablename = 'compression_cache'")
    ).fetchall()
    assert any("expires_at" in r[0] for r in idx), "the sweep needs an expiry index"


def test_inline_or_uri_but_the_table_permits_both_null(conn):
    """A pending row exists briefly before its content lands."""
    cols = _columns(conn, "compression_cache")
    assert cols["inline_text"] == "YES"
    assert cols["storage_uri"] == "YES"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd services/migrations && python -m pytest tests/test_migration_034.py -v`
Expected: FAIL — no such table. If tests SKIP, start Postgres: `docker compose up -d postgres`. **A skip is not a pass.**

- [ ] **Step 3: Write the migration**

```python
"""Klio compression: cache of pre-compression originals.

Lossy compression is only acceptable if the original can be recovered, so this
table is the recovery contract. It is deliberately NOT part of session_memories:
compressing a log must never publish it into a team's shared brain, never be
curated, and never appear in recall.

Siloed by (org_id, user_id) exactly like memory — a handle belonging to another
user must be unreachable. user_id is nullable because an unattributable caller
still deserves a working compressor; such a row is simply unreachable by any
user-scoped read, which is the correct outcome rather than a special case.

Retention is 24 hours by default: retrieval happens within a session or not at
all. Storage mirrors artifacts — small originals inline, large ones in object
storage.

Revision ID: 034
Revises: 033
Create Date: 2026-08-01
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "034"
down_revision = "033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "compression_cache",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("org_id", sa.String(64), nullable=False),
        sa.Column("user_id", sa.String(64), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("inline_text", sa.Text(), nullable=True),
        sa.Column("storage_uri", sa.Text(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    # Dedup: agents re-read the same file constantly. Scoped by org so one
    # tenant's content can never satisfy another's lookup.
    op.create_index(
        "ix_compression_cache_dedup",
        "compression_cache",
        ["org_id", "content_hash"],
    )
    # The expiry sweep scans by time and must not seq-scan the table.
    op.create_index(
        "ix_compression_cache_expires",
        "compression_cache",
        ["expires_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_compression_cache_expires", table_name="compression_cache")
    op.drop_index("ix_compression_cache_dedup", table_name="compression_cache")
    op.drop_table("compression_cache")
```

- [ ] **Step 4: Apply and verify**

```bash
cd services/migrations
DATABASE_URL="postgresql://agentguard:agentguard_dev@localhost:5432/agentguard" alembic upgrade head
python -m pytest tests/test_migration_034.py -v
```
Expected: migration runs `033 -> 034`, tests pass.

- [ ] **Step 5: Exercise the downgrade**

```bash
alembic downgrade 033 && alembic upgrade head && python -m pytest tests/test_migration_034.py -q
```
A downgrade nobody runs is a downgrade that does not work.

- [ ] **Step 6: Commit**

```bash
ruff format services/ && ruff check services/
git add services/migrations/alembic/versions/034_compression_cache.py services/migrations/tests/test_migration_034.py
git commit -m "feat(db): migration 034 — compression originals cache

Lossy compression is only acceptable if the original is recoverable, so this
table is that contract. Deliberately not session_memories: compressing a log
must never publish it into a team's brain, be curated, or appear in recall.

Siloed by (org_id, user_id) like memory. user_id is nullable because an
unattributable caller still deserves a working compressor — that row is
simply unreachable by any user-scoped read, which is correct rather than a
special case."
```

---

### Task 5: Cache access, siloed

**Files:**
- Create: `services/mcp-server/app/compression_cache.py`
- Test: `services/mcp-server/tests/test_compression_cache_isolation.py`

**Interfaces:**
- Consumes: migration `034`, `resolve_user_id` from `app/identity.py`
- Produces: `store_original(db, *, org_id, user_id, text) -> str` (returns handle) and `load_original(db, *, org_id, user_id, handle) -> str | None`

- [ ] **Step 1: Write the adversarial test**

```python
"""The cache is a tenancy boundary, not a convenience store.

An original is whatever the user compressed — a log, a source file, an API
response. It is at least as sensitive as a memory, and these tests attack it
the same way test_user_silo_isolation.py attacks recall.
"""

import pytest

from app.compression_cache import load_original, store_original

ALICE = "11111111-1111-1111-1111-111111111111"
BOB = "22222222-2222-2222-2222-222222222222"


def test_owner_can_load_their_own(db):
    """Positive control. Without this, a cache that returns nothing to
    anyone would pass every test below."""
    handle = store_original(db, org_id="acme", user_id=ALICE, text="secret log")
    assert load_original(db, org_id="acme", user_id=ALICE, handle=handle) == "secret log"


def test_another_user_gets_nothing(db):
    handle = store_original(db, org_id="acme", user_id=ALICE, text="secret log")
    assert load_original(db, org_id="acme", user_id=BOB, handle=handle) is None


def test_another_org_gets_nothing(db):
    handle = store_original(db, org_id="acme", user_id=ALICE, text="secret log")
    assert load_original(db, org_id="other", user_id=ALICE, handle=handle) is None


def test_unattributed_caller_cannot_read_an_owned_entry(db):
    """user_id=None must not degrade into match-anything."""
    handle = store_original(db, org_id="acme", user_id=ALICE, text="secret log")
    assert load_original(db, org_id="acme", user_id=None, handle=handle) is None


def test_expired_entry_is_not_returned(db):
    """An expired handle returns None; the caller reports `expired`."""
    handle = store_original(
        db, org_id="acme", user_id=ALICE, text="x", ttl_seconds=-1
    )
    assert load_original(db, org_id="acme", user_id=ALICE, handle=handle) is None


def test_unknown_handle_returns_none(db):
    assert load_original(db, org_id="acme", user_id=ALICE, handle="kc_nope") is None


def test_identical_content_dedups_within_an_org(db):
    a = store_original(db, org_id="acme", user_id=ALICE, text="same bytes")
    b = store_original(db, org_id="acme", user_id=ALICE, text="same bytes")
    assert a == b


def test_identical_content_does_not_dedup_across_orgs(db):
    a = store_original(db, org_id="acme", user_id=ALICE, text="same bytes")
    b = store_original(db, org_id="other", user_id=ALICE, text="same bytes")
    assert a != b, "one tenant's content must never satisfy another's lookup"
```

**Note for the implementer:** the `db` fixture must be a real Postgres session.
Mirror the fixture style used by `services/mcp-server/tests/` neighbours that
already touch the database. A mocked session would return whatever the test
queued and would pass against a query with no `WHERE` clause at all.

- [ ] **Step 2: Run to verify it fails**

Run: `cd services/mcp-server && python -m pytest tests/test_compression_cache_isolation.py -v`
Expected: FAIL — no module `app.compression_cache`.

- [ ] **Step 3: Implement**

`store_original` and `load_original` in `services/mcp-server/app/compression_cache.py`.
Requirements, all load-bearing:

- The handle is `kc_` + the first 16 hex characters of a SHA-256 over
  `(org_id, content)`. Deterministic within an org, so identical content dedups;
  org-salted, so it cannot collide across tenants.
- `load_original` filters on **`org_id = :org AND user_id = :user AND expires_at > NOW()`**
  as SQL predicates. Never fetch then filter in Python.
- A `user_id` of `None` must produce a predicate that matches nothing, not one
  that matches every row. `user_id = NULL` is never true in SQL, which gives the
  right behaviour — but assert it with a test rather than relying on it.
- Both functions use bound parameters via `text()`.

- [ ] **Step 4: Verify**

Run: `cd services/mcp-server && python -m pytest tests/test_compression_cache_isolation.py -v`
Expected: all PASS.

- [ ] **Step 5: Mutation check**

Delete `AND user_id = :user` from `load_original` and re-run. At least three
tests must fail. If they do not, the tests are not testing the boundary —
fix them before proceeding. Restore afterwards.

- [ ] **Step 6: Commit**

```bash
ruff format services/ && ruff check services/
git add services/mcp-server/app/compression_cache.py services/mcp-server/tests/test_compression_cache_isolation.py
git commit -m "feat(compression): siloed originals cache

An original is whatever the user compressed — a log, a source file, an API
response — and is at least as sensitive as a memory. Siloed by the same
(org_id, user_id) SQL predicates, with the same adversarial tests.

Handles are org-salted: identical content dedups within a tenant and cannot
collide across tenants."
```

---

### Task 6: `klio_retrieve`

**Files:**
- Modify: `services/mcp-server/app/mcp_app.py`
- Test: `services/mcp-server/tests/test_retrieve_tool.py`

- [ ] **Step 1: Write the failing test**

```python
"""klio_retrieve returns an original, to its owner and nobody else."""

import pytest


@pytest.mark.asyncio
async def test_owner_retrieves_the_original(storage):
    ...


@pytest.mark.asyncio
async def test_another_users_handle_is_not_found_not_forbidden(storage):
    """`forbidden` would confirm the handle exists. Refusals must be
    indistinguishable from a handle that was never issued."""
    ...


@pytest.mark.asyncio
async def test_expired_handle_says_expired(storage):
    """Silence would look like an empty original. Name the reason."""
    ...
```

**Note for the implementer:** mirror the fixture style of the existing
`test_forget_tool.py` / `test_recall_tool.py` in this directory.

- [ ] **Step 2: Run to verify it fails**

Expected: the tool does not exist.

- [ ] **Step 3: Implement**

Register `klio_retrieve(handle: str, offset: int = 0)` alongside the other MCP
tools. It resolves `user_id` via `resolve_user_id`, calls `load_original`, and
pages at 51,200 characters with `offset` and `has_more` — the same contract as
`artifact_get`, so an agent that knows one knows the other.

Responses:
- found → `{"text": ..., "offset": ..., "has_more": ...}`
- not owned / unknown → `{"error": "not_found"}`
- expired → `{"error": "expired", "detail": "..."}`

- [ ] **Step 4: Verify the full suite**

```bash
cd services/mcp-server && python -m pytest -q
cd ../shared && python -m pytest -q
cd ../migrations && python -m pytest -q
```

- [ ] **Step 5: Commit**

```bash
ruff format services/ && ruff check services/
git add services/mcp-server/app/mcp_app.py services/mcp-server/tests/test_retrieve_tool.py
git commit -m "feat(mcp): klio_retrieve — recover a compressed original

Same paging contract as artifact_get so an agent that knows one knows the
other. A handle owned by another user returns not_found rather than
forbidden: forbidden confirms the handle exists."
```

---

### Task 7: Ship

- [ ] **Step 1: Full verification**

```bash
cd services/shared && python -m pytest -q
cd ../mcp-server && python -m pytest -q
cd ../migrations && DATABASE_URL="postgresql://agentguard:agentguard_dev@localhost:5432/agentguard" python -m pytest -q
cd ../.. && ruff format --check services/ && ruff check services/
docker build -f services/mcp-server/Dockerfile -t klio-compress-check . \
  && docker run --rm --network none klio-compress-check python -c "import app.main; print('IMPORT OK')"
```

Migration tests must RUN, not skip.

- [ ] **Step 2: Push and open a PR**

Report the measured recall saving from Task 2 Step 5 in the PR body. A
compression PR without a before/after number is not reviewable.

---

## Self-Review

**Spec coverage (stages 1–2 only):**

| Spec requirement | Task |
|---|---|
| Protected tokens never altered | 1 |
| Structural compressor, lossless | 2 |
| Applied to recall automatically | 3 |
| `compression_cache` table | 4 |
| Siloed by org + user | 5 |
| Dedup by content hash | 5 |
| 24h retention, expired ≠ silence | 4, 6 |
| Not a memory, no quota | 4 (separate table) |
| `klio_retrieve`, artifact_get paging | 6 |
| Fail-open on every path | 2, 3 |

**Deliberately deferred to stage 3+:** the local proxy, `klio init` wiring, the
supervised service, `ENABLE_TOOL_SEARCH`, the code and prose compressors, the
model, and Codex support.

**Type consistency:** `compress_payload(payload) -> tuple[Any, CompressionStats]`
is used identically in Tasks 2 and 3. `store_original`/`load_original` signatures
match between Tasks 5 and 6. Handles are `str` throughout, prefixed `kc_`.

**Known softness:** Tasks 3, 6 and part of 5 specify contracts rather than
literal test bodies, because their fixtures depend on existing harnesses in this
repo that the implementer must read first. Every such task names the file to
mirror. This is deliberate — inventing a parallel fixture style would be worse
than deferring to the one that exists.
