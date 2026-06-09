# Memory Engine Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved spec at `docs/superpowers/specs/2026-06-10-memory-engine-hardening-design.md`: secret/PII redaction at the write chokepoint (fail closed), all-org backfill, `X-Vex-Project` scoped recall with two-tier ranking, recall-hidden raw observations with a 7-day TTL, curator supersession + a `forget` tool, and retrieval quality (relevance scores, `limit`, blended recency ranking).

**Architecture:** All memory writes funnel through `shared.brain.write_scoped_memory`, so redaction lives there (plus pre-LLM redaction in the capture transcript path and recall-event logging). Recall changes live in `shared.memory.retrieve_memories_scoped` and the two recall handlers (`app/mcp_app.py`, `app/capture.py`). The curator gains a bounded reconciliation pass. One new Alembic migration (**026** — the spec said 023, but the chain is already at 025).

**Tech Stack:** Python 3.11, FastAPI + FastMCP, SQLAlchemy raw-SQL (`text()`), pgvector, litellm, Alembic, pytest. House patterns: async handlers with inline sync psycopg2 I/O in `shared/*` (do NOT add threading there), raw SQL via `text()`, tenant scoping by `org_id` on every query.

**Working directory:** all paths relative to `vex_private/vex_engine/`. Run shared tests from `services/shared/`, mcp-server tests from `services/mcp-server/` (both have existing pytest suites; follow `tests/conftest.py` fixtures in each).

**Commit rule:** conventional commits, no co-author lines.

---

## File structure (what gets created/modified)

| File | Responsibility |
|---|---|
| Create `services/shared/shared/redaction.py` | Pure detection + masking. No I/O. |
| Create `services/shared/tests/test_redaction.py` | Detector catalog unit tests. |
| Modify `services/shared/shared/brain.py` | Call `redact()` in `write_scoped_memory` before embedding; fail closed; `recall_hidden` param. |
| Create `services/shared/tests/test_brain_redaction.py` | Chokepoint integration tests. |
| Modify `services/shared/shared/memory.py` | Hardened prompts; `relevance` on `MemoryRecord`; two-tier + blended-score + `recall_hidden`/`status` filters in `retrieve_memories_scoped`; `retract_memory` helper. |
| Modify `services/mcp-server/app/capture.py` | Pre-LLM transcript redaction; hidden observation writes; recency-path filters; `/capture/forget`. |
| Modify `services/mcp-server/app/recall_log.py` | Redact `query_text` before logging. |
| Modify `services/mcp-server/app/auth.py` | `X-Vex-Project` header constant + resolution helper. |
| Modify `services/mcp-server/app/mcp_app.py` | Header-scoped recall, `limit` arg, `relevance` + project name in results, `forget` tool (8 tools). |
| Modify `services/mcp-server/app/curator.py` | Reconciliation pass (supersede contradicted facts, bounded). |
| Create `services/migrations/alembic/versions/026_recall_hidden.py` | `recall_hidden` column + backfill of hook observations + partial index. |
| Modify `services/scripts/run_retention.py` | 7-day TTL sweep for hidden raw rows. |
| Create `services/scripts/run_redaction_backfill.py` | All-org mask-in-place backfill (dry-run default). |
| Create `services/shared/tests/test_redaction_backfill.py` | Backfill behavior tests. |
| Modify `services/mcp-server/tests/test_recall_tool.py`, `test_write_tools.py`, `test_capture.py`, `test_curator.py` | Extend for new behavior (each task lists exact additions). |

---

### Task 1: Redaction module — detection catalog

**Files:**
- Create: `services/shared/shared/redaction.py`
- Create: `services/shared/tests/test_redaction.py`

- [ ] **Step 1: Write the failing tests**

```python
# services/shared/tests/test_redaction.py
"""Unit tests for shared.redaction — detection catalog + masking.

False positives are product damage (memories get mangled), so the
false-positive corpus matters as much as the true-positive corpus.
"""

import pytest

from shared.redaction import Finding, RedactionResult, redact

# ---- true positives: every kind in the catalog --------------------------

TRUE_POSITIVES = [
    # (input, expected kind)
    (
        "User provided Stripe test secret key: "
        "sk_test_51TcfsTP4cXbUl5jnQJG2Ou9BUcBChSVqEDNj7d65uQjZXtAVzI1XR2"
        "eewh5KDq8GmCfrr1a1xVRNcSdB1c7TtWrU00GsK8y8Je",
        "stripe_secret_key",
    ),
    ("publishable: pk_live_abc123DEF456ghi789JKL012", "stripe_publishable_key"),
    ("whsec_8f2e1a9b0c3d4e5f6a7b8c9d0e1f2a3b", "stripe_webhook_secret"),
    ("aws key AKIAIOSFODNN7EXAMPLE", "aws_access_key_id"),
    ("ghp_16C7e42F292c6912E7710c838347Ae178B4a", "github_token"),
    ("github_pat_11ABCDEFG0abcdefghijklmnopqrstuvwxyz1234567890ABCD", "github_token"),
    ("OPENAI_API_KEY=sk-proj-abc123DEF456ghi789JKLmno", "openai_api_key"),
    ("sk-ant-api03-abc123DEF456ghi789JKLmno-pqrstu", "anthropic_api_key"),
    ("xoxb-2624-7034-abcdefGHIJKL", "slack_token"),
    ("google key AIzaSyA1234567890abcdefghijklmnopqrstu", "google_api_key"),
    (
        "token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "jwt",
    ),
    (
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\n"
        "-----END RSA PRIVATE KEY-----",
        "private_key_block",
    ),
    ("password=hunter2secret", "credential_assignment"),
    ('api_key: "9f8e7d6c5b4a3210fedcba9876543210"', "credential_assignment"),
    ("postgres://admin:s3cretPW@db.internal:5432/prod", "connection_string"),
    ("Authorization: Bearer abc123DEF456ghi789JKLmnop", "bearer_token"),
    ("reach me at jane.doe@example.com please", "email"),
    ("card 4111 1111 1111 1111 exp 12/28", "credit_card"),
    ("call +14155552671 tomorrow", "phone"),
]


@pytest.mark.parametrize("content,kind", TRUE_POSITIVES)
def test_detects_and_masks(content, kind):
    result = redact(content)
    assert result.redacted is True
    assert kind in {f.kind for f in result.findings}
    # The raw secret must be gone from the clean text.
    assert "sk_test_51Tcfs" not in result.clean_text
    assert "hunter2secret" not in result.clean_text
    assert f"[REDACTED:{kind}:" in result.clean_text or result.redacted


def test_placeholder_format_and_fingerprint():
    secret = "sk_test_51TcfsTP4cXbUl5jnQJG2Ou9BUcBChSVqEDNj7d65uQjZXtAVzI1XR2eewh5KDq8GmCfrr1a1xVRNcSdB1c7TtWrU00GsK8y8Je"
    r1 = redact(f"key A: {secret}")
    r2 = redact(f"key B mentioned again: {secret}")
    ph1 = [f for f in r1.findings if f.kind == "stripe_secret_key"][0]
    ph2 = [f for f in r2.findings if f.kind == "stripe_secret_key"][0]
    # Same secret → same 4-hex-char fingerprint, correlatable without exposure.
    assert ph1.fingerprint == ph2.fingerprint
    assert len(ph1.fingerprint) == 4
    assert f"[REDACTED:stripe_secret_key:{ph1.fingerprint}]" in r1.clean_text


# ---- false positives: must NOT redact ------------------------------------

FALSE_POSITIVES = [
    "The migration id is 2d07c2d4-b3c5-4b49-8ff9-08961a737b61",        # UUID
    "commit c2c3c09 fixed the bug",                                     # short SHA
    "commit 5bfab80d528255fb2342c9395f2c3bf1e817f145ab90a702",          # long hex sha
    "ORDER BY embedding <=> :embedding LIMIT 10",                       # SQL
    "The user wants to connect Stripe to moonforge-prod.",              # prose w/ 'Stripe'
    "Set MEMORY_EMBEDDING_MODEL to the default model name",             # env var name only
    "the meeting is at 10:30, room 415",                                # not a phone
    "version 4.1.1 released",                                           # not a card
]


@pytest.mark.parametrize("content", FALSE_POSITIVES)
def test_does_not_redact_ordinary_content(content):
    result = redact(content)
    assert result.redacted is False
    assert result.clean_text == content
    assert result.findings == ()


# ---- idempotency ----------------------------------------------------------

def test_redaction_is_idempotent():
    content = "key: sk_test_51TcfsTP4cXbUl5jnQJG2Ou9BUcBChSVqEDNj7d65uQjZXtAVzI1XR2eewh5KDq8GmCfrr1a1xVRNcSdB1c7TtWrU00GsK8y8Je"
    once = redact(content)
    twice = redact(once.clean_text)
    assert twice.redacted is False
    assert twice.clean_text == once.clean_text


def test_result_is_immutable():
    result = redact("nothing secret here")
    with pytest.raises(Exception):
        result.clean_text = "mutated"  # frozen dataclass


def test_empty_and_none_safe():
    assert redact("").redacted is False
    assert redact("").clean_text == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd services/shared && python -m pytest tests/test_redaction.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.redaction'`

- [ ] **Step 3: Implement the module**

```python
# services/shared/shared/redaction.py
"""Secret/PII detection and masking for memory content.

Pure functions, no I/O. Every memory write funnels through
:func:`shared.brain.write_scoped_memory`, which calls :func:`redact` BEFORE
embedding — neither stored text nor vectors ever contain a detected secret.

Detection is pattern + entropy based (the detect-secrets approach this repo
already uses in pre-commit). The generic entropy detector only fires inside a
credential-shaped context (key-value assignment), because entropy alone
misfires on UUIDs, git SHAs, and hashes — which are fine to store.

Placeholder format: ``[REDACTED:<kind>:<fingerprint>]`` where fingerprint is
the last 4 hex chars of SHA-256 of the matched secret, so repeated mentions of
the same secret correlate without exposure. Placeholders never re-match
(idempotent).
"""

import hashlib
import math
import re
from dataclasses import dataclass

# Order matters: more specific kinds first so e.g. a Stripe key is tagged
# stripe_secret_key, not generic credential_assignment.
_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("stripe_secret_key", re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{20,}\b")),
    ("stripe_publishable_key", re.compile(r"\bpk_(?:live|test)_[A-Za-z0-9]{20,}\b")),
    ("stripe_webhook_secret", re.compile(r"\bwhsec_[A-Za-z0-9]{20,}\b")),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    (
        "github_token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    ),
    ("anthropic_api_key", re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{20,}\b")),
    ("openai_api_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9\-_]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z\-_]{30,}\b")),
    ("supabase_key", re.compile(r"\bsbp_[A-Za-z0-9]{20,}\b")),
    (
        "jwt",
        re.compile(
            r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"
        ),
    ),
    (
        "private_key_block",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    (
        "connection_string",
        re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:([^\s@]+)@[^\s]+"),
    ),
    ("bearer_token", re.compile(r"(?i)\bBearer\s+([A-Za-z0-9\-._~+/]{16,}=*)")),
    (
        "credential_assignment",
        re.compile(
            r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)"
            r"\s*[:=]\s*[\"']?([^\s\"',;]{8,})[\"']?"
        ),
    ),
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    (
        "credit_card",
        re.compile(r"\b(?:\d[ \-]?){13,19}\b"),
    ),
    ("phone", re.compile(r"(?<![\d.])\+\d{10,14}\b")),
)

# For credential_assignment / bearer_token / connection_string, only the
# captured VALUE group is the secret; everything else replaces the full match.
_VALUE_GROUP_KINDS = {"credential_assignment": 2, "bearer_token": 1, "connection_string": 1}

_PLACEHOLDER_RE = re.compile(r"\[REDACTED:[a-z_]+:[0-9a-f]{4}\]")

_ENTROPY_THRESHOLD = 3.5
_MIN_VALUE_LEN_FOR_ENTROPY = 12


@dataclass(frozen=True)
class Finding:
    kind: str
    fingerprint: str


@dataclass(frozen=True)
class RedactionResult:
    clean_text: str
    findings: tuple[Finding, ...]
    redacted: bool


def _fingerprint(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[-4:]


def _shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for ch in value:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(value)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _luhn_valid(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alt = not alt
    return total % 10 == 0


def redact(content: str) -> RedactionResult:
    """Detect and mask secrets/PII in ``content``. Pure; never raises on str input."""
    if not content:
        return RedactionResult(clean_text=content or "", findings=(), redacted=False)

    findings: list[Finding] = []
    clean = content

    for kind, pattern in _PATTERNS:
        out: list[str] = []
        last = 0
        for match in pattern.finditer(clean):
            value_group = _VALUE_GROUP_KINDS.get(kind)
            secret = match.group(value_group) if value_group else match.group(0)

            # Skip already-masked content (idempotency).
            if _PLACEHOLDER_RE.search(match.group(0)):
                continue
            # credit_card: require Luhn so versions/dates never match.
            if kind == "credit_card":
                digits = re.sub(r"[ \-]", "", match.group(0))
                if not (13 <= len(digits) <= 19) or not _luhn_valid(digits):
                    continue
                secret = digits
            # credential_assignment: entropy-gate the VALUE so prose like
            # "password= the dog's name" or low-entropy words don't fire,
            # while real tokens do.
            if kind == "credential_assignment":
                if len(secret) >= _MIN_VALUE_LEN_FOR_ENTROPY:
                    if _shannon_entropy(secret) < 3.0 and not re.search(
                        r"\d", secret
                    ):
                        continue
                elif _shannon_entropy(secret) < 2.5:
                    continue

            fp = _fingerprint(secret)
            findings.append(Finding(kind=kind, fingerprint=fp))
            placeholder = f"[REDACTED:{kind}:{fp}]"
            if value_group:
                # Replace only the secret value inside the match.
                s, e = match.span(value_group)
                out.append(clean[last:s])
                out.append(placeholder)
                last = e
            else:
                s, e = match.span(0)
                out.append(clean[last:s])
                out.append(placeholder)
                last = e
        out.append(clean[last:])
        clean = "".join(out)

    return RedactionResult(
        clean_text=clean, findings=tuple(findings), redacted=bool(findings)
    )
```

- [ ] **Step 4: Run tests until green; tune the catalog against the corpus**

Run: `cd services/shared && python -m pytest tests/test_redaction.py -v`
Expected: PASS (all parametrized cases). If a false-positive case fires, tighten that pattern's guard — never delete the false-positive test.

- [ ] **Step 5: Commit**

```bash
git add services/shared/shared/redaction.py services/shared/tests/test_redaction.py
git commit -m "feat(redaction): secret/PII detection catalog with masking"
```

---

### Task 2: Redaction at the write chokepoint (fail closed)

**Files:**
- Modify: `services/shared/shared/brain.py` (function `write_scoped_memory`, currently lines 439–594)
- Create: `services/shared/tests/test_brain_redaction.py`

- [ ] **Step 1: Write the failing tests**

```python
# services/shared/tests/test_brain_redaction.py
"""write_scoped_memory must mask secrets before embedding, fail closed."""

import pytest

import shared.brain as brain_mod
import shared.memory as memory_mod
from shared.brain import write_scoped_memory

SECRET = (
    "sk_test_51TcfsTP4cXbUl5jnQJG2Ou9BUcBChSVqEDNj7d65uQjZXtAVzI1XR2"
    "eewh5KDq8GmCfrr1a1xRNcSdB1c7TtWrU00GsK8y8Je"
)


class FakeResult:
    def fetchone(self):
        return None

    def fetchall(self):
        return []


class FakeDB:
    """Records executed SQL + params; enough for the write path."""

    def __init__(self):
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((str(sql), params))
        return FakeResult()

    def commit(self):
        pass


@pytest.fixture
def fake_embed(monkeypatch):
    captured = {}

    async def _embed(text_input):
        captured["text"] = text_input
        return [0.1, 0.2, 0.3]

    # write_memory and find_similar_memory both call shared.memory.embed_text.
    monkeypatch.setattr(memory_mod, "embed_text", _embed)
    return captured


@pytest.mark.anyio
async def test_secret_is_masked_before_store_and_embedding(fake_embed):
    db = FakeDB()
    result = await write_scoped_memory(
        db,
        org_id="org-1",
        agent_id="agent-1",
        content=f"User provided Stripe key: {SECRET}",
        memory_type="memory",
        session_id="klio-brain:test",
        dedup=False,
    )
    assert result["stored"] is True
    assert result["redacted"] is True
    assert "stripe_secret_key" in result["redaction_kinds"]
    # The INSERT params and the embedded text must not contain the raw secret.
    insert_params = [p for _, p in db.calls if p and "content" in p][0]
    assert SECRET not in insert_params["content"]
    assert "[REDACTED:stripe_secret_key:" in insert_params["content"]
    assert SECRET not in fake_embed["text"]


@pytest.mark.anyio
async def test_clean_content_passes_untouched(fake_embed):
    db = FakeDB()
    result = await write_scoped_memory(
        db,
        org_id="org-1",
        agent_id="agent-1",
        content="The deploy finished at 10:30.",
        memory_type="memory",
        session_id="klio-brain:test",
        dedup=False,
    )
    assert result["stored"] is True
    assert result.get("redacted") is False


@pytest.mark.anyio
async def test_redactor_failure_fails_closed(fake_embed, monkeypatch):
    def _boom(content):
        raise RuntimeError("redactor crashed")

    monkeypatch.setattr(brain_mod, "redact", _boom)
    db = FakeDB()
    result = await write_scoped_memory(
        db,
        org_id="org-1",
        agent_id="agent-1",
        content=f"key {SECRET}",
        memory_type="memory",
        session_id="klio-brain:test",
        dedup=False,
    )
    assert result == {"stored": False, "error": "redaction_failed"}
    # Nothing was written.
    assert not any("INSERT INTO session_memories" in sql for sql, _ in db.calls)


@pytest.mark.anyio
async def test_disable_flag_skips_redaction(fake_embed, monkeypatch):
    monkeypatch.setenv("MEMORY_REDACTION_DISABLED", "1")
    db = FakeDB()
    result = await write_scoped_memory(
        db,
        org_id="org-1",
        agent_id="agent-1",
        content=f"key {SECRET}",
        memory_type="memory",
        session_id="klio-brain:test",
        dedup=False,
    )
    assert result["stored"] is True
    insert_params = [p for _, p in db.calls if p and "content" in p][0]
    assert SECRET in insert_params["content"]
```

(If the suite uses `pytest-asyncio` instead of `anyio`, mark with
`@pytest.mark.asyncio` — match whichever marker `tests/test_brain_write.py`
already uses.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd services/shared && python -m pytest tests/test_brain_redaction.py -v`
Expected: FAIL — `KeyError: 'redacted'` / no fail-closed behavior.

- [ ] **Step 3: Integrate redaction into `write_scoped_memory`**

In `services/shared/shared/brain.py`:

Add imports at the top (after the existing imports, line ~22):

```python
import os

from shared.redaction import redact
```

Inside `write_scoped_memory`, immediately after the `scope` validation block (after line 524 `)` of the `raise ValueError`), insert:

```python
    # --- redaction (fail closed) -----------------------------------------
    # Mask secrets/PII BEFORE dedup-embedding and storage so neither the
    # stored text nor any embedding ever contains a detected secret. A
    # redactor crash rejects the write: unscreened content is never stored.
    # MEMORY_REDACTION_DISABLED=1 is a self-hosted escape hatch only.
    redaction_kinds: list[str] = []
    was_redacted = False
    if os.environ.get("MEMORY_REDACTION_DISABLED") != "1":
        try:
            redaction = redact(content)
        except Exception:
            logger.exception("redaction failed; rejecting write (fail closed)")
            return {"stored": False, "error": "redaction_failed"}
        if redaction.redacted:
            content = redaction.clean_text
            redaction_kinds = sorted({f.kind for f in redaction.findings})
            was_redacted = True
            metadata = dict(metadata or {})
            metadata["redactions"] = redaction_kinds
```

Then extend BOTH return dicts to carry the flag. The dedup-hit return (currently lines 538–544) becomes:

```python
            return {
                "stored": False,
                "deduplicated": True,
                "id": str(existing.id),
                "memory_type": existing.memory_type,
                "scope": scope,
                "redacted": was_redacted,
                "redaction_kinds": redaction_kinds,
            }
```

And the success return (currently lines 586–594) becomes:

```python
    return {
        "stored": True,
        "deduplicated": False,
        "id": str(record.id),
        "memory_type": memory_type,
        "scope": scope,
        "space_id": space_id,
        "project_id": project_id,
        "redacted": was_redacted,
        "redaction_kinds": redaction_kinds,
    }
```

- [ ] **Step 4: Run the new tests AND the existing brain suites**

Run: `cd services/shared && python -m pytest tests/test_brain_redaction.py tests/test_brain_write.py tests/test_brain.py -v`
Expected: PASS. Existing `write_scoped_memory` callers are unaffected (new keys are additive).

- [ ] **Step 5: Commit**

```bash
git add services/shared/shared/brain.py services/shared/tests/test_brain_redaction.py
git commit -m "feat(brain): redact secrets at the write chokepoint, fail closed"
```

---

### Task 3: Pre-LLM transcript redaction + recall-log query redaction

**Files:**
- Modify: `services/mcp-server/app/capture.py` (`_combine_messages`, line 278)
- Modify: `services/mcp-server/app/recall_log.py`
- Modify: `services/mcp-server/tests/test_capture.py` (add tests)
- Modify: `services/mcp-server/tests/test_recall_log.py` (add test)

- [ ] **Step 1: Write the failing tests**

Append to `services/mcp-server/tests/test_capture.py`:

```python
SECRET_T3 = (
    "sk_test_51TcfsTP4cXbUl5jnQJG2Ou9BUcBChSVqEDNj7d65uQjZXtAVzI1XR2"
    "eewh5KDq8GmCfrr1a1xRNcSdB1c7TtWrU00GsK8y8Je"
)


def test_combine_messages_redacts_before_llm():
    from app.capture import TranscriptMessage, _combine_messages

    combined = _combine_messages(
        [
            TranscriptMessage(role="user", content=f"here is my key {SECRET_T3}"),
            TranscriptMessage(role="assistant", content="thanks, configuring now"),
        ]
    )
    assert SECRET_T3 not in combined
    assert "[REDACTED:stripe_secret_key:" in combined
    assert "configuring now" in combined
```

Append to `services/mcp-server/tests/test_recall_log.py`:

```python
def test_log_recall_event_redacts_query_text():
    from app.recall_log import log_recall_event

    class _Rec:
        def __init__(self):
            self.params = None

        def execute(self, sql, params=None):
            self.params = params

        def commit(self):
            pass

    db = _Rec()
    log_recall_event(
        db,
        org_id="org-1",
        agent_id="a",
        project_id=None,
        query_text=(
            "find sk_test_51TcfsTP4cXbUl5jnQJG2Ou9BUcBChSVqEDNj7d65uQjZXtAVzI1XR2"
            "eewh5KDq8GmCfrr1a1xRNcSdB1c7TtWrU00GsK8y8Je"
        ),
        result_count=0,
        source="mcp",
    )
    assert "sk_test_51Tcfs" not in (db.params or {}).get("query_text", "")
```

- [ ] **Step 2: Run to verify both fail**

Run: `cd services/mcp-server && python -m pytest tests/test_capture.py::test_combine_messages_redacts_before_llm tests/test_recall_log.py::test_log_recall_event_redacts_query_text -v`
Expected: FAIL (raw secret present).

- [ ] **Step 3: Implement**

In `services/mcp-server/app/capture.py`, add to imports (line ~53 area):

```python
from shared.redaction import redact
```

In `_combine_messages` (line 278), change the final lines from:

```python
    combined = "\n".join(parts)
    if len(combined) > _MAX_TRANSCRIPT_CHARS:
        combined = combined[-_MAX_TRANSCRIPT_CHARS:]
    return combined
```

to:

```python
    combined = "\n".join(parts)
    if len(combined) > _MAX_TRANSCRIPT_CHARS:
        combined = combined[-_MAX_TRANSCRIPT_CHARS:]
    # Redact BEFORE this text is sent to the external LLM for fact
    # extraction/summarisation — secrets never leave the infrastructure via
    # the mining path. write_scoped_memory re-screens whatever comes back
    # (defense in depth), so a redaction failure here can fail open to the
    # chokepoint guarantee.
    try:
        combined = redact(combined).clean_text
    except Exception:
        logger.warning("capture: transcript redaction failed; relying on chokepoint")
    return combined
```

In `services/mcp-server/app/recall_log.py`, inside `log_recall_event`, redact `query_text` before binding it (the function currently passes `query_text` straight into its INSERT params):

```python
from shared.redaction import redact

# ... inside log_recall_event, before the execute():
    if query_text:
        try:
            query_text = redact(query_text).clean_text
        except Exception:
            query_text = None  # best-effort log; never store unscreened
```

- [ ] **Step 4: Run the capture + recall-log suites**

Run: `cd services/mcp-server && python -m pytest tests/test_capture.py tests/test_recall_log.py -v`
Expected: PASS, including pre-existing tests.

- [ ] **Step 5: Commit**

```bash
git add services/mcp-server/app/capture.py services/mcp-server/app/recall_log.py services/mcp-server/tests/test_capture.py services/mcp-server/tests/test_recall_log.py
git commit -m "feat(capture): redact transcripts pre-LLM and recall-log queries"
```

---

### Task 4: Prompt hardening for fact extraction / summaries

**Files:**
- Modify: `services/shared/shared/memory.py` (`generate_summary` line 194, `extract_facts` line 224)
- Modify: `services/shared/tests/test_memory.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `services/shared/tests/test_memory.py`:

```python
def test_extract_facts_prompt_forbids_credentials(monkeypatch):
    import shared.memory as memory_mod

    seen = {}

    async def _fake_call_llm(prompt, system=None):
        seen["prompt"] = prompt
        return {"facts": []}

    monkeypatch.setattr(memory_mod, "call_llm", _fake_call_llm)
    import anyio

    anyio.run(memory_mod.extract_facts, "some output")
    assert "Never reproduce credentials" in seen["prompt"]


def test_generate_summary_prompt_forbids_credentials(monkeypatch):
    import shared.memory as memory_mod

    seen = {}

    async def _fake_call_llm(prompt, system=None):
        seen["prompt"] = prompt
        return {"summary": "ok"}

    monkeypatch.setattr(memory_mod, "call_llm", _fake_call_llm)
    import anyio

    anyio.run(memory_mod.generate_summary, "some output")
    assert "Never reproduce credentials" in seen["prompt"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd services/shared && python -m pytest tests/test_memory.py -k credentials -v`
Expected: FAIL (assertion on prompt text).

- [ ] **Step 3: Harden both prompts**

In `services/shared/shared/memory.py`, define once near the top (after `_JSON_MODE_PREFIXES`, line ~41):

```python
_NO_CREDENTIALS_RULE = (
    "Never reproduce credentials, API keys, tokens, passwords, or other "
    "secrets — not even partially. If the content contains one, refer to it "
    "generically (e.g. 'a Stripe test key was provided'). "
)
```

In `generate_summary` (line 210), change the prompt to:

```python
    prompt = (
        "Summarize the following agent turn in 1-2 concise sentences. "
        "Focus on what the agent did and the outcome. "
        + _NO_CREDENTIALS_RULE
        + "\n\n"
        + "\n".join(parts)
        + '\n\nRespond with JSON: {"summary": "<your summary>"}'
    )
```

In `extract_facts` (line 240), change the prompt to:

```python
    prompt = (
        "Extract discrete, verifiable facts from the following agent turn. "
        "Each fact should be a short, self-contained statement. "
        + _NO_CREDENTIALS_RULE
        + "\n\n"
        + "\n".join(parts)
        + '\n\nRespond with JSON: {"facts": ["fact1", "fact2", ...]}'
    )
```

- [ ] **Step 4: Run the memory suite**

Run: `cd services/shared && python -m pytest tests/test_memory.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/shared/shared/memory.py services/shared/tests/test_memory.py
git commit -m "feat(memory): forbid credential reproduction in synthesis prompts"
```

---

### Task 5: All-org redaction backfill script

**Files:**
- Create: `services/scripts/run_redaction_backfill.py`
- Create: `services/shared/tests/test_redaction_backfill.py`

- [ ] **Step 1: Write the failing tests**

```python
# services/shared/tests/test_redaction_backfill.py
"""Backfill: dry-run reports, apply masks + re-embeds, idempotent."""

import sys
import os

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
)
from run_redaction_backfill import process_memory_batch  # noqa: E402

SECRET = (
    "sk_test_51TcfsTP4cXbUl5jnQJG2Ou9BUcBChSVqEDNj7d65uQjZXtAVzI1XR2"
    "eewh5KDq8GmCfrr1a1xRNcSdB1c7TtWrU00GsK8y8Je"
)


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


async def _fake_embed(text_input):
    return [0.0, 0.1]


@pytest.mark.anyio
async def test_dry_run_reports_but_writes_nothing():
    db = FakeDB()
    rows = [_Row("m1", "org-a", f"the key is {SECRET}"), _Row("m2", "org-a", "clean")]
    report = await process_memory_batch(db, rows, apply=False, embed_fn=_fake_embed)
    assert report == {("org-a", "stripe_secret_key"): 1}
    assert db.updates == []


@pytest.mark.anyio
async def test_apply_masks_and_reembeds():
    db = FakeDB()
    rows = [_Row("m1", "org-a", f"the key is {SECRET}")]
    report = await process_memory_batch(db, rows, apply=True, embed_fn=_fake_embed)
    assert report == {("org-a", "stripe_secret_key"): 1}
    assert len(db.updates) == 1
    sql, params = db.updates[0]
    assert "UPDATE session_memories" in sql
    assert SECRET not in params["content"]
    assert "[REDACTED:stripe_secret_key:" in params["content"]
    assert params["embedding"] == str([0.0, 0.1])


@pytest.mark.anyio
async def test_embed_failure_leaves_row_unmodified():
    async def _fail(_):
        return None

    db = FakeDB()
    rows = [_Row("m1", "org-a", f"the key is {SECRET}")]
    report = await process_memory_batch(db, rows, apply=True, embed_fn=_fail)
    assert report == {("org-a", "stripe_secret_key"): 1}
    assert db.updates == []  # skipped, reported, not corrupted


@pytest.mark.anyio
async def test_idempotent_second_pass_is_noop():
    db = FakeDB()
    masked = "the key is [REDACTED:stripe_secret_key:ab12]"
    rows = [_Row("m1", "org-a", masked)]
    report = await process_memory_batch(db, rows, apply=True, embed_fn=_fake_embed)
    assert report == {}
    assert db.updates == []
```

- [ ] **Step 2: Run to verify failure**

Run: `cd services/shared && python -m pytest tests/test_redaction_backfill.py -v`
Expected: FAIL — `ModuleNotFoundError: run_redaction_backfill`

- [ ] **Step 3: Implement the script**

```python
# services/scripts/run_redaction_backfill.py
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
                               || :extra_meta::jsonb,
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
                text(
                    "UPDATE brain_recall_events SET query_text = :q WHERE id = :id"
                ),
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
                    WHERE {' AND '.join(where)}
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
```

- [ ] **Step 4: Run the tests**

Run: `cd services/shared && python -m pytest tests/test_redaction_backfill.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/scripts/run_redaction_backfill.py services/shared/tests/test_redaction_backfill.py
git commit -m "feat(scripts): all-org redaction backfill with dry-run default"
```

> **Operational note (do NOT automate in this plan):** after deploy, run dry-run against production, review the report, run `--apply`, then **rotate the two exposed Stripe test keys** (`sk_test_51Tcfs…` MoonForge, `sk_test_51TfGK5…` LoopReel sandbox).

---

### Task 6: Migration 026 — `recall_hidden`

**Files:**
- Create: `services/migrations/alembic/versions/026_recall_hidden.py`

- [ ] **Step 1: Write the migration** (style mirrors 025)

```python
# services/migrations/alembic/versions/026_recall_hidden.py
"""Klio brain: recall_hidden flag for raw passive captures.

ADDITIVE. Adds ``session_memories.recall_hidden`` (default FALSE). Rows with
``recall_hidden = TRUE`` are curator fuel only: both recall paths exclude
them. The data backfill hides existing passive hook observations
(``memory_type='observation'`` AND ``metadata->>'source'`` IN
('hook','hook-tool')) — explicit MCP ``observe`` rows (source='mcp') stay
visible.

Revision ID: 026
Revises: 025
Create Date: 2026-06-10
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "026"
down_revision: Union[str, None] = "025"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "session_memories",
        sa.Column(
            "recall_hidden",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
    )
    # Hide existing passive hook observations (curator fuel, not recall noise).
    op.execute(
        """
        UPDATE session_memories
        SET recall_hidden = TRUE
        WHERE memory_type = 'observation'
          AND metadata->>'source' IN ('hook', 'hook-tool')
        """
    )
    # Partial index: recall always filters recall_hidden = FALSE.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_session_memories_recall_visible
        ON session_memories (org_id, scope, status)
        WHERE recall_hidden = FALSE
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_session_memories_recall_visible;")
    op.drop_column("session_memories", "recall_hidden")
```

- [ ] **Step 2: Run the migration against the local dev DB**

Run: `cd services/migrations && alembic upgrade head`
Expected: `Running upgrade 025 -> 026`. Verify: `psql "$DATABASE_URL" -c "\d session_memories" | grep recall_hidden` shows `boolean | not null | default false`.

- [ ] **Step 3: Commit**

```bash
git add services/migrations/alembic/versions/026_recall_hidden.py
git commit -m "feat(migrations): add recall_hidden flag, hide passive hook observations"
```

---

### Task 7: Hidden writes + recall filtering (`status` + `recall_hidden`)

**Files:**
- Modify: `services/shared/shared/memory.py` (`retrieve_memories_scoped` line 582; `write_memory` line 296)
- Modify: `services/shared/shared/brain.py` (`write_scoped_memory` — pass-through param)
- Modify: `services/mcp-server/app/capture.py` (`capture_event` line 355, `_recent_memories` line 316)
- Modify: `services/shared/tests/test_brain_recall.py`, `services/mcp-server/tests/test_capture.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `services/shared/tests/test_brain_recall.py`:

```python
def test_recall_sql_excludes_hidden_and_inactive(monkeypatch):
    """The generated recall SQL must filter recall_hidden and status."""
    import anyio

    import shared.memory as memory_mod

    captured = {}

    async def _embed(_):
        return [0.1, 0.2]

    monkeypatch.setattr(memory_mod, "embed_text", _embed)

    class _DB:
        def execute(self, sql, params=None):
            captured["sql"] = str(sql)

            class R:
                def fetchall(self_inner):
                    return []

            return R()

    anyio.run(
        lambda: memory_mod.retrieve_memories_scoped(_DB(), "org-1", "query")
    )
    assert "recall_hidden = FALSE" in captured["sql"]
    assert "status = 'active'" in captured["sql"]
```

Append to `services/mcp-server/tests/test_capture.py`:

```python
def test_capture_event_observation_is_recall_hidden(client_with_auth, monkeypatch):
    """Passive hook observations are written recall_hidden=True."""
    import shared.brain as brain_mod

    seen = {}

    async def _fake_write(db, **kwargs):
        seen.update(kwargs)
        return {"stored": True, "deduplicated": False, "id": "x", "scope": "org"}

    monkeypatch.setattr(brain_mod, "write_scoped_memory", _fake_write)
    resp = client_with_auth.post(
        "/capture/event",
        json={"content": "Used tool Bash", "memory_type": "observation"},
    )
    assert resp.status_code == 200
    assert seen["recall_hidden"] is True


def test_capture_event_note_is_visible(client_with_auth, monkeypatch):
    import shared.brain as brain_mod

    seen = {}

    async def _fake_write(db, **kwargs):
        seen.update(kwargs)
        return {"stored": True, "deduplicated": False, "id": "x", "scope": "org"}

    monkeypatch.setattr(brain_mod, "write_scoped_memory", _fake_write)
    resp = client_with_auth.post(
        "/capture/event", json={"content": "decision made", "memory_type": "note"}
    )
    assert resp.status_code == 200
    assert seen["recall_hidden"] is False
```

(Use the existing authenticated-client fixture name from
`tests/conftest.py` — if it is not `client_with_auth`, substitute the
suite's actual fixture for an auth-bypassed `TestClient`.)

- [ ] **Step 2: Run to verify failure**

Run: `cd services/shared && python -m pytest tests/test_brain_recall.py -k hidden -v && cd ../mcp-server && python -m pytest tests/test_capture.py -k recall_hidden -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

**`services/shared/shared/memory.py`** — in `write_memory` (line 296), add a keyword param and column:

```python
async def write_memory(
    db_session: Any,
    session_id: str,
    org_id: str,
    agent_id: str,
    execution_id: str,
    memory_type: str,
    content: str,
    confidence: Optional[float] = None,
    sequence_number: Optional[int] = None,
    metadata: Optional[dict] = None,
    recall_hidden: bool = False,
) -> Optional[MemoryRecord]:
```

and in its INSERT, change the column list and VALUES to include the flag:

```sql
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, execution_id,
                memory_type, content, embedding, confidence,
                sequence_number, scope, status, metadata,
                created_at, updated_at, recall_hidden
            ) VALUES (
                :id, :session_id, :org_id, :agent_id, :execution_id,
                :memory_type, :content, :embedding, :confidence,
                :sequence_number, 'session', 'active', :metadata,
                :created_at, :updated_at, :recall_hidden
            )
```

with `"recall_hidden": recall_hidden` added to the params dict.

In `retrieve_memories_scoped` (line 669), extend the base clauses:

```python
    clauses = [
        "org_id = :org_id",
        "status = 'active'",
        "superseded_by IS NULL",
        "recall_hidden = FALSE",
    ]
```

**`services/shared/shared/brain.py`** — `write_scoped_memory` gains a pass-through keyword `recall_hidden: bool = False` (add to the signature after `scope`) and forwards it to `write_memory`:

```python
    record = await write_memory(
        db_session,
        session_id,
        org_id,
        agent_id,
        None,
        memory_type,
        content,
        metadata=metadata,
        recall_hidden=recall_hidden,
    )
```

**`services/mcp-server/app/capture.py`** — in `capture_event` (line 391), forward the flag (hidden only for passive observations):

```python
        result = await brain.write_scoped_memory(
            db,
            org_id=org_id,
            agent_id=agent_id,
            content=req.content,
            memory_type=memory_type,
            session_id=_capture_session_id(org_id, agent_id, req.session_id),
            project_id=project_id,
            metadata=metadata,
            scope=req.scope,
            # Raw tool/prompt observations are curator fuel, not recall
            # results; deliberate types (note/plan/decision/memory) stay
            # visible.
            recall_hidden=(memory_type == "observation"),
        )
```

and in `_recent_memories` (line 328), add both filters to the WHERE clause:

```sql
            WHERE org_id = :org_id
              AND scope = 'org'
              AND status = 'active'
              AND recall_hidden = FALSE
```

- [ ] **Step 4: Run both suites + existing recall/write tests**

Run: `cd services/shared && python -m pytest tests/ -v && cd ../mcp-server && python -m pytest tests/test_capture.py tests/test_recall_tool.py tests/test_write_tools.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/shared/shared/memory.py services/shared/shared/brain.py services/mcp-server/app/capture.py services/shared/tests/test_brain_recall.py services/mcp-server/tests/test_capture.py
git commit -m "feat(memory): recall-hidden raw observations excluded from recall"
```

---

### Task 8: 7-day TTL for hidden raw rows

**Files:**
- Modify: `services/scripts/run_retention.py`
- Modify: `services/shared/tests/test_run_memory_retention.py` (add test)

- [ ] **Step 1: Write the failing test**

Append to `services/shared/tests/test_run_memory_retention.py`:

```python
def test_hidden_raw_ttl_sql_present():
    """run_retention must sweep recall_hidden rows on a fixed short TTL."""
    import importlib.util
    import os

    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "scripts", "run_retention.py"
    )
    source = open(path).read()
    assert "recall_hidden = TRUE" in source
    assert "RAW_OBSERVATION_TTL_DAYS" in source
```

- [ ] **Step 2: Run to verify failure**

Run: `cd services/shared && python -m pytest tests/test_run_memory_retention.py -k hidden -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `services/scripts/run_retention.py`, after the per-account loop and **before** `tsdb_conn.commit()` (line 107), add:

```python
            # Hidden raw captures are curator fuel with a short shelf life:
            # sweep them on a fixed TTL regardless of plan retention.
            raw_ttl_days = int(os.environ.get("RAW_OBSERVATION_TTL_DAYS", "7"))
            deleted = tsdb_conn.execute(
                text(
                    """
                    DELETE FROM session_memories
                    WHERE recall_hidden = TRUE
                      AND created_at < NOW() - (:days || ' days')::interval
                    """
                ),
                {"days": raw_ttl_days},
            )
            logger.info(
                "Swept hidden raw observations older than %d days (%s rows)",
                raw_ttl_days,
                getattr(deleted, "rowcount", "?"),
            )
```

- [ ] **Step 4: Run the retention tests**

Run: `cd services/shared && python -m pytest tests/test_run_memory_retention.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/scripts/run_retention.py services/shared/tests/test_run_memory_retention.py
git commit -m "feat(retention): 7-day TTL sweep for recall-hidden raw rows"
```

---

### Task 9: `X-Vex-Project` header resolution

**Files:**
- Modify: `services/mcp-server/app/auth.py`
- Modify: `services/mcp-server/tests/test_auth.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `services/mcp-server/tests/test_auth.py`:

```python
def test_project_header_constant():
    from app.auth import PROJECT_HEADER

    assert PROJECT_HEADER == "X-Vex-Project"


def test_resolve_project_header_returns_none_without_header():
    from app.auth import resolve_project_from_request

    class _Req:
        headers = {}

    assert resolve_project_from_request(_Req(), object(), org_id="o") is None


def test_resolve_project_header_resolves_via_find(monkeypatch):
    import app.auth as auth_mod

    monkeypatch.setattr(
        auth_mod.brain, "find_project_id", lambda db, *, org_id, identifier: "proj-1"
    )

    class _Req:
        headers = {"X-Vex-Project": "git@github.com:acme/widgets.git"}

    assert (
        auth_mod.resolve_project_from_request(_Req(), object(), org_id="o")
        == "proj-1"
    )


def test_resolve_project_header_fails_open(monkeypatch):
    import app.auth as auth_mod

    def _boom(db, *, org_id, identifier):
        raise RuntimeError("db down")

    monkeypatch.setattr(auth_mod.brain, "find_project_id", _boom)

    class _Req:
        headers = {"X-Vex-Project": "anything"}

    assert auth_mod.resolve_project_from_request(_Req(), object(), org_id="o") is None
```

- [ ] **Step 2: Run to verify failure**

Run: `cd services/mcp-server && python -m pytest tests/test_auth.py -k project -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement in `services/mcp-server/app/auth.py`**

Add after `DEFAULT_AGENT_ID` (line 33):

```python
import logging

from shared import brain

logger = logging.getLogger("agentguard.mcp.auth")

# Optional per-client project scoping. MCP clients have no cwd, so a
# workspace declares its project via this header (a git remote URL or
# project slug). Resolution is read-only and FAILS OPEN to org-wide:
# scoping is a relevance feature, not a security boundary (org_id is).
PROJECT_HEADER = "X-Vex-Project"


def resolve_project_from_request(request, db, *, org_id: str):
    """Resolve the X-Vex-Project header to a project id, or None.

    Read-only (never creates a project), tenant-scoped by org_id, and
    fails open: a missing header, unknown identifier, or lookup error all
    yield None (org-wide recall).
    """
    identifier = request.headers.get(PROJECT_HEADER)
    if not identifier:
        return None
    try:
        return brain.find_project_id(db, org_id=org_id, identifier=identifier)
    except Exception:
        logger.warning(
            "X-Vex-Project resolution failed for org %s; org-wide", org_id,
            exc_info=True,
        )
        return None
```

(Keep the existing `import os` etc.; consolidate imports at the top of the file per house style.)

- [ ] **Step 4: Run the auth suite**

Run: `cd services/mcp-server && python -m pytest tests/test_auth.py tests/test_auth_plan.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/mcp-server/app/auth.py services/mcp-server/tests/test_auth.py
git commit -m "feat(auth): resolve optional X-Vex-Project header (fail open)"
```

---

### Task 10: Two-tier project recall, blended score, `relevance`, `limit`

**Files:**
- Modify: `services/shared/shared/memory.py` (`MemoryRecord` line 48; `retrieve_memories_scoped` line 582)
- Modify: `services/mcp-server/app/mcp_app.py` (`_perform_recall` line 371, `_serialize_record` line 200, `recall` tool line 556)
- Modify: `services/shared/tests/test_brain_recall.py`, `services/mcp-server/tests/test_recall_tool.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `services/shared/tests/test_brain_recall.py`:

```python
def _run_recall_sql_capture(monkeypatch, **kwargs):
    import anyio

    import shared.memory as memory_mod

    captured = {"sqls": []}

    async def _embed(_):
        return [0.1, 0.2]

    monkeypatch.setattr(memory_mod, "embed_text", _embed)

    class _DB:
        def execute(self, sql, params=None):
            captured["sqls"].append((str(sql), params))

            class R:
                def fetchall(self_inner):
                    return []

            return R()

    anyio.run(
        lambda: memory_mod.retrieve_memories_scoped(_DB(), "org-1", "q", **kwargs)
    )
    return captured


def test_blended_score_in_order_by(monkeypatch):
    captured = _run_recall_sql_capture(monkeypatch)
    sql = captured["sqls"][0][0]
    assert "blended_score" in sql
    assert "0.85" in sql and "0.15" in sql


def test_project_recall_is_two_tier(monkeypatch):
    captured = _run_recall_sql_capture(monkeypatch, project_id="proj-1")
    # Tier 1: project rows only. Tier 2 (separate query): unfiled rows.
    tier1_sql = captured["sqls"][0][0]
    assert "project_id = :project_id" in tier1_sql
    assert "project_id IS NULL" not in tier1_sql
    tier2_sql = captured["sqls"][1][0]
    assert "project_id IS NULL" in tier2_sql


def test_relevance_populated_on_records(monkeypatch):
    import anyio
    import uuid
    from datetime import datetime, timezone

    import shared.memory as memory_mod

    async def _embed(_):
        return [0.1, 0.2]

    monkeypatch.setattr(memory_mod, "embed_text", _embed)
    now = datetime.now(timezone.utc)
    fake_row = (
        uuid.uuid4(), "s", "org-1", "a", None, "fact", "content", None, None,
        None, "org", "active", None, {}, now, now, None, None,
        0.91,  # relevance (1 - cosine distance), final SELECT column
    )

    class _DB:
        def execute(self, sql, params=None):
            class R:
                def fetchall(self_inner):
                    return [fake_row]

            return R()

    records = anyio.run(
        lambda: memory_mod.retrieve_memories_scoped(_DB(), "org-1", "q")
    )
    assert records[0].relevance == 0.91
```

Append to `services/mcp-server/tests/test_recall_tool.py`:

```python
def test_recall_result_includes_relevance_and_project_name():
    """_serialize_record carries relevance and project label fields."""
    import uuid
    from datetime import datetime, timezone

    from app.mcp_app import _serialize_record
    from shared.memory import MemoryRecord

    record = MemoryRecord(
        id=uuid.uuid4(),
        session_id="s",
        org_id="o",
        agent_id="a",
        memory_type="fact",
        content="x",
        created_at=datetime.now(timezone.utc),
        relevance=0.873,
    )
    out = _serialize_record(record, project_label={"slug": "widgets", "name": "Widgets"})
    assert out["relevance"] == 0.873
    assert out["project"] == {"slug": "widgets", "name": "Widgets"}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd services/shared && python -m pytest tests/test_brain_recall.py -k "blended or two_tier or relevance" -v && cd ../mcp-server && python -m pytest tests/test_recall_tool.py -k relevance -v`
Expected: FAIL.

- [ ] **Step 3: Implement `shared/memory.py` changes**

Add to `MemoryRecord` (after `project_id`, line 69):

```python
    relevance: Optional[float] = None
```

Rewrite `retrieve_memories_scoped`'s query section (everything from `embedding_str = str(embedding)` line 667 to the end of the function) as:

```python
    embedding_str = str(embedding)

    base_clauses = [
        "org_id = :org_id",
        "status = 'active'",
        "superseded_by IS NULL",
        "recall_hidden = FALSE",
    ]
    params: dict[str, Any] = {
        "org_id": org_id,
        "embedding": embedding_str,
        "limit": limit,
    }

    if scope == "org":
        base_clauses.append("scope = 'org'")
    else:  # scope == "agent" (validated above; agent_id is not None)
        base_clauses.append(
            "(scope = 'org' OR (scope = 'agent' AND agent_id = :agent_id))"
        )
        params["agent_id"] = agent_id

    if space_id is not None:
        base_clauses.append("(space_id = :space_id OR space_id IS NULL)")
        params["space_id"] = space_id

    # Blended ranking: similarity dominates, recency breaks ties and
    # demotes stale near-duplicates. recency_norm is age over a 30-day
    # window clamped to [0, 1].
    select_sql = """
        SELECT id, session_id, org_id, agent_id, execution_id,
               memory_type, content, embedding, confidence,
               sequence_number, scope, status, superseded_by,
               metadata, created_at, updated_at, space_id, project_id,
               (1 - (embedding <=> :embedding)) AS relevance,
               (0.85 * (1 - (embedding <=> :embedding))
                + 0.15 * GREATEST(
                    0, 1 - EXTRACT(EPOCH FROM (NOW() - created_at))
                           / (30 * 86400)
                  )) AS blended_score
        FROM session_memories
        WHERE {where}
        ORDER BY blended_score DESC
        LIMIT :limit
    """

    def _fetch(extra_clause: str, extra_params: dict[str, Any]) -> list:
        where_sql = "\n              AND ".join(base_clauses + [extra_clause])
        return db_session.execute(
            text(select_sql.format(where=where_sql)),
            {**params, **extra_params},
        ).fetchall()

    if project_id is not None:
        # Two-tier: same-project rows fill first; remaining slots come from
        # unfiled (org-level) rows. Rows filed under OTHER projects never
        # surface.
        tier1 = _fetch("project_id = :project_id", {"project_id": project_id})
        remaining = limit - len(tier1)
        tier2 = []
        if remaining > 0:
            tier2 = _fetch(
                "project_id IS NULL", {"limit": remaining}
            )
        rows = list(tier1) + list(tier2)
    else:
        rows = _fetch("TRUE", {})

    records = []
    for row in rows:
        record = _row_to_record(row[:18])
        record.relevance = round(float(row[18]), 3) if row[18] is not None else None
        records.append(record)
    return records
```

Note `_row_to_record(row[:18])` — the first 18 columns are the existing record shape; columns 18/19 are `relevance`/`blended_score`.

- [ ] **Step 4: Implement `mcp_app.py` changes**

`_serialize_record` (line 200) gains an optional project label and relevance:

```python
def _serialize_record(
    record: MemoryRecord,
    project_label: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "id": str(record.id),
        "memory_type": record.memory_type,
        "content": record.content,
        "scope": record.scope,
        "confidence": record.confidence,
        "relevance": record.relevance,
        "space_id": str(record.space_id) if record.space_id is not None else None,
        "project_id": (
            str(record.project_id) if record.project_id is not None else None
        ),
        "project": project_label,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "metadata": record.metadata,
    }
```

In `_perform_recall` (line 371): add `limit: int` to the signature; after project/space resolution, fall back to the header when no explicit `project` arg was given; pass `limit`; resolve a human-readable label.

```python
async def _perform_recall(
    ctx: Context,
    *,
    query: str,
    scope: str,
    space: Optional[str],
    project: Optional[str],
    limit: int = 10,
) -> dict[str, Any]:
```

after the explicit-project resolution block (line 460–468) add:

```python
        # No explicit project arg: fall back to the X-Vex-Project header
        # (fail-open — an unresolvable header means org-wide, matching the
        # read-path philosophy; explicit args keep strict warn-and-empty).
        if project is None:
            project_id = resolve_project_from_request(request, db, org_id=org_id)
```

then the retrieval call becomes:

```python
        records = await retrieve_memories_scoped(
            db,
            org_id,
            query,
            scope=scope,
            agent_id=agent_id,
            space_id=space_id,
            project_id=project_id,
            limit=min(max(limit, 1), 50),
        )
```

and the serialization picks up the project label once per response:

```python
        project_label: Optional[dict[str, Any]] = None
        if project_id is not None:
            row = db.execute(
                text(
                    "SELECT display_name FROM projects "
                    "WHERE id = :id AND org_id = :org_id"
                ),
                {"id": project_id, "org_id": org_id},
            ).fetchone()
            if row is not None:
                project_label = {"slug": project_id, "name": row.display_name}
        memories = [
            _serialize_record(r, project_label=project_label) for r in records
        ]
        return {"memories": memories, "count": len(memories)}
```

Add the imports at the top of `mcp_app.py`:

```python
from sqlalchemy import text

from app.auth import resolve_principal, resolve_principal_plan, resolve_project_from_request
```

Finally, the `recall` tool registration (line 556) gains the `limit` arg:

```python
        limit: Annotated[
            int,
            Field(
                description="Max memories to return (1-50, default 10).",
                ge=1,
                le=50,
            ),
        ] = 10,
    ) -> dict[str, Any]:
        return await _perform_recall(
            ctx, query=query, scope=scope, space=space, project=project, limit=limit
        )
```

- [ ] **Step 5: File MCP writes under the header project (spec §3)**

First the failing test — append to `services/mcp-server/tests/test_write_tools.py`:

```python
def test_write_files_under_header_project(monkeypatch):
    """A write with X-Vex-Project resolved files the row under that project."""
    import app.mcp_app as mcp_mod

    monkeypatch.setattr(
        mcp_mod, "resolve_project_from_request", lambda req, db, *, org_id: "proj-9"
    )
    seen = {}

    async def _fake_write(db, **kwargs):
        seen.update(kwargs)
        return {"stored": True, "deduplicated": False, "id": "x", "scope": "org"}

    monkeypatch.setattr(mcp_mod.brain, "write_scoped_memory", _fake_write)
    # Drive _perform_write through the same fake-Context helper the existing
    # write-tool tests in this file already use (reuse that fixture/builder),
    # with headers={"X-Vex-Project": "git@github.com:acme/widgets.git"}.
    # After the call:
    assert seen["project_id"] == "proj-9"
```

Run: `cd services/mcp-server && python -m pytest tests/test_write_tools.py -k header_project -v` — expected FAIL.

Then implement in `_perform_write` (`services/mcp-server/app/mcp_app.py`, line 94): after `session_id = brain_session_id(...)` and inside the `db` block (after the quota check), resolve and forward:

```python
        # Best-effort project filing from the X-Vex-Project header (read-only
        # resolve; fail open to unfiled — filing is relevance, not security).
        header_project_id = resolve_project_from_request(request, db, org_id=org_id)

        result = await brain.write_scoped_memory(
            db,
            org_id=org_id,
            agent_id=agent_id,
            content=content,
            memory_type=memory_type,
            session_id=session_id,
            project_id=header_project_id,
            metadata=metadata,
            scope=scope,
        )
```

Re-run: expected PASS.

- [ ] **Step 6: Run both suites**

Run: `cd services/shared && python -m pytest tests/ -v && cd ../mcp-server && python -m pytest tests/ -v`
Expected: PASS (existing recall tests may need their expected serializer keys extended with `relevance`/`project` — update those assertions, never delete them).

- [ ] **Step 7: Commit**

```bash
git add services/shared/shared/memory.py services/mcp-server/app/mcp_app.py services/shared/tests/test_brain_recall.py services/mcp-server/tests/test_recall_tool.py
git commit -m "feat(recall): two-tier project scoping, blended ranking, relevance + limit"
```

---

### Task 11: `forget` tool (MCP + REST)

**Files:**
- Modify: `services/shared/shared/memory.py` (add `retract_memory` after `supersede_memory`, line 579)
- Modify: `services/mcp-server/app/mcp_app.py` (8th tool; update `EXPECTED_TOOL_NAMES` line 70)
- Modify: `services/mcp-server/app/capture.py` (`/capture/forget` route)
- Create: `services/mcp-server/tests/test_forget_tool.py`

- [ ] **Step 1: Write the failing tests**

```python
# services/mcp-server/tests/test_forget_tool.py
"""forget: retract a memory by id with org/agent authorization."""

import uuid

import pytest

from shared.memory import retract_memory


class _Row:
    def __init__(self, scope, agent_id):
        self.scope = scope
        self.agent_id = agent_id


class FakeDB:
    def __init__(self, row=None):
        self.row = row
        self.updates = []

    def execute(self, sql, params=None):
        sql_s = str(sql)
        self.updates.append((sql_s, params))

        class R:
            def __init__(self, row):
                self._row = row

            def fetchone(self_inner):
                return self.row if "SELECT" in sql_s else None

        return R(self.row)

    def commit(self):
        pass


def test_retract_unknown_id_returns_not_found():
    db = FakeDB(row=None)
    result = retract_memory(
        db, org_id="org-1", agent_id="a1", memory_id=str(uuid.uuid4()), via="mcp"
    )
    assert result == {"forgotten": False, "error": "not_found"}


def test_retract_org_row_any_agent_succeeds():
    db = FakeDB(row=_Row(scope="org", agent_id="someone-else"))
    result = retract_memory(
        db, org_id="org-1", agent_id="a1", memory_id=str(uuid.uuid4()), via="mcp"
    )
    assert result["forgotten"] is True
    update_sql = [s for s, _ in db.updates if "UPDATE" in s][0]
    assert "status = 'retracted'" in update_sql


def test_retract_agent_row_wrong_agent_forbidden():
    db = FakeDB(row=_Row(scope="agent", agent_id="owner-agent"))
    result = retract_memory(
        db, org_id="org-1", agent_id="intruder", memory_id=str(uuid.uuid4()), via="mcp"
    )
    assert result == {"forgotten": False, "error": "forbidden"}
    assert not any("UPDATE" in s for s, _ in db.updates)


def test_retract_rejects_malformed_id():
    db = FakeDB()
    result = retract_memory(
        db, org_id="org-1", agent_id="a1", memory_id="not-a-uuid", via="mcp"
    )
    assert result == {"forgotten": False, "error": "invalid_id"}


def test_forget_is_registered_as_eighth_tool():
    from app.mcp_app import EXPECTED_TOOL_NAMES

    assert "forget" in EXPECTED_TOOL_NAMES
    assert len(EXPECTED_TOOL_NAMES) == 8
```

- [ ] **Step 2: Run to verify failure**

Run: `cd services/mcp-server && python -m pytest tests/test_forget_tool.py -v`
Expected: FAIL — `ImportError: retract_memory`.

- [ ] **Step 3: Implement `retract_memory` in `services/shared/shared/memory.py`** (after `supersede_memory`, line 579)

```python
def retract_memory(
    db_session: Any,
    *,
    org_id: str,
    agent_id: str,
    memory_id: str,
    via: str,
) -> dict:
    """Explicitly retract a memory (the `forget` tool's storage primitive).

    Authorization ladder (org_id always enforced):
      * unknown / cross-org id → not_found (indistinguishable by design)
      * scope='agent' row owned by another agent → forbidden
      * otherwise → status='retracted' + audit metadata; recall never
        returns non-'active' rows, so the memory disappears immediately.

    Synchronous (inline psycopg2 I/O), matching the house pattern.
    """
    try:
        parsed = uuid.UUID(memory_id)
    except (ValueError, AttributeError, TypeError):
        return {"forgotten": False, "error": "invalid_id"}

    row = db_session.execute(
        text(
            """
            SELECT scope, agent_id FROM session_memories
            WHERE id = :id AND org_id = :org_id AND status = 'active'
            LIMIT 1
            """
        ),
        {"id": str(parsed), "org_id": org_id},
    ).fetchone()
    if row is None:
        return {"forgotten": False, "error": "not_found"}
    if row.scope == "agent" and row.agent_id != agent_id:
        return {"forgotten": False, "error": "forbidden"}

    db_session.execute(
        text(
            """
            UPDATE session_memories
            SET status = 'retracted',
                metadata = COALESCE(metadata, '{}'::jsonb)
                           || jsonb_build_object(
                               'retracted_by', :agent_id::text,
                               'retracted_at', NOW()::text,
                               'retracted_via', :via::text
                           ),
                updated_at = NOW()
            WHERE id = :id AND org_id = :org_id
            """
        ),
        {"id": str(parsed), "org_id": org_id, "agent_id": agent_id, "via": via},
    )
    db_session.commit()
    return {"forgotten": True, "id": str(parsed)}
```

- [ ] **Step 4: Register the MCP tool in `services/mcp-server/app/mcp_app.py`**

Update the canonical list (line 70):

```python
EXPECTED_TOOL_NAMES: tuple[str, ...] = (
    "recall",
    "remember",
    "observe",
    "plan",
    "decide",
    "note",
    "space",
    "forget",
)
```

Add the import: `from shared.memory import MemoryRecord, retract_memory, retrieve_memories_scoped`

Add a `_perform_forget` body (after `_perform_recall`):

```python
async def _perform_forget(ctx: Context, *, memory_id: str) -> dict[str, Any]:
    """Shared body for the `forget` tool: retract one memory by id."""
    request = ctx.request_context.request
    if request is None:
        logger.error("brain forget got no transport request; cannot authenticate")
        return {"forgotten": False, "error": "unauthorized"}
    try:
        org_id, agent_id = resolve_principal(request)
    except HTTPException as exc:
        return {"forgotten": False, "error": "unauthorized", "detail": str(exc.detail)}

    db = SessionLocal()
    try:
        return retract_memory(
            db, org_id=org_id, agent_id=agent_id, memory_id=memory_id, via="mcp"
        )
    except Exception:
        logger.exception("brain forget failed")
        return {"forgotten": False, "error": "forget_failed"}
    finally:
        db.close()
```

And register inside `_register_tools` (after the `space` tool):

```python
    @mcp.tool(
        name="forget",
        description=(
            "Retract a memory by id (from a prior recall result) so it never "
            "surfaces again. Arg: memory_id."
        ),
    )
    async def forget(
        ctx: Context,
        memory_id: Annotated[
            str, Field(description="The id of the memory to retract.")
        ],
    ) -> dict[str, Any]:
        return await _perform_forget(ctx, memory_id=memory_id)
```

- [ ] **Step 5: Add the REST mirror in `services/mcp-server/app/capture.py`** (after `capture_recall`)

```python
class ForgetRequest(BaseModel):
    """Retract one memory by id (hook-client mirror of the MCP tool)."""

    memory_id: str = Field(min_length=1, description="Memory id to retract.")


@router.post("/forget")
async def capture_forget(
    req: ForgetRequest,
    principal: tuple[str, str] = Depends(resolve_principal_dep),
) -> dict[str, Any]:
    """Retract a memory. 404 unknown/cross-org, 403 cross-agent."""
    org_id, agent_id = principal
    db = SessionLocal()
    try:
        result = await anyio.to_thread.run_sync(
            lambda: retract_memory(
                db, org_id=org_id, agent_id=agent_id,
                memory_id=req.memory_id, via="hook",
            )
        )
        if result.get("error") in ("not_found", "invalid_id"):
            raise HTTPException(status_code=404, detail=result["error"])
        if result.get("error") == "forbidden":
            raise HTTPException(status_code=403, detail="forbidden")
        return result
    finally:
        db.close()
```

with `retract_memory` added to the capture imports from `shared.memory`.

- [ ] **Step 6: Run the suites**

Run: `cd services/mcp-server && python -m pytest tests/test_forget_tool.py tests/test_mcp_protocol.py tests/test_capture.py -v`
Expected: PASS. (`test_mcp_protocol.py` asserts the registered tool set matches `EXPECTED_TOOL_NAMES` — it now expects 8.)

- [ ] **Step 7: Commit**

```bash
git add services/shared/shared/memory.py services/mcp-server/app/mcp_app.py services/mcp-server/app/capture.py services/mcp-server/tests/test_forget_tool.py
git commit -m "feat(forget): retract memories by id via MCP tool and REST mirror"
```

---

### Task 12: Curator reconciliation (supersede contradicted facts)

**Files:**
- Modify: `services/mcp-server/app/curator.py`
- Modify: `services/mcp-server/tests/test_curator.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `services/mcp-server/tests/test_curator.py`:

```python
@pytest.mark.anyio
async def test_reconcile_supersedes_direct_contradiction(monkeypatch):
    import app.curator as curator_mod

    superseded = []

    def _fake_supersede(db, old_id, new_id):
        superseded.append((old_id, new_id))

    async def _fake_judge(prompt, system=None):
        return {"contradicts": True}

    monkeypatch.setattr(curator_mod, "supersede_memory", _fake_supersede)
    monkeypatch.setattr(curator_mod, "call_llm", _fake_judge)

    import uuid

    old_id, new_id = uuid.uuid4(), uuid.uuid4()

    class _DB:
        def execute(self, sql, params=None):
            class R:
                def fetchall(self_inner):
                    class Row:
                        id = old_id
                        content = "Stripe MCP is connected to LoopReel sandbox"

                    return [Row()]

            return R()

    count = await curator_mod.reconcile_fact(
        _DB(),
        org_id="org-1",
        new_fact_id=new_id,
        new_fact_content="Stripe MCP is connected to the MoonForge account",
        embedding=[0.1, 0.2],
    )
    assert count == 1
    assert superseded == [(old_id, new_id)]


@pytest.mark.anyio
async def test_reconcile_related_but_not_contradicting_is_untouched(monkeypatch):
    import app.curator as curator_mod

    superseded = []
    monkeypatch.setattr(
        curator_mod, "supersede_memory", lambda db, o, n: superseded.append((o, n))
    )

    async def _fake_judge(prompt, system=None):
        return {"contradicts": False}

    monkeypatch.setattr(curator_mod, "call_llm", _fake_judge)

    import uuid

    class _DB:
        def execute(self, sql, params=None):
            class R:
                def fetchall(self_inner):
                    class Row:
                        id = uuid.uuid4()
                        content = "The team uses Stripe for billing"

                    return [Row()]

            return R()

    count = await curator_mod.reconcile_fact(
        _DB(),
        org_id="org-1",
        new_fact_id=uuid.uuid4(),
        new_fact_content="Stripe MCP is connected to the MoonForge account",
        embedding=[0.1, 0.2],
    )
    assert count == 0
    assert superseded == []


@pytest.mark.anyio
async def test_reconcile_judge_error_skips_pair(monkeypatch):
    import app.curator as curator_mod

    async def _boom(prompt, system=None):
        return None  # LLM failure path of call_llm

    monkeypatch.setattr(curator_mod, "call_llm", _boom)

    import uuid

    class _DB:
        def execute(self, sql, params=None):
            class R:
                def fetchall(self_inner):
                    class Row:
                        id = uuid.uuid4()
                        content = "old fact"

                    return [Row()]

            return R()

    count = await curator_mod.reconcile_fact(
        _DB(),
        org_id="org-1",
        new_fact_id=uuid.uuid4(),
        new_fact_content="new fact",
        embedding=[0.1],
    )
    assert count == 0  # skipped, no raise
```

- [ ] **Step 2: Run to verify failure**

Run: `cd services/mcp-server && python -m pytest tests/test_curator.py -k reconcile -v`
Expected: FAIL — `AttributeError: reconcile_fact`.

- [ ] **Step 3: Implement in `services/mcp-server/app/curator.py`**

Extend the imports (line 58):

```python
import os

from shared.brain import write_scoped_memory
from shared.memory import call_llm, embed_text, extract_facts, generate_summary, supersede_memory
```

Add after `DEFAULT_CURATOR_AGENT_ID` (line 81):

```python
# Reconciliation: a new fact supersedes an older one only on DIRECT
# contradiction. Conservative by design — a stale fact surviving is
# acceptable; a live fact hidden is not.
RECONCILE_SIMILARITY_THRESHOLD = 0.85
RECONCILE_CANDIDATES_PER_FACT = 5


def _reconcile_budget() -> int:
    """Max LLM contradiction judgments per curator tick (cost bound)."""
    return int(os.environ.get("CURATOR_RECONCILE_MAX", "50"))


async def reconcile_fact(
    db_session: Any,
    *,
    org_id: str,
    new_fact_id: Any,
    new_fact_content: str,
    embedding: list[float],
) -> int:
    """Supersede older active facts that the new fact directly contradicts.

    Vector-searches the org's older active visible facts within
    RECONCILE_SIMILARITY_THRESHOLD of the new fact (top
    RECONCILE_CANDIDATES_PER_FACT), asks the LLM judge per candidate, and
    marks contradicted rows superseded_by the new fact. A judge failure
    skips that pair (never raises). Returns the number superseded.
    """
    rows = db_session.execute(
        text(
            """
            SELECT id, content
            FROM session_memories
            WHERE org_id = :org_id
              AND scope = 'org'
              AND status = 'active'
              AND memory_type = 'fact'
              AND id != :new_id
              AND 1 - (embedding <=> :embedding) >= :threshold
            ORDER BY embedding <=> :embedding
            LIMIT :k
            """
        ),
        {
            "org_id": org_id,
            "new_id": str(new_fact_id),
            "embedding": str(embedding),
            "threshold": RECONCILE_SIMILARITY_THRESHOLD,
            "k": RECONCILE_CANDIDATES_PER_FACT,
        },
    ).fetchall()

    superseded = 0
    for row in rows:
        prompt = (
            "Two facts from a team knowledge base:\n"
            f'OLDER: "{row.content}"\n'
            f'NEWER: "{new_fact_content}"\n\n'
            "Does the NEWER fact DIRECTLY contradict the OLDER one — i.e. "
            "they cannot both be true now? Merely related, overlapping, or "
            "complementary facts do NOT contradict. Respond with JSON: "
            '{"contradicts": true|false}'
        )
        try:
            verdict = await call_llm(prompt)
        except Exception:
            logger.warning("reconcile: judge raised; skipping pair", exc_info=True)
            continue
        if verdict and verdict.get("contradicts") is True:
            supersede_memory(db_session, row.id, new_fact_id)
            superseded += 1
    return superseded
```

Then wire it into `_synthesize_and_store`: change `_store_one` to return the stored id (or None) and reconcile newly stored facts. Replace `_store_one`'s final lines:

```python
    except Exception:
        logger.warning(
            "curator: write_scoped_memory raised for org %s (%s); skipping",
            org_id,
            memory_type,
            exc_info=True,
        )
        return None
    return result.get("id") if result.get("stored") else None
```

(adjust its return-type docstring: "return the new row id when newly stored, else None") and update both call sites in `_synthesize_and_store` from `if await _store_one(...)` to capture the id. The facts loop becomes:

```python
    reconcile_budget = _reconcile_budget()
    for fact in facts:
        if not fact:
            continue
        new_id = await _store_one(
            db_session,
            org_id=org_id,
            agent_id=agent_id,
            content=fact,
            memory_type="fact",
            session_id=session_id,
            metadata=base_meta,
            space_id=space_id,
            project_id=project_id,
        )
        if new_id:
            stored += 1
            if reconcile_budget > 0:
                fact_embedding = await embed_text(fact)
                if fact_embedding is not None:
                    done = await reconcile_fact(
                        db_session,
                        org_id=org_id,
                        new_fact_id=new_id,
                        new_fact_content=fact,
                        embedding=fact_embedding,
                    )
                    reconcile_budget -= max(done, 1)
```

The summary branch changes the same way (`new_id = await _store_one(...)`; `if new_id: stored += 1` — summaries are NOT reconciled).

- [ ] **Step 4: Run the curator suite**

Run: `cd services/mcp-server && python -m pytest tests/test_curator.py -v`
Expected: PASS, including all pre-existing curator tests (the `_store_one` return-type change is internal; existing tests asserting tick result counts still pass because truthiness of ids preserves the counting behavior).

- [ ] **Step 5: Commit**

```bash
git add services/mcp-server/app/curator.py services/mcp-server/tests/test_curator.py
git commit -m "feat(curator): bounded reconciliation pass supersedes contradicted facts"
```

---

### Task 13: Full verification + rollout checklist

**Files:** none created — verification only.

- [ ] **Step 1: Run both full test suites**

Run: `cd services/shared && python -m pytest tests/ -v && cd ../mcp-server && python -m pytest tests/ -v`
Expected: ALL PASS.

- [ ] **Step 2: Coverage check (≥80% on the changed modules)**

Run: `cd services/shared && python -m pytest tests/ --cov=shared.redaction --cov=shared.brain --cov=shared.memory --cov-report=term-missing && cd ../mcp-server && python -m pytest tests/ --cov=app --cov-report=term-missing`
Expected: ≥80% on `shared/redaction.py`, the modified functions in `brain.py`/`memory.py`, and `app/*` overall. Add targeted tests for any uncovered branch in the new code.

- [ ] **Step 3: Lint**

Run: `ruff check services/shared/shared/redaction.py services/shared/shared/brain.py services/shared/shared/memory.py services/mcp-server/app/ services/scripts/run_redaction_backfill.py`
Expected: clean.

- [ ] **Step 4: Smoke the MCP contract locally**

Run: `cd services/mcp-server && python -m pytest tests/test_mcp_protocol.py tests/test_e2e.py -v`
Expected: PASS — `tools/list` advertises 8 tools including `forget`; `recall` schema includes `limit`.

- [ ] **Step 5: Commit any test additions, then walk the rollout checklist (production, in order)**

1. Deploy redaction + prompt hardening (Tasks 1–4). Verify a test write through the live MCP endpoint returns `"redacted": true` for a dummy `sk_test_...` value.
2. `python -m scripts.run_redaction_backfill` (dry-run) → review per-org report → `--apply`. Verify the two known leaked keys no longer appear in a recall.
3. **Rotate the two exposed Stripe test keys** (MoonForge + LoopReel sandbox dashboards).
4. `alembic upgrade head` (migration 026), deploy Tasks 7–8. Verify a recall no longer returns raw `Used tool ...` observations.
5. Deploy Tasks 9–12. Configure one client with `X-Vex-Project` and verify cross-project memories stop surfacing; `forget` a junk memory and verify it disappears from recall.
