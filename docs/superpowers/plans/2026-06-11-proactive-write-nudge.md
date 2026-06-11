# Proactive-Write Nudge Layer (Subsystem A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved spec at `docs/superpowers/specs/2026-06-11-proactive-write-nudge-design.md`: make hosted MCP agents write memory more often by (1) appending a `guidance` write-nudge to every `recall` result and (2) hardening `SERVER_INSTRUCTIONS` and the write-tool descriptions into a proactive when-to-write protocol — steering toward durable facts/decisions/preferences and away from tool-call/step noise.

**Architecture:** Engine-only, one file (`services/mcp-server/app/mcp_app.py`). No new tool, no state, no storage change — purely the advisory surface the host model reads (server `instructions` at `initialize`, tool descriptions, and tool-result text). Best-effort by design: an MCP server cannot force a write; this maximizes the only lever for agents we don't control.

**Tech Stack:** Python 3.11, FastMCP. Test from repo root: `.venv/bin/python -m pytest services/mcp-server/tests -q`. Deploys via `git push origin main` → Railway. Conventional commits, no co-author lines.

---

## File structure

| File | Responsibility |
|---|---|
| Modify `services/mcp-server/app/mcp_app.py` | `RECALL_WRITE_NUDGE` constant; add `guidance` to all `_perform_recall` returns; rewrite `SERVER_INSTRUCTIONS` + write-tool descriptions |
| Modify `services/mcp-server/tests/test_recall_tool.py` | Assert `guidance` present on recall results |
| Modify `services/mcp-server/tests/test_mcp_protocol.py` | Assert proactive protocol in instructions; 8 tools unchanged |

---

### Task 1: `RECALL_WRITE_NUDGE` constant + guidance on every recall return

**Files:**
- Modify: `services/mcp-server/app/mcp_app.py` (`_perform_recall` returns at the unknown-space, unknown-project, success, and error paths; constant near `_PROJECT_ANY`)
- Modify: `services/mcp-server/tests/test_recall_tool.py`

- [ ] **Step 1: Write the failing tests** — append to `services/mcp-server/tests/test_recall_tool.py`:

```python
def test_recall_write_nudge_constant_steers_durable_not_noise():
    from app.mcp_app import RECALL_WRITE_NUDGE

    low = RECALL_WRITE_NUDGE.lower()
    assert "durable" in low
    # Must explicitly steer AWAY from tool-call/step noise (protects the
    # signal-quality cleanup we just shipped).
    assert "tool call" in low or "tool calls" in low
    assert "remember" in low


def test_recall_serializer_unchanged_and_guidance_is_separate():
    """guidance is a top-level recall-result field, not per-memory."""
    import uuid
    from datetime import datetime, timezone

    from app.mcp_app import _serialize_record
    from shared.memory import MemoryRecord

    out = _serialize_record(
        MemoryRecord(
            id=uuid.uuid4(), session_id="s", org_id="o", agent_id="a",
            memory_type="fact", content="x",
            created_at=datetime.now(timezone.utc),
        )
    )
    assert "guidance" not in out  # nudge lives on the envelope, not each record
```

(These two run without a live DB. The envelope-level `guidance` assertions on the actual `_perform_recall` returns are covered by the existing recall-tool tests that build a fake Context — Step 4 extends those; check `test_recall_tool.py` for the existing `_perform_recall`/fake-ctx harness and add envelope assertions there. If no such harness exists, these two unit tests plus the contract test in Task 2 are sufficient — `_perform_recall`'s returns are simple dict literals verified by reading the diff.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest services/mcp-server/tests/test_recall_tool.py -k "write_nudge or guidance_is_separate" -v`
Expected: FAIL — `ImportError: cannot import name 'RECALL_WRITE_NUDGE'`.

- [ ] **Step 3: Add the constant.** In `services/mcp-server/app/mcp_app.py`, after the `_PROJECT_ANY` definition (near line 197), add:

```python
# Appended to every recall result as a top-level ``guidance`` field. On hosted
# clients we cannot force a write, so we remind the agent — at the moment it is
# already engaged with Klio (each recall), refreshed rather than only at
# initialize — to persist what matters. Deliberately steers toward DURABLE
# knowledge and away from tool-call/step noise so it reinforces (not undoes)
# the signal-quality work.
RECALL_WRITE_NUDGE = (
    "If this conversation produced durable facts, decisions, or preferences, "
    "persist them now with remember / decide / note. Save durable knowledge "
    "— not tool calls, file reads, or transient steps."
)
```

- [ ] **Step 4: Add `guidance` to every `_perform_recall` return path.** In `_perform_recall`:

The unknown-space return (currently `{"memories": [], "count": 0, "warning": f"unknown space: {space}"}`) becomes:

```python
                return {
                    "memories": [],
                    "count": 0,
                    "warning": f"unknown space: {space}",
                    "guidance": RECALL_WRITE_NUDGE,
                }
```

The unknown-project return becomes:

```python
                return {
                    "memories": [],
                    "count": 0,
                    "warning": f"unknown project: {project}",
                    "guidance": RECALL_WRITE_NUDGE,
                }
```

The success return (currently `return {"memories": memories, "count": len(memories)}`) becomes:

```python
        return {
            "memories": memories,
            "count": len(memories),
            "guidance": RECALL_WRITE_NUDGE,
        }
```

The error return (currently `return {"error": "recall_failed"}`) becomes:

```python
        return {"error": "recall_failed", "guidance": RECALL_WRITE_NUDGE}
```

(Also add `"guidance": RECALL_WRITE_NUDGE` to the `unauthorized` and `quota_exceeded` early returns in `_perform_recall` for consistency — find them in the same function and append the key. They are dict literals; do not change their other fields.)

- [ ] **Step 5: Run** the new tests + full recall suite: `.venv/bin/python -m pytest services/mcp-server/tests/test_recall_tool.py -v` → PASS. If any existing recall test asserts exact dict equality on a `_perform_recall` return, update it additively to include `guidance` (never weaken other assertions).

- [ ] **Step 6: Commit**

```bash
git add services/mcp-server/app/mcp_app.py services/mcp-server/tests/test_recall_tool.py
git commit -m "feat(mcp): append write-nudge guidance to every recall result"
```

---

### Task 2: Hardened SERVER_INSTRUCTIONS + write-tool descriptions

**Files:**
- Modify: `services/mcp-server/app/mcp_app.py` (`SERVER_INSTRUCTIONS` ~line 83; write-tool `description=` strings in `_register_tools`)
- Modify: `services/mcp-server/tests/test_mcp_protocol.py`

- [ ] **Step 1: Write the failing tests** — append to `services/mcp-server/tests/test_mcp_protocol.py`:

```python
def test_server_instructions_state_proactive_write_protocol():
    from app.mcp_app import SERVER_INSTRUCTIONS

    low = SERVER_INSTRUCTIONS.lower()
    assert "proactively" in low
    assert "decision" in low or "decide" in low
    # Steers away from noise, consistent with the recall nudge + cleanup.
    assert "not" in low and ("tool call" in low or "tool calls" in low)


async def test_remember_description_is_proactive():
    """The remember tool must instruct proactive writing, not passive storage."""
    server = build_mcp_server()
    tools = await server.list_tools()
    remember = next(t for t in tools if t.name == "remember")
    desc = remember.description.lower()
    assert "proactively" in desc or "whenever" in desc
```

(Adapt to the suite's actual server-construction helper — `test_mcp_protocol.py` already builds a server and lists tools for `test_tools_list_returns_eight`; reuse that exact pattern. If `build_mcp_server`/`list_tools` is accessed differently there, match it.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest services/mcp-server/tests/test_mcp_protocol.py -k "proactive" -v`
Expected: FAIL.

- [ ] **Step 3: Rewrite `SERVER_INSTRUCTIONS`** in `services/mcp-server/app/mcp_app.py` to:

```python
SERVER_INSTRUCTIONS = (
    "Klio is the team's shared memory for AI agents. Memory is scoped to the "
    "org resolved from your X-Vex-Key and to an agent resolved from the "
    "X-Vex-Agent header; set X-Vex-Agent to a stable per-agent identifier "
    "(clients that omit it share one default agent). "
    "Use `recall` to retrieve relevant context before acting. "
    "PROACTIVELY persist durable knowledge as it arises — without being asked: "
    "when the user states a durable fact or preference, call `remember`; when a "
    "decision is made, call `decide`; when a plan is set, call `plan`; for other "
    "durable context, `note`. Do NOT persist tool calls, file reads, or "
    "transient steps. Use `forget` to retract a memory by id."
)
```

- [ ] **Step 4: Sharpen the write-tool descriptions.** Read the current `description=` strings in `_register_tools` and replace them (match each tool's current string exactly as `old_string`):

- `remember` → `"Store a durable fact or preference for the team. Call this PROACTIVELY whenever the user reveals one — e.g. 'I'm interviewing at Hive.co for a senior PM role' — not only when explicitly asked to remember. Arg: content."`
- `decide` → `"Record a decision and its rationale as soon as one is made. Args: content, optional rationale."`
- `plan` → `"Record a forward-looking plan when one is set. Arg: content."`
- `note` → `"Free-form note for durable context that isn't a fact, decision, or plan. Arg: content."`
- `observe` → leave unchanged (deliberate session logging; not the proactive target).
- `recall` → leave unchanged.

(Preserve each tool's `name=` and signature; only the `description=` text changes. The `remember` description retains its X-Vex-Agent note if present — append the proactive framing, don't drop existing guidance.)

- [ ] **Step 5: Run** `.venv/bin/python -m pytest services/mcp-server/tests/test_mcp_protocol.py -v` → PASS, including `test_tools_list_returns_eight` (no tool added/removed) and `test_tool_has_description_and_schema`.

- [ ] **Step 6: Commit**

```bash
git add services/mcp-server/app/mcp_app.py services/mcp-server/tests/test_mcp_protocol.py
git commit -m "feat(mcp): proactive when-to-write protocol in server instructions + tool descriptions"
```

---

### Task 3: Verification

**Files:** none — verification only.

- [ ] **Step 1: Full mcp-server suite** — `.venv/bin/python -m pytest services/mcp-server/tests -q` → ALL PASS (no regressions; existing recall tests updated additively for `guidance`).
- [ ] **Step 2: Shared suite unaffected** — `cd services/shared && python3 -m pytest tests/ -q` → green (no shared changes, sanity only).
- [ ] **Step 3: Lint** — `ruff check services/mcp-server/app/mcp_app.py services/mcp-server/tests/test_recall_tool.py services/mcp-server/tests/test_mcp_protocol.py` and `ruff format --check services/` → clean.
- [ ] **Step 4: MCP contract smoke** — `.venv/bin/python -m pytest services/mcp-server/tests/test_mcp_protocol.py -v` → 8 tools advertised, instructions carry the proactive protocol.
- [ ] **Step 5: Rollout + measurement.** `git push origin main` → Railway rebuilds. Then the measurement from the spec: in claude.ai, state a durable fact in normal conversation (e.g. "I'm interviewing at Hive.co for a senior PM role") WITHOUT asking it to remember; in a fresh session, `recall` it. If Claude now writes proactively → the nudge works. If not → strong signal that Subsystem B (SDK/middleware guaranteed capture) is the necessary path, and prioritize it.
