# Proactive-Write Nudge Layer (Subsystem A) — Design

**Date:** 2026-06-11
**Status:** Approved (design review with product owner)
**Scope:** vex_engine MCP server only (`mcp_app.py`). Hosted-client-facing.

## Problem

Klio only stores what an agent explicitly writes via `remember`/`observe`/
`note`/`plan`/`decide`. On hosted clients we don't control (claude.ai,
ChatGPT connectors) the agent treats Klio as read-mostly: it calls `recall`
when asked, but rarely writes during normal conversation. Proven concretely
this session — a claude.ai chat about the user's Hive.co PM interview was
never persisted (direct content search of all orgs found zero interview
memories), so a later `recall` returned nothing. The retrieval engine is
sound (write→recall→forget round-trips at 0.91 relevance); the gap is that
nothing writes.

## Constraint (the hard line)

An MCP server **cannot force the host model to call a tool** and cannot see
the conversation. For agents we don't control, the only lever is
**in-protocol influence**: the server `instructions` (sent once at
`initialize`), tool descriptions, and the **text returned in a tool result**
(read by the agent every time it calls a tool). This subsystem maximizes that
lever. It is honest best-effort — it raises the write rate, it cannot
guarantee a write. The *guarantee* path (SDK/middleware auto-capture for
agents that integrate code) is **Subsystem B, a separate spec.**

## Decisions (design review)

| Topic | Decision |
|---|---|
| Mechanism | Recall-result write-nudge + hardened server instructions/tool descriptions |
| Cadence | Append a short nudge to EVERY recall result (lightweight, stateless, always-fresh) |
| What to write | Durable facts/decisions/preferences ONLY — explicitly NOT tool calls/file reads/transient steps (protects the just-shipped signal-quality work) |
| New tool? | No — no new tool, no per-session state, no forced/faked writes |

## 1. Recall-result write-nudge

`services/mcp-server/app/mcp_app.py`, `_perform_recall`: the success return
dict gains a `guidance` field, present on every recall result:

```
"If this conversation produced durable facts, decisions, or preferences,
persist them now with remember / decide / note. Save durable knowledge —
not tool calls, file reads, or transient steps."
```

- Defined as a module constant `RECALL_WRITE_NUDGE` (single source of truth,
  testable).
- Added to the success payload alongside `memories`/`count`. The
  warning/empty and error payloads also carry it (an empty recall is exactly
  when the agent most needs prompting to contribute).
- Stateless and lightweight: one sentence, every call. No throttling, no
  per-session memory (which the stateless MCP transport can't cleanly hold).
- Reaches the agent mid-conversation, refreshed each recall — unlike the
  `initialize`-time instructions, which get buried in a long chat (the
  interview-test failure mode).

The nudge wording deliberately steers toward durable memory and away from
tool-call/step noise, reinforcing the signal-quality philosophy rather than
re-creating the noise the cleanup just removed.

## 2. Hardened server instructions + write-tool descriptions

`SERVER_INSTRUCTIONS` is rewritten to state an explicit **when-to-write
protocol**, proactively framed:

> Klio is the team's shared memory. **Proactively** persist durable knowledge
> as it arises — without being asked: when the user states a durable fact or
> preference, when a decision is made, or when a plan is set, call the matching
> write tool (`remember`/`decide`/`plan`/`note`). Do NOT persist tool calls,
> file reads, or transient steps. Use `recall` before acting; use `forget` to
> retract. (Plus the existing X-Vex-Agent header guidance, retained.)

Each write tool's description shifts from passive to active, with an example
that mirrors the real miss:

- `remember`: "Store a durable fact or preference for the team. Call this
  proactively whenever the user reveals one — e.g. 'I'm interviewing at
  Hive.co for a senior PM role' — not only when asked to remember."
- `decide`: "Record a decision + rationale as soon as one is made."
- `plan`: "Record a forward-looking plan when one is set."
- `note`: "Free-form note for durable context that isn't a fact/decision/plan."
- `observe`: keep as-is (deliberate session logging; not the proactive target).

## 3. Explicit non-goals

- No new tool (MCP contract stays at 8 tools).
- No per-session/throttle state.
- No attempt to force, fake, or infer a write from a conversation the server
  can't see.
- No change to write/recall storage behavior, scoping, redaction, or the
  curator — purely the advisory surface.

## 4. Error handling

No new failure modes. The `guidance` field is a static constant; it cannot
fail. If `_perform_recall` errors, the existing error path is unchanged
(guidance may be included on the error dict for consistency, but is never
required).

## 5. Testing

- **Unit:** `RECALL_WRITE_NUDGE` constant contains the durable-vs-noise steer
  ("durable", "not tool calls"); `_perform_recall` success result includes
  `guidance` equal to the constant; empty/warning result also includes it.
- **Contract:** `SERVER_INSTRUCTIONS` contains the proactive when-to-write
  protocol (asserts on key phrases: "proactively", "decision", "do not persist
  tool calls"); write-tool descriptions contain the proactive framing;
  `tools/list` still advertises exactly 8 tools (no tool added).
- **Regression:** existing recall/serialization tests updated additively for
  the new `guidance` key (never weakened); full mcp-server suite green.

## 6. Honest success criterion + measurement

This raises the proactive-write rate on hosted clients; it cannot guarantee
it. After deploy, the read is simple: re-run the interview scenario in
claude.ai (state a durable fact in normal conversation, then in a fresh
session `recall` it). If Claude now writes it proactively, the nudge works and
informs how hard Subsystem B needs to be. If it still doesn't, that's strong
evidence the only reliable path is code-level capture (Subsystem B), and we
prioritize accordingly.

## Out of scope (→ Subsystem B, separate spec)

- SDK/middleware auto-capture for developer-built agents (LangGraph/CrewAI/
  custom) extending the Python/TS `@watch`/`trace` SDKs — the *guaranteed*
  write path for agents that integrate code.
- Any claude.ai-side auto-capture (not possible; Anthropic-controlled runtime).
- MCP `sampling`-based server-driven summarization (thin host support; revisit
  only if the nudge underperforms and sampling becomes broadly available).
