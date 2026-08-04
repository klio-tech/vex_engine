# Klio `init` — onboarding that fills the workplace on day one

**Date:** 2026-08-03
**Status:** Design, approved in conversation.
**Builds on:** the user silo (2026-08-01), the curator as graph builder (2026-08-02), capture redaction.

---

## The problem

Klio is worthless on day one. A new user connects the MCP server, calls
`recall`, and gets nothing back — the workplace is empty, so the first week
feels like paying rent on an empty office. Today the only way to seed it is
to work for days while passive capture accumulates, or to paste context in
by hand. Both lose users before the product has shown anything.

Anthropic validated the fix in March 2026: their ChatGPT-import prompt
exists purely to kill this cold start. Klio can do the same, but into
neutral, cross-vendor ground.

## The UX contract (fixed, not negotiable)

> The user adds the MCP server, then says to their agent:
> **"Use the Klio init tool to onboard me on Klio and set up."**
> Everything after that sentence is the agent and the server talking.

No CLI step, no dashboard visit, no form. The tool description must be
written so that "onboard me", "set me up", "get me started with Klio" all
route the agent to this tool.

**Naming.** The MCP tool is `init`. The npm launcher
(`npx @klio-tech/klio init`) is a different surface; docs refer to it as
**the launcher** from now on to keep support conversations unambiguous.
This collision was flagged and accepted — the user-facing phrase "Klio
init" wins.

## What `init` can and cannot be

An MCP server cannot read the user's chat history, their repo, or their
other tools. The server only ever sees what the agent sends it. So `init`
is **not a scraper** — it is a **structured interview the server conducts
with the agent**, where the agent does the reading and the server does the
judging, deduplication, redaction, and filing.

This is the right division anyway: the agent has the context; the server
has the guardrails.

## Tool shape — one tool, three phases

`init(phase, payload?)` — stateful across one onboarding run.

### Phase `start`

Agent calls `init(phase="start")`. Server returns a **playbook**: an
ordered list of collection steps for the agent to perform, plus the rules
for what qualifies. The playbook is versioned server-side so it can improve
without a client update.

v1 playbook steps:

1. **The conversation.** Distil what you already know from this session:
   who the user is, what they are building, decisions already made,
   preferences already stated. Durable facts only — no transcripts, no
   play-by-play.
2. **The repo, if there is one.** Read `CLAUDE.md`, `AGENTS.md`,
   `.cursor/rules`, `README`, and the last ~50 commit subjects. Extract
   conventions, architecture decisions, active plans, and standing rules.
   Skip anything generated, boilerplate, or stale.
3. **Ask the user two questions, at most.** "Anything your agents keep
   getting wrong that I should record?" and "Anything here that should NOT
   be remembered?" The second is a consent gate, not decoration.

The playbook explicitly tells the agent what NOT to submit: secrets, keys,
personal data, tool-call logs, file contents, anything the user marked
off-limits.

### Phase `submit`

Agent calls `init(phase="submit", candidates=[...])`, one or more times.
Each candidate: `{content, kind, source}` where `kind` is one of the
existing RAW types (`observation | note | plan | decision | memory`) and
`source` is `conversation | repo | user`.

Server-side, every candidate goes through **the same door as capture** —
no bypass:

- secret/PII redaction exactly as `capture` applies it;
- dedup against existing memories (same threshold as `remember`);
- provenance: `EXTRACTED` when `source="user"` or the candidate quotes the
  user's stated fact; `INFERRED` for everything the agent derived;
- scope: **`private`, always, in v1.** Bulk-imported context is exactly
  the material that must not accidentally go org-wide on day one.
  Promotion to project/org happens later, deliberately, via `share`.
- project: resolved via the existing `find_project_id` path when the repo
  step supplied identifiers; otherwise unfiled.

The response tells the agent what was accepted, deduplicated, redacted, or
refused — so the agent can report honestly to the user.

### Phase `finish`

Server returns a summary: counts by kind and source, what was redacted
(kinds only, never contents), and a one-paragraph "here is what Klio now
knows" the agent relays verbatim. It also writes one marker memory
(`kind=note`, content "Onboarded via init on <date>, N memories seeded")
so re-runs are detectable.

## Caps and quality — the god-entity lesson, doubled

An eager agent will happily submit 400 "facts" from a README. Hard limits,
server-enforced:

| Limit | Value | On overflow |
|---|---|---|
| candidates per submit call | 25 | reject the call, tell the agent to prioritise |
| accepted candidates per run | 60 | refuse further submits with a clear message |
| content length per candidate | 2,000 chars | reject the candidate, not the batch |
| runs per user per day | 3 | refuse `start` |

The curator then distils the accepted batch on its normal tick — `init`
seeds RAW fuel; it does not write facts, entities, or edges directly. The
graph grows from onboarding the same way it grows from everything else.

## Security invariants (tested, mutation-checked)

1. **No redaction bypass.** `init` submits flow through the same
   redaction/strip path as `capture`. Mutation: removing the redaction
   call from the `init` path must fail ≥2 tests (one secret, one PII).
2. **Private only.** No parameter, header, or payload field can make an
   `init` candidate land at `project` or `org` scope. Mutation: adding a
   scope passthrough must fail the isolation test.
3. **Identity is server-resolved.** `org_id` from the key, `user_id` via
   `resolve_request_user_id` — never from the payload. Same rule as
   everywhere; asserted for this path specifically.
4. **Caps are server-side.** A client that ignores the playbook still
   cannot exceed them. Mutation: deleting the per-run cap must fail ≥1 test.
5. **Idempotence.** Running `init` twice on the same context dedups to
   ~zero new rows. Test seeds a run, repeats it, asserts accepted ≈ 0.

## Out of scope (v1)

- **Paste-import from Claude memory / ChatGPT exports** — v1.1. It is the
  same `submit` phase with `source="import"`; the work is the playbook
  text and docs, not the engine. Strategically this is the Anthropic
  import play pointed at neutral ground; it deserves its own release note.
- Org/project-scoped seeding, bulk artifact upload, dashboard-side
  onboarding UI.

## Why this also fixes the demo

`init` on any repo produces a populated workplace in about two minutes —
recall that answers, an `explore` that has entities once the curator
ticks. That is the screen you want in front of a team, instead of an
empty ledger.

## Acceptance

- Tool inventory: 18 → 19, names asserted in the inventory test.
- The sentence "use the Klio init tool to onboard me" causes Claude Code
  (with the server connected) to run start → submit → finish with no
  human step in between — verified manually against production before
  announcing.
- All five invariants above have observed-RED tests and reported mutation
  results, per house discipline.
