# Klio Compression — cut what reaches the model, automatically

**Date:** 2026-08-01
**Status:** Design, approved in brainstorming. Not yet planned.
**Ships in:** the open engine (`klio-tech/klio`) and the private engine (`vex_engine`).

---

## The problem

Two separate wastes, measured rather than assumed.

**1. Klio wastes its own tokens.** A `recall(limit=10)` returns 1,442 tokens, of
which **only 17% is memory content**. One row costs 454 characters to deliver
105 characters of memory:

```json
{ "id": "80590c00-cb5f-45c8-b813-952054a10b11",
  "content": "The prototype includes a Memory Manager…",   ← 105 chars
  "confidence": null, "space_id": null, "project": null,
  "project_id": "93e3f7df-788a-4de0-9c3f-654e1259ce00",
  "created_at": "2026-07-27T16:02:38.526112+00:00",
  "scope": "org", "provenance": "EXTRACTED", … }
```

At the documented `limit=50`, that is roughly 7,000 tokens to deliver 1,300.

**2. Everything else an agent reads is worse.** Published figures for 2026:
three MCP servers (GitHub, Slack, Sentry) consume **143,000 of a 200,000-token
context — 72%** before the first user message. Tool *definitions* now have
platform fixes (Anthropic's Tool Search, 85%, on by default). Tool *results*,
logs, files and RAG chunks have none.

## What this is

A compression layer with two deployments of one library:

```
  Agent (Claude Code, Codex…)
    │  every request/response
    ▼
  ┌─────────────────────────────────┐
  │  klio proxy   (LOCAL, port 8787)│  ← compresses agent ↔ LLM traffic
  │  Router → Compressors → Cache   │
  └─────────────────────────────────┘
    │
    ▼  LLM provider (Anthropic / OpenAI)

  ┌─────────────────────────────────┐
  │  Klio engine  (mcp.klio.tech)   │  ← compresses its own recall payloads
  │  same compressor library        │
  └─────────────────────────────────┘
```

### Why the proxy is local, despite private memory being server-side

Private *memory* is siloed on our server — that decision stands and is
unrelated to this. The proxy is local for a different reason: **latency**.
Routing every LLM request through our servers before Anthropic would add a
round trip to every single call. The compressor runs where the data already is.

Server-side compression applies only to data Klio already holds (recall
payloads). Nothing new is sent anywhere by that path.

## Why an MCP server cannot do this alone

Klio sees only calls made to Klio. It cannot see what the GitHub MCP server
returned, what the agent read off disk, or anything travelling to the model.

There are exactly two positions in an agent's data path:

| | Sees | Setup |
|---|---|---|
| MCP server (Klio today) | only its own traffic | none |
| **Proxy** | everything | base-URL redirect |

Claude Code confirms there is no third option: *"There is no hook event for
intercepting Claude's API calls themselves."* Hooks fire on tool use and
lifecycle events, never on model traffic.

---

## Automatic, after one command

The user runs `klio init` **once**. It writes to `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8787",
    "ENABLE_TOOL_SEARCH": "true"
  }
}
```

Every session after that routes through the proxy. No wrapper command, no shell
export, nothing to remember. This is documented and supported; the `env` block
is honoured identically to the real process environment.

### `ENABLE_TOOL_SEARCH` is not optional

Pointing `ANTHROPIC_BASE_URL` at a non-Anthropic host **disables MCP Tool Search
by default**. Tool Search is the platform feature that cuts tool-schema tokens
by ~85% and is on by default in Claude Code v2.1.7+.

Without this flag we would save ~20% on prose while silently costing ~85% on
schemas — **a net loss the user cannot see**. Re-enabling it only works if the
proxy correctly forwards `tool_reference` blocks, so that is a hard requirement
on the proxy, not a nice-to-have.

Also disabled when pointing at a non-Anthropic host: Remote Control (v2.1.196+).
Say so in `klio init` output rather than letting someone discover it.

### The proxy must never be the reason a session fails

Once `ANTHROPIC_BASE_URL` points at localhost, **a dead proxy means the agent
cannot reach a model at all.** Not degraded — dead. That is far worse than
having no compression.

Two mitigations, both required:

1. **Supervised service.** launchd agent with `KeepAlive` (macOS), systemd user
   unit with `Restart=always` (Linux), scheduled task at logon (Windows).
2. **Fail-open, always.** Any internal error — model load failure, cache miss,
   malformed content, OOM — passes the request through **untouched**.
   Compression failing costs tokens. It must never cost a working session.

`klio doctor` verifies the settings entry is present, the service is alive, and
end-to-end traffic flows. It re-applies the entry if another writer removed it.

### Settings files have more than one writer

Claude Code writes to `settings.json` itself (theme, `effortLevel`,
permissions). Klio already writes hooks there. A clobbering write by any party
drops another's entries.

Every write is **read → merge → write**, never a wholesale replace. This is
fiddly enough that comparable tools devote a dedicated module to it; budget for
that rather than treating it as a one-liner.

Evidence it survives: Klio's hooks are in that file today and have persisted
across Claude Code updates. User config is not regenerated by upgrades.

---

## The compression core

A router selects a compressor by content type.

| Compressor | Handles | Method | Typical | Lossy |
|---|---|---|---|---|
| **Structural** | JSON, tool output, recall payloads | Fold repeated keys, drop nulls, shorten ids, hoist constants | 50–80% | no |
| **Code** | Source, diffs | Tree-sitter AST: keep signatures and structure, drop bodies | 40–70% | yes |
| **Prose** | Logs, docs, transcripts | `klio-compress-v1`, keep/drop score per token | **~1.4%** (measured) | yes |
| **Passthrough** | Unrecognised, or under ~500 tokens | nothing | 0% | no |

> **Corrected 2026-08-01, after stage 4 shipped.** This table originally
> projected 15–20% for prose. Measured on our own markdown at the conservative
> default it is **1.4%**, and on log files **0% at every threshold** — 56% of a
> log's subwords sit inside protected spans (timestamps, dotted identifiers,
> byte counts), and only 13 of 1,502 scored below 0.30 at all.
>
> That is the conservative threshold behaving as designed on technical text,
> not a defect. But 15–20% was a guess, it was wrong, and it must not be
> quoted. **The value in this system is code (72%) and JSON (66–82%), not the
> model.** Prose compression earns its place only on genuinely prose-heavy
> input, which agent traffic largely is not.

**Structural ships first and is the most valuable.** It is pure waste removal,
needs no model, and fixes the measured 83% envelope on Klio's own recall.

### Nothing technical is ever altered

Protected before compression, restored after: fenced code blocks, inline code,
URLs, filesystem paths, `CONST_CASE`, `dotted.method()`, function calls, version
numbers, error strings, numbers, and identifiers.

A compressor that mangles a stack trace is worse than no compressor. This is the
single most important correctness property in the system and gets adversarial
tests, not just unit tests.

### Threshold

The prose model emits a keep-probability per subword. Published operating points
on the base model:

| Threshold | Dropped | Must-keep recall |
|---|---|---|
| 0.30 | 8% | 0.994 |
| 0.40 | 13% | 0.987 |
| 0.50 | 18% | 0.974 |

**Ship 0.30 (conservative) as the default.** A first release that silently drops
a needed line destroys trust permanently; one that saves 8% instead of 18% just
under-delivers. Configurable for users who want more.

---

## Reversibility

### What the caller receives

```json
{
  "compressed": "FATAL db pool exhausted at 14:22 …",
  "original_tokens": 10144,
  "compressed_tokens": 1260,
  "handle": "kc_7f3a91b2",
  "lossy": true
}
```

The handle is a **separate field, never a marker inside the text.** Inline
markers get echoed back by models, corrupt diffs, and break code. No handle is
issued when `lossy` is false — there is nothing to recover.

### The originals cache

Table `compression_cache`, mirroring how artifacts already work:

| Column | Purpose |
|---|---|
| `id`, `org_id`, `user_id` | Siloed exactly like memory — same ladder, same predicate |
| `content_hash` | Dedup. Agents re-read the same file constantly |
| `inline_text` **or** `storage_uri` | Small in Postgres, large in R2 — the artifacts split |
| `size_bytes`, `created_at`, `expires_at` | Retention |

### Retrieval

```
klio_retrieve(handle, offset?) → original text, paged
```

Same paging contract as `artifact_get` (51,200 characters, `offset`,
`has_more`). An agent that knows one knows the other.

**Siloed like everything else.** A handle owned by another user resolves to
`not_found` — never `forbidden`, which would confirm it exists. Same adversarial
tests as the memory ladder: another user, an org admin, another org.

### Three settled defaults

- **Retention 24 hours.** Retrieval happens in the same session or not at all.
  An expired handle returns an explicit `expired` error naming the reason —
  never silence, never a wrong answer.
- **Not a memory.** A separate table. It never appears in `recall`, is never
  curated, and never becomes team knowledge. Compressing a log must not
  accidentally publish it.
- **No storage quota.** Transient and not user-controlled. Rate limits handle
  abuse; quota must not punish someone for compressing.

---

## The model

Base: **`answerdotai/ModernBERT-base`** (149M) with a LoRA adapter and a dual
head (token classifier + 1-D span conv), as published in
**`chopratejas/kompress-v2-base`** — Apache 2.0, ungated, ONNX weights included
so it runs on CPU.

We ship it as **`klio-compress-v1`**, fine-tuned on our own corpus (agent traces
and code — not the healthcare/finance/government domains in the base training
set, which we do not need).

**Attribution.** Apache 2.0's condition is that attribution survives. We keep
the upstream copyright in a `NOTICE` file and `base_model:` on the model card.
That is one file nobody reads and one line of metadata — it does not appear in
the product UI, and it is exactly what the upstream authors themselves do for
ModernBERT.

All pipeline code is ours. Nothing is copied.

**Training our own from scratch was considered and rejected**: the compute is
cheap (LoRA on a 149M model, hours on one GPU), but the labelled dataset is not
— 126,617 accepted examples produced by a two-stage compressor-plus-judge
pipeline across 17 domains, which is months and tens of thousands of dollars in
API spend to reproduce, for a result at best equal to something already free.

---

## Agent coverage

| Agent | Mechanism | Automatic? |
|---|---|---|
| **Claude Code** | `~/.claude/settings.json` → `env` | ✅ |
| **Codex** | `~/.codex/config.toml` → provider `base_url` | ✅ |
| Copilot, Aider | environment variable | later |
| **Cursor** | base URL is a **GUI setting** | ❌ manual instructions |
| Gemini, Hermes | no known config target | ❌ |

**Launch with Claude Code and Codex done properly.** Everything else gets
printed instructions until there is demand.

Wiring N agents is a permanent maintenance cost — every agent that ships,
updates, or changes config format is work forever. Comparable tools carry 15+
providers and a dedicated install registry. Shipping two well beats shipping
fifteen badly.

---

## Open vs private

| Component | Repo | Licence |
|---|---|---|
| Compressors, router, protections | `klio-tech/klio` | AGPL-3.0 |
| Local proxy + `klio init` wiring | `klio-tech/klio` | AGPL-3.0 |
| `klio-compress-v1` weights | HuggingFace | Apache-2.0 |
| Server-side recall compression | `vex_engine` | private |
| Hosted originals cache | `vex_engine` | private |

The proxy is the thing people install, so it is open — it drives adoption and it
is the piece with no competitive secret in it. The hosted memory integration
stays private, consistent with the existing open-core line.

---

## Risks

| Risk | Mitigation |
|---|---|
| **Proxy down = agent completely broken** | Supervised service, fail-open on every error path, `klio doctor` |
| **Tool Search silently disabled → net token loss** | `ENABLE_TOOL_SEARCH=true` **and** the proxy must forward `tool_reference` blocks. Verify with a measured before/after, not an assumption. |
| Compression mangles a stack trace or code | Protected-token restore, adversarial tests, conservative default threshold |
| Another writer drops our `settings.json` entry | Read-merge-write; `doctor` re-applies |
| We claim savings we do not deliver | Publish a measured benchmark per content type before launch. No number goes in marketing that we have not reproduced. |
| Scope creep into 15 agents | Two agents at launch, by decision |

## Out of scope

- Image compression
- Compressing other MCP servers' tool schemas (the platform already fixed this)
- Output-token steering (trimming what the model writes back)
- Fine-tuning `klio-compress-v1` — v1 ships the base weights; the fine-tune
  follows once there is a corpus and an eval to judge it

## Build order

Each stage is independently shippable and verifiable.

1. **Structural compressor + recall integration.** Lossless, no model, no proxy.
   Fixes the measured 83% envelope. Value on day one with nothing to install.
2. **Originals cache + `klio_retrieve`.** The reversibility contract, with the
   siloing tests, before anything lossy exists.
3. **Local proxy + `klio init` + supervised service.** The delivery mechanism.
   Ships passthrough-only first — prove the plumbing cannot break a session
   before it is allowed to change a single byte.
4. **Code and prose compressors.** The lossy ones, behind the cache built in 2.
5. **Codex support**, then the fine-tune.

Stage 3 shipping as a pure passthrough is deliberate. The riskiest thing in this
design is being in the request path at all; that risk should be proven survivable
while the blast radius is zero.

## Open question for planning

Whether the proxy should compress **requests only**, or responses too. Requests
are where the tokens are and where the risk is lowest. Response compression
would change what the agent shows the user, which is a different and larger
promise. This spec assumes requests only; revisit with data.
