# User Silo — private memory, project ACLs, and a dashboard that respects both

**Date:** 2026-08-01
**Status:** Design, approved in brainstorming. Not yet planned.
**Supersedes nothing. Blocks:** the dashboard memory views, and the graphify provenance work folded in below.

---

## The problem

Klio has two principals — `org_id` and `agent_id` — and neither is a person.

`agent_id` looks like a person but is not. For OAuth callers it is derived as
`oauth:<first 8 chars of user uuid>/<client slug>`, so the same human running
Claude Code and Cursor is two different agents, and two humans can collide on
eight hex characters. There is no way to ask "what does Priya's context look
like", to keep something personal across your own tools, or to grant one
teammate access to one project.

That gap blocks three things at once:

1. **The dashboard.** Admin views, per-user views and project views all need a
   user principal that does not exist.
2. **Signal.** 82% of writes arrive from passive hooks and all land at `org`
   scope. The shared brain holds 37,605 memories of which **11 are decisions**.
   A second developer joining today would recall against 33,000 rows of one
   person's ambient session mechanics.
3. **The category claim.** The single strongest external signal we have is a
   skeptical reviewer independently describing the winning product as "a
   hierarchical memory space that is easy to use". We have `org`, a fail-open
   project filter, and no user.

## What this is not

This is not a privacy feature bolted onto a shared product. It inverts the
default: capture becomes personal, and sharing becomes deliberate. The shared
brain gets smaller and much better.

---

## Decisions

Settled during brainstorming. Do not re-litigate without new evidence.

| # | Decision | Rationale |
|---|---|---|
| 1 | Identity work lands **before** the dashboard | Half the dashboard is unbuildable without a user principal; building the visible half first would bake in the wrong data model. |
| 2 | `private` is a **real** boundary — teammates cannot read it and **neither can admins** | Chosen explicitly over the Slack/Workspace model. It is the strongest promise and the only one that makes personal-by-default capture safe. |
| 3 | Ambient hook capture → `private`. Explicit tools → shared | Calling `remember`/`decide`/`plan`/`note` **is** the act of sharing. Hooks are not a decision, so they must not publish. |
| 4 | The curator runs **per user**, output stays private, the user promotes | The curator currently distils hook captures into org facts. Under decision 3 that would publish private material to the team — a privacy breach shipped as a feature. |
| 5 | API keys are **personal**; the key's `created_by` is the attribution | All 55 existing keys already carry `created_by` (36-char UUIDs, 14 distinct creators), so this resolves for every current install with no flag day. |
| 6 | Existing 37,605 memories **stay** `org`-scoped, `user_id` NULL | They were written under different rules. Retroactively hiding a team's accumulated brain is a worse surprise than leaving it. |

### The consequence to expect

On day one the shared brain drops to the 65 explicit MCP writes plus whatever
people promote. **It will feel empty.** That is the trade being made
deliberately — 33,000 rows of hook noise exchanged for a brain where everything
present was put there on purpose — but it is a visible regression in volume and
should not come as a surprise after deploy.

### Admins cannot read private memories, and this is permanent

Decision 2 is a product commitment, not an implementation detail.

Admins see every org memory, every project memory, and every artifact outside
private scope. They do **not** see anyone's private memories. No admin role, no
support workflow, no future tier and no "break glass" path changes that.

To anyone reading this later: the absence of an admin override is deliberate. It
is not a gap to be closed. A private scope an administrator can read is not
private, and shipping it under that name would break the promise the product
makes to the person who wrote the memory.

---

## The scope model

| Scope | Owner | Readable by |
|---|---|---|
| `private` **(new)** | `user_id` | **Only that user.** Not teammates. Not admins. |
| `project` **(new)** | `project_id` | Project members, and org admins |
| `org` | `org_id` | Everyone in the org |
| `session` | unchanged | Verification only; still excluded from brain recall |
| `agent` | deprecated | Superseded by `private`. Stop writing; keep reading. |

### `project_id` now means two different things — this must not be conflated

`project_id` already exists as a **column** on 32,568 rows whose scope is `org`.
There it is a *relevance filter*: it ranks a project's rows first and backfills
with unfiled ones, and it fails open (an unknown project yields org-wide
recall). It is explicitly **not** a security boundary.

`scope = 'project'` is new and **is** a boundary: the row is readable only by
members of that project.

Both coexist deliberately, and decision 6 means every existing row keeps the
first meaning. Concretely:

| Row | scope | project_id | Who reads it |
|---|---|---|---|
| Existing (all 32,568) | `org` | set | Everyone in the org; project only ranks it higher |
| New explicit write with `X-Vex-Project` | `project` | set | Project members + admins |

An implementer who assumes `project_id IS NOT NULL` implies restricted access
will wrongly hide 32,568 rows from the org. An implementer who assumes it
implies open access will leak project-scoped rows. **The predicate is always
`scope`, never the presence of `project_id`.**

### Write routing

| Path | scope | user_id |
|---|---|---|
| `capture.py` hook endpoints (`hook-transcript`, `hook-tool`) | `private` | resolved caller |
| `remember` / `decide` / `plan` / `note` | `org`, or `project` when `X-Vex-Project` resolves | resolved caller, as **author** |
| `artifact_put` | as explicit tools | resolved caller |
| curator | `private` | the user it ran for |

`user_id` is recorded on shared rows too. On a shared row it is **authorship,
not visibility** — it answers "who taught the team this", which the dashboard
wants and which provenance (below) depends on.

### Identity resolution

One function, one rule:

```
resolve_user_id(request) -> str | None
    OAuth    -> principal.user_id       # the real UUID, not the [:8] prefix
    API key  -> key's created_by        # already stored; never reaches KeyInfo today
    never    -> X-Vex-Agent
```

`X-Vex-Agent` is client-declared. Deriving identity from it would let any caller
read another person's private memories by setting a header. It stays a label for
*which tool*, never *which person*.

**When `user_id` is `None`:** write `org`-scoped and log a warning. Not private,
not refused. Nothing breaks for a caller we cannot attribute, and nothing leaks —
a row with no owner cannot be private.

---

## Engine changes

### Migration `031`

Current head is `030`.

| Change | Notes |
|---|---|
| `session_memories.user_id` — nullable varchar | `NULL` = pre-existing or org-authored |
| **CHECK constraint on `scope`** | There is none today. A typo'd scope currently creates rows no query returns. Adding `private` is the moment to close that. |
| Partial index `(org_id, user_id)` WHERE `scope = 'private'` | Private recall is the hot path |
| `project_members(project_id, user_id, role, granted_by, granted_at)` | PK `(project_id, user_id)` |
| `brain_curator_state` PK → `(org_id, user_id)` | Per-user cursor |
| `session_memories.provenance` — see Provenance below | |

### Recall — the visibility ladder

```sql
private:  scope = 'private' AND user_id = :me
project:  scope = 'project' AND project_id IN (
            SELECT project_id FROM project_members WHERE user_id = :me)
org:      scope = 'org'
```

Default recall returns the **union**: my private + my projects + the org.
Requesting `scope='private'` narrows to mine alone.

**`user_id = :me` is a SQL predicate, exactly like `org_id`.** Never a Python
filter over a wider result set. This is a tenancy boundary now and earns the
same rigor as org isolation:

- SQL-level enforcement only
- Tests that attempt a cross-user read and assert not-found
- RLS on the dashboard side

Note the existing two-tier project behaviour is unchanged: project-filtered
recall returns that project's rows first, then backfills with unfiled rows, and
never surfaces rows filed under a different project.

### Curator

Reads `scope='private' AND user_id = X`, RAW types, `recall_hidden = FALSE`.
Writes `scope='private', user_id = X`. It never reads or writes `org`. The
keyset cursor advances per `(org_id, user_id)`.

### Promotion

New MCP tool `share(memory_id, to='org'|'project')` plus the equivalent
dashboard action. It **moves** the row's scope and retains `user_id` as author.

Promotion is also where a curator-derived fact is stamped `INFERRED` — see
below.

---

## Provenance (folded in from graphify)

Klio cannot currently distinguish a fact the user stated outright from a fact an
LLM deduced from ten observations. Both come back from `recall` looking
identical in kind; the only hint is `metadata.source = 'curator'`, which is a
breadcrumb, not a claim about confidence.

This was the specific concern raised in external review: *"if it starts driving
inferences that aren't correct, and god forbid if it starts getting numbers
wrong."*

Adopt graphify's three labels (Apache-2.0, attribution required in NOTICE):

| Label | Meaning in Klio |
|---|---|
| `EXTRACTED` | Stated directly by the user or an explicit tool call |
| `INFERRED` | Deduced by the curator or an LLM pass |
| `AMBIGUOUS` | Uncertain; surfaced for review rather than trusted |

Concretely: a `session_memories.provenance` column (varchar, CHECK-constrained
to the three labels), defaulting to `EXTRACTED`. Explicit tool writes and hook
captures are `EXTRACTED` — they record what was said or done. Everything the
curator synthesises is `INFERRED`. `recall` returns the label so a caller can
weigh it, and the dashboard renders inferred facts differently from stated ones.

**The column is `provenance`, not `confidence`.** `session_memories.confidence`
already exists — a nullable `DOUBLE PRECISION` from migration 016 with 4,435
rows populated (0.398–1.0), written by `store_memory()`, read into
`MemoryRecord.confidence`, and returned verbatim in the `recall` payload. Reusing
the name would have destroyed live data and changed an observable API type. The
float column is untouched and an invariant test asserts so.

Existing rows backfill to `EXTRACTED` **except** the 1,841 with
`metadata.source = 'curator'`, which backfill to `INFERRED`. That is the one
backfill this spec does perform, and it is safe: it adds information to rows
without changing who can read them.

It belongs in **this** spec rather than a later one because it lands on the same
curator code and the same promotion path. The moment an LLM's deduction becomes
team truth is exactly where the label must be attached, and that moment is being
built here.

---

## Dashboard

### Three views, three loaders

| Tab | Contents | Visibility |
|---|---|---|
| **Mine** | My private captures + my curator-distilled facts; promotion action | Only me — identical for admins |
| **Projects** | Project picker → that project's memories and artifacts | Members see their projects; admins see all |
| **Team** | The org-scoped shared brain | Everyone in the org |

Three separate loaders, not one with a `scope` argument. A single
`loadMemories(filters)` can be called wrong; `loadMyPrivate(userId)` cannot,
because the predicate is baked in and `user_id` is never optional. With a
privacy boundary, *impossible to call wrong* beats *documented to call
correctly*.

The existing loader filters `org_id + scope + status` and explicitly excludes
project- and space-scoped rows. It must learn about `user_id` in the same change
or it becomes the leak.

### What an admin gets

Everything except anyone's private scope:

- Every org memory
- Every project memory, including projects they are not a member of
- Every artifact outside private scope
- An **org-wide** storage total, for capacity and billing

Nothing else. No per-member private counts, no "last active in private", no
list of which projects someone has memories in. Billing needs one number for
the organisation; it does not need a per-person breakdown, and a per-person
count is still a disclosure about that person — who is most active, who is
storing a lot, who went quiet. The private scope is opaque from the outside,
including its shape.

### Artifacts

No dashboard UI exists today. Artifacts are rows with `memory_type='artifact'`
and `metadata.artifact_id`; they render as cards (title, summary, kind, size)
with a download that calls `artifact_get(mode='url')`. Same scope rules, no
exceptions.

### Projects on both surfaces

**Dashboard:** `/projects` list, create form, detail page with a members tab.

**MCP:** `project_create`, `project_list`, `project_grant`, `project_members`.

Two details:

- **Granting is by email, not UUID.** An agent does not know a teammate's user
  id. `project_grant(project: "dashboard", user: "priya@oppla.ai")` resolves
  against org membership and fails closed if they are not in the org.
- **`ensure_project` already auto-creates projects on write** — that is how 106
  exist. Explicit creation is for naming and organising ahead of use, and the
  tool description must say so or agents will call it redundantly.

### Security note on `project_grant`

Exposing ACL changes over MCP means an LLM can alter permissions. This was
requested on both surfaces and is designed in, with two constraints:

- A caller may only grant access to projects they administer. It is not a
  general-purpose ACL tool.
- Every grant is audit-logged: who, what, when, and via which surface.

If an agent is prompt-injected through a malicious repo file, `project_grant` is
the tool that converts that into real data access. The constraints hold the
blast radius at "a project you already control".

---

## Risks

| Risk | Mitigation |
|---|---|
| A recall bug is now a **privacy breach**, not a bad result | SQL-level predicates only; adversarial cross-user tests; RLS |
| The shared brain looks empty on day one | Expected and accepted — see "consequence" above. Communicate before deploy. |
| A shared API key attributes everyone's captures to one person's private scope | Keys are personal by policy; the dashboard states this at issue time. Residual risk accepted. |
| Dashboard RLS may already be leaky | Klio memory records a prior RLS gate returning LEAKS with three breaches; it is unclear whether that referred to this dashboard or another project. **Confirm before writing tests** — adding a privacy boundary on leaky RLS builds on sand. |
| Promotion never becomes a habit, so nothing reaches the team | Measure it: promotions per user per week. If it stays near zero, the model is wrong and should change rather than be defended. |

## Out of scope

- Backfilling existing memories to private (decision 6)
- Retiring `scope='agent'` rows (deprecated, still read)
- The `project_only` strict recall mode
- Reconciling the `space` / `project` overlap
- graphify's MinHash dedup and clustering; caveman's provider table — all
  deliberately deferred to a later phase

## Suggested phasing for the plan

This spec is deliberately larger than one sitting. It should become a plan in
three phases, each independently shippable and verifiable:

1. **Identity** — migration `031`, `resolve_user_id`, `KeyInfo.user_id`,
   `provenance` column and backfill. No behaviour change visible to users.
   Verifiable by unit tests alone.
2. **Routing and recall** — hook captures go private, the ladder lands, the
   curator goes per-user, `share` ships. This is the phase where a bug is a
   privacy breach; it needs the adversarial cross-user tests before merge.
3. **Dashboard and projects** — three views, artifacts UI, `/projects`, the four
   MCP project tools, audit logging.

Do not collapse 1 and 2. Phase 1 landing alone leaves the product working
exactly as it does now, which makes it a safe place to stop if phase 2 needs
rethinking.

## Open question for planning

Whether promotion should also exist as a bulk action ("share everything the
curator produced this week"), or only per-memory. Per-memory is safer and is
what this spec assumes; bulk is what will actually get used. Decide during
planning with a bias toward per-memory first.
