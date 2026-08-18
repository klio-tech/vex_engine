# Per-project default capture scope

**Date:** 2026-08-18
**Status:** Proposed. Requested by Abhishek after asking how to share a project
so that other users' agents get access to what he is doing.
**Extends:** `2026-08-01-user-silo-design.md`. That design stands; this changes
one default it deliberately set, on the trigger it deliberately wrote.

---

## The finding this rests on

**Nothing was removed.** A per-project default scope has never existed, and no
`projects.default_scope` column was ever created. What changed is narrower and
was deliberate.

Before 2026-08-01, every write landed at `scope='org'` — the team brain.
`test_write_tool_scope_defaults_to_org` and `test_event_scope_defaults_to_org`
still exist and still pass, which is why this feels like something regressed.

On 2026-08-01 the user-silo design routed **passive session capture** to
`private` whenever the request can be attributed to a person
(`capture.py::_hook_write_scope`, reached from `/capture/event` and
`/capture/transcript`). The docstring is explicit that an attributable caller's
*declared* scope is ignored on purpose, because private is strictly narrower and
overriding it could only widen who sees the row. It fails closed.

**Explicit MCP writes were not changed.** `remember`, `note`, `decide` and
`plan` still default to `scope="org"` (`mcp_app.py:292`). So the split today is:

| Path | Scope | Visible to |
|---|---|---|
| Hook / session capture | `private` | the capturing user only |
| `remember` / `decide` / `plan` / `note` | `org` | everyone in the org |
| Anything promoted via `share` | `org` or `project` | org, or project members |

That split is coherent. It is also, in practice, the reason the team brain is
empty: almost all volume comes through hooks.

## The trigger has fired

The user-silo spec did not merely anticipate this outcome. It named it as a
risk and wrote the condition for reversing itself:

> **Promotion never becomes a habit, so nothing reaches the team** — Measure it:
> promotions per user per week. **If it stays near zero, the model is wrong and
> should change rather than be defended.**

And left it open:

> Whether promotion should also exist as a bulk action ... Per-memory is safer
> and is what this spec assumes; **bulk is what will actually get used.**

The measurement, taken 2026-08-18 against production:

| | |
|---|---|
| Private, active, reference org | **6,543** |
| Org-scoped, active, reference org | **1** |
| Promotions ever, that org | **1** |
| Project-scoped rows, entire database | **12** |
| Projects in that org | 34 |

Promotions per user per week is not near zero; it **is** zero, over seventeen
days. By the standard the original design set for itself, the default is wrong
and changes now. This spec is not overriding that decision — it is executing it.

## What this adds

A project may declare the scope its captures land at. That is the whole feature.

`projects.default_scope`, one of:

- **`private`** — today's behaviour. The default for every existing and new
  project. Nothing changes unless somebody changes it.
- **`project`** — captures filed under this project land at `scope='project'`,
  readable by that project's members (the recall arm for this already exists and
  is already tested).
- **`org`** — captures land in the team brain.

### Where it applies, and where it must not

It applies **only** to passive capture that has already resolved a project:
`_hook_write_scope` gains the project's default in place of its hardcoded
`private`.

It does **not** apply to:

- **Captures with no project.** Unfiled context stays `private`. A default that
  leaked to unfiled capture would silently share everything a user does outside
  any repo, which is the opposite of the ask.
- **Explicit writes.** `remember`/`decide`/`plan`/`note` already take an explicit
  scope and already default to `org`. Routing them through a project default
  would change behaviour nobody complained about.
- **Existing rows.** No backfill, ever. Changing a project's default changes what
  happens next, never what already happened. A setting that retroactively
  published 6,543 private rows is a privacy incident, not a feature.

### Who may set it

The same gate as `project_grant`: the caller must **administer** the project —
not merely belong to it, and not merely belong to the org. Every change is
audit-logged with the old and new value, exactly as grants are.

This matters because the setting converts a prompt injection into data
disclosure if it is loosely gated. `project_grant` already reasons this through
in its docstring; this follows it deliberately rather than inventing a second
model.

### The per-user override

A shared project must not force a person to publish. `project_members` gains
`scope_override`, nullable:

- `NULL` — follow the project default.
- `'private'` — this member's captures in this project stay private regardless.

Only the member themself may set their own override. An admin cannot set it for
someone else in either direction: an admin who could clear an override could
publish a colleague's context without their consent, which breaks the user-silo
promise that private is a real boundary *including from admins*.

## Making it visible

A default that silently changes where your work goes is worse than no default.
Three surfaces, none optional:

1. **The capture response** already returns the stored scope. The hook surfaces
   that already; no change needed, but it becomes load-bearing and gets a test.
2. **The dashboard project view** states the current default in words a person
   reads without decoding: *"Captures in this project are shared with the team"*
   / *"...visible to project members"* / *"...private to you"*.
3. **The `space` / `project_list` MCP output** includes it, so an agent can tell
   the user where their context is going before it goes there.

## Data and migration

```
ALTER TABLE projects
  ADD COLUMN default_scope text NOT NULL DEFAULT 'private'
  CHECK (default_scope IN ('private','project','org'));

ALTER TABLE project_members
  ADD COLUMN scope_override text
  CHECK (scope_override IN ('private'));
```

Both additive, both defaulting to today's behaviour. Deploying the migration
alone changes nothing observable — which is the property that makes it safe to
land before the code.

## Testing

The boundary is a SQL predicate and a routing decision, so both get real
Postgres, not mocks.

- A capture in a project defaulting to `org` lands `org`; the same capture in a
  `private` project lands `private`. Both asserted on the stored row, not the
  response.
- A capture with **no project** lands `private` even when some project in the
  org defaults to `org`.
- A member with `scope_override='private'` captures privately in a project
  defaulting to `org`.
- A non-administering member cannot change the default; the attempt is refused
  and audited.
- An admin cannot set or clear another member's override.
- Changing a default does not alter a single pre-existing row — asserted by
  counting rows per scope before and after.
- Adversarial: a project in another org with `default_scope='org'` never
  influences this org's routing.
- Mutation check, stated as a requirement rather than a hope: removing the
  "no project means private" branch must fail a test.

## Success criteria

- A project can be set to shared capture, by someone who administers it, and the
  next capture in that project is readable by the intended audience.
- No existing row changes scope, at any point.
- A member can opt out of a shared project's default, and nobody else can undo
  that.
- Every surface that shows a project states where its captures go.
- Promotions stop being the only path to a shared brain — measurable as
  org-or-project-scoped rows per week rising above zero.

## Explicitly not in this spec

- **Bulk promotion of existing memories.** The original open question. It is a
  separate, riskier feature: it acts on 6,543 rows that were captured under a
  private promise. Defaults change the future; bulk promotion rewrites the past.
- **Retiring `scope='agent'`.** Still deprecated, still read.
- **Reconciling `space` and `project`.** Deferred by the user-silo spec and still
  deferred.
- **Changing explicit-write defaults.** They already reach the team.
