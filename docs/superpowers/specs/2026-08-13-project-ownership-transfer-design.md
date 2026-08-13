# Project ownership transfer on departure — design

**Date:** 2026-08-13
**Status:** Approved by Abhishek 2026-08-12; engine freeze lifted for this work.
**Follows:** `2026-08-12-project-listing-visibility-design.md` (projects are members-only, shipped as
migration 039) and `2026-08-12-user-scoped-api-keys-design.md` (a key stops working when its owner
leaves).

---

## The problem

Migration 039 seated exactly one owner per project. Measured in production immediately after:

| Projects with… | Count |
| --- | --- |
| exactly **1** member | **133** |
| 2 members | 1 |

Every project now depends on a single person. When that person leaves the workspace their key stops
working — correct, and shipped — but their `project_members` row is the only one, so the project
becomes **invisible to everyone**. Nobody can list it and its project-scope content is unreachable.

Concretely: if the largest owner (`1211d286…`) left, **118 projects** would go dark at once. Across
all single-owner projects sit **35,162 org-scope memories** and **16,987 private** ones.

Nothing is deleted in any of this — the memories remain in the database, and org-scope content stays
searchable. What is lost is the *project*: its existence, its listing, and its project-scope tier.

## What must be true

1. A project never has zero members. That state is unreachable, not merely unlikely.
2. A departure transfers the project to the **workspace owner**, who can then re-grant it.
3. Nobody gains visibility while the departing person is still on the team. This is a transfer at
   the moment of departure, **not** a standing org-admin bypass — Abhishek explicitly rejected that
   in the previous spec ("no org-admin bypass in the engine").
4. **Private memories are never transferred or exposed.** 16,987 of them sit in these projects.
   Private means private; a workspace owner does not inherit someone's private notes because they
   left. Only the *project ownership* moves.

## Mechanism: a reconciliation sweep, not a webhook

Departure happens in Supabase (`accounts_memberships`), which the engine does not own. Two ways to
learn about it:

- **Push (webhook).** There is **no Supabase webhook consumer anywhere in the engine** — a grep for
  `SUPABASE_DB_WEBHOOK_SECRET` finds no handler. Building one means new plumbing on both sides,
  including the Dashboard repo, which is currently off-limits. It is also lossy: a missed or failed
  delivery leaves a project dark permanently, with nothing to correct it.
- **Pull (sweep).** `async-worker` already runs a polling loop (`app/main.py`). A periodic
  reconciliation is idempotent, self-healing, needs no cross-repo coupling, and — importantly —
  repairs projects that are *already* orphaned rather than only catching future departures.

**Decision: sweep.** A webhook can be layered on later purely for immediacy; the sweep remains the
backstop either way.

## Design

Run periodically in `async-worker`:

1. Collect the distinct orgs that have `project_members` rows.
2. For each org, resolve its Supabase account and read the **full current membership set** — one
   query per org, not per row.
3. For each `project_members` row whose `user_id` is absent from that set, the user has departed.
   For each affected project, **in a single transaction**:
   a. Seat the workspace owner as `admin` (`ON CONFLICT DO NOTHING`, so an existing membership or
      role is never clobbered or downgraded).
   b. Then remove the departed row.
   **Order is the invariant: seat before removing.** Never leave a project memberless, even
   transiently, even if the process dies between statements.
4. Record each transfer — org, project, from-user, to-user, timestamp — so the change is
   explicable months later.

Reuse the owner-resolution helper written for migration 039 (`organizations.account_slug` → Supabase
`accounts`, preferring `accounts.id` when `org_id` is a UUID, restricted to non-personal accounts,
`primary_owner_user_id` with earliest membership as the documented tie-break). Do not write a second
one.

## Safety — the failure mode that matters

**A false "this user has departed" reading would strip memberships across the whole database.** The
sweep must be paranoid in exactly one direction:

- If the Supabase query **fails**, skip that org entirely. Do not treat an error as "nobody is a
  member."
- If it **succeeds but returns an empty membership set**, skip that org and log loudly. A real
  workspace always has at least its owner; an empty result means something is wrong, not that
  everyone left simultaneously.
- If the workspace owner **cannot be resolved**, leave the project exactly as it is. A dark project
  is recoverable; a memberless one after we deleted the last row is not.
- Cap transfers per run (a configurable ceiling, defaulting low). A cascade should trip the cap and
  surface, not run to completion.
- Ship with a **dry-run mode** that logs what it would do and changes nothing, so the first
  production run can be inspected before it is armed.

## Out of scope

- Deleting or reassigning any memory, of any scope. This spec moves project membership only.
- Removing the departed user's authored content or authorship attribution — `session_memories.user_id`
  is history and stays.
- Re-granting to specific teammates. The owner receives it and re-grants deliberately; guessing who
  should inherit is not the engine's call.
- Dashboard surfaces. Note the dashboard still grants org admins a project-listing bypass, which now
  contradicts the engine — tracked separately, and it should follow the engine.

## Test plan

Adversarial and mutation-verified, matching the repo's existing conventions:

1. Departed sole owner → workspace owner seated as admin, departed row removed, project still
   listable by the owner.
2. Departed member of a **multi-member** project → row removed, no transfer needed, project
   untouched for the others.
3. **Current** member → nothing happens at all (the regression that matters most: the sweep must be
   a no-op for healthy data).
4. Supabase query fails → org skipped, no rows changed.
5. Supabase returns an empty set → org skipped, warning logged, no rows changed.
6. Owner unresolvable → project left alone, departed row **not** removed (never memberless).
7. Existing membership/role of the workspace owner is never downgraded by the transfer.
8. Idempotent: running twice changes nothing the second time.
9. Dry-run changes nothing and reports the same set it would have acted on.

## Rollout

Deploy with dry-run enabled. Inspect one production cycle's log. Arm it. No migration; no schema
change.
