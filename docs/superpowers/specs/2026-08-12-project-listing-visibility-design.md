# Project listing visibility — design

**Date:** 2026-08-12
**Status:** Decided and implemented. See "Decision" at the bottom — Abhishek ruled private-by-default
for every project, rejecting this document's §4 recommendation. The path leak was already fixed (see
"Already shipped"); the migration and listing change described below now ship the ruling.
**Related:** the Dashboard repo's `2026-08-12-project-membership-decision.md` — **these two must be
decided together**, for reasons in "Why this can't be decided alone".

---

## The problem

`shared.projects.list_projects` returns every project in the org to any caller:

```sql
SELECT p.id, p.display_name, p.git_remote, p.repo_root_path, …
FROM projects p WHERE p.org_id = :org_id
```

No membership filter. Its docstring defends this — *"Listing projects is not a disclosure of their
CONTENTS"* — and that reasoning is sound as far as it goes. Memory contents remain correctly gated:
private stays private, project scope requires membership.

Two things it did not anticipate:

1. **Projects are auto-created, not deliberately created.** `shared/brain.py` upserts a `projects`
   row from the git remote or repo root the first time an agent writes in a directory. Nobody
   "creates" most of these. In one production org this yields **118 projects**.
2. **The row carries the creating machine's local path.** Directory names disclose a great deal —
   job hunting, personal relationships, health, side businesses — none of it memory content, all of
   it disclosure to every org member and key holder.

## Already shipped

`fix(projects): stop listing local filesystem paths to the whole org` (`517e30a`, branch
`fix/project-listing-privacy`). `repo_root_path` no longer appears in `project_list` output. A full
`services/` audit found `_serialize_project()` in `mcp_app.py` was the only site where it reached a
response; every other use is input or internal. The key is **omitted rather than nulled**, so the
presence of the key cannot itself signal which projects have a local path on file. Lookup *by* path
still works and is regression-tested.

That removes the sensitive field. It does not address who can see that a project exists at all.

## The requirement, and an ambiguity worth resolving

Abhishek, 2026-08-12: *"I do not want everyone to see all projects — only the user from whose local
this was created should be shown to the other users."*

That admits two readings:

- **A — restrict:** only the creator (plus members) sees the project; others don't see it listed.
- **B — attribute:** everyone still sees the project, but it shows *who* created it instead of the
  path.

The first clause points at A. The wording of the second points at B. **This spec assumed A, with B
as a natural addition** — where a project *is* listed, show the creator rather than the path.

**Ruling (see "Decision" below): A, and ONLY A.** Abhishek's follow-up ruling ("private by
default — everything") resolves the ambiguity harder than this section anticipated: a project is
visible only to its **members**, full stop, with the creator seated as the project's first admin
member rather than granted some separate creator-only visibility. B — showing *who* created a
project that is otherwise listed org-wide — does not arise, because no project is listed org-wide
regardless of who created it. `project_list`'s response shape is unchanged by the ruling; it was
already creator/path-silent per "Already shipped" above, and stays that way. If a future product
call wants B's attribution UI for the projects a caller CAN already see, that is a separate,
smaller change layered on top of this one — not something this migration or listing change
attempts.

## The blocker

**`projects` has no creator column.** Per migration `020_klio_brain_spaces_projects.py`, the table
is: `id`, `org_id`, `space_id`, `git_remote`, `repo_root_path`, `display_name`, timestamps. Nothing
records who caused the row to exist. Neither reading of the requirement is implementable today.

## Design

### 1. Record the creator

Add `created_by` (nullable, a user id) to `projects`. Populate it in `brain.py`'s upsert from the
resolved user — which is now available on both auth rungs, since API keys carry a user
(`KeyInfo.user_id`, shipped 2026-08-01). `ON CONFLICT` must **not** overwrite an existing
`created_by`: first writer wins, so a teammate later working in the same repo doesn't reassign
ownership.

Nullable because a key without a user, or a pre-fix row, has no honest answer. Do not invent one.

### 2. Backfill

Existing rows can be attributed from authorship: the earliest `session_memories.user_id` for that
project is a strong proxy for whose machine created it. It is a heuristic, not a fact — a project
whose first write was org-scope by an unattributed key yields `NULL`, and should stay `NULL` rather
than guess.

### 3. Listing rule — SUPERSEDED BY THE RULING

> **This section described the pre-ruling proposal. It is kept for the record; the "Decision"
> section at the bottom of this document is the actual rule that shipped.**

The original proposal: a project is listed to a caller who is any of:

- its **creator** (`created_by = caller`), or
- a **member** (`project_members`), or
- an **org admin**.

Everyone else does not see it. Where a project *is* listed, include the creator's display identity
in place of the path (reading B, folded in).

**What the ruling changed:** "creator" and "org admin" are not separate arms of the predicate.
There is exactly one gate — **membership** (`project_members`) — and no org-admin bypass in the
engine. The creator arm survives only indirectly: `ensure_project`'s auto-create path now seats the
creator as the project's first `project_members` admin at the moment the row is inserted (migration
039 backfills the same for existing rows), so "the creator can see their own project" is *true in
practice* but is implemented as "the creator is a member," not as a second, independent visibility
rule the listing query has to check. An org admin who is not a project member does not see the
project — the ruling states this explicitly ("No org-admin bypass in the engine").

### 4. The git-remote heuristic — considered and REJECTED

> **Abhishek's ruling rejected this section outright: "No git-remote exception."** It is kept below,
> unmodified, as the record of what was proposed and turned down — not as a live option, a fallback,
> or a "revisit later." Do not resurrect it without a fresh, explicit ruling.

There is a sharper rule hiding in the data. A project with a `git_remote` is a **shared repository**
— by definition collaborative, and org-wide visibility is defensible. A project with only a
`repo_root_path` and no remote is a **local folder** — inherently personal, and is exactly the
category leaking `/Users/…/Job Application`.

So the default could be: **remote-backed projects are org-visible; path-only projects are visible to
their creator, members and admins.** This matches intent better than a blanket rule, keeps team
repos discoverable (which is the product's whole point), and confines restriction to the rows that
actually embarrass. Recommended, but it is a product call.

**Why it was rejected in practice, beyond the ruling being explicit:** a git remote is a
collaborative *artifact*, not a collaborative *ACL*. Treating "has a remote" as "org-visible" would
have meant a project's visibility flipped the moment someone added a remote to a previously
path-only repo — a side effect of a `git remote add` command changing who can see a project's
existence, with no one having granted anything. Membership is a decision a person makes
(`project_grant`, or auto-seating on creation); a remote is a fact about a repository. The ruling
keeps those two things from being conflated.

## Why this can't be decided alone

`project_members` currently holds **3 rows across the entire production database**. Gate the listing
on membership alone with no backfill and effectively every project disappears for everyone — the
feature would look broken, and would take the owner's own project away from them along with
everyone else's. This is why the migration (039) is mandatory, not optional: it seats each existing
project's earliest-attributed author as its `project_members` admin BEFORE the listing query starts
enforcing membership, so "gate on membership alone" is safe precisely because membership is no
longer sparse by the time the gate goes live.

Conversely, if the membership decision lands on "authorship implies membership," much of this
collapses into it. Decide them in the same sitting. *(Resolution: it did — see "Decision" below.
Auto-seating the creator as admin, both going forward in `ensure_project` and retroactively in
migration 039's backfill, IS "authorship implies membership," just spelled as a membership row
rather than a second predicate arm.)*

## Interim option (no migration) — NOT TAKEN

If a ruling is wanted before schema work, the creator can be derived on read: join each project to
the earliest `session_memories.user_id` for it. Same predicate as the backfill, evaluated live. It
costs a subquery per listing and cannot see projects with no memories, but it needs no migration and
proves the UX before committing to schema.

Not taken: the ruling landed the same day as this document, so there was no gap to fill with an
interim read-time version. Migration 039 does the equivalent computation once, at migrate time,
which is strictly better for a value (`created_by`, and the resulting admin seat) that a caller's
`project_grant` may then build on — a live-derived value cannot be added to as a real membership row
can.

## Test plan

Superseded by the ruling — the git-remote arm (old #2) and the org-admin arm (old #4) do not exist
in the shipped rule. The actual test plan (real Postgres, `KLIO_REQUIRE_DB=1`, adversarial and
mutation-verified) is in the task that implemented this:

1. A member sees their project; a non-member in the same org does not see it at all — not even that
   it exists.
2. An unattributed caller (no user id) sees nothing, not the org's full project set.
3. The creator lands as an admin member (via `ensure_project`'s auto-seat, or migration 039's
   backfill for pre-existing rows) and can use `project_grant` to add someone else — proving the
   seat is real membership, not a cosmetic label.
4. A second writer into the same repo does NOT become owner and does NOT overwrite `created_by`.
5. Backfill: a project whose earliest attributed memory is user X yields X as admin; a project with
   no attributed memories yields no membership row and stays invisible to everyone.
6. Every project tool (`project_list`, `project_members`, `project_grant`) gives a non-member the
   same `not_found` shape a genuinely missing project gets — never a distinguishable `forbidden`.
7. Migration 039's `downgrade()` runs cleanly against a real database, forward and back.
8. No `repo_root_path` in any response (already covered by `517e30a`; kept covered).

## Out of scope

- **All dashboard changes.** Note the dashboard *already* gates project listing by membership
  (`loadVisibleProjects`), so the two surfaces used to disagree — the MCP showed everything, the
  dashboard showed almost nothing. This change makes the MCP side agree with the dashboard's
  existing stance; reconciling the dashboard's own membership source of truth with the engine's is
  still a separate piece of work.
- `git_remote` exposure *to members*. A private repo URL does reveal a name; judged acceptable
  because a remote is a deliberately shared artifact, and because — post-ruling — only members see
  it at all. Revisit if that proves wrong.
- Retroactively deleting projects with no attributed creator. Migration 039 leaves them exactly as
  invisible-to-everyone as the ruling implies; deleting them outright is a separate, destructive
  cleanup conversation this task does not open.
- Surfacing the creator's identity in `project_list` output (reading B). The ruling settled the
  visibility question (reading A) without asking for an attribution UI on top of it; `created_by` is
  schema and ACL plumbing here, not something the tool echoes back yet.

## Decision

Abhishek, 2026-08-12, after this document's first draft was reviewed: **"Private by default —
everything."**

**Reading:** A (restrict) — and ONLY A. Reading B (attribute: show the creator on an otherwise
org-visible project) does not apply, because no project is org-visible regardless of who created it.

**Default rule:** blanket membership-only. §4's git-remote heuristic (remote-backed projects stay
org-visible) was **considered and explicitly rejected** — "No git-remote exception." A project's
visibility does not depend on whether it has a `git_remote`.

**Admins:** projects have admins; an admin may add other members (`project_grant`, already
role-gated to admins — see `shared.projects.is_admin`). **No org-admin bypass in the engine** — an
org admin who is not a project member does not see the project. (The dashboard may layer its own
org-admin surface on top; that is dashboard scope, not engine scope, and is not addressed here.)

**Implementation:** migration 039 (`created_by` column + backfill from earliest attributed
`session_memories.user_id` + one `project_members` admin row per attributable project, existing rows
never touched via `ON CONFLICT DO NOTHING`), `shared.brain.ensure_project` auto-seating the creator
as admin on actual insert (first-writer-wins), and `shared.projects.list_projects` gated on
membership with no `user_id` failing closed to an empty list.

**Date:** 2026-08-12
