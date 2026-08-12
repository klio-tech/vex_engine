# Project listing visibility — design

**Date:** 2026-08-12
**Status:** Proposed. The path leak is already fixed (see "Already shipped"); the rest needs a ruling
plus a migration.
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

The first clause points at A. The wording of the second points at B. **This spec assumes A, with B
as a natural addition** — where a project *is* listed, show the creator rather than the path. If
that reading is wrong, this design changes materially and should be revised before implementation.

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

### 3. Listing rule

A project is listed to a caller who is any of:

- its **creator** (`created_by = caller`), or
- a **member** (`project_members`), or
- an **org admin**.

Everyone else does not see it. Where a project *is* listed, include the creator's display identity
in place of the path (reading B, folded in).

### 4. The git-remote heuristic — worth considering as the default

There is a sharper rule hiding in the data. A project with a `git_remote` is a **shared repository**
— by definition collaborative, and org-wide visibility is defensible. A project with only a
`repo_root_path` and no remote is a **local folder** — inherently personal, and is exactly the
category leaking `/Users/…/Job Application`.

So the default could be: **remote-backed projects are org-visible; path-only projects are visible to
their creator, members and admins.** This matches intent better than a blanket rule, keeps team
repos discoverable (which is the product's whole point), and confines restriction to the rows that
actually embarrass. Recommended, but it is a product call.

## Why this can't be decided alone

`project_members` currently holds **3 rows across the entire production database**. Gate the listing
on membership alone and effectively every project disappears for everyone — the feature would look
broken. Creator-based visibility is what makes gating safe, because the creator always retains sight
of their own work regardless of whether membership is ever granted.

Conversely, if the membership decision lands on "authorship implies membership," much of this
collapses into it. Decide them in the same sitting.

## Interim option (no migration)

If a ruling is wanted before schema work, the creator can be derived on read: join each project to
the earliest `session_memories.user_id` for it. Same predicate as the backfill, evaluated live. It
costs a subquery per listing and cannot see projects with no memories, but it needs no migration and
proves the UX before committing to schema.

## Test plan

Adversarial and mutation-verified, matching the tenancy tests already in this repo:

1. Creator sees their own path-only project; a non-member, non-admin org member does not.
2. Remote-backed projects stay visible org-wide (under the §4 rule).
3. Members see projects they are members of, regardless of creator.
4. Org admins see all — matching the dashboard's existing admin arm.
5. `created_by` is never overwritten by a second writer in the same repo.
6. A `NULL` creator does not become a wildcard match — the null case must not widen visibility.
7. No `repo_root_path` in any response (already covered by `517e30a`; keep it covered).

## Out of scope

- **All dashboard changes.** Note the dashboard *already* gates project listing by membership
  (`loadVisibleProjects`), so the two surfaces currently disagree — the MCP shows everything, the
  dashboard shows almost nothing. Whatever is decided should make them agree.
- `git_remote` exposure. A private repo URL does reveal a name; judged acceptable because a remote
  is a deliberately shared artifact. Revisit if that proves wrong.
- Retroactively deleting the 118 auto-created rows. Worth a separate cleanup conversation —
  particularly whether path-only projects with no memories should be reaped at all.

## Decision

> _To be filled in by Abhishek._

**Reading (A restrict / B attribute / both):**

**Default rule (blanket vs the §4 git-remote heuristic):**

**Date:**
