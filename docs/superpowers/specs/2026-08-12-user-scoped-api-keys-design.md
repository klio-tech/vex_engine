# User-scoped API keys — design

**Date:** 2026-08-12
**Status:** Proposed. Needs an explicit exemption from the 2026-08-07 engine freeze before implementation.
**Scope:** Engine only. Dashboard changes are deliberately **out of scope** (Abhishek, 2026-08-12).

---

## The problem

An API key identifies an **organization**, never a person. Keys are stored as elements of the
`organizations.api_keys` JSONB array and matched by SHA-256 hash; validation returns
`KeyInfo{org_id, key_id, scopes, plan, plan_overrides}` (`services/shared/shared/auth.py:132`).
There is no user field. `resolve_principal` returns `(org_id, agent_id)` and nothing else
(`services/mcp-server/app/auth.py`).

The consequence is not a leak — it is the opposite. Recall builds its visibility ladder like this
(`services/shared/shared/memory.py:1723`):

```python
arms = ["scope = 'org'"]
if scope == "agent":
    arms.append("(scope = 'agent' AND agent_id = :agent_id)")
if user_id:
    arms.append("(scope = 'private' AND user_id = :user_id)")
    arms.append("(scope = 'project' AND project_id IN (<membership subquery>))")
```

The private and project arms are **only constructed when a `user_id` exists**. The function's own
docstring states it plainly: *"Without a `user_id` this is exactly `scope = 'org'`, byte-for-byte
the historical behaviour."* Writes are gated the same way — `scope='private'` raises `ValueError`
without an owner, deliberately, so a private write can never silently widen.

So a key-connected agent:

- reads **org scope only** — never another member's private rows, never project rows;
- cannot hold private memories of its own;
- writes everything at org scope, visible to the whole tenant.

It is fail-closed and safe, but key users are second-class citizens. They cannot participate in
the middle or lower rungs of the ladder at all.

This compounds a second finding (see the dashboard repo's
`2026-08-12-project-membership-decision.md`): `project_members` contains **one row across the
entire production database**. `project` scope is therefore dead twice over — nobody is a member,
and most agents authenticate by key and so would never get the project arm even if they were.

## What already exists

Most of this feature is built. `services/mcp-server/app/identity.py:69` already resolves a user
across both rungs and **already reads the key's user id**:

```python
if key_info is not None:
    user_id = getattr(key_info, "user_id", None)
    if user_id:
        return str(user_id)
```

The `getattr(..., None)` is defensive precisely because `KeyInfo` has no such attribute yet, so
this branch always yields `None`. `resolve_request_user_id` wraps it for callers without a
`KeyInfo` in hand, and `mcp_app.py` consumes it at **13 call sites** — writes, recall, project
permission checks, artifacts. The codebase also already treats the unattributed principal as a
first-class state (`compression_cache.py:18`).

In other words: the consumption side is finished and waiting. Only the producer is missing.

## The change

Three edits. No migration.

1. **`KeyInfo` gains `user_id: Optional[str] = None`** (`services/shared/shared/auth.py:132`).
   Additive-with-default, following the precedent `plan_overrides` set in the same class — that
   field carries a comment explaining the default keeps direct constructors in ingestion-api and
   sync-gateway valid. The same reasoning applies here.

2. **`_query_db` selects the field out of the JSONB element** (`services/shared/shared/auth.py`,
   the `jsonb_array_elements(api_keys)` query around line 347) and populates `KeyInfo.user_id`.
   `organizations.api_keys` is JSONB, so elements lacking the field simply yield `None` — **no
   schema migration and no backfill**.

3. **Nothing else.** The ladder SQL does not change. `resolve_principal` keeps its
   `(org_id, agent_id)` shape, so its ~40 call sites across three variants stay untouched. The 13
   `resolve_request_user_id` sites light up on their own.

## Back-compat contract

A key without `user_id` must behave **byte-for-byte as it does today**: recall returns
`scope = 'org'` only, `scope='private'` writes raise, project rows stay invisible. This is the
single most important property to test, because every key in production today is in that state.

## Security analysis

**This change has one genuinely new risk, and it is the reason the feature needs a decision rather
than just an implementation.**

Today a key is *safe to share*. It only ever sees org scope, so handing it to a teammate, a CI job,
or a shared runner exposes nothing personal. After this change, a key bound to a user reads that
user's **private memories and their project memories**. A shared key becomes an impersonation of
its owner.

Keys are bearer credentials, so binding a user is **attribution, not authentication** — anyone
holding the key acts as that user. That is true of any bearer token, but the blast radius grows
here from "the shared tier" to "one person's private tier."

Mitigation — make the two intents explicit at creation rather than implicit:

- **Personal key** — bound to a user, full ladder, documented as not shareable. The right default
  for a developer connecting their own agent.
- **Service key** — unattributed, org-scope only, safe for CI and shared infrastructure. Exactly
  today's behaviour, and the correct target for every key that already exists.

This also gives existing keys an honest name instead of leaving them as an unlabelled legacy state.

Secondary notes:

- `X-Klio-Agent` remains self-asserted. A key holder can still claim any agent id; agent
  attribution is a label, not an identity. Unchanged by this work, but worth not overstating in
  docs.
- The private **dedup** window (`memory.py:1112`) already requires a user and raises without one.
  Once keys carry users, key writers get the same privacy-preserving dedup behaviour OAuth writers
  have — a private write stops deduping against the org brain. That is a correctness improvement,
  not a regression, but it changes dedup outcomes for anyone who moves a key from service to
  personal.

## Test plan

Adversarial and mutation-verified, in the style the tenancy tests already use:

1. **Legacy key unchanged** — a key with no `user_id` recalls org scope only; the private and
   project arms are absent from the emitted SQL. Mutate by injecting a user and assert the test
   fails.
2. **Bound key gains exactly the right arms** — a key with `user_id` produces the same three arms
   an OAuth principal produces, with the user bound as a parameter and never interpolated.
3. **No cross-user widening** — user A's key must not return user B's private rows. Seed both;
   assert the fixture is findable by its owner (positive control) and absent for the other.
4. **Project arm still requires membership** — a bound key whose user is not in `project_members`
   sees no project rows. The existing subquery's `JOIN projects ... AND p.org_id` guard must remain
   covered by the adversarial cross-tenant membership test that already exists.
5. **Private write now succeeds for a bound key** and still raises for an unbound one.
6. **Rate-limited path** — `resolve_request_user_id` propagates 429 rather than returning `None`
   (`identity.py:163`); a rate-limited bound key must not silently degrade to an org-scope write.
   That regression is documented in the existing docstring and must stay covered.

## Rollout

No migration, no backfill, no downtime. Existing keys keep `user_id = None` and remain service
keys. New keys become personal only when the creating surface stamps a user id — which is a
dashboard change and is **out of scope here** by decision; until that ships, this work is inert in
production and safe to merge behind the freeze exemption.

## Out of scope

- **All dashboard changes**, including stamping `user_id` at key creation, the onboarding connect
  card, and any key-management UI. Explicitly deferred by Abhishek on 2026-08-12.
- Populating `project_members` — tracked separately in the dashboard repo's
  `2026-08-12-project-membership-decision.md`. Note the two are related: user-bound keys only
  unlock the project arm for people who are actually members, so that decision gates the practical
  value of this one.
- Per-key data scoping narrower than the user's own entitlements (e.g. a key restricted to one
  project). Different feature; the `scopes` field is a capability list, not a data filter.

## Open questions

1. **Personal vs service keys — one type or two?** Recommended: two, explicit at creation. Without
   the distinction, a shared key silently becomes an impersonation vector.
2. **Should a bound key's writes default to `private` or stay `org`?** Recommended: **stay `org`.**
   Write scope is already an explicit per-call parameter defaulting to `"org"`
   (`mcp_app.py:239`), and Klio's product thesis is a shared workplace. Binding a user should add
   attribution and unlock private as a *choice*, not silo every write by default.
3. Should existing keys be relabelled "service" in place, or left unlabelled until touched?
