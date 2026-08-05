# Review remediation — design

**Date:** 2026-08-04
**Status:** Design, approved in conversation.
**Baseline:** `main` @ `218e0c5`.

---

## Where this came from

Three independent review bodies landed on the Klio Brain on 2026-08-04:

1. **Tom's Claude rig** — 3 reviewers + adversarial refuters. Verdict: ship with conditions.
2. **Tom's Codex rig** — same shape. Verdict: do not ship.
3. **A three-model sweep** (opus / gpt-5.6-sol / gemini 3.1 pro) — 122 raw findings → 75 distinct,
   explicitly **unverified**.

We then ran our own adversarial pass over all 75 with five verifiers, read-only, against the exact
reviewed commit. That pass is the reason this document exists in its current shape.

### What verification bought

| | |
|---|---|
| Refuted outright | ~12 |
| Narrowed materially | ~15 |
| Confirmed | ~40 |
| **Worse than filed** | 5 |
| **Found by verification, in no reviewer's list** | 6 |

Roughly a third of the sweep was wrong or overstated. Two examples that justify the whole exercise:

- **Finding #33 recommended giving org admins project access.** That directly contradicts the
  strict-no-override decision, and `projects.py` documents why in three places: `project_grant` is
  LLM-reachable, so blast radius must stop at projects the caller already controls. Implementing it
  would have widened a deliberately narrow boundary on an injectable surface.
- **Finding #61 called `recall_log`'s commit a bug.** It is deliberate, commented, and pinned by a
  live-Postgres regression test for the exact incident it fixes. "Fixing" it breaks a working guard.

## Product decisions this design assumes

These were settled in conversation and are **not** open:

1. **Project scope is strict. No org-admin override.** An admin gets no automatic access to project
   rows. The dashboard's existing admin override is an inconsistency to reconcile, not a precedent.
2. **Private scope is invisible to everyone, including admins.**
3. **`recall_hidden` rows are deliberate curator fuel** — excluded from recall, fed to synthesis.
4. **Observability ships as Prometheus `/metrics` → Grafana Cloud**, not structured logs alone.
   Rationale: real retained production data from day one, and alerting without infrastructure to run.

## The shape of the work

The findings collapse into **seven fix shapes**. Organising by shape rather than by finding is the
central decision of this design — several "findings" are one defect at many sites, and two pairs
must land together or the fix makes things worse.

### Shape 1 — Tenancy and visibility

The ship-gates. Everything here either discloses data across a boundary or lands a write wider than
the caller was told.

| Defect | Site | Note |
|---|---|---|
| `retrieve_memories` has no `scope`/`recall_hidden` predicate | `memory.py:630` | **Live in production.** Client-supplied `session_id` on the verification path reaches private and hidden rows, contents go to the consistency LLM |
| Artifact tools never check project membership | `mcp_app.py` get/put/forget | Worst arm: unconditional `artifact_put` into a project you are not a member of |
| `retract_memory` has no project arm | `memory.py:893` | Private and agent arms exist; a project row is retractable by any same-org caller with the id |
| Promotion UPDATE lacks a scope guard | `promotion.py:104` | Lost update can land a row wider than reported, defeating the `already_shared` guard |
| Swallowed 429 → `user_id=None` → project write lands at org | `identity.py:87` | Fires at half the configured rpm; this is a visibility bug wearing a rate-limit costume |
| Capture dedup oracle | `capture.py:264` | Client `session_id` verbatim + attributed/unattributed writes sharing one bucket |
| `supersede_memory` takes no `org_id` | `memory.py:830` | Latent; the primitive has no tenancy argument at all |

### Shape 2 — The rollback pattern

One defect, ten sites, **two** fixes. An `except` that swallows a DB error without `rollback()`
poisons a session the caller keeps using.

- **Seven sites take `db.rollback()`**: `usage.py:115`, `capture.py:313`, `capture.py:336`,
  `onboarding.py:366`, `mcp-server/app/auth.py:89`, `projects.py:307`, `graph_store.py:319`.
- **Three curator sites take `begin_nested()`**: `curator.py:797`, `:306`, `:333`. Prior writes are
  in flight there; a bare rollback discards good work.

Three of these sites were not in any reviewer's list.

### Shape 3 — Curator correctness

**Ordering constraint: shapes 2 and 3 must land in the same change.** The missing rollback at
`curator.py:797` currently causes `_advance_cursor` to raise, which is the only thing preventing
permanent row abandonment on write failure. Fixing the rollback alone converts a self-limiting retry
into silent data loss.

| Defect | Site | Note |
|---|---|---|
| Entity budget decremented on upsert **hits** | `curator.py:303` | Drops ~40% of a tick's entities **and their edges**, permanently. Bites above 12 distinct entities per tick — week one. Fix: `RETURNING id, (xmax = 0) AS inserted` |
| Cursor advances past unsynthesized rows | `curator.py:899` | Two branches, not one: whole-batch below threshold **preserves** the cursor (correct); **per-partition** below threshold advances past skipped rows (abandonment) |
| `entity_brief` selects the **oldest** 50 | `graph_store.py:585` | `explore` freezes in the past past 50 memories, with no truncation flag |
| Reconcile budget re-read per partition | `curator.py:702` | Documented per-tick, actually per-partition; worst case 16× |
| Fact filter shifts indices under LLM `src` pointers | `curator.py:231` | Replace invalid elements, do not filter |

The comment at `curator.py:878` claiming sub-threshold rows "stay recallable" is false and must be
corrected — `run_retention.py:206` deletes them on a 7-day TTL.

### Shape 4 — Availability

| Defect | Site | Note |
|---|---|---|
| No embedding timeout; connection pinned across it | `memory.py:198` | **Two** embeds per write; the second runs inside an open transaction. Pool is 5+10; the 16th concurrent call queues 30s then fails |
| MCP surface violates the threading contract | `mcp_app.py` | `capture.py` honours it at 14 sites; the tool surface honours it for S3 only. ~2.4 rps per replica ceiling |
| Scheduler sweeps block the event loop | `scheduler.py:228` | Plus an unbounded `collect_purgeable`; with a slow object store, ~1.5h of wedged loop |
| One malformed plan override hard-downs a tenant | `plan_limits.py:265` | `TypeError` escapes `verify_api_key`, which catches only `AuthError`. Not self-healing |
| Pool timeout surfaces as `403 not_connected` | `oauth.py:324` | Tells a correctly-connected user to re-authorize |

### Shape 5 — Redaction and egress

All four regex gaps were **executed**, not read:

```
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/...  -> redacted=False   (underscore defeats \b)
{"password": "s3cretV4lue!"}             -> redacted=False   (closing quote breaks \s*[:=])
redis://:hunter2secret@host:6379         -> redacted=False   (host segment needs 1+ non-colon char)
password=Passw0rdPassw0rd                -> redacted=False   (entropy 2.7500 vs 3.0 bar)
password: hunter2hunter2                 -> redacted=False   (entropy 2.8074 vs 3.0 bar)
```

There is **no AWS secret-value pattern at all** — only the access key id. The entropy gate is
strictest exactly where secrets get longer.

Separately: `sync-gateway` and `async-worker` call `write_memory` directly and **neither imports the
redactor**. LLM-generated summaries and facts land unredacted on those paths.

These gaps are the *reachable* version of the "redaction fails open to the LLM" concern — a
transcript containing any of the above passes through the redactor untouched and reaches the vendor
with no exception required.

### Shape 6 — Config hardening

Guarded parsers (`_parse_bool`, `_parse_int`, `_parse_dedup_threshold`) already exist and are
correct. They are simply **confined to two modules**. Every config trap is a place the existing
pattern was not applied.

**`RAW_OBSERVATION_TTL_DAYS` is the only data-destruction finding in the set.** Proven against real
Postgres: `0` puts the cutoff at `NOW()`, negatives put it in the future, and the DELETE has **no
`org_id` predicate** — it is a global cross-org delete by design, with age as the only bound. The
plausible operator action ("disable the sweep" with `0` or `-1`) is the destructive one.

Also: unbounded `agent_id` into `String(128)`, and a client-supplied `mcp-session-id` header taken
verbatim producing a 411-char value for a `VARCHAR(256)` column.

### Shape 7 — Observability

Prometheus `/metrics` with `remote_write` to Grafana Cloud, plus a dependency-aware `/readyz` that
reports actual embedder and DB state. `/healthz` stays a liveness probe.

Counters live in **one module** so the same instrumentation can point elsewhere later without
touching call sites.

## Data reconciliation

Three defects left residue in production. Each reconciliation job **must have a dry-run mode that
reports counts before touching anything**, and dry-run output must be reviewed before execution.

1. **Curator-skipped rows** — re-queue for distillation where the 7-day TTL has not already deleted them.
2. **Unredacted `decide` rationales** — scrub existing metadata through the redactor.
3. **Orphaned R2 objects** — reconcile the bucket against the artifacts table. Nobody currently knows
   whether orphans exist; this is the only way to find out whether the deletion promise has held.

## Explicitly out of scope

Rejected on evidence, recorded so they are not re-litigated:

- **Org-admin override on project scope** (#33) — contradicts the product decision; harmful.
- **`recall_log` borrowed-session commit** (#61) — deliberate, tested, load-bearing.
- **Keyset pagination monotonic sequence** (#69) — premise false; `created_at` is a Python clock read.
  The real 2ms skip window closes for free when the two-commit write is made atomic.
- **`_BRIEF_SUPERSEDED_LIMIT`** (#73) — structurally bounded at 250; 57 edges exist in total.
- **`SELECT DISTINCT` in curator discovery** (#66) — 0.004% duty cycle at current size; revisit at ~500K private rows.
- **`scheduler.py:204` UUID/text mismatch** (#18) — `collect_purgeable` already returns `str()`.
- **`fact_noise.py` Python 3.9 floor** (#53) — nothing runs 3.9; four other files share it. Fix the
  packaging metadata, not the file.
- **Unclosed `StreamingBody`** (#34) — the S3 client is rebuilt per call, so there is no shared pool
  to exhaust. Worth three lines for consistency, not prioritisation.

## Success criteria

- Every confirmed defect has a test that fails when the fix is reverted, reported as an observed
  mutation count.
- No existing test count drops. Baselines: shared 593, mcp-server 442, migrations 64.
- Adversarial tests carry a positive control proving the fixture is not vacuous.
- The two ordering constraints are honoured in the commit history, not just in intent.
