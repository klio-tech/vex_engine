# Klio artifacts — design

**Status:** proposed, awaiting approval
**Date:** 2026-07-30
**Scope:** `services/migrations`, `services/mcp-server`, `services/shared/shared/plan_limits.py`,
a new `klio-artifacts` object-storage bucket

---

## The ask

A customer asked whether they can store their own artifacts in Klio and share them with
other agents as needed. "Artifact" was confirmed to mean three things at once:

- **files** — PDFs, images, spreadsheets, archives
- **documents** — specs, PRDs, generated code, transcripts: text too large to be a memory
- **records** — JSON one agent produces and another consumes programmatically

Klio holds the bytes in all three cases. Storing only a pointer to someone else's
storage was explicitly rejected.

The answer today is no. `session_memories` holds text and an embedding; there is no
blob column and no file concept anywhere in the memory path. The eight MCP tools each
take a single `content: str`.

## Goal

An agent can hand a file, document, or record to every other agent in the org, and a
second agent that never knew it existed finds it through the same `recall` it already
uses.

## Non-goals

- **Text extraction from binaries.** v1 does not parse PDFs or Office files into text.
  Text-native kinds return text; binaries return a download URL. Extraction is a
  follow-up with its own spec.
- **Dashboard UI and CLI commands.** Both are thin layers over the tools defined here
  and get their own specs, so this one stays shippable.
- **Artifact-to-artifact relationships**, diffing, or merge. Versioning is a linear
  supersedes chain, nothing richer.
- **Editing in place.** Artifacts are immutable; a change is a new version.

---

## Approach

Three shapes were considered.

**A — artifacts as columns on `session_memories`.** Smallest diff: recall, spaces,
scopes, `forget`, and quota all work unchanged. Rejected on two counts. Free's memory
retention is one day, so a design document would be deleted the next morning — an
indefensible behaviour that would require a second retention rule inside the same table
anyway. And it loads a second concept onto the table already carrying the vector index
and the highest write volume in the system.

**B — a separate `artifacts` table with its own embedding, UNIONed into `recall`.**
Rejected because it puts a 40KB document's embedding in the same ranked list as a
110-character fact. Long-text embeddings average out into mush; that comparison either
swamps the memories or vanishes beneath them, and no amount of score normalisation
makes it meaningful. Two vector indexes for a worse result.

**C — artifact in object storage, a short card in memory. Chosen.**

The bytes live in an `artifacts` row (small text inline in Postgres) or in S3. On write,
the same transaction inserts **one short `session_memories` row** — the *card* — holding
the title, a one-line summary, the kind, the size, and the artifact id. The card is
embedded and recalled exactly like any other memory.

This wins for four reasons:

1. **Ranking stays homogeneous.** Cards are short, like every other memory, so B's
   hardest problem does not arise.
2. **Recall payloads are bounded by construction.** `recall` returns up to 50 results;
   if artifacts returned full text, one call could push megabytes into a context window.
   Returning a card cannot.
3. **Artifacts get their own lifecycle** — retention, versioning, quota — without
   touching memory's.
4. **The card is independently useful.** It is what a human reads in the Memory and
   Sessions views.

The cost is one extra hop: `recall` surfaces the card, and the agent calls
`artifact_get` when it actually wants the content. That hop is the deliberate trade.

---

## Data model

### `artifacts` (new table, migration 027)

| Column | Type | Notes |
| --- | --- | --- |
| `id` | `uuid` PK | |
| `org_id` | `varchar(64)` NOT NULL | The tenancy boundary. Present in every query. |
| `space_id` | `uuid` NULL → `spaces` | Relevance scoping. Fails open, as spaces do today. |
| `project_id` | `uuid` NULL | |
| `agent_id` | `varchar` NOT NULL | Who wrote it. |
| `title` | `varchar(200)` NOT NULL | |
| `summary` | `text` NOT NULL | Agent-supplied. Becomes the card body. |
| `kind` | `varchar(16)` NOT NULL | `document` \| `record` \| `file` |
| `mime_type` | `varchar(128)` NOT NULL | |
| `size_bytes` | `bigint` NULL | Measured server-side and set when the row goes `active`. Null while `pending` — the client's declared size is never stored. |
| `checksum` | `varchar(64)` NULL | SHA-256, computed server-side at the same moment. Null while `pending`, because the bytes do not exist yet. |
| `inline_text` | `text` NULL | Set when the artifact is text ≤ 64KB. |
| `storage_uri` | `text` NULL | `s3://klio-artifacts/{org_id}/{artifact_id}/v{version}` otherwise. |
| `version` | `int` NOT NULL default 1 | |
| `supersedes` | `uuid` NULL → `artifacts` | Linear chain, same idiom as memory. |
| `status` | `varchar(16)` NOT NULL default `'pending'` | `pending` \| `active` \| `superseded` \| `retracted` |
| `created_at` | `timestamptz` NOT NULL default `now()` | |

**Invariant:** exactly one of `inline_text` and `storage_uri` is non-null on an `active`
row. Enforced by a `CHECK` constraint, because a violation here means an artifact that
reads as present but has no bytes.

Indexes:

- `(org_id, status)` — the list path.
- `(org_id, space_id, status)` — space-scoped listing.
- unique `(org_id, checksum, status) WHERE status = 'active'` — deduplication. Re-putting
  identical bytes returns the existing artifact rather than storing a second copy.

Text ≤ 64KB lives inline in Postgres. That covers essentially every spec, JSON record,
and code file, and it means the common read path never touches S3.

### The card

Written into `session_memories` in the same transaction as the artifact row:

```
content     = "[artifact] {title} — {summary} ({kind}, {human_size})"
memory_type = 'artifact'
metadata    = {"artifact_id": "<uuid>", "source": "mcp"}
scope       = 'org'          -- copied from the artifact's intended scope
space_id    = artifact.space_id
project_id  = artifact.project_id
agent_id    = artifact.agent_id
```

`memory_type = 'artifact'` is new. Every existing consumer treats `memory_type` as an
opaque string, so nothing breaks; the dashboard can key a chip off it later.

Cards are written **only for `active` artifacts** — see the confirm step below. An
abandoned upload therefore never appears in recall.

---

## MCP tools

Two new tools. `recall` needs no code change.

### `artifact_put`

```
artifact_put(
  title: str,                     # ≤ 200 chars
  summary: str,                   # ≤ 500 chars, becomes the card
  kind: 'document'|'record'|'file',
  content: str        = None,     # inline path
  mime_type: str      = None,     # required when content is omitted
  size_bytes: int     = None,     # required when content is omitted (declared, then verified)
  space: str          = None,
  project: str        = None,
  supersedes: str     = None,     # artifact id this replaces
  artifact_id: str    = None,     # confirm path only
) -> dict
```

Three behaviours, selected by which arguments are present:

**1. Inline** (`content` given). Content is stored in `inline_text`, the artifact goes
straight to `active`, the card is written, and the call returns
`{artifact_id, version, status: 'active'}`. One round trip. Rejected with a clear error
if `content` exceeds 64KB — the error names the presigned path.

**2. Presigned request** (`content` omitted). Creates a `pending` artifact and returns
`{artifact_id, upload_url, expires_at, method}`. For `size_bytes` over 100MB it instead
returns `{artifact_id, upload_id, part_urls[], expires_at}` for a multipart upload.
No card is written yet. URLs expire in 15 minutes.

**3. Confirm** (`artifact_id` given, no `content`). The server issues a `HEAD` against
the object, takes the **real** size and computes the checksum, enforces the per-file and
pool limits against those measured values, flips the row to `active`, and writes the
card. Returns the same shape as the inline path.

Splitting request from confirm is what makes the size limit enforceable: a client's
declared `size_bytes` only selects the upload strategy, and the measured size is what
the quota is charged against.

**Garbage collection:** `pending` artifacts older than 24 hours are deleted along with
any uploaded object, by the existing scheduler in `services/mcp-server/app/scheduler.py`.

Inline is not a convenience — it is the only write path available to ChatGPT connectors
and Claude.ai custom connectors, which can make MCP tool calls but cannot issue an
arbitrary HTTP PUT. Without it those clients could read artifacts but never create one.

### `artifact_get`

```
artifact_get(
  artifact_id: str,
  mode: 'text'|'url' = 'text',
  offset: int = 0,
) -> dict
```

- `mode='text'` — returns `inline_text`, or the object's contents for text MIME types,
  capped at **50KB per call** with `offset` for paging and `has_more` in the response.
  For a binary artifact it returns an error naming `mode='url'`; v1 does no extraction.
- `mode='url'` — returns a presigned GET valid for 15 minutes. For an inline artifact,
  Klio writes the text to a temporary object so the URL contract holds for every kind.

### `forget`

Extended to accept an artifact id. Sets the artifact to `retracted` and its card to
`recall_hidden = true` in one transaction — one lifecycle, no orphaned cards. Bytes are
deleted from object storage by the same scheduler pass that collects abandoned uploads.

### `recall`

No code change. Cards surface as ordinary results carrying
`memory_type: 'artifact'` and `metadata.artifact_id`. The tool **description** gains one
sentence directing agents to call `artifact_get` when a result is an artifact — the
description is how an agent learns the follow-up exists.

---

## Plan limits

Storage is the value metric: a running level, not a monthly flow, so it does not belong
in `memory_usage_counters` (which is per-month and resets).

Two fields added to `PlanConfig` in `services/shared/shared/plan_limits.py`, beside the
existing memory fields and under the same "mirror in `Dashboard/apps/landing/lib/pricing.ts`"
rule:

```python
artifact_max_file_bytes: int = 0     # 0 disables artifacts entirely
artifact_storage_bytes: int = 0      # total active bytes for the org
```

| Plan | Per file | Pool |
| --- | --- | --- |
| `free` | 10 MB | 100 MB |
| `starter` ($29) | 50 MB | 5 GB |
| **`max` ($499)** | **1 GB** | **1 TB** |

`pro` and `team` are left at the `starter` values until the pricing restructure that
introduces `max`; artifacts do not force that restructure now. **`max` must be added as a
key to `PLAN_LIMITS`** — `get_plan_config` falls back to `free` for an unknown plan, so a
`max` customer would silently get 10MB otherwise.

Artifacts are available on every plan. Capacity is what is paid for, not the feature:
connecting agents and handing work between them is the wedge, which is why every tier
already advertises unlimited connected agents. A Free user who cannot share a single file
never reaches the moment the product is sold on.

**Enforcement.** Current usage is `SELECT COALESCE(SUM(size_bytes), 0) FROM artifacts
WHERE org_id = $1 AND status = 'active'`. A running counter would be faster but can drift;
a sum cannot, and artifact counts are in the thousands, not millions. Revisit if that
stops being true.

Over the pool, **writes are rejected and reads always succeed** — the existing hard-limit
idiom, and we do not hold a customer's data hostage. The error names the current usage,
the limit, and the plan needed.

## Retention

**Artifacts are exempt from the memory retention sweep.** Free's one-day memory retention
deleting an uploaded design document would be indefensible. Artifacts live until they are
forgotten, superseded, or the org's pool fills. The pool is the pressure.

**The card must be exempt from the sweep too, and today it would not be.** Migration
`025_enforce_memory_retention` defines:

```sql
DELETE FROM session_memories
WHERE org_id = p_org_id
  AND created_at < NOW() - (p_retention_days || ' days')::interval;
```

There is no `memory_type` filter. On Free, an artifact card would be deleted the day
after it was written while its artifact survived — leaving bytes that are paid for,
listed in the dashboard, and unreachable by `recall`. This is the single most likely
way to ship a silently broken feature.

**Migration 028 replaces the function** to exclude cards whose artifact is still live:

```sql
DELETE FROM session_memories sm
WHERE sm.org_id = p_org_id
  AND sm.created_at < NOW() - (p_retention_days || ' days')::interval
  AND NOT (
    sm.memory_type = 'artifact'
    AND EXISTS (
      SELECT 1 FROM artifacts a
      WHERE a.id = (sm.metadata->>'artifact_id')::uuid
        AND a.status = 'active'
    )
  );
```

A card therefore outlives the retention window exactly as long as its artifact does, and
becomes collectable the moment the artifact is retracted or superseded.

## Versioning

`artifact_put` with `supersedes` creates version *n+1*, sets the previous row to
`superseded`, and sets the previous card's `recall_hidden = true`. Recall only ever
surfaces the newest version. Superseded bytes count against the pool until forgotten,
which is deliberate: versions are not free.

---

## Security

- **Tenancy.** `org_id` is a predicate inside every query. An artifact id guessed from
  another org reads as not-found, matching the memory and session paths.
- **Presigned URLs are bearer credentials.** 15-minute TTL, scoped to a single object,
  never written to logs. This is the one new exposure the design introduces and is the
  reason for the short TTL.
- **Server-measured size and checksum.** A client's declared size selects a strategy and
  nothing more; quota is charged on what was actually stored.
- **Spaces remain a relevance filter, not a security control** — unchanged from memory.
  Anything in the org is reachable by anything else in the org. This must be stated in
  the docs alongside the same warning memory already carries.
- **MIME type is recorded, not trusted.** Nothing in Klio executes an artifact, and
  download responses set `Content-Disposition: attachment` plus
  `X-Content-Type-Options: nosniff`.

## Testing

Mirroring the loader-test pattern already used in the repo:

- Parameters are bound, never interpolated; `kind` and `mode` come from whitelists.
- Cross-tenant `artifact_get` and `forget` return not-found.
- The one-of-`inline_text`/`storage_uri` constraint rejects a row violating it.
- No card exists for a `pending` artifact; a card appears exactly on confirm.
- GC removes a `pending` artifact older than 24h and its object.
- Over-pool write is rejected while `artifact_get` on an existing artifact still succeeds.
- Declared `size_bytes` under the limit with real bytes over it is rejected at confirm.
- Deduplication: identical bytes return the existing artifact id.
- Retention: on a 1-day window, an artifact card whose artifact is `active` survives the
  sweep; the same card becomes collectable once the artifact is retracted.
- Supersede hides the old card and leaves exactly one active version in recall.
- End-to-end: agent A writes, agent B recalls a query it never saw and gets the card.

## Migration

Two migrations. `027_artifacts.py` is additive only — a new table, no change to
`session_memories`. `028_retention_excludes_artifact_cards.py` replaces the
`enforce_memory_retention` function as shown above; it must land **before** any card is
written, or Free-plan artifacts start orphaning within a day. The
new `memory_type` value needs no migration. The `klio-artifacts` bucket is created
separately from `agentguard-traces`; the two must not share a bucket, because trace
payloads are internal and artifacts are customer data with different retention.

Rollback is a table drop plus bucket removal. Cards would remain and point at nothing, so
the down migration also deletes `session_memories` rows with `memory_type = 'artifact'`.

---

## Open decisions

1. **Does `recall` need an artifacts-only filter?** An agent that knows it wants a file
   currently has to search semantically and hope. A `memory_type` filter on `recall`
   would fix it cheaply. Recommendation: defer — ship without, add if it is asked for.
2. **Should the card's summary be LLM-generated when an agent supplies a poor one?**
   v1 requires the agent to write it. Recommendation: keep it required; a curator pass
   over weak summaries is a natural follow-up given the curator already exists.
3. **`pro` and `team` storage values.** Left at `starter` levels pending the pricing
   pass. They must be set deliberately before `max` ships publicly.
