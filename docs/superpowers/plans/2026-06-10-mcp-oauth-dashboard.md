# MCP OAuth (Dashboard Side) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Dashboard half of `docs/superpowers/specs/2026-06-10-mcp-oauth-design.md` (this private engine repo): the `oauth_grants` Supabase migration, the OAuth consent page with workspace picker, and a Connected-apps settings page with revoke — in the **public open-source** Dashboard repo at `/Users/thakurg/Hive/Research/AgentGuard/vex_public/Dashboard`.

**Architecture:** Supabase Auth (OAuth 2.1 Server, enabled in the Supabase dashboard) redirects authorize requests to our consent page with an authorization id; the page authenticates the user, lists their workspaces, and on approval (a) upserts an `oauth_grants` row binding `(user, oauth_client)` → `account_slug` and (b) completes the Supabase OAuth approval. The engine (already merged, dark) resolves grants via `SELECT account_slug FROM oauth_grants WHERE user_id=… AND oauth_client_id=… AND revoked_at IS NULL` — **that contract is frozen**; the migration must match it exactly. UI work mirrors the existing api-keys settings feature file-for-file.

**Tech Stack:** Next.js 16 App Router (React 19), Makerkit kit packages (`@kit/supabase`, `@kit/ui`, `@kit/next`), Supabase migrations + RLS, Zod + `enhanceAction` server actions, Vitest. Repo: turbo monorepo, app at `apps/web`. Commands: `pnpm --filter web typecheck`, `pnpm --filter web lint`, `pnpm --filter web test` (vitest), `pnpm supabase:web:reset` (local DB).

**OPEN-SOURCE RULE (applies to every task):** this repo is public. No secrets, no internal/production URLs or identifiers (Neon hosts, org ids, Supabase project refs), no references to private-repo internals in code, comments, tests, migration comments, or commit messages. Test fixtures use obviously fake values (`user-1111…`, `client_fake_abc`). The only product URLs allowed are the public ones already used in the repo (app.klio.tech / klio.tech docs).

**Engine contract (frozen — from the merged engine half):**
- Table `public.oauth_grants` with columns `user_id uuid`, `oauth_client_id text`, `account_slug text`, `created_at timestamptz`, `revoked_at timestamptz null`.
- Active grant = `revoked_at IS NULL`; at most one active grant per `(user_id, oauth_client_id)`.
- The engine queries via the service-level Postgres connection (bypasses RLS) — RLS here protects the *Dashboard's* user-facing reads/writes.

**Working branch:** create `feat/mcp-oauth-consent` off `origin/main` (the checkout currently sits on a stale feature branch — `git fetch origin && git checkout -b feat/mcp-oauth-consent origin/main`).

---

## File structure

| File | Responsibility |
|---|---|
| Create `apps/web/supabase/migrations/<ts>_oauth_grants.sql` | Table + RLS + partial unique index |
| Create `apps/web/lib/oauth/consent-service.ts` | Wrapper over Supabase's OAuth-server consent API (details pinned in Task 2) |
| Create `apps/web/app/oauth/consent/page.tsx` + `_components/consent-card.tsx` + `_lib/server/consent-actions.ts` + `_lib/server/consent.loader.ts` + `_lib/schema/consent.schema.ts` | Consent page: auth gate, workspace picker, approve/deny |
| Create `apps/web/app/home/(user)/settings/connected-apps/…` (path adapted to the repo's actual user-settings segment) | Connected-apps list + revoke (mirrors api-keys feature) |
| Modify `apps/web/public/locales/en/agentguard.json` | `connectedApps.*` + `oauthConsent.*` strings |
| Modify `apps/web/config/paths.config.ts` | Consent + settings paths |
| Tests beside each `_lib/server/*` module (vitest `*.test.ts`) | Actions/loaders with mocked Supabase client |

---

### Task 1: `oauth_grants` migration + RLS

**Files:**
- Create: `apps/web/supabase/migrations/20260611000000_oauth_grants.sql`

- [ ] **Step 1: Read precedents** — `apps/web/supabase/migrations/20221215192558_schema.sql` (RLS idiom: enable RLS → revoke all → grant to authenticated/service_role → policies via `has_role_on_account`) and `20260219000000_add_vex_plan_columns.sql` (style for newer migrations).

- [ ] **Step 2: Write the migration:**

```sql
/*
 * MCP OAuth grants: which workspace a user connected to an OAuth client.
 *
 * A row binds (user, oauth client) -> one workspace (account slug). The
 * memory engine treats `revoked_at IS NULL` as active; revoking from the
 * Connected-apps settings page sets `revoked_at` and takes effect within
 * the engine's auth-cache TTL. At most one ACTIVE grant per (user, client):
 * re-consenting to a different workspace revokes the previous grant first
 * (handled by the approve server action; the partial unique index enforces
 * the invariant).
 */
create table if not exists public.oauth_grants (
  id uuid primary key default extensions.uuid_generate_v4(),
  user_id uuid not null references auth.users (id) on delete cascade,
  oauth_client_id text not null,
  account_slug text not null,
  created_at timestamptz not null default now(),
  revoked_at timestamptz
);

comment on table public.oauth_grants is
  'Workspace consents for OAuth-connected MCP clients (active when revoked_at is null)';

-- One ACTIVE grant per (user, client).
create unique index if not exists uq_oauth_grants_active
  on public.oauth_grants (user_id, oauth_client_id)
  where revoked_at is null;

-- Settings page lists a user's grants newest-first.
create index if not exists ix_oauth_grants_user
  on public.oauth_grants (user_id, created_at desc);

alter table public.oauth_grants enable row level security;

revoke all on public.oauth_grants from authenticated, service_role;
grant select, insert, update on table public.oauth_grants to authenticated;
grant select, insert, update, delete on table public.oauth_grants to service_role;

-- Users see and manage only their own grants. INSERT additionally requires
-- membership of the chosen workspace: the slug must resolve to an account
-- the user can read under the accounts RLS (membership or ownership).
create policy oauth_grants_select on public.oauth_grants
  for select to authenticated
  using ((select auth.uid()) = user_id);

create policy oauth_grants_insert on public.oauth_grants
  for insert to authenticated
  with check (
    (select auth.uid()) = user_id
    and exists (
      select 1 from public.accounts a
      where a.slug = account_slug
         or (a.is_personal_account and a.id = (select auth.uid()))
    )
  );

create policy oauth_grants_update on public.oauth_grants
  for update to authenticated
  using ((select auth.uid()) = user_id)
  with check ((select auth.uid()) = user_id);
```

NOTE for the implementer: check how `accounts.slug` is populated for PERSONAL accounts in this schema (some Makerkit versions leave personal-account slug NULL and key personal workspaces by account id). If personal slugs are NULL, the spec's `account_slug` contract still holds for the ENGINE (the engine's `organizations.account_slug` rows exist for every org it tracks) — but the consent flow must then store the value that matches `organizations.account_slug` for personal workspaces. Inspect `apps/web/supabase/migrations/*schema*.sql` for the slug trigger and how the api-keys feature's `resolveOrgId(accountSlug)` maps slugs → engine org ids (`apps/web/app/home/[account]/settings/api-keys/_lib/server/`); reuse the SAME identifier the api-keys flow uses. Adjust the INSERT policy subquery accordingly and document the choice in the migration comment. This is the one open mapping question — resolve it from the code, do not guess.

- [ ] **Step 3: Apply locally** — `pnpm supabase:web:reset` (or `supabase db push` against the local stack per repo scripts) and verify: `\d public.oauth_grants` shows the columns/indexes; inserting a row as an authenticated test user for someone else's `user_id` is rejected by RLS. If the repo has SQL tests (`pnpm supabase:web:test`), add one for the unique-active-grant invariant.

- [ ] **Step 4: Typegen** — `pnpm supabase:web:typegen` so `oauth_grants` lands in the generated DB types (server actions need it).

- [ ] **Step 5: Commit** — `feat(oauth): oauth_grants table with RLS (one active grant per user+client)`. NO Co-Authored-By.

---

### Task 2: Pin the Supabase consent API and build `consent-service.ts`

**Files:**
- Create: `apps/web/lib/oauth/consent-service.ts`
- Create: `apps/web/lib/oauth/consent-service.test.ts`

- [ ] **Step 1 (MANDATORY discovery — do not skip):** Pin the exact Supabase OAuth-server consent contract against current docs. Fetch and read:
  - https://supabase.com/docs/guides/auth/oauth-server/getting-started
  - https://supabase.com/docs/guides/auth/oauth-server/oauth-flows
  - https://supabase.com/docs/guides/auth/oauth-server/mcp-authentication
  Determine: (a) the query param the authorize redirect sends to our consent page (expected: an `authorization_id`); (b) the API to fetch authorization details (client name, scopes) — SDK method (e.g. `supabase.auth.oauth.getAuthorizationDetails(...)`) or REST endpoint under `<project>/auth/v1/oauth/...` using the signed-in user's access token; (c) the approve/deny API and its redirect contract (it returns the `redirect_to` URL for the client callback); (d) whether `@supabase/supabase-js` 2.95.3 exposes these (check `node_modules/@supabase/auth-js` exports) or a version bump / direct `fetch` is needed. Record findings as a comment block at the top of consent-service.ts (public-safe: docs links only).

- [ ] **Step 2: Implement `consent-service.ts`** with this interface (stable regardless of transport):

```typescript
export interface AuthorizationDetails {
  authorizationId: string;
  clientId: string;
  clientName: string | null;
  scopes: string[];
  redirectHint: string | null;
}

export async function getAuthorizationDetails(
  client: SupabaseClient,
  authorizationId: string,
): Promise<AuthorizationDetails>;

export async function approveAuthorization(
  client: SupabaseClient,
  authorizationId: string,
): Promise<{ redirectTo: string }>;

export async function denyAuthorization(
  client: SupabaseClient,
  authorizationId: string,
): Promise<{ redirectTo: string }>;
```

Transport per Step 1's findings (SDK calls if available; otherwise `fetch` against the project auth URL with the user's session token from `client.auth.getSession()`). Every function throws a typed `ConsentServiceError` with a user-presentable message on failure — no silent nulls. If a supabase-js bump is required, bump the catalog pin and run the full typecheck.

- [ ] **Step 3: Tests** — vitest with a mocked transport (mock `fetch` or the SDK namespace): details-parse happy path, approve returns redirectTo, HTTP error → ConsentServiceError, expired/unknown authorization id → ConsentServiceError.

- [ ] **Step 4: Commit** — `feat(oauth): consent service wrapping Supabase OAuth-server API`.

---

### Task 3: Consent page (auth gate + workspace picker + approve/deny)

**Files:**
- Create: `apps/web/app/oauth/consent/page.tsx`
- Create: `apps/web/app/oauth/consent/_components/consent-card.tsx`
- Create: `apps/web/app/oauth/consent/_lib/server/consent-actions.ts` (+ `.test.ts`)
- Create: `apps/web/app/oauth/consent/_lib/server/consent.loader.ts` (+ `.test.ts`)
- Create: `apps/web/app/oauth/consent/_lib/schema/consent.schema.ts`
- Modify: `apps/web/config/paths.config.ts`, `apps/web/public/locales/en/agentguard.json`

Mirror the api-keys feature's structure exactly (`apps/web/app/home/[account]/settings/api-keys/…` is the template for page/loader/actions/schema separation, `withI18n`, `enhanceAction`, `requireUser`).

- [ ] **Step 1: Loader** (`consent.loader.ts`): given the authenticated user, return `{ details: AuthorizationDetails, workspaces: Array<{ slug: string; name: string; isPersonal: boolean }> }`. Workspaces come from the accounts the user can read (`client.from('accounts').select('slug, name, is_personal_account')` — RLS scopes it to memberships; map with the personal-slug decision from Task 1). Details via `getAuthorizationDetails`. Unauthenticated → the PAGE redirects to sign-in with `next=` back to the consent URL (reuse the repo's existing sign-in redirect helper — find it where other authenticated pages redirect, e.g. `requireUserInServerComponent` / paths config).

- [ ] **Step 2: Schema** (`consent.schema.ts`):

```typescript
import { z } from 'zod';

export const ConsentDecisionSchema = z.object({
  authorizationId: z.string().min(1),
  decision: z.enum(['approve', 'deny']),
  accountSlug: z.string().min(1).optional(),
}).refine((v) => v.decision === 'deny' || !!v.accountSlug, {
  message: 'accountSlug is required to approve',
});
```

- [ ] **Step 3: Actions** (`consent-actions.ts`): one `submitConsentAction = enhanceAction(async (data) => {...}, { schema: ConsentDecisionSchema })`:
  - `requireUser(getSupabaseServerClient())`.
  - **deny** → `denyAuthorization(...)` → return `{ redirectTo }`.
  - **approve** → server-side membership re-validation (query accounts by slug under RLS; reject if not readable — never trust the picker), then in order: (1) revoke any existing active grant for `(user, clientId)` (`update oauth_grants set revoked_at = now() where user_id = ... and oauth_client_id = ... and revoked_at is null`), (2) insert the new grant row, (3) `approveAuthorization(...)`. If the Supabase approval call FAILS after the grant insert, revoke the just-inserted grant (compensating update) and rethrow — never leave a grant active for an approval that didn't complete. Return `{ redirectTo }`.
  - The client id comes from `getAuthorizationDetails` server-side (NOT from the form) so a forged form can't bind a grant to a different client.

- [ ] **Step 4: Page + card component**: server component reads `searchParams` (authorization id param name per Task 2 findings), runs the loader, renders `ConsentCard` (client component): client name, "wants to read and write your team's memory" copy (i18n: `agentguard:oauthConsent.*`), a workspace radio list (personal first, then teams), Approve/Deny buttons with `useTransition`, on success `window.location.assign(redirectTo)`. Invalid/expired authorization id → friendly error card with a link to docs. Use `@kit/ui` Card/Button/RadioGroup primitives; match the repo's visual conventions (check an existing standalone card page like the join-team flow for layout).

- [ ] **Step 5: Tests** — actions: approve happy path (grant row written, approval called, order verified), deny path (no grant), forged-slug rejection (membership validation fails), compensating revoke when approval throws. Mock `@kit/supabase` server client + consent-service.

- [ ] **Step 6:** `pnpm --filter web typecheck && pnpm --filter web lint && pnpm --filter web test` → green. Commit `feat(oauth): consent page with workspace picker`.

---

### Task 4: Connected-apps settings page (list + revoke)

**Files:**
- Create: `apps/web/app/home/(user)/settings/connected-apps/page.tsx` — FIRST verify the actual user-level settings path segment in this repo (`ls apps/web/app/home` — Makerkit uses `(user)` for personal context; if this repo only has `[account]` settings, place it there and scope the QUERY to the signed-in user regardless of account context, noting grants are user-scoped).
- Create: `…/connected-apps/_components/connected-apps-table.tsx`
- Create: `…/connected-apps/_lib/server/connected-apps-actions.ts` (+ `.test.ts`)
- Create: `…/connected-apps/_lib/server/connected-apps.loader.ts` (+ `.test.ts`)
- Create: `…/connected-apps/_lib/schema/connected-apps.schema.ts`
- Modify: i18n json, paths config, and the settings navigation config (find where api-keys registers its settings-nav entry and mirror it).

- [ ] **Step 1: Loader**: `client.from('oauth_grants').select('id, oauth_client_id, account_slug, created_at, revoked_at').is('revoked_at', null).order('created_at', { ascending: false })` — RLS scopes to the user. (Client display names: show `oauth_client_id` plus the workspace; if Task 2 discovered a client-metadata lookup API, enrich best-effort, else skip — YAGNI.)

- [ ] **Step 2: Action**: `revokeGrantAction = enhanceAction(...)` with `z.object({ grantId: z.string().uuid() })`: `update oauth_grants set revoked_at = now() where id = :id and revoked_at is null` under the user's RLS client (policy enforces ownership); return `{ success }`. Test: revoke marks the row; unknown/foreign id → no row updated → action returns failure without leaking existence.

- [ ] **Step 3: Page + table**: mirror `api-keys-table.tsx` (Table, Badge for workspace, AlertDialog confirm on revoke, `useTransition` + `router.refresh()`). Empty state: "No connected apps yet — connect Klio from Claude and it will appear here." (i18n `agentguard:connectedApps.*`).

- [ ] **Step 4:** typecheck + lint + test green. Commit `feat(oauth): connected-apps settings with revoke`.

---

### Task 5: Full verification + runbook

- [ ] **Step 1:** `pnpm --filter web typecheck && pnpm --filter web lint && pnpm --filter web test` — all green.
- [ ] **Step 2:** Local end-to-end sanity with the local Supabase stack (`pnpm supabase:web:reset`, run the app): hit `/oauth/consent?<param>=fake-id` signed out → redirected to sign-in → back to consent → friendly invalid-authorization error (real authorization ids need the Supabase OAuth server enabled, which is a hosted-dashboard step — the error path IS the testable path locally). Settings page renders empty state.
- [ ] **Step 3: OPEN-SOURCE sweep:** `git diff origin/main...HEAD` and grep the full diff for secrets/internal identifiers (`sk_`, `whsec_`, `postgresql://`, `neon`, `supabase.co/` project refs, engine-internal hostnames). Must be clean.
- [ ] **Step 4:** Commit any remaining test additions. The PRODUCTION runbook (lives in the PRIVATE engine repo, not here): enable OAuth 2.1 Server + dynamic client registration in the hosted Supabase dashboard and set the authorization/consent URL to `https://app.klio.tech/oauth/consent`; apply the migration to hosted Supabase; deploy Dashboard; then set the engine's `MCP_OAUTH_ENABLED=1` + `SUPABASE_AUTH_ISSUER` + `MCP_PUBLIC_URL` (+ `SUPABASE_DATABASE_URL` already set) and dogfood from Claude Code/claude.ai.
