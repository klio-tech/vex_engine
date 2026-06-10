# MCP OAuth for Klio — Design

**Date:** 2026-06-10
**Status:** Approved (design review with product owner)
**Scope:** Two repos — vex_engine (resource server) and Dashboard (consent UI +
Supabase config). One spec, two sequenced implementation plans.

## Problem

Connecting Klio as an MCP server today requires manually pasting an
`X-Vex-Key` (plus optional `X-Vex-Agent` / `X-Vex-Project` headers) into
client config. That blocks: one-click connect on claude.ai and Claude
Desktop (remote custom connectors effectively require OAuth), listing in
connector directories, and self-serve onboarding. The MCP authorization
spec defines the fix: OAuth 2.1 with PKCE, dynamic client registration,
and RFC 9728 protected-resource metadata.

## Decisions (made in design review)

| Topic | Decision |
|---|---|
| Authorization server | Supabase Auth's OAuth 2.1 Server (already the product's identity stack). Engine builds NO authorization endpoints. |
| Org binding for multi-org users | Org picker on the consent page (like Slack/Notion installs) |
| Default agent identity | `oauth:<user_short>/<client_name>` (client name from MCP `initialize` clientInfo); `X-Vex-Agent` header overrides |
| Header auth | Kept forever — headless agents, hooks, CI stay on `X-Vex-Key`. OAuth is additive. |
| Approach rejected | Building an AS into FastAPI (weeks of security-critical plumbing); Auth0/WorkOS (second identity system + cost) |

References: Supabase [OAuth 2.1 Server](https://supabase.com/docs/guides/auth/oauth-server),
[MCP Authentication](https://supabase.com/docs/guides/auth/oauth-server/mcp-authentication).
Note: the MCP ecosystem is moving from DCR toward Client ID Metadata
Documents (CIMD); riding Supabase inherits that transition.

## 1. Architecture

Three parties, none new:

- **Supabase Auth** — the OAuth 2.1 authorization server: discovery metadata
  at `/.well-known/oauth-authorization-server`, PKCE, dynamic client
  registration (enabled in the Supabase dashboard), token issuance, JWKS.
- **Dashboard (app.klio.tech)** — hosts login + consent + org picker.
  Supabase's OAuth server delegates the consent/authorization UI to the
  application by design.
- **vex_engine MCP server** — pure resource server. Serves RFC 9728
  protected-resource metadata via FastMCP `AuthSettings`; validates Bearer
  JWTs against Supabase's JWKS locally (no per-request Supabase round trip
  for signature checks).

Connection flow:

1. MCP client calls `/mcp` without a token → 401 +
   `WWW-Authenticate: Bearer resource_metadata="https://<mcp-host>/.well-known/oauth-protected-resource"`.
2. Client fetches the metadata → discovers the Supabase authorization
   server, registers (DCR), starts the PKCE authorize flow.
3. Browser: user logs in at app.klio.tech, picks a workspace, approves.
4. Client exchanges the code for tokens; every subsequent MCP request
   carries `Authorization: Bearer <jwt>`.

## 2. Grant model

New Supabase table `oauth_grants`:

```
user_id uuid NOT NULL          -- auth.users
oauth_client_id text NOT NULL  -- Supabase OAuth client id
account_slug text NOT NULL     -- chosen workspace
created_at timestamptz NOT NULL DEFAULT now()
revoked_at timestamptz NULL
UNIQUE (user_id, oauth_client_id) WHERE revoked_at IS NULL
```

RLS: users read/write only their own grants.

- **Consent page** (`/oauth/consent` in the Dashboard): shows the requesting
  client's name + requested access ("read and write your team's memory"),
  lists the user's account memberships (personal + team), user picks one,
  approval (a) upserts the grant row and (b) completes Supabase's OAuth
  approval call. Deny cancels the flow.
- **Connected apps** (dashboard settings): lists active grants
  (client, workspace, connected date), each with a Revoke button that sets
  `revoked_at`. Takes effect within the engine's verifier cache TTL (60 s).
- **Org resolution:** token → `user_id` (sub) + `client_id` → grant →
  `account_slug` → `organizations.org_id` — the same
  `organizations.account_slug` join `KeyValidator` already uses for plans.
  A grant whose `account_slug` no longer resolves is treated as revoked.

## 3. Engine: dual-mode principal resolution

New `SupabaseTokenVerifier` implementing the MCP SDK's `TokenVerifier`
protocol (available in the deployed SDK), validating:

1. JWT signature against Supabase JWKS (keys cached with refresh-on-miss),
2. `exp` / `iat` (small clock-skew allowance),
3. `iss` == the Supabase project's auth issuer,
4. audience/authorized-party: the token must be issued to the OAuth client
   that holds a grant (reject tokens minted for other resources).

Then it resolves the grant (Supabase query, cached 60 s per
`(user_id, client_id)` — same pattern as the key cache) and produces the
SAME principal shape the key path produces: `(org_id, agent_id, plan,
plan_overrides)`. Plan comes via the existing `_query_plan_from_supabase`.

`resolve_principal` ladder:

1. `Authorization: Bearer <jwt>` present → OAuth path (above).
2. Else `X-Vex-Key` / `X-AgentGuard-Key` → existing `KeyValidator` path,
   byte-for-byte unchanged.
3. Else → 401 with `WWW-Authenticate` resource metadata.

Because the principal shape is identical, everything downstream — memory
quotas, plan caps, redaction, recall scoping, usage metering — works
unmodified. Rate limiting for OAuth principals keys on
`user_id:client_id` instead of API-key id, with the same sliding-window
implementation.

Feature flag: `MCP_OAUTH_ENABLED` (default off). Off → today's behavior
exactly, including the current 401 without resource metadata.

## 4. Identity mapping

- **Agent:** default `oauth:<user_short>/<client_name>` where
  `user_short` is the first 8 chars of the Supabase user UUID and
  `client_name` is the MCP `initialize` `clientInfo.name` slugified
  (fallback: the OAuth client id). A user's claude.ai and Claude Code
  connections are distinct agents; `scope='agent'` private memory works
  per-surface. `X-Vex-Agent` header overrides when present.
- **Project:** `X-Vex-Project` unchanged. claude.ai web cannot send it →
  org-wide recall (correct for a chat surface); Claude Code config can.
- **Authorization scope:** an OAuth grant carries the `memory` scope ONLY.
  Ingest/verify remain key-gated. (Future per-scope consent is out of
  scope.)

## 5. Error handling

| Failure | Behavior |
|---|---|
| Missing / expired / bad-signature token | 401 + `WWW-Authenticate` with resource metadata (drives client re-auth) |
| Valid token, no active grant | 403, body: `{"error": "not_connected", "detail": "Connect this workspace at https://app.klio.tech/settings/connected-apps"}` |
| Grant's account no longer exists | Same as revoked → 403 `not_connected` |
| JWKS unreachable (cold cache) | Fail closed for Bearer requests (503 with Retry-After); key auth unaffected |
| Grant lookup query fails | Fail closed (authn is never fail-open, unlike relevance lookups) |
| Token valid but flag off | Bearer ignored; falls through ladder (key or 401) |

No token contents are ever logged; verifier logs carry `user_short` +
client id only.

## 6. Security notes

- PKCE enforced by Supabase (OAuth 2.1 mandatory).
- Tokens are validated for issuer AND audience/azp — a Supabase session JWT
  from the dashboard web app is NOT acceptable at the MCP endpoint unless
  minted through the OAuth client flow with an active grant.
- `oauth_grants` under RLS; consent approval verifies the user is a member
  of the chosen account server-side (not just UI filtering).
- Revocation is grant-level; Supabase refresh-token revocation is a bonus
  but the grant check is the enforcement point (≤60 s lag).
- DNS-rebinding posture unchanged (auth remains the boundary).

## 7. Testing

- **Unit (engine):** verifier against a self-signed JWKS fixture — valid,
  expired, wrong issuer, wrong audience, revoked grant, unknown account,
  cache expiry, JWKS refresh-on-miss.
- **Integration (engine):** 401 carries correct resource metadata when flag
  on (and today's 401 when off); Bearer → principal → tool call end-to-end
  with mocked JWKS + grants; key-auth regression suite untouched and green.
- **Dashboard:** consent page renders memberships, server-side membership
  validation, grant upsert, revoke flow.
- **Manual E2E (we are user zero):** connect from Claude Code and claude.ai
  custom connectors; verify org picker, tool calls, revoke + reconnect.

## 8. Rollout

1. Engine: verifier + ladder + metadata behind `MCP_OAUTH_ENABLED=0`.
   Deploy dark.
2. Supabase: enable OAuth 2.1 Server + dynamic client registration; set
   the authorization/consent URL to the Dashboard page.
3. Dashboard: consent page + connected-apps settings + `oauth_grants`
   migration. Deploy.
4. Flip `MCP_OAUTH_ENABLED=1`. Dogfood from Claude Code + claude.ai.
5. Docs + marketing: "Connect Klio to Claude in one click" (feeds the
   positioning fix).

## Out of scope

- CIMD (inherit from Supabase when it ships).
- Per-scope consent granularity.
- OAuth on the REST capture/hook path (stays key-based by design).
- Multi-org tokens / org switching without re-consent.
- Connector-directory listing work (follows once OAuth is live).
