# Dashboard API

Real-time WebSocket API that streams execution updates to connected dashboard clients.

## Responsibilities

- Subscribes to the `executions.stored` Redis Stream (using XREAD, not a consumer group)
- Authenticates every `/ws` handshake against a short-lived, org-scoped signed token
- Routes each `execution.new` event to the connections belonging to that event's org
- Manages WebSocket connection lifecycle with automatic cleanup of disconnected clients

## Tenancy

Connections are held in a `dict[org_id, set[WebSocket]]`. `broadcast()` takes an
`org_id` and can only reach that bucket — there is no all-orgs fan-out. A stream
payload with no `org_id` is logged and dropped rather than sent to everyone.

Events on this feed carry `org_id`, `agent_id`, `session_id`, execution and parent
ids, latency, token counts, cost estimates, and the S3 key of the trace payload.
Treat any change that widens delivery as a tenancy change.

## Dependencies

### External
- **Redis** — reads from `executions.stored` stream (non-destructive XREAD)

### Internal
- None (standalone service). Authentication is a keyed hash, so it needs no
  database and no Redis: an auth path that depends on a flaky backend either
  fails open, which is a leak, or takes the feed down.

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | none | Health check |
| WS | `/ws` | WS token | Org-scoped real-time execution updates |

## Authenticating to `/ws`

The client presents a short-lived token minted by the dashboard web tier, which
signs it only after authenticating the session and asserting account membership.
The long-lived `X-Vex-Key` is **never** sent to a browser, and is not accepted
here — see the module docstring in `app/ws_auth.py` for why the `KeyValidator`
path does not apply to this endpoint.

**Preferred — `Sec-WebSocket-Protocol`.** A request header, so it stays out of
access logs, `Referer`, and browser history, and is still settable from browser
JavaScript:

```js
new WebSocket('wss://…/ws', ['vex.ws.v1', token]);
```

The server negotiates `vex.ws.v1` and never echoes the credential back.

**Fallback — query string.** For clients that cannot negotiate a subprotocol:

```
wss://…/ws?token=<token>
```

Query strings are recorded verbatim by most reverse proxies, so this path logs a
warning on every use. Prefer the header.

Any failure closes with code `1008` (policy violation) **without accepting** the
socket, and with no reason string — "no token", "wrong signature" and "expired"
are indistinguishable from outside.

### Token format

```
v1.<b64url(org_id)>.<exp>.<b64url(hmac_sha256(secret, "v1.<b64url(org_id)>.<exp>"))>
```

Base64url is unpadded, so every character is a legal RFC 7230 token character.
`exp` is an integer Unix timestamp; tokens claiming a lifetime longer than
`MAX_TOKEN_LIFETIME_S` (15 min) are refused even when correctly signed.

The Dashboard repo mints the identical string in TypeScript. Both suites pin the
same fixed vector (`app/ws_auth.TEST_VECTOR`), so the implementations cannot
drift apart without one of them going red.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `REDIS_URL` | No | Redis connection string (default: `redis://localhost:6379`) |
| `DASHBOARD_WS_TOKEN_SECRET` | **Yes** | HMAC secret for `/ws` tokens. Must match the web tier's `AGENTGUARD_WS_TOKEN_SECRET`. Unset means every handshake is refused — `/health` still answers, so the service stays up rather than flapping, but it serves nobody. |

Rotating the secret invalidates every live token; clients reconnect and mint a
new one, so the visible effect is a few seconds of stale charts.

## Running Locally

```bash
cd services/dashboard-api
pip install -e ".[dev]"
DASHBOARD_WS_TOKEN_SECRET=dev-secret uvicorn app.main:app --reload
```

## Testing

```bash
pytest tests/ -v
```
