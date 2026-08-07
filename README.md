# Vex Engine

Private repository for Vex's verification engine and backend services, including
the Klio memory brain that serves `mcp.klio.tech`.

## Services

| Service | Type | Purpose |
|---------|------|---------|
| sync-gateway | FastAPI | Sync verification + correction cascade |
| ingestion-api | FastAPI | Async event ingestion |
| async-worker | Redis Consumer | Background verification |
| storage-worker | Redis Consumer | PostgreSQL + S3 persistence |
| alert-service | Redis Consumer | Webhook alerting |
| dashboard-api | FastAPI + WS | Real-time dashboard backend |
| mcp-server | FastMCP (Streamable HTTP) | Klio memory-brain MCP service, org-scoped memory tools, host port 8085 |
| verification-engine | Library | 6-check verification pipeline |
| shared | Library | Models, auth, rate limiting |
| migrations | Alembic | Database schema management |

### mcp-server notes

- Requires API keys provisioned with the `memory` scope, which is separate from `ingest`.
- MCP transport DNS-rebinding protection defaults to **off**, because auth is the real
  boundary. For a production deployment behind a fixed host, set `MCP_ALLOWED_HOSTS`,
  `MCP_ALLOWED_ORIGINS` and `MCP_DNS_REBINDING_PROTECTION=true`.

## Development

```bash
docker compose up --build
```

Tests need a migrated database. Point `DATABASE_URL` at it, run
`alembic upgrade head` from `services/migrations`, then run `pytest` per service.
Set `KLIO_REQUIRE_DB=1` so database-gated tests fail loudly instead of skipping.

`sync-gateway` and `async-worker` also import `verification-engine`, so install it
(`pip install -e services/verification-engine`) or their tests will not collect.

## Operational endpoints

The mcp-server exposes three endpoints that answer three different questions. Wiring
them to the wrong probe causes outages rather than preventing them.

| Endpoint | Question | Touches the database |
|----------|----------|----------------------|
| `/healthz` | Is the process alive? | Never |
| `/readyz` | Can the process serve? | Yes, a bounded `SELECT 1` |
| `/metrics` | Prometheus text exposition | No |

Point the platform's **liveness** check at `/healthz` and its **readiness** check at
`/readyz`. Pointing liveness at `/readyz` turns a database outage into a restart loop
across every replica.

`/metrics` requires `KLIO_METRICS_TOKEN`. With that unset it returns 404 to everyone,
so metrics stay dark until someone deliberately turns them on. With it set, a caller
without the token gets 401 and a `WWW-Authenticate: Bearer` challenge. See
[docs/observability/grafana-cloud.md](docs/observability/grafana-cloud.md) for the
full configuration, why those two answers differ, and the metrics that are collected.

## Database constraints

Production connects to **Neon's pooled endpoint**, which is PgBouncer in transaction
pooling mode. Two consequences bind anything that touches connection setup:

1. **Never pass libpq startup parameters** through SQLAlchemy's
   `connect_args["options"]`. The pooler refuses the entire connection when it does
   not recognise one, and the refusal is total: every connection fails, so every
   request fails. Apply session settings with a `SET` on connect instead, as
   `services/mcp-server/app/db.py` does for `statement_timeout`.
2. **Local Postgres cannot reproduce this.** The test suite runs against a direct
   connection, which accepts startup parameters happily. A change to connection
   configuration can pass every test and still take production down, so
   `shared/engine_bounds.py` centralises the settings that are safe on a pooled
   endpoint and a test pins the form.

Assume prepared statements, `LISTEN`/`NOTIFY`, session-level advisory locks and temp
tables are unreliable through the pooler unless proven otherwise.

Every long-lived engine takes its bounds from `shared/engine_bounds.py`
(`pool_recycle`, `connect_timeout`). `pool_pre_ping` does not substitute for these: it
detects a connection that is already dead, not one that is slow to open.

## Deployment

Railway builds each service from GitHub source on push to `main`. That is the only
deploy path.

The `docker-publish.yml` workflow is **not** part of it. It currently fails for lack
of Docker Hub credentials, and fixing it would not change how anything ships.

Migrations do **not** run automatically. There is no migrations service in Railway, so
`alembic upgrade head` must be run against the production database by hand before
deploying a change that depends on a new schema.

Health checks use `/healthz`, not `/health`.
