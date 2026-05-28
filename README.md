# Vex Engine

Private repository for Vex's verification engine and backend services.

## Services

| Service | Type | Purpose |
|---------|------|---------|
| sync-gateway | FastAPI | Sync verification + correction cascade |
| ingestion-api | FastAPI | Async event ingestion |
| async-worker | Redis Consumer | Background verification |
| storage-worker | Redis Consumer | PostgreSQL + S3 persistence |
| alert-service | Redis Consumer | Webhook alerting |
| dashboard-api | FastAPI + WS | Real-time dashboard backend |
| mcp-server | FastMCP (Streamable HTTP) | Klio memory-brain MCP service (FastMCP over Streamable HTTP; org-scoped memory tools) — host port 8085 |
| verification-engine | Library | 6-check verification pipeline |
| shared | Library | Models, auth, rate limiting |
| migrations | Alembic | Database schema management |

### mcp-server notes

- Requires API keys provisioned with the `memory` scope (separate from `ingest`).
- MCP transport DNS-rebinding protection defaults **OFF** (auth is the real boundary). For production deployments behind a fixed host, set `MCP_ALLOWED_HOSTS` / `MCP_ALLOWED_ORIGINS` and `MCP_DNS_REBINDING_PROTECTION=true`.

## Development

```bash
docker compose up --build
```

## Deployment

Push to `main` triggers Docker image publish to Docker Hub (`7hakurg/*`).
