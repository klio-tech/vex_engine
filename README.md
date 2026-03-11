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
| verification-engine | Library | 6-check verification pipeline |
| shared | Library | Models, auth, rate limiting |
| migrations | Alembic | Database schema management |

## Development

```bash
docker compose up --build
```

## Deployment

Push to `main` triggers Docker image publish to Docker Hub (`7hakurg/*`).
