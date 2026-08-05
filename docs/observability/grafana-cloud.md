# Metrics: `/metrics`, `/readyz`, and shipping to Grafana Cloud

## What exists

The MCP server exposes two operational endpoints, plus the liveness probe it
already had:

| Endpoint   | Question it answers          | Touches the database |
|------------|------------------------------|----------------------|
| `/healthz` | Is the process alive?        | **No, ever**         |
| `/readyz`  | Can the process serve?       | Yes, `SELECT 1`      |
| `/metrics` | Prometheus text exposition   | No                   |

`/healthz` and `/readyz` are deliberately different questions. On 2026-08-04 a
`statement_timeout` applied as a libpq startup parameter was refused by Neon's
pooled endpoint, because a PgBouncer transaction pooler rejects unrecognised
startup parameters. Every database connection was refused and every recall
failed, while `/healthz` returned 200 for the whole incident. `/readyz` is the
signal that was missing, and it goes 503 on exactly that failure.

Point your platform's **readiness** check at `/readyz` and its **liveness** check
at `/healthz`. Never point liveness at `/readyz`: a database outage would then
restart every replica in a loop.

`/readyz` bounds its own database round-trip at 3 seconds, so a hung database
cannot make the probe itself hang. Note that a suspended Neon compute can exceed
that on a cold start, which reads as a brief 503 until the first connection
succeeds.

## Required configuration

### `KLIO_METRICS_TOKEN`, required to expose metrics at all

`/metrics` is **not world-readable**. The exposition describes write and recall
volume, error and quota rates, curator progress and connection-pool pressure.

```
KLIO_METRICS_TOKEN=<a long random string you generate>
```

Generate it with `openssl rand -hex 32`. Scrapers send it as
`Authorization: Bearer <token>`.

**With the variable unset, `/metrics` returns 404 to everyone.** This fails
closed on purpose. The alternative is serving openly until someone remembers to
set it, which makes the safe state opt-in and the default state a silent leak. A
denied caller gets a 404 rather than a 401, so it cannot even confirm the
endpoint exists.

### Grafana Cloud remote_write, optional and inert when unset

```
GRAFANA_CLOUD_PROMETHEUS_URL=https://prometheus-prod-XX-prod-YY.grafana.net/api/prom/push
GRAFANA_CLOUD_USER=<your numeric Grafana Cloud instance ID>
GRAFANA_CLOUD_API_KEY=<a Grafana Cloud access policy token with metrics:write>
```

All three must be set together. With any of them missing the configuration reads
as `None` and nothing is enabled, because a URL without a credential would
retry-loop against a 401 forever, which is worse than no metrics.

Find all three in Grafana Cloud under **Home → Connections → Add new connection →
Hosted Prometheus metrics**. The URL and username are shown there, and the token
is created under **Access Policies**.

The API key is never logged, never included in a `repr`, and never returned by
any endpoint. Do not commit it. Set it in the platform's environment settings.

## Shipping the samples

This repository exposes `/metrics` and reads the configuration above. It does not
run a shipper itself, because the deployment target runs one process per service
and an in-process shipper thread is a second thing to keep alive. Run Grafana
Alloy (or `prometheus-agent`) alongside, pointed at the endpoint:

```alloy
prometheus.scrape "klio" {
  targets    = [{ __address__ = "klio-mcp.internal:8000" }]
  metrics_path = "/metrics"
  scrape_interval = "30s"

  // The token from KLIO_METRICS_TOKEN.
  bearer_token = sys.env("KLIO_METRICS_TOKEN")

  forward_to = [prometheus.remote_write.grafana.receiver]
}

prometheus.remote_write "grafana" {
  endpoint {
    url = sys.env("GRAFANA_CLOUD_PROMETHEUS_URL")
    basic_auth {
      username = sys.env("GRAFANA_CLOUD_USER")
      password = sys.env("GRAFANA_CLOUD_API_KEY")
    }
  }
}
```

## What is measured

Every metric is declared in `services/shared/shared/metrics.py` and nowhere else,
so the backend can be swapped without touching a call site.

- **Writes and recalls**: `klio_memory_writes_total`,
  `klio_memory_recalls_total` (by outcome: `ok`, `quota`, `error`) and the
  matching `_seconds` histograms. `klio_tool_calls_total` breaks the same down by
  tool.
- **Embedding**: `klio_embedding_calls_total` (`ok`, `timeout`, `error`),
  `klio_embedding_seconds`, `klio_embedding_memo_total` for the hit rate, and
  `klio_embedding_concurrency_saturated_total` for the concurrency cap. A single
  write embeds the same string twice, so a memo hit rate far below 50% means the
  memo is broken.
- **Database pool**: `klio_db_pool_checkouts_in_use`,
  `klio_db_pool_checkout_wait_seconds`, `klio_db_pool_timeouts_total`. Rising
  timeouts are pool exhaustion. `klio_db_connect_failures_total` is what the
  2026-08-04 outage looks like from inside the process.
- **Curator**: ticks, rows read against rows distilled (the gap is the skip
  rate), entities created against entities merely referenced, cursor advanced
  against cursor held, and budget exhaustion. A cursor held for many consecutive
  passes is a stalled curator.
- **Capture**: events by outcome, and redaction hits.

### Label discipline

Labels are **bounded by construction**, not by convention. The registry refuses
identifier-shaped label names (`org_id`, `user_id`, `session_id`, `agent_id`, and
so on) at declaration time, and every label must enumerate the finite set of
values it may take. An undeclared value raises rather than minting a new time
series. One series per tenant, retained forever, is how a metrics backend dies.

If you add a metric, give every label a closed value set. There is a test that
fails if you do not.

## Suggested first alerts

```promql
# Recall error rate over 5%, five minutes running.
sum(rate(klio_memory_recalls_total{outcome="error"}[5m]))
  / sum(rate(klio_memory_recalls_total[5m])) > 0.05

# The 2026-08-04 outage: connections being refused.
rate(klio_db_connect_failures_total[5m]) > 0

# Pool exhaustion.
rate(klio_db_pool_timeouts_total[5m]) > 0

# A stalled curator: the cursor held on every pass for an hour.
rate(klio_curator_cursor_total{action="advanced"}[1h]) == 0
  and rate(klio_curator_ticks_total[1h]) > 0
```

Also alert on the readiness probe from the platform side. `/readyz` returning 503
while `/healthz` returns 200 is precisely a dependency outage.
