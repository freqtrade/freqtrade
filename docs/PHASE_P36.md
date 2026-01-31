# Phase P36: Metrics Exporter (Textfile)

## Objective

Implement a robust metrics exporter that produces atomic JSON and Prometheus TextFile outputs for observability integration.

## Design

The exporter follows the sidecar pattern, running as a separate process or periodic hook that:

1. Loads the internal `health.json` snapshot.
2. Reads count data from `alerts.jsonl` and `paper_ledger`.
3. Calculates derived metrics (p50 latency).
4. Writes `metrics.json` and `metrics.prom` atomically via temp files.

## Metrics

### Counters

- `policy_blocks_total`: Number of policy interventions (Market Hours, etc).
- `degraded_failures_total`: Errors caught by degraded mode.
- `circuit_open_total`: 1 if circuit breaker is open, 0 otherwise (Gauge behavior).
- `orders_paper_total`: Count of paper orders executed.
- `orders_live_blocked_total`: Count of blocked live orders.
- `alerts_total`: Count of runtime alerts generated.

### Latency (Gauge)

- `latency_ms_p50{method="..."}`: p50 latency for `fetch_ticker`, `fetch_ohlcv`, `create_order`.

## Artifacts

- `adapters/ccxt_shim/metrics_exporter.py`: Core logic.
- `adapters/ccxt_shim/health_snapshot.py`: Enhanced to store duration samples.
- `scripts/gates/p36_metrics_exporter.sh`: Gate script for verification.

## Verification

Run `bash scripts/accept_all.sh p36_metrics_exporter`.
