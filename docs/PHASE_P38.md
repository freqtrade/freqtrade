# Phase P38: Soak Stability Gate

## Objective

Verify the stability of the Trading System under sustained operation and fault injection conditions.

## Components

### 1. Soak Test Gate

`scripts/gates/p38_soak_stability.sh`

- Runs `freqtrade trade --dry-run` in mock mode.
- Duration: 90 seconds (Positive path).
- Verifies:
  - Metric generation (Health + Exporter).
  - Traceback absence.
  - Process persistence.

### 2. Failure Injection (Negative Path)

- Injects "Circuit Open" state into `health.json` during runtime.
- Verifies that the Metrics Exporter correctly reports `circuit_open_total=1`.
- Verifies system stability (process does not crash).
*Implementation Note: Uses external state injection as intrinsic failure hooks are not exposed in the shim.*

## Verification

Run `bash scripts/accept_all.sh p38_soak_stability`.

- Positive: Runs for 90s, validates system health.
- Negative: Runs for 10s, detects injected fault state in metrics.
