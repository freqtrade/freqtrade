# Phase 35: Operations Runbook

## Objective

This runbook guides operators on Day 2 tasks: Monitoring, alerting, backup/restore, and troubleshooting.

## 1. Monitoring

- **Logs**: Located in `user_data/logs/`.
  - `freqtrade.log`: General application logs.
  - `user_data/generated/runtime/health.json`: Recent health snapshots and error counters.
- **Health Check**:
  - Check `health.json` for `degraded_failures` count.
  - Check `health.json` for `circuit_breaker.tripped` status.

## 2. Alerts

- High Priority alerts are prefixed with `[ALERT:HIGH]`.
- Triggered on:
  - Circuit Breaker Trip (`DEGRADED_ENTER`).
  - Risk Block (`RISK_BLOCK`).

## 3. Backup & Restore

- **Backup**:

  ```bash
  bash scripts/backup_state.sh
  # Output: PATH=...
  ```

- **Restore**:

  ```bash
  bash scripts/restore_state.sh <backup_file>
  ```

  **Warning**: This replaces the current `user_data` directory. Old state is saved to `user_data.old_TIMESTAMP`.

## 4. Troubleshooting

- **Circuit Breaker Tripped**:
  - Investigate logs around the trip time.
  - Fix the underlying issue (API, connectivity).
  - Restart the bot to reset (if timeout passed) or manually edit `health.json` to untrip (advanced).
- **Orders Blocked**:
  - Check `MarketHoursGuard` (is market open?).
  - Check `RiskGuard` (daily limits exceeded?).

## 5. Deployment Verification

- Use `scripts/accept_all.sh` to verify system integrity after updates.
