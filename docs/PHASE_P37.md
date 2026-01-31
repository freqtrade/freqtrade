# Phase P37: Scheduler Templates & Locking

## Objective

Implement a robust mechanism for scheduling operational tasks with exclusive locking to prevent race conditions.

## Components

### 1. Atomic Locking Utility

`scripts/ops/with_lock.py`

- Wraps any command with an exclusive, non-blocking file lock.
- Ensures single-instance execution.
- Usage: `python with_lock.py --lock /path/to.lock --cmd "command args"`

### 2. Systemd Templates

Located in `docs/ops/systemd/`.

- `p25_security_master.service` + `.timer`: Daily refresh of instrument master data at 06:00 UTC.
- `p33_backup.service` + `.timer`: Daily backup at 02:00 UTC.
- All services use `with_lock.py` in `ExecStart` for safety.

## Verification

Run `bash scripts/accept_all.sh p37_scheduler_templates`.

- Verifies unit tests for lock contention.
- Validates systemd template syntax (ExecStart, OnCalendar).
