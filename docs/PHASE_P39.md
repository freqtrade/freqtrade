# Phase P39: Ops Hardening Gate

## Objective

Establish a composite gate that enforces security, observability, and hygiene standards across the codebase.

## Checks

### 1. Observability Audit (P19)

- Verifies that all `except` blocks use proper logging (`logger.exception` or `exc_info=True`).
- Ensures tracebacks are captured for unexpected errors.

### 2. Open Ports Scan (P20)

- Scans `netstat` to ensure no unauthorized ports are listening (e.g. 0.0.0.0 bindings).

### 3. Secrets Hygiene (P21)

- Scans codebase for potential hardcoded secrets.

### 4. Codebase Hygiene

- Scans for `TODO` and `FIXME` comments.
- Warns if found (non-blocking for now).

### 5. Ops Runbook

- Verifies existence of `docs/OPS_RUNBOOK.md`.

## Verification

Run `bash scripts/accept_all.sh p39_ops_hardening`.

- Aggregates results from P19, P20, P21 and new checks.
