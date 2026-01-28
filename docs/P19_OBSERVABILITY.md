# P19 Observability & Audit

This phase enforces strict observability contracts to ensure the trading bot is auditable and debuggable in production.

## 1. Exception Logging Policy

All code owned by this repository (scripts, strategies, adapters) must log full stack traces when catching exceptions.

**Rule:**

- **DO NOT** use `logger.error("msg: %s", e)` inside an `except` block.
- **DO** use `logger.exception("msg")` (preferred) or `logger.error("msg", exc_info=True)`.

This ensures that all operational failures contain sufficient context (tracebacks) for debugging without requiring code changes or restarts.

## 2. Startup State Observability

The bot must emit a clearly identifiable log message when it transitions to the `RUNNING` state. This allowed external monitors (like P05 gate) to verify startup success reliably.

**Marker:** `Changing state to: RUNNING`

## 3. Tooling & Enforcement

### P19 Gate: `scripts/gates/p19_observability_audit.sh`

This gate runs automatically in CI/CD and verfies:

1. **Static Analysis**: Scans all `.py` files to ensure compliance with the exception logging policy.
   - Scanner: `scripts/ops/p19_scan_exc_logging.py`
2. **Runtime Verification**: Intentionally raises an exception and parses logs to ensure the traceback is recorded.
   - Helper: `scripts/p19_raise_and_log.py`
3. **State Transition**: Runs a dry-run smoke test and greps for the RUNNING state marker.

## 4. Maintenance

If the static scanner flags a file, fix it by converting `logger.error` to `logger.exception` in the relevant `except` block.
If the state transition marker fails, ensure `freqtrade` core or strategy startup logic hasn't been altered to suppress this log.
