# Phase 34: Circuit Breaker Persistence

## Objective

Make the Circuit Breaker (Degraded Mode) persistent across restarts to prevent "crash loops" from resetting the failure counter and allowing bad orders again.

## Implementation

- Utilize `HealthSnapshot` (P31) to store circuit breaker state.
- Add `circuit_breaker` key to health state:

  ```json
  "circuit_breaker": {
      "tripped": true,
      "tripped_at": 167...,
      "failures": 3
  }
  ```

- `DegradedModeGuard` will read this on `__init__`.

## Gate (`p34_circuit_breaker`)

- **Pos**:
    1. Trigger failures -> Trip CB.
    2. Verify `health.json` has `tripped: true`.
    3. Restart process (new check script).
    4. Assert `DegradedModeGuard` initializes in `DEGRADED` state.
- **Neg**:
    1. Wait for timeout (mocked).
    2. Restart.
    3. Assert `DegradedModeGuard` initializes in `HEALTHY` state (expiry check).
