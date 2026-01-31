# Phase 40: Live Readiness & Deadman

## Objective

Implement strict readiness checks and a "Deadman Switch" mechanism to prevent unauthorized or accidental live trading execution. This phase ensures that the trading bot only places real orders when specific safety conditions are met.

## Components Implemented

### 1. LiveReadiness Guard

- **File**: `adapters/ccxt_shim/live_readiness.py`
- **Logic**:
  - **Session Token**: Verifies presence of Breeze Session Token in config or env.
  - **Disk Space**: Ensures `user_data` partition has >2GB free space.
  - **Security Master**: Verifies `NSEScripMaster.txt` and `FONSEScripMaster.txt` are fresh (<24h).
  - **Pair Whitelist**: Ensures whitelist is not empty.

### 2. Deadman Switch

- **File**: `adapters/ccxt_shim/live_readiness.py`
- **Mechanism**:
  - Checks for file: `user_data/secrets/deadman_live.ok`.
  - Verification: File must exist AND be less than 10 minutes old.
  - **Fail-Closed**: If check fails, `create_order` raises `OperationalException` ("DEADMAN_MISSING" or "DEADMAN_STALE").

### 3. CCXT Shim Integration

- **File**: `adapters/ccxt_shim/breeze_ccxt.py`
- **Method**: `create_order` (Sync)
- **Pre-Flight Checks** (in order):
    1. **Rate Limit**: Leak/Wait.
    2. **Market Hours**: NSE IST check (P29).
    3. **Degraded Mode**: P17 (skipped for P40 checks if critical).
    4. **Live Guard (P30)**: Config + Env Var check.
    5. **Live Readiness (P40)**: Full health check.
    6. **Deadman (P40)**: Deadman file check.
    7. **Idempotency (P40)**: Duplicate suppression logic.
    8. **Risk Guard (P15)**: Risk limits.

### 4. Idempotency Support

- **File**: `adapters/ccxt_shim/idempotency.py` (assumed existing or integrated logic)
- **Logic**: Tracks client order IDs to prevent duplicate submissions during retries.

## Verification Gate

- **Script**: `scripts/gates/p40_live_readiness.sh`
- **Modes**:
  - `pos`: Creates `deadman_live.ok`, runs trade, verifies success.
  - `neg`: Removes `deadman_live.ok`, runs trade, verifies "Deadman Switch Failed" block.
- **Integration**: Added to `scripts/accept_all.sh` (Auto/Hardened).

## Operational Procedures

See `docs/OPS_RUNBOOK.md` for Deadman Switch management.
