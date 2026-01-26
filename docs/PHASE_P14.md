# Phase 14: Market Hours Guard

## Overview

Phase 14 introduces deterministic enforcement of NSE trading hours (09:15 - 15:30 IST) at the shim boundary (`BreezeCCXT`). This prevents accidental order placement outside valid execution windows and enables safer automated operations.

## Features

- **Deterministic Guard**: Uses `MarketHoursGuard` in `adapters/ccxt_shim`.
- **Safety**: Blocks `create_order` (buy/entry) if market is closed.
- **Exits Allowed**: `create_order` (sell/exit) is always allowed for safety.
- **Overrides**: Environment variables control behavior for testing/backfilling.

## Environment Overrides

Use these variables to force state regardless of actual time:

- `FT_FORCE_MARKET_OPEN=1`: Treat market as OPEN.
- `FT_FORCE_MARKET_CLOSED=1`: Treat market as CLOSED.

## Acceptance Gate (P14)

The acceptance gate `scripts/gates/p14_market_hours.sh` verifies:

1. **Closed Mode**: Ensures entry orders are blocked and logged with `market_hours_block`.
2. **Open Mode**: Ensures entry orders proceed without shim-level blocking.

## Verification

Run the P14 gate:

```bash
bash scripts/accept_all.sh p14_market_hours
```
