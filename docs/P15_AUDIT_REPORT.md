# P15 Context Audit Report

## 1. Choke Points for Order Creation

The `BreezeCCXT` class in `adapters/ccxt_shim/breeze_ccxt.py` has a clear choke point for order creation.

- **Async Wrapper**: `create_order` (Line 820)
- **Sync Implementation**: `create_order` (Line 539)

The sync implementation currently calls `self.market_hours.assert_can_create_order(side, symbol)` at Line 542. This is the ideal location to inject the `RiskGuard` check.

## 2. Configuration Access

`BreezeCCXT` is initialized with the full configuration dictionary.

- **Init Method**: `__init__` (Line 73)
- **Storage**: `self.config = config` (Line 77)

This confirms that `RiskGuard` can be instantiated inside `__init__` passing `self.config`.

## 3. Price Surface for Spread Checks

Bid/Ask prices are available via `fetch_ticker`.

- **Method**: `fetch_ticker` (Line 440-441) returns `bid` and `ask` keys.
- **Mock Data**: `_MOCK_BASE_PRICES` provides reference prices for mock mode.

The `RiskGuard` will need to call `fetch_ticker` (or receive ticker data) to validate spread checks.

## 4. Acceptance Gate Discovery

The `scripts/accept_all.sh` script utilizes a static array `ALL_GATES` to define the execution order.

- **Definition**: Line 16
- **Current Last Gate**: `p14_market_hours`

To add P15, we must append `p15_risk_guardrails` (or simplified `p15`) to this array. The script already supports `--mode=pos|neg` propagation, which aligns with the P15 requirements.
