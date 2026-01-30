# Phase P30: Live Order Execution (Guarded)

## Objective

Enable ACTUAL live order placement to ICICI Breeze, protected by a strict double-lock mechanism. By default, live orders are BLOCKED even if config says enabled.

## Double-Lock Mechanism

To place a live order, TWO conditions must be met:

1. **Configuration**: `config.icicibreeze.live_trading.enabled = true`
2. **Environment**: `FT_ENABLE_LIVE_ORDERS=1`

If either is missing, `create_order` raises `OperationalException("Live Trading Guard: Blocked")`.

## Implementation

### `BreezeCCXT.create_order`

- Fall-through from P29 (Paper Check).
- Check `live_trading.enabled`.
- Check `env["FT_ENABLE_LIVE_ORDERS"]`.
- If Passed:
  - Map Parameters to Breeze SDK `place_order`.
  - Handle `order_type` (limit, market, stop).
  - Return valid CCXT Order Structure.
- If Blocked:
  - Log Critical Warning.
  - Raise Exception.

## Acceptance Gate (P30)

### Positive Case (P30_POS_PASS)

- Only if double-lock is ON.
- Since we can't safely test this in CI without real money, the "Positive" gate actually verifies the **Checking Logic** using a Mock.
- We mock `breeze.place_order` and verify that with double-lock ON, it CALLS the mock.

### Negative Case (P30_BLOCK_SUCCESS)

- Run without Env Var.
- assert `create_order` RAISES "Live Trading Guard: Blocked".
- This is the critical safety property.
