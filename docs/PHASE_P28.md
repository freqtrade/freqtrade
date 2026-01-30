# Phase P28: Execution Microstructure (FR-403+)

## Objective

Implement advanced execution controls in the CCXT Shim layer to optimize order placement and management without strategy involvement.

## Features

### 1. GTT Hysteresis & Virtual GTT

- **GTT Hysteresis**: Prevents excessive modification churn. If a requested modification is within `rearm_seconds` AND price change is < `min_price_move_ticks`, the modification is skipped.
- **Virtual GTT**: If broker GTT isn't supported, emulates it by polling and cancelling/replacing orders.

### 2. Sniper Cancel (3-second rule)

- Automatically cancels orders that remain OPEN after `cancel_after_seconds` (default 3s).
- Optionally replaces if desired, but primarily cleans up stale limit orders to avoid "chasing".

### 3. ATR Limit Buffer

- Applies purely to LIMIT orders.
- Adjusts limit price slightly relative to `last_price` using ATR-derived ticks.
- **Buy**: `min(limit, last + buffer)` (Don't overpay, but capture aggressive moves).
- **Sell**: `max(limit, last - buffer)`.

### 4. Order Slicing

- Splits large orders into `max_child_orders` chunks to hide intent/reduce impact.
- Respects lot size constraints.

### 5. Partial Fill Management

- Reconciliation logic to handle partially filled orders correctly in the `paper_ledger` and `OrderRouter`.
- Determines remaining quantity accurately for subsequent logic.

## Configuration Schema

```json
"microstructure": {
    "enabled": true,
    "gtt_hysteresis": { "enabled": true, "rearm_seconds": 20, "min_price_move_ticks": 2 },
    "virtual_gtt_fallback": { "enabled": true, "poll_interval_seconds": 2 },
    "sniper_cancel": { "enabled": true, "cancel_after_seconds": 3 },
    "atr_limit_buffer": { "enabled": true, "buffer_mult": 0.15, "min_ticks": 1, "max_ticks": 20 },
    "order_slicing": { "enabled": true, "max_child_orders": 4 },
    "partial_fill_management": { "enabled": true }
}
```

## Acceptance Criteria

- **Positive**: Unit tests for each component pass. Gate marks `P28_POS_PASS`.
- **Negative**: Test forcing lot size violation in slicing fails as expected. Gate marks `P28_NEG_EXPECTED_FAIL`.
