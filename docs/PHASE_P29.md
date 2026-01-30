# Phase P29: Real-Mode Paper Trading Simulation

## Objective

Run the bot with `BREEZE_MOCK=0` (Real Market Data) but ensure ALL order routing is intercepted and sent to a local Paper Ledger. This allows testing "Real Data + Paper Execution" with zero risk of accidental live orders.

## Implementation Contract

### 1. Config Flag

```json
"icicibreeze": {
    "paper_trading": {
        "enabled": true,
        "ledger_path": "user_data/generated/paper_ledger.sqlite"
    }
}
```

### 2. Enforcement

- **Create/Cancel/Modify Order**:
  - Check `config.icicibreeze.paper_trading.enabled`.
  - If TRUE:
    - DO NOT call any Broker API for execution.
    - Record action in `PaperLedger`.
    - Return synthetic Order/Response.
  - If FALSE:
    - Proceed to Live execution (subject to P30 Guard).

- **Real Mode**:
  - `BREEZE_MOCK=0` allows real market data (TICKS/OHLCV).
  - Paper Trading config isolates Execution only.

## Acceptance Gate (P29)

### Positive Case (P29_POS_PASS)

- Run with `BREEZE_MOCK=0`.
- Config enables `paper_trading`.
- Strategy places an order.
- Verify:
  - Order appears in `paper_ledger.sqlite`.
  - No "Broker Order Placed" logs.
  - Gate passes.

### Negative Case (P29_SKIP_MISSING_CREDS)

- If credentials missing for Real Mode, gate skips gracefully (as per P22 pattern).
- If Config disables paper trading but Live Guard (P30) blocks it, that's P30's domain.
- For P29, we focus on proving Paper Route works.
