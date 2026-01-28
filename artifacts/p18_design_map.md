# P18 Design Map: Paper Forward Test & Ledgers

## Code Insertion Points

| File Path | Responsibility | Planned Change |
|-----------|----------------|----------------|
| `adapters/ccxt_shim/breeze_ccxt.py` | Order Entry/Exit | 1. Read `icicibreeze_paper_forward_test` from config.<br>2. In `create_order`: if paper mode, route to `_create_paper_order` (bypass SDK).<br>3. In `cancel_order`: if paper mode, route to `_cancel_paper_order`.<br>4. In `fetch_order/s`: merge paper orders with real orders (or support paper only). |
| `adapters/ccxt_shim/breeze_ccxt.py` | Execution Engine | Implement `_create_paper_order`: <br>- Get price from `fetch_ticker` (real market data).<br>- Apply `paper_slippage_bps` and `paper_fee_bps`.<br>- Generate deterministic ID (`paper-xxx`).<br>- Return filled order struct immediately (v1). |
| `adapters/ccxt_shim/paper_ledger.py` | Persistence | **[NEW FILE]**<br>- Class `PaperLedger`<br>- `append_trade(trade_dict)`: Writes to `user_data/generated/paper_ledger/paper_trades.csv`.<br>- `update_daily_summary(trade_dict)`: Upserts `paper_daily_summary.csv`. |
| `scripts/paper_ledger_report.sh` | Reporting | **[NEW FILE]**<br>- CLI tool to `tail` trades and show summary. |
| `scripts/gates/p18_paper_forward_test_pos.sh` | Acceptance Gate | **[NEW FILE]**<br>- Runs pytest specific to P18.<br>- Runs a smoke test using a script that invokes paper order.<br>- Checks for existence of ledger files. |

## Global Ownership & Data Flow

- **Owner**: `BreezeCCXT` owns `PaperLedger` instance.
- **Initialization**: `BreezeCCXT.__init__` checks config. If paper mode -> init `PaperLedger`.
- **Data Flow**:
    `Strategy` -> `freqtrade` -> `BreezeCCXT.create_order` -> `[Paper Guard]` -> `_create_paper_order` -> `PaperLedger.append_trade` -> `CSV`.

## Configuration Schema (User Data)

```json
{
  "icicibreeze_paper_forward_test": true,
  "paper_slippage_bps": 5,
  "paper_fee_bps": 10
}
```
