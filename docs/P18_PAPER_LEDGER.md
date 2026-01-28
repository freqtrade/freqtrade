# P18: Paper Forward Test & Ledger

The ICICI Breeze integration supports a "Paper Forward Test" mode (`p18`) which allows running strategies with real market data but executing orders simulating fills against a paper ledger.

## Architecture

- **Flag**: `icicibreeze_paper_forward_test: true` in `config.json`.
- **Logic**: `BreezeCCXT` intercepts `create_order` and simulates fills.
  - Uses real ticker data (`fetch_ticker`).
  - Applies configurable slippage and fees.
  - Generates deterministic order IDs (`paper-xxx`).
- **Persistence**: Trades and daily summaries are written to `user_data/generated/paper_ledger/`.
  - `paper_trades.csv`: Append-only log of all trades.
  - `paper_daily_summary.csv`: Daily aggregate stats (trades, notional, fees).
- **Ownership**: `BreezeCCXT` owns the `PaperLedger` instance.

## Configuration

Add the following to your config (e.g., `user_data/config_icicibreeze.json`):

```json
{
  "icicibreeze_paper_forward_test": true,
  "paper_slippage_bps": 5,
  "paper_fee_bps": 10
}
```

## Running Paper Mode

Run Freqtrade as usual (dry-run or live command, but ensure dry-run logic is used if you want safety, though the adapter forces paper execution even in live mode if the flag is set).

**Recommended**: Use `dry_run: true` in config as well to avoid Freqtrade internal database conflicts, although the adapter handles the API level.

```bash
freqtrade trade --config user_data/config_icicibreeze.json --strategy MyStrategy
```

## Reporting

Use the provided script to view recent activity:

```bash
bash scripts/paper_ledger_report.sh
```
