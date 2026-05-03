# Backtest Report: LongOnlyFreqAIStrategy

## Summary

- Total return: -0.000617 (-0.06%)
- Trade count: 2
- Win rate: 0.00%
- Profit factor: 0.000000
- Max drawdown: 0.061699%
- CAGR: -0.106523
- Sharpe: -123.751514
- Sortino: -123.751514
- Calmar: -955.248659
- Expectancy: -0.003150
- Fee paid: n/a
- Period: 2025-01-05 00:00:00 to 2025-01-07 00:00:00

## Initial Gate

- Recommendation: fail
- Promotion recommendation: retry_with_modification

## Gate Checks

- FAIL: min_trades (2 vs >= 200)
- FAIL: min_profit_factor (0.000000 vs >= 1.25)
- PASS: max_drawdown_pct (0.061699 vs <= 15.0)
- FAIL: min_sortino (-123.751514 vs >= 1.2)

## Gate Thresholds

- Minimum trades: 200
- Minimum profit factor: 1.25
- Maximum drawdown pct: 15.0
- Minimum sortino: 1.200000

## Reviewer Notes

- FreqAI training factory verification only; no paper or live promotion.
- Phase 2 FreqAI training factory verification only; no paper or live promotion.
- FreqAI labels are backtest labels, not live trading instructions.

## Notes

- This report is generated from a backtest only.
- It is not approval for paper trading or live trading.
- Production promotion requires walk-forward, paper trading, and human approval.
