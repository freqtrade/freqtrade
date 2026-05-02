# Backtest Report: LongOnlyFreqAIStrategy

## Summary

- Total return: -0.001225 (-0.12%)
- Trade count: 3
- Win rate: 33.33%
- Profit factor: 0.287425
- Max drawdown: 0.171882%
- CAGR: -0.200414
- Sharpe: -18.174831
- Sortino: -111.596917
- Calmar: -680.686630
- Expectancy: -0.004236
- Fee paid: n/a
- Period: 2025-01-07 00:00:00 to 2025-01-09 00:00:00

## Initial Gate

- Recommendation: fail
- Promotion recommendation: retry_with_modification

## Gate Checks

- FAIL: min_trades (3 vs >= 200)
- FAIL: min_profit_factor (0.287425 vs >= 1.25)
- PASS: max_drawdown_pct (0.171882 vs <= 15.0)
- FAIL: min_sortino (-111.596917 vs >= 1.2)

## Gate Thresholds

- Minimum trades: 200
- Minimum profit factor: 1.25
- Maximum drawdown pct: 15.0
- Minimum sortino: 1.200000

## Reviewer Notes

- Walk-forward historical FreqAI verification only; no paper or live promotion.
- Phase 2 walk-forward verification only; no paper or live promotion.
- FreqAI labels are backtest labels, not live trading instructions.

## Notes

- This report is generated from a backtest only.
- It is not approval for paper trading or live trading.
- Production promotion requires walk-forward, paper trading, and human approval.
