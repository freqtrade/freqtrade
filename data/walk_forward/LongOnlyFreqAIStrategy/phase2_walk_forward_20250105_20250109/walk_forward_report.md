# Walk-Forward Report

## Summary

- Recommendation: fail
- Windows: 2/2 completed
- Pass rate: 0.00%
- Profitable windows ratio: 0.00%
- Total return: -0.001842 (-0.18%)
- Max drawdown in any window: 0.171882%
- Max single-window profit dependency: n/a

## Gate Checks

- PASS: all_windows_completed (2/2 vs all windows complete)
- FAIL: min_pass_rate (0.000000 vs >= 0.7)
- FAIL: min_profitable_windows_ratio (0.000000 vs >= 0.6)
- PASS: max_drawdown_pct_any_window (0.171882 vs <= 20.0)
- FAIL: max_single_window_profit_dependency (n/a vs <= 0.4)

## Windows

- Window 1: 20250105-20250107 | status=completed | gate=fail | return=-0.061699% | trades=2
- Window 2: 20250107-20250109 | status=completed | gate=fail | return=-0.122479% | trades=3

## Notes

- This report is generated from historical backtests only.
- Passing walk-forward gates does not authorize paper trading or live trading.
- FreqAI labels are backtest labels, not live trading instructions.
