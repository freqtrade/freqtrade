# FreqAI Training Factory Report

## Summary

- Strategy: LongOnlyFreqAIStrategy
- Run ID: phase2_training_20250105_20250107
- Status: failed
- Recommendation: fail
- Stages completed: 0/1
- Timerange: 20250105-20250107
- Timeframe: 5m
- FreqAI model: LightGBMRegressor

## Stages

- freqai_backtest: status=failed, recommendation=n/a, returncode=2
  Error: 2026-05-02 22:33:01,635 - freqtrade - ERROR - Could not load markets, therefore cannot start. Please investigate the above error for more details.

## Artifacts

- training_manifest: `data\freqai_training\LongOnlyFreqAIStrategy\phase2_training_20250105_20250107\training_manifest.json`
- training_report: `data\freqai_training\LongOnlyFreqAIStrategy\phase2_training_20250105_20250107\training_report.md`
- command: `data\freqai_training\LongOnlyFreqAIStrategy\phase2_training_20250105_20250107\command.txt`
- freqai_env: `data\freqai_training\LongOnlyFreqAIStrategy\phase2_training_20250105_20250107\freqai_env.json`
- logs: `data\freqai_training\LongOnlyFreqAIStrategy\phase2_training_20250105_20250107\logs`

## Notes

- This training factory uses checked historical Freqtrade backtesting wrappers only.
- Local JSON, CSV, log, and Markdown files remain the source of truth.
- Passing training or walk-forward gates does not authorize paper trading or live trading.
- FreqAI labels are backtest labels, not live trading instructions.

- FreqAI training factory verification only; no paper or live promotion.
- Local artifacts remain the source of truth.
- Phase 2 FreqAI training factory verification only; no paper or live promotion.