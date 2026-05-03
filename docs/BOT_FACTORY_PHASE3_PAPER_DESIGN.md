# Bot Factory Phase 3 Paper Readiness Design

This document covers the first Phase 3 increment: a no-startup readiness layer
for future paper trading.

It does not authorize `freqtrade trade`, paper startup, dry-run startup, live
trading, canary live trading, exchange order placement, API keys, leverage above
`1.0`, or shorting.

## Goal

The readiness layer answers one question from local evidence only:

```text
Is this strategy candidate eligible for a tightly scoped future paper run?
```

The answer is written as `pass`, `fail`, or `blocked`.

- `pass`: all local evidence, gates, static checks, long-only checks, config
  safety checks, and reviewer-note requirements pass.
- `fail`: the check completed, but candidate quality gates or review
  requirements do not pass.
- `blocked`: required evidence is missing or a safety/config/static issue makes
  paper readiness unsafe to evaluate.

## Scope

Allowed in this increment:

- Read historical Phase 2 FreqAI artifacts.
- Read walk-forward artifacts.
- Read training factory artifacts.
- Run or consume static safety checks.
- Inspect strategy source for long-only constraints.
- Inspect a proposed dry-run config without writing credential values.
- Write local JSON and Markdown readiness artifacts.

Not allowed in this increment:

- Starting any bot process.
- Running `freqtrade trade`.
- Paper, dry-run, canary, or live startup.
- API keys or secrets.
- Exchange order endpoints.
- Shorting or leverage above `1.0`.
- Promotion based on failed Phase 2 gates.

## CLI

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_check_paper_readiness.py `
  --config user_data\config_freqai_phase2_safe.json `
  --strategy LongOnlyFreqAIStrategy `
  --historical-dir data\freqai\LongOnlyFreqAIStrategy\phase2_safe_20250105_20250107 `
  --walk-forward-dir data\walk_forward\LongOnlyFreqAIStrategy\phase2_walk_forward_20250105_20250109 `
  --training-dir data\freqai_training\LongOnlyFreqAIStrategy\phase2_training_20250105_20250107 `
  --run-id phase3_readiness_20260503 `
  --reviewer-note "Phase 3 no-startup paper readiness check only; do not start paper trading."
```

This command reads local files and writes reports. It does not call Freqtrade.

## Required Evidence

The checker requires these local files:

- historical backtest `metrics.json`
- historical backtest `report.md`
- historical backtest `freqai_metadata.json`
- historical backtest `trades.csv`
- walk-forward `walk_forward_metrics.json`
- walk-forward `walk_forward_report.md`
- walk-forward child window `metrics.json`, `trades.csv`, and
  `freqai_metadata.json` for each recorded window
- training factory `training_manifest.json`
- training factory `training_report.md`
- training factory `freqai_backtest` child `metrics.json`, `trades.csv`, and
  `freqai_metadata.json`

Failed historical, walk-forward, or training recommendations produce `fail` and
block paper readiness. Missing top-level or child files produce `blocked`.
Historical, walk-forward child, and training child trade exports must contain no
short trades and no leverage above `1.0`.

## Config Safety

The proposed config must be dry-run only and sanitized:

- `dry_run=true`
- explicit strategy and timeframe
- explicit positive `max_open_trades`
- `max_open_trades <= 3`
- explicit positive numeric `stake_amount <= 1000`
- explicit positive numeric `dry_run_wallet <= 10000`
- `stake_amount <= dry_run_wallet`
- explicit non-empty `exchange.pair_whitelist`
- `force_entry_enable` absent or `false`
- `initial_state=stopped`
- explicit boolean `cancel_open_orders_on_exit`
- API server disabled
- no non-empty API keys, secrets, passwords, UIDs, tokens, or credential-like
  values
- no private environment variable references
- no leverage above `1.0`
- no private or order endpoint overrides

The generated `config_safety.json` contains sanitized metadata only. It records
credential key paths when unsafe values are present, but never writes the values.
It also records the accepted simulation limits used by the policy check.

## Long-Only Strategy Safety

The checker parses the strategy source and requires:

- `can_short = False`
- no `enter_short` or `exit_short` signal references
- no `leverage()` hook, or only statically capped returns at `1.0`
- historical exported trades contain no shorts and no leverage above `1.0`
- walk-forward child exported trades contain no shorts and no leverage above
  `1.0`
- training child exported trades contain no shorts and no leverage above `1.0`

The existing static checker is also run or consumed, and any static safety error
blocks readiness.

## Artifacts

Readiness artifacts are written under:

```text
data/paper_readiness/<strategy>/<run_id>/
```

Files:

- `paper_readiness.json`
- `paper_readiness_report.md`
- `candidate_artifacts.json`
- `config_safety.json`
- `command.txt`

Local JSON, CSV, and Markdown artifacts remain the source of truth. MLflow is
not involved in this readiness layer.

## Current Limitation

The verified `LongOnlyFreqAIStrategy` Phase 2 artifacts are pipeline
verification artifacts. Their recent historical, walk-forward, and training
gates fail, so the readiness checker must return `fail` for that candidate.
The hardened evidence and config checks pass for the current local artifacts,
so the current `fail` status is driven by those Phase 2 quality gates rather
than missing child evidence or unsafe config.

A future human-approved infrastructure-only smoke test is a separate path and
is not implemented here.
