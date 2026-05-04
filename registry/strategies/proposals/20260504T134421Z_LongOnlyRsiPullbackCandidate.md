# Strategy Proposal: LongOnlyRsiPullbackCandidate

## Metadata

- created_at: 2026-05-04T13:44:21+00:00
- created_by_agent: codex
- strategy_type: mean_reversion
- target_exchange: bybit
- target_symbols: BTC/USDT:USDT
- timeframe: 5m
- spot_or_futures: futures
- long_short: long-only
- proposal_status: accepted
- safety_scope: long-only, leverage=1.0, historical-evaluation-only, no live data, no order endpoints, no secrets, no process control

## Summary

Long-only RSI pullback candidate for historical evaluation.

## Hypothesis

After sharp short-term pullbacks in a liquid BTC futures market, mean reversion may occur when volume and volatility filters confirm liquidity.

## Market Condition

Liquid BTC/USDT futures, historical OHLCV only.

## Entry Logic

Enter long after RSI pullback and recovery confirmation using closed candles only.

## Exit Logic

Exit on mean-reversion target, momentum failure, or timeout using closed candles only.

## Risk Logic

Use strategy stoploss and no leverage above 1.0; no shorting.

## Required Data

- OHLCV closed candles only

## Parameters

- RSI window, recovery threshold, stoploss, timeout candles

## Expected Failure Cases

- Trend continuation after pullback

## Backtest Plan

Run static checks, OHLCV quality check, historical backtest, walk-forward, and training factory if FreqAI is added later.

## Rejection Conditions

- Future data is required
- Trade count is too low
- Profit depends on one narrow period

## Reviewer Notes

- Strategy proposal generation smoke test only; do not generate code, backtest, start paper trading, or promote.

## Generation Boundary

- Strategy proposal generation writes local Markdown and metadata artifacts only. It does not generate strategy code, run backtests, start paper or dry-run trading, call exchange order endpoints, promote candidates, or manage any bot process.
- This proposal is not eligible for strategy code generation unless the sidecar metadata status is `accepted`.
- Local JSON, CSV, Markdown, and log artifacts remain the source of truth.
