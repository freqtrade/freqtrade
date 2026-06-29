---
name: scalping-microstructure-research
description: Use when researching crypto 1m/5m scalping, short momentum scalping, mean reversion, grid, market making, order book, spread, maker/taker, or microstructure strategies. Covers when OHLCV backtests are insufficient and how to design safe Freqtrade experiments without overclaiming.
---

# Scalping Microstructure Research

## Purpose

Separate K-line scalping research from true microstructure or market-making research.

Use when the user asks about:
- 1m/5m scalping
- short momentum scalping
- grid or mean reversion
- market making
- order book /盘口
- spread, slippage, maker/taker assumptions

## Data Boundary

OHLCV can test:
- simple momentum
- breakout
- mean reversion
- time stops
- fee sensitivity
- rough trade frequency

OHLCV cannot validate:
- queue position
- maker fill probability
- spread capture
- order book imbalance persistence
- adverse selection from fills
- real market-making profitability

If the strategy depends on order book, trades, spread, or maker fills, say so and require the right data/simulator.

## Scalping Diagnostics

For short-cycle K-line strategies, require:

- trades/day
- PF
- win rate and payoff ratio
- average duration
- avg MFE/MAE
- fee and slippage stress
- long vs short lane results
- comparison at 1x/3x/5x before high leverage

High trade count with PF < 1 is not progress. It is usually a negative expectancy machine.

## Experiment Menu

Use at most 3 entry confirmations for K-line scalping. Prefer simple experiments:

- inverse signal retest
- delayed entry by 3/5/10 candles
- price moves in favor before entry
- volatility floor filter
- time stop exit
- fee sensitivity grid
- long-only / short-only split
- regime matrix

For true microstructure:

- collect order book snapshots or trade prints
- define maker/taker model
- simulate fills and adverse selection
- compare with a taker-only baseline

## Promotion Block

Do not promote scalping strategies if:
- base-fee PF < 1.05
- stress-fee return turns strongly negative
- MFE/MAE shows bad entry timing
- results depend on unmodeled maker fills
- only one short window works

## Output Contract

When explaining scalping results:

1. state whether data is OHLCV or order book/trades
2. state whether strategy is K-line scalping or true microstructure
3. report cost sensitivity
4. say what can and cannot be concluded
5. list next experiments or data needed
