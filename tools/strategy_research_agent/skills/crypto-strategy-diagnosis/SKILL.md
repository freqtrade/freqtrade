---
name: crypto-strategy-diagnosis
description: Use when diagnosing why a crypto spot or futures strategy backtest performed poorly or unexpectedly, especially in Freqtrade research. Covers separating signal edge, entry timing, leverage, ROI/stoploss, fees/slippage, regime fragility, sample size, long/short lane issues, and risk controls before proposing changes.
---

# Crypto Strategy Diagnosis

## Purpose

Diagnose strategy results like a trading researcher before changing parameters.

Use this skill when a crypto strategy has:
- negative return or weak PF
- high win rate but loses money
- high trade count but loses money
- low trade count and unclear validity
- long/short imbalance
- poor results after leverage, fee, or stop changes
- scalping or futures results that may be cost-sensitive

## First Split

Always separate:

1. **Signal edge**: do entries have positive expectancy before leverage?
2. **Entry timing**: does price move against the trade before it moves for it?
3. **Exit quality**: are winners cut early or losers held to stop?
4. **Cost drag**: do fees, slippage, funding, spread, or churn erase gross edge?
5. **Risk sizing**: does leverage or stake size magnify a weak signal?
6. **Regime dependency**: does it only work in bull, bear, range, or high-vol windows?
7. **Sample validity**: are there enough trades across enough windows?

Do not call a strategy “fixed” from one aggregate return.

## Diagnostic Rules

| Symptom | Likely Diagnosis | Response |
|---|---|---|
| Many trades, PF < 0.8, negative return | Negative expectancy | Stop increasing leverage; test inverse, delayed entry, fee stress, low leverage |
| Long and short both negative | Entry/exit logic failure | Split long-only and short-only lanes; test each by regime |
| High win rate but negative return | Payoff asymmetry | Increase target/trim losses; inspect avg win vs avg loss |
| Low win rate but positive return | Tail/trend profile | Check drawdown, losing streaks, and regime fragility |
| Positive base result but stress fee negative | Cost sensitivity | Require higher edge per trade or lower churn |
| Few trades | Insufficient sample | Relax one condition at a time; do not add complex filters |
| High MAE vs MFE | Bad entry timing | Test delayed/confirmed entry and price-moves-in-favor filters |
| Large loss clusters | Bad regime/risk gating | Add cooldown/circuit breaker and regime stop |

## Required Evidence

Before recommending changes, look for:
- total return, PF, max drawdown, trade count, trades/day
- long vs short trades and PnL
- win rate, avg win, avg loss, payoff ratio
- MFE/MAE if available
- stop-loss and exit-reason breakdown
- pair/tag breakdown
- base fee and stress fee results
- market change benchmark
- regime matrix and walk-forward results if available

## Output Contract

Answer in this order:

1. top diagnosis
2. evidence
3. what not to do
4. next experiments
5. promotion block, if any

For personal Freqtrade work, prefer these local commands when available:

```bash
user_data/strategy_research/start_manual_research.sh --mature-researcher
user_data/strategy_research/start_manual_research.sh --mature-researcher-queue
user_data/strategy_research/start_manual_research.sh --trade-behavior
user_data/strategy_research/start_manual_research.sh --failure-attribution
```
