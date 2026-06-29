---
name: futures-risk-design
description: Use when designing, reviewing, or tuning crypto futures risk controls for Freqtrade or custom bots, including leverage, isolated vs cross margin, stake sizing, stoploss, ROI, max drawdown pauses, consecutive-loss guards, cooldowns, liquidation risk, and dry-run/live promotion safety.
---

# Futures Risk Design

## Purpose

Design futures risk controls before optimizing strategy returns.

Use for crypto perpetual/futures strategies, especially leveraged BTC/ETH USDT-M research.

## First Principles

- Leverage magnifies price movement against margin; it does not create edge.
- Judge signal quality at low leverage first.
- Risk limits should be expressed in account/dry-run equity terms, not only price movement.
- Keep live trading blocked until dry-run and promotion gates pass.

## Required Parameters

Always identify:

| Parameter | Meaning |
|---|---|
| `leverage` | Notional multiplier on margin |
| stake / margin per trade | How much account equity is posted per trade |
| max open trades | Portfolio-level exposure cap |
| stoploss | Price move loss threshold before leverage effect |
| `minimal_roi` | Exit targets by holding time |
| protections | Cooldown, stoploss guard, drawdown guard |
| circuit breakers | Pause after account drawdown or loss streak |
| isolated/cross | Whether margin is per position or shared account-wide |
| fee/slippage/funding | Cost assumptions |

## Leverage Rules

- 50x is research-only until signal edge is proven at 1x/3x/5x.
- If a strategy has PF < 1 at 5x, do not test 30x/50x as a “fix”.
- A 1% adverse price move at 50x is about 50% loss on that position’s margin before fees/funding, not 50% account loss unless all account equity is posted.

## Circuit Breakers

Recommended defaults for research dry-run:

| Guard | Default |
|---|---|
| per-trade margin | 10% of tradable allocation |
| daily/account drawdown pause | pause when 24h account drawdown exceeds configured threshold |
| consecutive losses | pause after 4 losses |
| high volatility | reduce leverage or block entries |
| stress cost failure | block promotion |

When user asks “杠杆前还是杠杆后”, clarify:
- account drawdown guard is after realized leveraged PnL on account equity
- stoploss price move is pre-leverage price movement
- position margin loss is price move × leverage × margin

## Output Contract

For any risk proposal, give:

1. leverage
2. per-trade margin
3. stoploss as price move
4. approximate margin loss and account loss
5. ROI targets
6. protections/circuit breakers
7. what blocks dry-run/live

Do not present aggressive leverage as safe because the backtest return looks good.
