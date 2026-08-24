# Proposal: Crypto Strategy Discovery & Validation Engine

> Phase 2 roadmap — builds on the research/validation gate delivered by
> `docs/superpowers/plans/2026-08-23-freqtrade-research-mvp.md`. Not yet brainstormed or
> planned; saved here as the source proposal for that future session.

## Objective

Extend our Freqtrade-based trading system so that a user who is **not a trading guru** can
systematically investigate common crypto trading strategy families, identify potentially
persistent edges, and promote only strategies that survive rigorous validation.

The objective is **not**:

> Find the strategy with the highest historical return.

The objective is:

> **Identify strategies whose observed edge is most likely to generalize to future crypto markets after fees, slippage, execution effects, regime changes, and multiple-testing/selection bias.**

No profitability should ever be assumed or guaranteed.

---

## 1. Strategy families to investigate

Build the research system around explicit strategy families.

### Tier 1

Start with:

1. Trend following
2. Momentum
3. Mean reversion
4. Breakout
5. Cross-sectional momentum

### Tier 2

Then investigate:

6. Funding-rate effects
7. Futures basis / cash-and-carry
8. Liquidation-driven effects
9. Market-regime strategies
10. Volatility strategies

### Tier 3

Only after the research framework is proven:

11. Statistical arbitrage
12. Pairs trading
13. Cross-exchange arbitrage
14. Market making
15. ML/AI-based strategies

Do **not** implement all of these initially. The framework should make strategy families
pluggable.

---

## 2. Strategy-family interface

Create a common abstraction allowing the research engine to treat each strategy family
consistently.

Conceptually:

```python
class StrategyHypothesis:
    name
    family
    description
    required_data
    parameters
    generate_signals()
    generate_hypothesis()
```

The research system should be able to ask:

> "What happens if we test this family across these assets, timeframes, market regimes and
> parameter ranges?"

without hard-coding research logic into every strategy.

---

## 3. Start with transparent strategies

The initial implementations should be intentionally understandable.

### Trend following

Examples: moving-average relationships, price vs long-term moving average, trend strength,
Donchian-style breakouts, trailing exits. Do not assume these are profitable — they are
research hypotheses.

### Momentum

Test: 1h, 4h, 24h, 3d, 7d momentum; risk-adjusted momentum; volume-confirmed momentum. Test
both time-series momentum and cross-sectional momentum.

### Mean reversion

Test whether unusually large deviations predict subsequent reversal. Potential features:
return deviation, distance from moving average, z-score, RSI, volatility-adjusted move,
volume spike, liquidation spike. Do not assume "oversold" means reversal — measure the
conditional return distribution.

### Breakouts

Test: recent high/low, Donchian channels, volatility compression → expansion,
volume-confirmed breakouts, multi-timeframe breakouts. Measure both continuation and failed
breakout/reversal.

### Cross-sectional momentum

Priority. For a defined liquid crypto universe: rank assets → momentum score → top quantile
→ long (bottom quantile → short if supported, else compare against market/BTC benchmark).
The system must prevent survivorship bias in the universe.

---

## 4. Crypto-specific research

After Tier 1 is working, add features unavailable in traditional equity research.

### Funding

Investigate whether extreme funding conditions predict continuation, reversal, volatility,
or liquidation events. Examples: funding percentile, funding change, funding acceleration,
funding × momentum, funding × volatility.

### Basis

Investigate futures premium/discount. Hypothesis: large, persistent futures basis may
provide an exploitable return independent of directional BTC exposure. Include spot price,
futures price, basis, funding, transaction costs, holding period.

### Liquidations

Investigate whether unusually large liquidation events create continuation, temporary
dislocation, mean reversion, or volatility expansion. Normalize liquidation size against
typical market activity.

---

## 5. Market regimes

Every strategy should be evaluated by regime: Bull, Bear, Sideways, High volatility, Low
volatility, Crash, Recovery. Do not merely label regimes after seeing strategy results —
define the regime methodology independently and record it as part of the experiment. Output
example:

```
Strategy: Momentum-24h
Bull              PASS
Bear              FAIL
Sideways          WEAK
High volatility   PASS
Low volatility    WEAK
Crash             FAIL
Recovery          PASS
```

This should influence robustness scoring.

---

## 6. Data requirements

Before implementing strategies, inventory the data available from Freqtrade and determine
what additional data is required.

- **Price data:** OHLCV, trades if available, timeframe, volume
- **Derivatives:** funding, futures prices, open interest, liquidation data
- **Market structure:** spreads, order book where realistically available

Do not build strategies around data that cannot be reliably obtained during live trading. A
feature unavailable at decision time must never be allowed into research.

---

## 7. Prevent lookahead and leakage

Hard requirement. Every feature must have an explicit `available_at` timestamp. The
research engine must guarantee: a signal at time T can only use information that was
actually available at or before T. Run existing Freqtrade lookahead/recursive-analysis
tooling where applicable. Add tests specifically designed to detect: future candles, future
funding, future liquidation information, future universe membership, future normalization,
future ranking, future regime labels.

---

## 8. Universe construction

Crypto survivorship bias is a major concern. Do not simply backtest today's top 100 coins
against historical data. If possible, reconstruct what assets were actually tradable at each
historical point — record listing date, delisting date, liquidity, exchange availability,
minimum volume, historical universe membership. If historical membership cannot be
reconstructed reliably, explicitly downgrade the evidence quality.

---

## 9. Transaction-cost modeling

Every strategy must be tested with realistic exchange fees, spread, slippage, funding,
execution assumptions. Then stress test: baseline, 1.25× costs, 1.5× costs, 2× slippage. A
strategy that only works under optimistic execution assumptions should fail robustness
testing.

---

## 10. Discovery vs validation

Never allow strategy discovery and final evaluation to use the same data. Implement:
Discovery → Validation → LOCK → OOS. Prefer walk-forward validation (Train → Validate →
Test, repeated across sliding windows), exact windows configurable. Every result must record
dataset period, universe, timeframe, parameters, code version, experiment ID.

---

## 11. Multiple-testing correction

Critical component. The system must record: how many hypotheses were tested, how many
parameter combinations, how many assets, how many timeframes, how many holding periods, how
many variants. A strategy discovered after 50 experiments should not be treated identically
to one discovered after 50,000 experiments. Implement an experiment ledger. Eventually
incorporate effective trial count, deflated Sharpe, selection-bias penalties,
false-discovery considerations. The precise statistical methodology should be documented
before implementation.

---

## 12. Robustness testing

For every serious candidate, test:

- **Parameter stability** — does profitability exist across a region of parameter space (not just one lucky value)?
- **Time stability** — does the strategy work across different historical periods?
- **Asset stability** — does it work only on BTC, or across BTC/ETH/SOL/...?
- **Regime stability** — does it depend on one particular market environment?
- **Cost stability** — does modestly worse execution destroy the edge?
- **Trade-count stability** — is the result dependent on only a handful of trades?

---

## 13. Monte Carlo

For promising candidates, estimate drawdown distribution, losing streak distribution,
probability of severe drawdown, return uncertainty, probability of negative future
performance. Do not present one backtest equity curve as the expected future path.

---

## 14. Strategy scoring

Create a Robustness Score, not simply a profitability score. Potential components: OOS
performance, OOS consistency, drawdown, Sharpe/Sortino, deflated Sharpe, parameter
stability, time stability, asset stability, regime stability, cost sensitivity, trade count,
Monte Carlo risk, profit concentration, IS/OOS degradation. The exact weighting must be
documented and tested.

---

## 15. Strategy report

Every candidate should produce a standardized report, e.g.:

```
========================================
STRATEGY: Cross-Sectional Momentum #17
========================================
Universe:        Liquid top-100 crypto
Timeframe:       4h
Holding period:  3 days

DISCOVERY    Return +62%   Sharpe 1.72   Max DD -19%
VALIDATION   Return +24%   Sharpe 1.21   Max DD -16%
OOS          Return +18%   Sharpe 1.08   Max DD -17%

ROBUSTNESS
Parameter: PASS   Time: PASS   Asset: PASS   Regime: CONDITIONAL
Costs: PASS   Lookahead: PASS   Universe: PASS

Experiments:     1,284
Deflated Sharpe: 0.71

VERDICT: PAPER TRADE
```

The report should also explain *why* the strategy passed or failed.

---

## 16. Non-expert user workflow

User should not need to understand strategy coding: select assets → select risk tolerance →
select maximum acceptable drawdown → select trading frequency → start research. The research
engine tests strategy families, ranks robust candidates, explains evidence, recommends
candidates. The user chooses whether to paper trade.

---

## 17. Paper-trading promotion

No strategy should automatically go from BACKTEST to LIVE. Require: Backtest → Validation →
OOS → Paper trading → Live eligibility. Paper trading should verify actual exchange
connectivity, signal timing, order behavior, fills, slippage, data availability, operational
reliability.

---

## 18. Live strategy health

Once deployed, continue evaluating whether the edge remains intact. Monitor expected vs.
actual expectancy, win rate, drawdown. States: HEALTHY, WATCH, DEGRADED, SUSPENDED. Do not
automatically assume every losing streak means the edge is dead — use statistical
thresholds.

---

## 19. AI research assistant — later phase

After the statistical framework is reliable, add AI: generate hypotheses, inspect existing
research, suggest features/strategy combinations, explain failures, identify suspicious
results, propose follow-up experiments, summarize evidence. But AI-generated hypotheses must
enter exactly the same experimental pipeline as human-generated hypotheses — AI gets no
special authority.

---

## 20. What NOT to build initially

Do not initially build: autonomous AI trading, LLM-generated buy/sell decisions,
high-frequency trading, market making, cross-exchange arbitrage, complicated neural
networks, hundreds of indicators, massive hyperparameter searches. First prove the research
framework can reliably distinguish a robust candidate from a beautiful overfit.

---

## MVP (for this proposal — a future session, not the current one)

**Strategy families:** Momentum, Trend following, Mean reversion, Breakout, Cross-sectional
momentum.

**Research capabilities:** train/validation/OOS, walk-forward, realistic costs, parameter
stability, regime analysis, experiment ledger, multiple-testing accounting, Monte Carlo,
robustness score, standardized strategy report.

**Execution:** use existing Freqtrade functionality wherever possible.

**User interface:** a CLI is sufficient initially.

---

## Success criterion

The MVP is not successful because it discovers a strategy with a 500% backtest. The MVP is
successful if it can demonstrate, using controlled tests, that it:

1. Finds plausible strategy candidates.
2. Rejects obvious overfit strategies.
3. Detects lookahead/data leakage.
4. Penalizes excessive experimentation.
5. Distinguishes IS performance from OOS performance.
6. Identifies regime dependence.
7. Models realistic costs.
8. Produces reproducible research.
9. Gives a non-expert understandable evidence-based recommendations.
10. Does not claim profitability without sufficient evidence.

---

## Final product vision

> "I don't know which crypto strategy works. Find out for me."
>
> "We tested 4,812 hypotheses across five strategy families. Most failed out-of-sample. Six
> survived the initial robustness tests. Two remain promising after transaction-cost stress
> testing and multiple-testing correction. One has the best evidence/risk profile for your
> stated risk tolerance. It is currently recommended for paper trading, not live capital."

That is the product. Freqtrade remains the execution engine. The new system becomes the
research scientist. Central engineering principle: **optimize for evidence of a persistent
edge, not impressive backtests.**
