# Proposal: Public Crypto Trader / Wallet Strategy Mining

> Independent research-subsystem proposal, parallel to
> `CRYPTO_STRATEGY_DISCOVERY_PROPOSAL.md` (the strategy-family discovery/validation engine).
> Not yet brainstormed or planned; saved here as the source proposal for a future session.

## Objective

Add a research subsystem to the crypto trading fork that discovers potentially profitable
trading behavior from publicly observable crypto trader activity, reconstructs that activity
into normalized trades, and produces testable strategy hypotheses for the existing
backtesting/paper-trading infrastructure.

The goal is not to build a copy-trading system.

The goal is to answer:

> Can profitable behavior observed in real crypto traders be identified, reconstructed, and
> converted into a strategy that demonstrates a repeatable out-of-sample edge?

The system must treat every discovered trader as a research subject, not as proof that a
strategy works.

## Phase 1 — Hyperliquid Research Adapter

Start with Hyperliquid because its public API exposes wallet-specific fills and closed P&L
directly, making it substantially easier to reconstruct actual trading activity than
arbitrary EVM transactions.

Implement a provider abstraction so Hyperliquid is the first provider rather than a permanent
architectural dependency.

Conceptually:

```python
class TraderDataProvider(Protocol):
    def get_fills(self, trader: str, ...): ...
    def get_account_summary(self, trader: str, ...): ...
```

The first implementation should be: `HyperliquidTraderDataProvider`.

The provider must be read-only. No order placement, account credentials, or trading
functionality should be introduced.

## Phase 2 — Normalize Public Fills

Create an internal representation independent of the upstream provider.

At minimum, retain: `trader`, `timestamp`, `symbol`, `side`, `price`, `quantity`, `notional`,
`position`, `closed_pnl`, `order_id`, `transaction_id`.

Preserve the original provider payload as well, so that normalization bugs can be
investigated later.

The normalized data should be persisted locally rather than requiring repeated API calls.

The ingestion process must be: idempotent, resumable, rate-limit aware, tolerant of duplicate
fills, explicit about incomplete history, timestamped with the source retrieval time.

Do not silently treat missing historical data as zero activity.

## Phase 3 — Trade Reconstruction

Raw fills are not equivalent to trades.

Implement a reconstruction layer that groups fills into logical position/trade records.

A reconstructed trade should contain approximately: `trader`, `symbol`, `direction`,
`entry_timestamp`, `entry_price`, `exit_timestamp`, `exit_price`, `quantity`, `gross_pnl`,
`fees`, `net_pnl`, `holding_time`.

Handle: partial entries, partial exits, position increases, position reductions, reversals,
multiple fills at different prices, positions spanning multiple fills.

The reconstruction algorithm should be heavily unit tested using hand-built fill sequences.

Do not assume that one fill equals one trade.

## Phase 4 — Trader Performance Analysis

Build a research analyzer that calculates performance statistics from reconstructed trades.

At minimum: trade count, total volume, gross P&L, fees, net P&L, win rate, average win,
average loss, profit factor, expectancy, median trade return, maximum drawdown, average
holding period, median holding period, long/short distribution, symbol concentration.

Where possible, report both absolute P&L and normalized/risk-adjusted measures.

Raw P&L must not be the primary ranking metric. A wallet making $2M while taking enormous
risk should not automatically rank above a wallet making $500K with substantially better
risk-adjusted performance.

## Phase 5 — Candidate Trader Discovery

Initially support importing an explicit list of wallet addresses.

Do not begin by attempting to discover every profitable wallet on the blockchain. That
creates unnecessary scale and selection-bias problems.

Example, `candidate_wallets.txt`:

```
wallet_A
wallet_B
wallet_C
...
```

Later, add automated discovery using provider-specific capabilities.

Candidate filtering should require configurable minimums such as: minimum trade count,
minimum trading volume, minimum history duration, minimum realized P&L, minimum profit
factor, maximum acceptable drawdown.

These thresholds must be configuration, not hard-coded assumptions.

## Phase 6 — Time-Aware Evaluation

This is the most important research requirement.

Do not rank a trader using their entire historical record and then claim that their strategy
works.

Historical information must be divided chronologically. For example:

```
TRAIN        2024      -> discover candidate behavior
VALIDATION   2025 H1    -> freeze the hypothesis
TEST         2025 H2    -> no parameter changes
FORWARD      2026        -> paper trading
```

The exact dates should be configurable.

The system must prevent future observations from contaminating earlier research decisions.

## Phase 7 — Behavioral Analysis

For high-quality candidate traders, generate descriptive statistics about how they trade.

Examples:

- **Market selection** — BTC, ETH, SOL, ...
- **Holding period** — < 15 minutes, 15m–1h, 1h–4h, 4h–24h, 1d+
- **Position behavior** — average position size, position-size variability, scaling in,
  scaling out, averaging down, reversals
- **Direction** — long % / short %
- **Timing** — do entries cluster around volatility spikes, breakouts, large drawdowns,
  funding-rate extremes, volume changes, market-wide moves?

This stage should be descriptive first. Do not automatically turn correlations into trading
rules.

## Phase 8 — Strategy Hypothesis Generation

For promising traders, produce a human-readable research report. Example:

```
Trader: wallet_A

Observed behavior:
- primarily trades BTC and ETH perpetuals
- median holding period: 6.4 hours
- 72% of positions are long
- entries cluster after large short-term declines
- rarely averages down
- exits frequently occur after approximately 1-2 ATR moves

Candidate hypothesis:
"BTC/ETH long entries following unusually large short-term
drawdowns may have positive expectancy when volatility remains
elevated."

Evidence:
TRAIN:      1,204 trades   expectancy: +0.18R
VALIDATION:   411 trades   expectancy: +0.11R
TEST:         397 trades   expectancy: +0.04R
```

The generated hypothesis is not automatically considered a strategy. It must be
independently implemented and tested.

## Phase 9 — Existing Backtester Integration

The resulting hypotheses should be expressible using the existing strategy/backtesting
infrastructure:

```
observed trader behavior -> candidate rule -> Freqtrade strategy
  -> historical backtest -> walk-forward testing -> paper trading
```

The original trader's actual performance must never be used as the backtest result for the
reconstructed strategy. The purpose of reconstruction is to generate a hypothesis that can be
independently tested.

## Phase 10 — Research Integrity Requirements

This feature is specifically intended to combat strategy-selection bias, so the
implementation should enforce research discipline.

**Never:** optimize directly against the trader's complete history; select traders because
they performed well during the eventual test period; report cherry-picked profitable trades;
equate one profitable wallet with a validated strategy; assume wallet identity represents a
single strategy; silently fill missing historical data; treat unrealized P&L as realized
P&L; ignore fees; ignore partial fills; assume all fills can be perfectly reconstructed.

**Always:** preserve raw source data; record ingestion timestamps; preserve provider/source
metadata; separate train/validation/test periods; report sample sizes; report drawdown;
report fees; distinguish observed behavior from inferred rules; maintain an audit trail from
source fill → reconstructed trade → analysis → hypothesis → backtest.

## Proposed Architecture

```
research/
    trader_mining/
        providers/
            base.py
            hyperliquid.py
        models.py       # RawFill, NormalizedFill, ReconstructedTrade, TraderPerformance
        ingestion.py
        reconstruction.py
        performance.py
        discovery.py
        behavior.py
        hypotheses.py
        reports.py
```

The provider interface should make future implementations possible: Hyperliquid → Bitquery →
Dune → other providers. The core research models must not depend on any one provider.

### Storage

Use the project's existing database/storage conventions rather than introducing a second
database technology without a concrete reason.

At minimum persist: `trader_sources`, `raw_fills`, `normalized_fills`,
`reconstructed_trades`, `trader_metrics`, `analysis_runs`, `strategy_hypotheses`.

Every derived record should be traceable back to its source data.

### CLI

A first useful CLI could look like:

```
freqtrade trader import hyperliquid WALLET
freqtrade trader analyze WALLET
freqtrade trader report WALLET
freqtrade trader compare WALLET_A WALLET_B WALLET_C
```

Later: `freqtrade trader discover`. Automated discovery should come after the single-wallet
pipeline is proven.

## Testing Strategy

Follow the project's existing testing conventions. The highest-risk areas deserve extensive
tests:

**Fill reconstruction** — one entry/one exit, multiple entries, multiple exits, partial
close, reversal, repeated fills, zero-size/invalid records, out-of-order records if the
provider can return them, duplicate ingestion.

**Performance calculations** — deterministic hand-calculated fixtures verifying P&L, fees,
win rate, profit factor, expectancy, drawdown.

**Time boundaries** — trade exactly at start, trade exactly at end, trade spanning a
boundary, missing history, duplicate history.

**Provider** — mock the external API; exercise the project's own normalization/reconstruction
code for real.

## Initial Scope

The first implementation should deliberately be small.

**In scope:** Hyperliquid provider; explicit wallet-address input; historical fill
ingestion; persistent raw/normalized fills; logical trade reconstruction; performance
metrics; basic behavioral analysis; human-readable research report; comprehensive tests; CLI
entry points; provider abstraction for future data sources.

**Out of scope initially:** automatic trading; copy trading; order execution; portfolio
mirroring; automated strategy deployment; automatic strategy optimization; automatic wallet
discovery across the entire blockchain; LLM-generated strategies accepted without testing;
social-media/signal scraping; CEX private-account data.

## Future Providers

Once the Hyperliquid pipeline is reliable, evaluate:

- **Bitquery** — broad multi-chain DEX trader discovery, historical DEX trades, wallet P&L.
- **Dune** — custom SQL research and independent validation of wallet/trader datasets.
- **Alchemy / GoldRush** — lower-level EVM transaction and wallet reconstruction where
  necessary.
- **Nansen / Arkham** — potentially wallet discovery/labeling sources rather than canonical
  trade data.

## Success Criteria

The feature is successful if the system can take a public wallet address and reproducibly
produce:

```
raw public activity -> normalized fills -> reconstructed trades
  -> performance statistics -> behavioral analysis -> candidate strategy hypothesis
```

with every stage auditable.

The first milestone is not finding a profitable strategy. The first milestone is proving that
we can reliably transform real public trader activity into trustworthy research data. Only
after that should we ask whether the observed behavior contains a persistent edge.

## Recommended Implementation Order

```
Task 1  Hyperliquid API client
Task 2  Raw-fill persistence
Task 3  Fill normalization
Task 4  Trade reconstruction
Task 5  Performance metrics
Task 6  Trader analysis/report
Task 7  Time-split research framework
Task 8  Behavioral feature extraction
Task 9  Strategy-hypothesis output
Task 10 Integration with existing backtesting/paper trading
Future  Bitquery / Dune / other providers
```

The key architectural principle: treat observed trader performance, reconstructed trades,
inferred behavior, and backtested strategy performance as four separate things. Never allow
the system to blur those boundaries.
