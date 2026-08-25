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

## Review notes (lmchatbot cross-check, 2026-08-25)

Reviewed by Gemini, cross-verified by ChatGPT (several of Gemini's draft claims were
factually corrected in the verify pass -- corrections below reflect the surviving,
fact-checked findings, not the raw draft). Not yet re-brainstormed against these notes;
recorded here so a future planning session starts from them instead of re-discovering them.

**Already decided independently, confirmed sound:** the Hyperliquid provider should be a
thin wrapper around `ccxt.async_support.hyperliquid` (its `fetch_my_trades` already accepts
a `params.user`/`params.address` override reading Hyperliquid's public, unauthenticated
`userFills`/`userFillsByTime` info endpoint for *any* wallet) rather than a hand-rolled REST
client; the CLI should be its own `python -m research.cli` subcommand, matching every
existing `research/` feature, not a new `freqtrade trader ...` subcommand on freqtrade's own
core CLI parser.

**Scope for Phase 1 -- smaller than the architecture section below implies.** Both reviewers
converged on the same simplification: four files, not nine --
`trader_mining/{provider,storage,engine,cli}.py`. Defer `discovery.py` (explicit
wallet-list input only, no scanning), `behavior.py`/`hypotheses.py` (basic descriptive stats
to JSON/CSV are enough before auto-generated English hypotheses), and `reports.py` (plain
console/Markdown output) until the single-wallet pipeline is proven -- the
`TraderDataProvider` Protocol abstraction itself should wait for a second provider to
actually exist, not be built speculatively against Bitquery/Dune now.

**Hyperliquid ingestion specifics to get right from Task 1** (fact-checked, not the
draft's original framing):

- **10,000-fill hard ceiling per wallet**, not a vague "90-180 day archival" issue --
  `userFillsByTime` exposes at most the 10,000 most recent fills, full stop. A wallet whose
  requested date range can't actually be recovered must produce an explicit
  `history_completeness = truncated_by_provider_limit` result, not a guessed
  `is_history_truncated` heuristic based on whether the oldest returned fill "looks right."
- **Rate limit is weighted** (1,200 requests/minute per IP, but a large fill-count response
  costs more than a plain request) -- a naive N-requests-per-second limiter under-throttles;
  budget by response size, not request count alone.
- **Fill identity is `tid`**, not `(hash, oid)` -- `oid` identifies an *order* (one order can
  produce many fills), and `hash` can be `0` for TWAP fills. Dedupe and paginate against
  `tid`.
- **Funding is a separate endpoint** (`userFunding`), not something to filter out of the fill
  stream as a "synthetic fill." Model funding P&L separately from execution P&L rather than
  treating it as fill-stream noise.
- **Do not hard-code FIFO** for trade reconstruction (Phase 3). Hyperliquid's own fills carry
  `startPosition`, `side`, `dir`, and `closedPnl` -- reconstruction should follow the
  exchange's own net-position transitions and realized P&L, with any lot-accounting
  convention (FIFO or otherwise) applied only where a logical-trade decomposition is
  genuinely needed and documented as a researcher-imposed convention, not the trader's
  actual economic decomposition.
- **Liquidation/ADL classification needs verification against the real fill schema during
  Task 1**, not an assumed field value (the draft guessed a `dir == "Liquidated"` encoding
  that wasn't independently confirmed) -- preserve the raw `dir`/`side`/`crossed`/`closedPnl`
  fields and classify from documented behavior, not a guess.
- **Order-book depth/liquidity modeling is out of scope for Phase 1** -- track maker/taker
  volume from the fill data's `crossed` field; defer historical L2 reconstruction unless a
  later phase actually demonstrates execution capacity is a material confounder.

**Phase 4/6 metric and threshold corrections:**

- **Drop CAGR** as a reconstructed-wallet metric -- there's no clean equity-curve denominator
  without accounting for deposits/withdrawals/funding/collateral changes a fills-only
  reconstruction can't see. Report trade-level P&L/volume statistics instead of an
  equity-curve-style headline number.
- **No hard "discard if expectancy degrades >50% from TRAIN to VALIDATION" rule.** Expectancy
  is noisy and can cross zero; an arbitrary percentage cutoff is unstable at realistic sample
  sizes. Report degradation as a diagnostic, not an automatic rejection threshold.

**Phase 5/6 -- the deeper, unresolved gap: wallet selection is its own multiple-comparisons
problem, separate from parameter selection.** Ranking N candidate wallets by TRAIN
performance and keeping the top few is a selection process over N trials, exactly the kind
of thing `research/pbo.py`/`research/statistics.py` already exist to guard against for
*parameters* -- but "apply Bonferroni or a Deflated-Sharpe-style correction across the wallet
count" is an oversimplified fix, not a complete one (DSR was derived for strategy/parameter
selection specifically; substituting "wallet" for "strategy" isn't automatically valid). The
right first move is a frozen candidate cohort with a documented, pre-registered
non-performance inclusion protocol (minimum trade count, minimum history length, active
across multiple regimes -- decided *before* looking at TRAIN performance, matching Phase 5's
existing configurable-minimums idea) plus a reserved confirmation set, not a mechanical
formula substituted for real methodology. This needs its own design pass before Phase 6 is
implemented, not just before it's trusted.

## Phased release plan (2026-08-25)

The proposal below is written as ten conceptual phases; this section regroups them into
seven **shippable releases**, each one a working increment someone could actually use, in
dependency order. Consolidates the review notes above with the proposal's own §-numbering
so a future planning session starts here rather than re-deriving it. Only Release 1 is ready
to brainstorm/spec/plan next -- everything after it is provisional until the one before it
has shipped and taught us something.

**Release 1 -- Hyperliquid ingestion (read-only, single wallet).** Proposal §1-2. Three
files, not the review notes' four -- `engine.py` does reconstruction (Release 2) and has
nothing to do until raw fills exist, so it doesn't belong in Release 1's scope:
`provider.py` (thin `ccxt.async_support.hyperliquid` wrapper), `storage.py`
(raw/normalized fill persistence, `research/db.py` conventions), `cli.py` (a
`python -m research.cli trader-import` command). Deliverable: pull one real wallet's fill
history into local storage, reproducibly and idempotently, with the 10k-fill ceiling
surfaced as an explicit `history_completeness` result rather than silently truncated. No
reconstruction, no analysis yet -- this release is "can we reliably get the data," full
stop.

**Release 2 -- Trade reconstruction.** Proposal §3. Groups raw fills into logical trades
following Hyperliquid's own `startPosition`/`closedPnl` position transitions (no hard-coded
FIFO), handling partial entries/exits and reversals. Deliverable: `trader-analyze` turns
Release 1's raw fills into `ReconstructedTrade` records. Heavily unit-tested against
hand-built fill sequences, per the proposal's own testing section.

**Release 3 -- Performance metrics + report.** Proposal §4 (trade-level stats only, no
CAGR) and §15 (report format, simplified to console/Markdown per the review notes, not a
separate `reports.py`). Deliverable: `trader-report WALLET` produces a readable,
risk-adjusted (not raw-P&L-ranked) summary for one wallet.

**Release 4 -- Time-aware evaluation framework.** Proposal §6, the proposal's own
"most important research requirement." TRAIN/VALIDATION/TEST/FORWARD chronological
splitting with configurable boundaries, applied to a single wallet's reconstructed trades.
This has to exist *before* Release 5 touches multiple wallets, or every later release
inherits an unvalidated foundation. Deliverable: Release 3's metrics reported per period,
not just in aggregate -- makes "does this wallet's edge hold up over time" answerable for
the first time.

**Release 5 -- Multi-wallet candidates + selection-bias guardrails.** Proposal §5
(explicit wallet-list input, configurable minimums) merged with the deeper gap flagged
above: wallet selection across N candidates is its own multiple-comparisons problem, not
solved by a mechanical Bonferroni/DSR substitution. This release's design (the frozen
cohort + pre-registered inclusion protocol + reserved confirmation set) needs its own
brainstorming pass when it's reached, not a rubber-stamp of the proposal's original text --
flagged explicitly so it isn't accidentally implemented as an afterthought inside a later
release. Deliverable: `trader compare WALLET_A WALLET_B ...` with the guardrails built in
from day one.

**Release 6 -- Behavioral analysis (descriptive).** Proposal §7. Market selection, holding
period distribution, position behavior, direction, entry-timing correlation with
volatility/breakouts/drawdowns. Descriptive only, per the proposal's own instruction not to
automatically turn correlations into trading rules. Only worth building once Release 5 has
produced two or more real candidate wallets to describe -- a single-wallet behavior report
has nothing to compare against.

**Release 7 -- Hypothesis generation + backtester integration.** Proposal §8-9, the
proposal's own closing vision. Human-readable hypothesis reports from Release 6's behavioral
findings, each one independently expressed as a real Freqtrade strategy and run through the
*existing* `research/gate.py` walk-forward infrastructure from
`CRYPTO_STRATEGY_DISCOVERY_PROPOSAL.md` -- the original trader's own performance is never
used as the backtest result, per proposal §9's explicit requirement. This is where the two
sibling proposals in this repo converge into one pipeline.

**Deliberately not a release yet:** proposal §5's automated wallet discovery, and every
provider beyond Hyperliquid (§"Future Providers": Bitquery, Dune, Alchemy/GoldRush,
Nansen/Arkham) -- both explicitly out of scope in the proposal's own "Initial Scope"
section, and nothing above changes that.

**Known issue carried forward, not yet fixed:** Hyperliquid uses `tid=0` as a sentinel for
`"Spot Dust Conversion"` fills rather than a genuine unique identifier -- found live-
validating Release 3, reproduced as a real cross-wallet `IntegrityError` in
`research/trader_mining/ingestion.py` (full detail, real repro, and the upgrade path
already proven for the same pattern in the ledger table: FIELD-NOTES.md, "Hyperliquid's
`tid` is not always a real per-fill identifier" entry). Low-impact today (one wallet at a
time, mostly), but Release 5 imports many wallets into the same database by design --
fix this before or alongside Release 5, not after, or the collision goes from "a coincidence
in a validation scratch db" to "a real failure mode on the first multi-wallet run."

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
