# Trader/Wallet Mining — Release 3 (Performance Metrics + Report) — Design

## Context

Third release of `TRADER_WALLET_MINING_PROPOSAL.md`'s phased plan, building directly on
Release 2's `ReconstructedTrade` records (merged, unchanged since). Turns a wallet's closed
trades into a readable performance summary -- Phase 4's metric set plus a normalized/
risk-adjusted measure, since the proposal is explicit that raw P&L must not be the primary
ranking metric.

The risk-adjusted-measure design was cross-checked with lmchatbot (gemini, verified by
chatgpt) before this release's `/loop` prompt was even written -- one real correction
survived that review and shapes this spec directly (see "Trade consistency score" below):
the first-draft framing conflated a notional-based return with a true risk-adjusted/
R-multiple measure, which `ReconstructedTrade` doesn't have the data (stop/risk distance
per trade) to compute honestly.

## What this is

`research/trader_mining/metrics.py`: a pure function
`compute_metrics(trades: list[ReconstructedTrade]) -> WalletMetrics` plus a
`trader-report WALLET` CLI subcommand that prints it as plain console/Markdown text. No new
ingestion, no new fill/ledger handling -- `ReconstructedTrade` is the only input.

## What this is not

**Not a risk-adjusted/Sharpe measure.** `ReconstructedTrade` has no per-trade stop-distance
or initial-risk field, so a true R-multiple-based measure (Van Tharp's SQN, in its
conventional form) can't be computed honestly. This release computes an SQN-*shaped*
statistic --  `sqrt(N) * mean(r) / stdev(r)` where `r` is **return on notional**
(`net_pnl / (entry_price * quantity)`), not a risk-normalized return -- and names it
`trade_consistency_score`, never "Sharpe" or "risk-adjusted", in every docstring, report
label, and field name. One trade at very high leverage and one at very low leverage can have
identical `r` despite radically different risk, and this score can't tell them apart --
documented as a known limitation, not silently glossed over.

**Not a significance test.** `sqrt(N)` gives `trade_consistency_score` a t-statistic-like
shape, but trade outcomes can be autocorrelated or regime-dependent (a wallet's 50 trades
during one strong trend aren't 50 independent draws) -- this number is a consistency
summary, not proof an edge is real or will persist. Real out-of-sample validation is
Release 4's job (the TRAIN/VALIDATION/TEST/FORWARD framework), not this release's. Stated
once in `metrics.py`'s module docstring and once in the report's own footer, not buried only
in a code comment nobody reading the report will see.

**Not mark-to-market drawdown.** Max drawdown is computed over cumulative `net_pnl` ordered
by `exit_timestamp` -- a **closed-trade P&L drawdown**, reported as an absolute dollar
figure. This is not the wallet's actual equity drawdown: `ReconstructedTrade` doesn't
capture unrealized P&L on positions that were open concurrently with others, so two trades
that overlapped in real time are still treated as sequential here. Not fixable within this
release's scope (would need position-level, not trade-level, data) -- named precisely
instead of allowed to imply something it isn't. No percentage/normalized drawdown either --
see "Design decisions" below for why.

**Not "Calmar".** No return-to-drawdown ratio in this release is annualized (that needs a
defined measurement period, out of scope here), so none of them are named Calmar. A plain
`return_to_drawdown_ratio` field, if computed, says exactly what it is instead.

**Not persisted.** No new `trader_metrics` table. `compute_metrics` is pure and cheap enough
(a handful of trades to a few thousand, not millions) to recompute from `ReconstructedTrade`
on every `trader-report` call -- see "Design decisions" below.

## Metric set (Phase 4 plus lmchatbot-vetted additions)

Phase 4's explicit minimum: trade count, total volume, gross P&L, fees, net P&L, win rate,
average win, average loss, profit factor, expectancy, median trade return, maximum drawdown,
average holding period, median holding period, long/short distribution, symbol
concentration.

Added, cross-checked against the lmchatbot review's *corrected* (not draft) recommendations
-- the draft's "max consecutive losses = psychological resilience" and "volume-weighted
return" suggestions were explicitly walked back in the verify pass and are NOT included:

- **`payoff_ratio`**: `avg_win / avg_loss` -- both already computed for Phase 4's own list,
  effectively free. Exposes a martingale-shaped wallet (high win rate, one huge loss) that
  win rate alone hides.
- **`max_losing_streak`**: longest run of consecutive losing trades (ordered by
  `exit_timestamp`). Framed neutrally as a risk/streak statistic in every doc/label --
  explicitly NOT "psychological resilience," a framing the lmchatbot verify pass called
  unjustified by the data (a losing streak can arise purely from the payoff distribution and
  trade frequency, independent of anything about the trader).
- **`pnl_concentration_top_5`**: fraction of total `net_pnl` contributed by the 5 largest
  winning trades. Flags a wallet whose track record is one or two lucky trades rather than a
  repeatable edge.

`total_volume` is `sum(entry_price * quantity)` across all trades (Phase 4's "total volume,"
using entry notional as the standard convention -- exit notional would double-count the same
economic position).

## Trade consistency score

```python
def trade_consistency_score(trades: list[ReconstructedTrade]) -> float | None:
    """sqrt(N) * mean(r) / stdev(r), r = net_pnl / (entry_price * quantity) per trade.
    NOT a risk-adjusted/Sharpe measure -- see module docstring. None (not 0.0, not NaN)
    when undefined: N < 2 (stdev needs at least 2 points), or stdev == 0 (every trade has
    the exact same return -- a real, if unusual, case: e.g. a wallet whose only two trades
    both returned exactly the same %). A capped/clamped value was considered instead of
    None for the zero-variance case; None chosen because "infinitely consistent" isn't a
    number worth reporting as if it were on the same scale as every other wallet's score.
    """
```

`N == 0`: also `None` (nothing to compute). `N == 1`: `None` (stdev undefined -- Bessel's
correction, `stdev` over one point is undefined, not zero). The report prints `n/a` for a
`None` score with a one-line reason (`"only 1 trade"`, `"0 trades"`, `"all trades had the
exact same return"`), not a blank or a `0.0` that could be misread as "bad".

## Max drawdown

```python
def max_drawdown(trades: list[ReconstructedTrade]) -> float:
    """Absolute dollar closed-trade P&L drawdown: sort by exit_timestamp, build the
    cumulative net_pnl series, track the running peak, return max(peak - cumulative) across
    the series. 0.0 for zero or one trade (no drawdown possible)."""
```

## Design decisions

- **No percentage/normalized drawdown.** `ReconstructedTrade` has no capital/equity concept
  at all -- inventing a denominator (e.g. summed entry notional) would be economically
  meaningless across wallets of different real capital. Absolute dollar figure only,
  documented as such.
- **Compute on the fly, no new table.** `trader_metrics` (listed in the proposal's Storage
  section) is deferred until something actually needs to query metrics across many wallets
  at once (Release 5's `trader compare`) -- YAGNI, matching Phase 1's own precedent of
  deferring `discovery.py`/`behavior.py`/`reports.py` until proven needed rather than built
  speculatively.
- **Report format**: plain text to stdout, Markdown-flavored (a `##` header, a metrics
  table), matching the review notes' explicit "console/Markdown, not a separate `reports.py`
  module" simplification. No new dependency, no templating engine.

## Architecture

```
research/
  trader_mining/
    metrics.py                  [new]
      WalletMetrics (dataclass) -- every field from the metric set above
      compute_metrics(trades: list[ReconstructedTrade]) -> WalletMetrics
        -- pure function, no DB access, mirrors reconstruct_trades' own "pure, DB-free
           core algorithm" precedent
      trade_consistency_score / max_drawdown -- as above, called by compute_metrics
      format_report(metrics: WalletMetrics, trader: str) -> str
        -- plain console/Markdown text
  cli.py                        [extended]
    trader-report subcommand: loads a wallet's ReconstructedTrade rows (optionally
    --symbol-scoped, matching trader-analyze's own flag), calls compute_metrics +
    format_report, prints the result
```

## Testing plan

- `test_metrics.py`: hand-built `ReconstructedTrade` sequences, PRIMARY correctness check
  (per the loop prompt's own stated priority -- this release's risk is statistical-formula
  correctness, not exchange-data-shape quirks, unlike Release 2's). Every expected value
  computed by hand, not eyeballed. Required edge cases: zero trades, one trade (stdev
  undefined), all-winning, all-losing, exact win/loss tie, all trades with the exact
  identical return (zero-variance guard -- a real `ZeroDivisionError` class of bug if
  missed).
- Real-data cross-check (supplementary, not primary): run `trader-report` against a real
  wallet via the working `trader-analyze` pipeline.
- External cross-check (the acceptance step for this release, gated on the user supplying
  reference numbers from Hyperliquid's own UI): compare `compute_metrics`' aggregate net_pnl
  against the wallet's leaderboard-displayed PNL/ROI, reconcile any delta explicitly
  (unrealized P&L, funding, truncated/unreconciled ingestion history), do not assume a
  mismatch is a bug without checking those first.
