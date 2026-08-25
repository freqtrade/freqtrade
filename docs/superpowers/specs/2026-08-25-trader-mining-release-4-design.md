# Trader/Wallet Mining — Release 4 (Temporal Validation) — Design

## Context

Fourth release of `TRADER_WALLET_MINING_PROPOSAL.md`'s phased plan, building directly on
Release 3's `compute_metrics`/`format_report` (merged, unchanged since). Implements Phase
6, the proposal's own "most important research requirement": every metric reported so far
is computed over a wallet's entire history at once, with no way to tell a real edge from
lookahead-contaminated noise. `metrics.py`'s own module docstring already names this gap
directly: *"real out-of-sample validation is Release 4's job (the TRAIN/VALIDATION/TEST/
FORWARD framework), not this module's."*

Priority of this release over the sibling `CRYPTO_STRATEGY_DISCOVERY_PROPOSAL.md` work was
cross-checked with lmchatbot (gemini, verified by chatgpt) before scoping began: the
strategy-discovery engine already has real OOS validation (`research/gate.py`'s walk-forward
splitting); wallet-mining does not, making this the higher-risk gap to close first.

Scope for this release comes directly from the user's own release plan: build the
chronological splitter and stop -- no multi-wallet work (Release 5), no new metrics beyond
what `compute_metrics` already computes, no automated hypothesis pipeline (Releases 6-7).

## What this is

Two new pure, DB-free modules under `research/trader_mining/`:

- `splitting.py` -- partitions a wallet's `ReconstructedTrade` list into TRAIN/VALIDATION/
  TEST/FORWARD buckets by a configurable set of chronological boundaries.
- `split_report.py` -- computes `compute_metrics` per bucket and formats a report showing
  each period's sample count, date range, and metrics, alongside a clearly labeled
  whole-history reference section.

Plus a `trader-report` CLI extension: three new optional flags
(`--train-end`/`--validation-end`/`--test-end`) that, when all given, print the split report
instead of today's aggregate-only report.

## What this is not

**Not multi-wallet.** `split_trades`/`compute_split_report` take one wallet's
`list[ReconstructedTrade]`, exactly like `compute_metrics` does today -- no `trader`
grouping parameter, no per-wallet dict-of-results shape. Multi-wallet candidate selection
and its own multiple-comparisons guardrails are Release 5's explicit job.

**Not a new metric set.** `compute_metrics` is imported and called per period, unmodified.
No period-specific statistic (e.g. a first-class "degradation score" field) is added to
`WalletMetrics`.

**Not an automatic rejection rule.** The proposal's own review-notes correction is explicit:
"No hard 'discard if expectancy degrades >50% from TRAIN to VALIDATION' rule ... Report
degradation as a diagnostic, not an automatic rejection threshold." `format_split_report`
prints the TRAIN-to-VALIDATION expectancy delta as free-floating diagnostic text only --
never a pass/fail verdict, never a threshold, never the words "reject" or "fail".

**Not a trade-splitting operation.** A trade that enters in one period but exits after that
period's boundary is not divided, duplicated, or excluded -- it counts wholly in its entry
period (see "Design decisions" below for why entry, not exit, decides membership), with the
straddle itself surfaced only as a diagnostic count (`n_straddling`).

**Not a change to `walkforward.py`/`gate.py`.** Confirmed by exploration: `generate_windows`
is pure date-math but shaped for *repeated rolling* train/test windows over OHLCV backtests,
not a single fixed four-way partition of discrete trade records, and `gate.py` has no
TRAIN/VALIDATION/TEST/FORWARD concept at all. Nothing else in `research/` partitions
discrete timestamped records. This release does not touch either file.

**Not the `tid=0` ingestion bug.** Already tracked in `FIELD-NOTES.md`
("Hyperliquid's `tid` is not always a real per-fill identifier"), explicitly deferred to
"before or alongside Release 5" -- a Release 1/2 ingestion concern, unrelated to splitting
already-reconstructed trades.

## Timezone landmine (verified empirically, not assumed)

Hyperliquid fills are ingested tz-aware UTC (`ingestion.py:70`,
`datetime.fromtimestamp(ms/1000, tz=UTC)`), but `ReconstructedTrade.entry_timestamp`/
`exit_timestamp` are plain SQLAlchemy `DateTime` columns (`models.py:179`, `:181`), not
`DateTime(timezone=True)`. Confirmed by writing a tz-aware trade to a real SQLite engine and
reading it back in a fresh session: `entry_timestamp.tzinfo` comes back `None`. Meanwhile
the existing `gate` CLI subcommand's own boundary-date convention is tz-aware
(`cli.py:100-101`, `datetime.fromisoformat(...).replace(tzinfo=UTC)`). A naive-vs-aware
comparison between a tz-aware boundary and a freshly-queried naive trade timestamp raises
`TypeError: can't compare offset-naive and offset-aware datetimes` -- exactly the situation
`research/cli.py`'s `trader-report` handler creates today (`cli.py:146-155`, a fresh
`query.all()`).

## Design decisions

- **Split key: `entry_timestamp`, not `exit_timestamp`.** Entry is the decision point being
  studied -- Phase 6's own framing is "TRAIN -> discover candidate behavior." Using
  `exit_timestamp` would let a trade's outcome, only knowable at exit, decide which research
  phase it is scored in: a lookahead channel built into the split itself, exactly the kind
  of thing this release exists to prevent.
- **Straddling trades count wholly in their entry period.** Splitting a single trade across
  two periods, or excluding it, would turn "report sample counts" into an ambiguous
  fractional-accounting problem for no research benefit. `straddles_boundary()` flags the
  case as a diagnostic (`SplitTrades.n_straddling`) without altering assignment.
- **Timezone handling: convert to UTC, then strip tzinfo, once.** A naive `datetime` in this
  codebase always means "already UTC" -- that is the ingestion invariant, not a guess.
  `PeriodBoundaries.__post_init__` and the trade-comparison path both apply the same
  `_to_naive_utc` conversion: tz-aware input is converted to UTC first (not just stripped,
  so a non-UTC offset like `+05:00` still normalizes correctly), naive input is passed
  through unchanged. This makes the module correct regardless of which convention a caller
  used, and regardless of whether a trade came from a fresh DB read or an in-process object
  -- no burden on any caller to remember the landmine above.
- **FORWARD is open-ended.** Matches the proposal's own "FORWARD 2026 -> paper trading"
  framing verbatim -- paper trading is ongoing by definition. Only three boundary dates
  (`train_end`, `validation_end`, `test_end`) configure four periods; no fourth date needed.
- **Boundaries validated eagerly, at `PeriodBoundaries` construction.** Rejecting
  non-strictly-increasing dates (`train_end < validation_end < test_end`) at construction,
  not at split time, gives every caller -- CLI, a future test, a notebook -- the same
  fail-fast guarantee, and is the single check that protects the no-overlap requirement
  (equal or inverted boundaries would otherwise silently produce an empty or overlapping
  period).
- **CLI flags are all-or-nothing; the split report replaces, never supplements, the printed
  aggregate output.** A single missing boundary is a config error (`subparser.error(...)`),
  not a valid partial state. When all three are given, `format_split_report`'s own output
  already contains a distinctly labeled "whole-history (reference only -- NOT
  out-of-sample)" section built from the same unmodified `compute_metrics(trades)` call the
  plain path uses -- satisfying "keep existing whole-history metrics, but clearly
  distinguish them" without printing the aggregate report twice. When no split flags are
  given, behavior is byte-identical to today's `trader-report` -- zero regression risk.
- **Two files, not one.** "Assign a trade to a period" (`splitting.py`) and "compute and
  format a report about the assignment" (`split_report.py`) are separable concerns with
  different failure modes (a boundary-validation bug vs. a formatting bug) and different
  test surfaces -- unlike `metrics.py`'s `compute_metrics`/`format_report`, which share one
  file because both operate on the same single `WalletMetrics` object, here the two halves
  operate on different objects (`SplitTrades` vs. `WalletMetrics` per period).

## Architecture

```
research/
  trader_mining/
    splitting.py                [new]
      PERIODS: tuple[str, str, str, str]  -- ("TRAIN", "VALIDATION", "TEST", "FORWARD")
      PeriodBoundaries (frozen dataclass) -- train_end/validation_end/test_end,
        tz-normalized and strictly-increasing-validated at construction
      assign_period(trade, boundaries) -> str
      straddles_boundary(trade, boundaries) -> bool
      SplitTrades (frozen dataclass) -- train/validation/test/forward lists + n_straddling
      split_trades(trades, boundaries) -> SplitTrades
        -- pure, no DB, no metrics; a trade's period is a function of its own
           entry_timestamp and the configured boundaries alone, never a statistic
           computed across the full trade set (the research/regime.py anti-lookahead
           lesson: that classifier ranks each window against a full-sample median
           computed across windows including future ones, safe only because it is a
           post-hoc whole-run report -- this splitter must not repeat that pattern)
    split_report.py             [new]
      PeriodSummary (dataclass) -- period, start, end, n_trades, metrics: WalletMetrics
      SplitReport (dataclass) -- boundaries, periods (4, PERIODS order), n_straddling,
        whole_history: WalletMetrics
      compute_split_report(trades, boundaries) -> SplitReport
        -- calls split_trades then compute_metrics per bucket, unmodified
      format_split_report(report, trader) -> str
        -- plain text; each period section reuses metrics.format_report verbatim;
           whole-history section labeled distinctly; TRAIN->VALIDATION expectancy
           delta printed as diagnostic-only text, never a verdict
  cli.py                        [extended]
    trader-report subcommand: three new optional flags --train-end/--validation-end/
    --test-end (YYYY-MM-DD, all-or-nothing). When all given, dispatches to
    compute_split_report + format_split_report instead of compute_metrics + format_report.
    Reuses cli.py's existing datetime.fromisoformat(...).replace(tzinfo=UTC) convention
    (already used by the gate subcommand) to parse them.
```

## Testing plan

- `test_splitting.py`: boundary validation (strictly-increasing enforcement, equal-dates
  rejection), tz normalization (tz-aware UTC, tz-aware non-UTC offset, naive-treated-as-UTC,
  all three producing the same stored value), `assign_period`'s half-open boundary semantics
  at every one of the 8 edge dates (just before/at/just before the next/at each of the three
  cuts, plus far future for FORWARD's open end), `straddles_boundary` true/false cases,
  `split_trades`'s no-overlap guarantee (every input trade appears in exactly one output
  bucket -- asserted, not assumed), empty-history handling, single-period-only histories.
  Also a real end-to-end regression test that writes a tz-aware trade to a real (not
  `:memory:`-shortcut-only) SQLite session and reads it back in a fresh session before
  calling `assign_period` -- proving the documented timezone landmine is actually handled,
  not just simulated with a hand-built naive `datetime` literal.
- `test_split_report.py`: per-period metrics correctness (each bucket's `compute_metrics`
  output matches hand-picking the same trades and calling `compute_metrics` directly), empty
  periods print `n/a` not `None` and don't crash, whole-history section is present and
  labeled, and an explicit test that no "reject"/"fail"/"threshold" language appears
  anywhere in the formatted report even when a large TRAIN-to-VALIDATION degradation is
  present.
- `test_cli.py`: partial-flags rejection (`SystemExit`, error message mentions "together"),
  split report printed when all three flags given, and a regression test proving output is
  byte-identical to today's plain report when no split flags are given.
- Real-data validation (supplementary, not primary): run `trader-report` with and without
  the three flags against an already-ingested/reconstructed wallet from an earlier release;
  confirm period counts plus `n_straddling` reconcile against the wallet's known total trade
  count, and that the no-flags path is unchanged from Release 3's previously-validated
  output.
