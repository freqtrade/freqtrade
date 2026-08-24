# Freqtrade Research Architecture

Phase 0 investigation + Phase 1 architecture proposal for adding a research/validation layer
on top of freqtrade, informed by MarketMind's "honest backtest funnel."

No code was modified to produce this document. All file:line citations below were verified
against `develop` (commit `10db8654c`, 2026-08-23) and against `C:\dev\MarketMind` on the same date.

---

## 1. Freqtrade architecture map

```
freqtrade/
├── freqtradebot.py, worker.py     live/dry-run bot loop
├── exchange/                       ccxt adapters per exchange
├── persistence/                    SQLAlchemy Trade/Order models, DB
├── strategy/interface.py           IStrategy base class (user subclasses this)
├── optimize/
│   ├── backtesting.py              backtest engine
│   ├── hyperopt/                   parameter search (Optuna)
│   ├── hyperopt_loss/              pluggable objective functions
│   └── analysis/                   lookahead.py, recursive.py (bias detectors)
├── freqai/                         ML plumbing (rolling retrain, feature/target hooks)
├── plugins/
│   ├── protections/                cooldown, drawdown, stoploss-guard
│   └── pairlist/                   whitelist filters (volume, spread, performance...)
├── rpc/                            telegram, webhook, REST API
└── data/                           OHLCV loading, metrics (Sharpe/Sortino/Calmar/...)
```

Everything a research layer needs — strategy execution, backtest simulation, param search,
persistence, live/dry-run gating — already exists. The question this document answers is
*what's missing on top of it*, not whether to rebuild it.

---

## 2. Execution flow (live/dry-run)

```
Worker._worker() → FreqtradeBot.process()                    worker.py:83, freqtradebot.py:257
  ├─ strategy.analyze()                                       freqtradebot.py:284
  ├─ exit_positions() → should_exit() → execute_trade_exit()  freqtradebot.py:1315 / interface.py:1419
  └─ enter_positions() → create_trade() → execute_entry()     freqtradebot.py:306 / 684 / 895
       ├─ wallets.get_trade_stake_amount()                    wallets.py:387  (sizing)
       ├─ exchange.create_order()                              exchange.py:1451
       │    ├─ dry-run → create_dry_run_order()                exchange.py:1159 (orderbook-based sim fill)
       │    └─ live → ccxt _api.create_order()                 exchange.py:1481
       └─ update_trade_state() → Trade.commit()                freqtradebot.py:2339 (SQLAlchemy persist)
```

Key findings:
- **Position sizing** default is flat-% or equal-weight-unlimited (`wallets.py:349-361`); the only
  risk-based-sizing hook is `IStrategy.custom_stake_amount()` (`interface.py:625`), which does
  nothing by default. **No Kelly criterion or volatility-based sizing exists anywhere in freqtrade.**
- **No strategy versioning.** `IStrategy.version()` (`interface.py:870`) is a cosmetic string,
  not tracked, hashed, or diffed. A strategy is a stateless `.py` file re-resolved on each reload.
- Persistent per-trade state beyond the `Trade`/`Order` tables exists via `trade_custom_data`
  (`persistence/custom_data.py:28`) — usable as a hook point for research-layer metadata.

---

## 3. Backtesting flow

```
load_bt_data() → history.load_data() → clean_ohlcv_dataframe()      backtesting.py:357 / converter.py:86
  → gaps filled with flat zero-volume synthetic candles                converter.py:126-174
strategy.advise_all_indicators() (whole dataframe at once)              backtesting.py:1826
backtest_loop() — candle-by-candle, pair-by-pair                        backtesting.py:1520
  entry fill: low ≤ rate ≤ high (OHLC-only, no tick data)                backtesting.py:787
  exit priority: signal/custom_exit → stoploss → ROI → trailing          interface.py:1504-1521
fees: single flat rate, same both legs, NO SLIPPAGE MODEL                backtesting.py:268, 1223
metrics: Sharpe/Sortino/Calmar/SQN/expectancy/drawdown                   data/metrics.py
```

**What freqtrade's own docs concede** (`docs/backtesting.md:555-646`):
- No slippage in backtest fills; historic exchange precision/limits unknown (today's limits
  applied retroactively); stoploss/trailing-stop ordering within a candle is an admitted,
  documented bias; "backtesting will never replace running a strategy in dry-run mode."
- **Zero train/validation/OOS split anywhere in the engine.** The only mechanism is the user
  manually passing different `--timerange` values. Nothing prevents (or even flags) reusing
  the same range for both discovery and "validation."

---

## 4. Hyperopt flow

```
prepare_hyperopt_data() — loads ONE --timerange, ONCE                    hyperopt_optimizer.py:454
optuna.create_study(sampler=NSGAIIISampler, direction="minimize")        hyperopt_optimizer.py:406-438
per epoch: generate_optimizer(params) → backtest(SAME full range)        hyperopt_optimizer.py:321
  → loss = IHyperOptLoss subclass (Sharpe/Sortino/Calmar/drawdown/multi) hyperopt_loss_*.py
_save_result() → append JSON line to user_data/hyperopt_results/*.fthypt hyperopt.py:95
```

Critical gaps (confirmed by grep — zero hits for "deflat", "false discovery", "bonferroni",
"multiple test" anywhere in `freqtrade/optimize/`):

- **Default sampler is NSGA-III (genetic), not Bayesian** — despite `docs/hyperopt.md:7,502`
  claiming "Bayesian search." A stale-docs finding worth fixing upstream separately.
- **Every epoch backtests the exact same data it's being scored on.** No held-out split exists.
- **No loss function corrects for trial count.** All 12 loss functions in `hyperopt_loss/`
  score a single epoch in isolation; none discount for having tried hundreds/thousands of
  parameter combinations against the same fixed dataset.
- **No cross-run trial ledger.** Each `.fthypt` file is independent; nothing aggregates "how
  many parameter combinations have I ever tried" across runs — a user would have to `ls` and
  sum manually.
- **Low-trade-count guard is weak and off by default.** `hyperopt_min_trades` (`cli_options.py:357`)
  defaults to **1** — a single lucky trade can produce a "best" epoch under `OnlyProfitHyperOptLoss`
  or `SharpeHyperOptLoss`.
- Reproducibility: seeded (`hyperopt_optimizer.py:406`) but sensitive to `-j` parallelism for
  stateful samplers (TPE/GP) — undocumented caveat.

This is the single biggest gap MarketMind's methodology targets, and it's real: **freqtrade
hyperopt has no defense against the multiple-comparisons problem at all.**

---

## 5. FreqAI flow

Rolling-window walk-forward retraining is genuinely implemented (`data_kitchen.py:315`,
`split_timerange`) and label truncation correctly prevents the most basic look-ahead
(`freqai_interface.py:347-354`). But three real gaps exist:

1. **Feature population happens once over the FULL backtest range, then is sliced** —
   `freqai_interface.py:341-345`. A non-causal feature (`.shift(-1)`, centered rolling, global
   mean) leaks future information into every training window despite the walk-forward
   structure. The docs flag this in prose (`docs/freqai-running.md:71`); nothing in code
   checks for it.
2. **No default early stopping.** LightGBM/XGBoost get an `eval_set` but no
   `early_stopping_rounds` unless the user adds it manually — the shipped example config
   (`config_freqai.example.json:84`) is empty. The held-out `test_size` split (default 0.1)
   is computed and then effectively wasted for the models the quick-start guide pushes.
3. **No prediction provenance.** `historic_predictions.pkl` carries no model-version column,
   and `purge_old_models` keeps only 2 model folders per pair by default — auditing which
   model made a specific historical prediction is often impossible after the fact.

**Bottom line: FreqAI is real ML plumbing, not a validity guarantee.** It does not solve the
problem this project exists to solve; it's a data source the research layer must treat with
the same skepticism as any hand-written strategy.

---

## 6-8. Existing protections, risk controls, data-integrity controls

| Category | What exists | Verdict |
|---|---|---|
| Lookahead detection | `optimize/analysis/lookahead.py` — truncates data at each trade's open/close date, diffs signals/indicators against a full-range baseline | Real, but only checks signals it actually observed; untriggered branches, FreqAI targets (always false-positive), and limit-order pricing are blind spots |
| Recursive-indicator detection | `optimize/analysis/recursive.py` — varies `startup_candle_count`, diffs last-row indicator values | Real but last-row-only, `populate_indicators` only, single pair by default |
| Protections | `CooldownPeriod`, `LowProfitPairs`, `MaxDrawdown`, `StoplossGuard` (`plugins/protections/`) | Reactive (fire after a trade closes), operate on realized P&L only, opt-in during backtest/hyperopt (`--enable-protections`) |
| Pairlist filters | 12 filters (`plugins/pairlist/`) | Mostly liquidity/quality; **`PerformanceFilter` sorts pairs by the bot's own recent live performance — a textbook selection-bias/performance-chasing mechanism, doesn't even run in backtest** |
| Position sizing | `wallets.py`, `custom_stake_amount()` hook | No built-in Kelly or volatility-based sizing |
| Drawdown circuit breaker | `MaxDrawdownProtection` | Realized-drawdown only, global-only, reactive between trade-close events — not continuous equity monitoring |
| Dry-run/live gating | `dry_run` defaults to `true` everywhere (`constants.py:196`, `configuration.py:470`) | Matches the "off by default, explicit false to go live" pattern MarketMind uses — **already present, no porting needed** |
| Strategy versioning | None | A `.py` file with a cosmetic `version()` string; no history, no hash, no lifecycle |

---

## 9. Proposed research-layer architecture

```
                    freqtrade (unmodified core)
              ┌────────────────────────────────┐
              │ IStrategy, Backtesting,         │
              │ Exchange, Trade/Order DB,       │
              │ dry-run, protections, pairlist  │
              └───────────────┬──────────────────┘
                               │ called in-process
                               │ (Backtesting class, IStrategy instances)
                               ▼
              ┌────────────────────────────────┐
              │        research/ package         │
              │  (new, separate DB + CLI)        │
              │                                  │
              │  statistics.py   — DSR, BH-FDR,  │
              │                    permutation   │
              │  pbo.py          — CSCV          │
              │  walkforward.py  — train/test    │
              │                    windowing     │
              │  ledger.py       — candidate/    │
              │                    trial ledger  │
              │  gate.py         — promotion     │
              │                    gate CLI       │
              └───────────────┬──────────────────┘
                               │
                        PASS ──┴── FAIL
                         │
                         ▼
              freqtrade dry-run (existing, unmodified)
                         │
                    live eligibility (future phase)
```

**Design principle: the research layer is a client of freqtrade, not a fork of it.** It
imports `freqtrade.optimize.backtesting.Backtesting` and `freqtrade.strategy.interface.IStrategy`
directly (in-process, same Python environment — this is a monorepo addition, not a subprocess
wrapper), runs its own experiments by calling `Backtesting.backtest()` repeatedly across
windows/params, and records results in its own SQLite DB. It never touches `freqtradebot.py`,
`exchange/`, or `persistence/models.py`.

---

## 10. Proposed database schema

Own SQLite DB (SQLModel, following MarketMind's `ledger.py` pattern — trivial, ~80 lines):

```python
class CandidateResult(SQLModel, table=True):
    id: int
    run_stamp: datetime
    strategy_id: str
    strategy_family: str          # alias-mapped, e.g. "ema_cross_v3" → "trend_following"
    params_json: str
    universe: str                 # e.g. "BTC/USDT,ETH/USDT"
    timeframe: str
    discovery_start: date
    discovery_end: date
    validation_start: date | None
    validation_end: date | None
    oos_start: date | None
    oos_end: date | None
    n_trials_this_run: int        # grid/search size for this run
    is_sharpe: float
    oos_sharpe: float | None
    deflated_sharpe: float | None
    permutation_p: float | None
    pbo: float | None
    survived: bool
    evidence_json: str            # full detail blob for the "advanced view"
```

`family_trial_count(family)` = `max(row count for family, declared n_trials)` — ported
directly from MarketMind's design (`ledger.py:28`) — is what makes deflation correct across
repeated research sessions instead of resetting to zero each run.

---

## 11. Integration points

- **Backtesting**: call `freqtrade.optimize.backtesting.Backtesting` directly per
  train/test window — reuse its fill/fee/stoploss/ROI simulation unchanged.
- **Hyperopt**: for the per-window "best-train-params" step, either (a) call freqtrade's own
  `Hyperopt`/`HyperOptimizer` with a small epoch budget per window, or (b) run a plain grid
  search via repeated `Backtesting.backtest()` calls when the param space is small — grid is
  simpler and keeps the trial count exactly known (needed for DSR deflation), so **prefer (b)
  for the MVP** and only reach for freqtrade's hyperopt when a space is too large for a grid.
- **Strategy objects**: `StrategyResolver.load_strategy()` — instantiate the user's existing
  `IStrategy` subclass unchanged; the research layer treats it as an opaque, parameterized
  experimental object per the user's Phase 1 spec.
- **Live/paper comparison**: read `Trade`/`Order` rows directly from freqtrade's own DB
  (`persistence/trade_model.py`) for the future edge-monitoring phase — no need to duplicate
  execution tracking.
- **Lookahead/recursive checks**: shell out to (or import) freqtrade's existing
  `optimize/analysis/lookahead.py` and `recursive.py` as a pre-gate step — don't reimplement.

---

## 12-14. Reuse / Extend / Replace

**Reuse unchanged:** `Backtesting` engine, `Exchange`/ccxt layer, exit/stoploss/ROI simulation,
`Trade`/`Order` persistence, protections, pairlist filters, dry-run infrastructure, RPC
(Telegram/API), lookahead/recursive analysis tools, `custom_stake_amount`/`leverage` hooks as
the integration point for risk-based sizing.

**Extend (new code, freqtrade untouched):** the entire `research/` package — statistics,
walk-forward runner, candidate ledger, promotion gate, (later) regime classifier, cost-stress
sweep, live edge monitor.

**Replace:** nothing in freqtrade core. The one *soft* recommendation: document (not code-change)
that `PerformanceFilter` is a selection-bias risk and shouldn't be used to select pairs for a
strategy still under validation.

---

## 15-16. MarketMind components — port vs. don't

**Worth porting (design and, mostly, code):**

| Component | MarketMind location | Portability |
|---|---|---|
| Deflated Sharpe Ratio | `backend/backtest/enhanced/statistics.py:168` | Near-verbatim; adjust `periods_per_year` for crypto's 24/7 market |
| Benjamini-Hochberg FDR | `statistics.py:14` | Reusable as-is — pure stats, zero domain coupling |
| Permutation tests (sign-flip + trade-sequence) | `statistics.py:41,301` | Near-verbatim |
| PBO via CSCV (+ Reality Check/SPA) | `backend/backtest/enhanced/reality_check.py:271` | Reusable as-is — the single highest-value file to port whole |
| Candidate ledger + effective-trial-count | `backend/research/ledger.py` | Port the design (count-then-write, family aliasing); reimplement schema against this project's own tables |
| Walk-forward runner | `backend/backtest/enhanced/walkforward.py` | Port orchestration logic; swap MarketMind's `_sim`/`get_ohlcv` calls for freqtrade's `Backtesting` API |
| Regime classification + attribution | `backend/regime/classifier.py`, `attribution.py` | Port the pattern (rule-cascade classifier + causal `asof` attribution); swap VIX/SPY for BTC dominance / funding rate / realized vol |
| Risk-budget multiplier + Kelly fraction | `backend/risk_ai/engine.py` | Near-verbatim pure functions |
| Live/paper edge monitoring | `backend/auto_trader/drift_monitor.py` | Port the mechanism (significance-gated drift vs. backtest baseline, demote-not-delete); no literal HEALTHY/WATCH/DEGRADED/SUSPENDED enum exists to copy — design one |
| Layered paper-trading gate | `backend/auto_trader/executor.py`, `signal_handler.py` | Concept only — freqtrade already has its own dry-run gate; replicate the *layering* (feature flag off by default + hard config refusal + fail-closed strategy-registry check), not the IBKR code |

**Do not port:** IB Gateway/ib_insync execution code (irrelevant — freqtrade has its own
ccxt-based dry-run/live layer); yfinance data fetching; the React/FastAPI/frontend scaffolding
(freqtrade has its own API/webUI/Telegram); MarketMind's `pbo.py` (Version A, superseded by
`reality_check.py`); cost-sensitivity stress testing — **doesn't exist in MarketMind, nothing
to port, must be built fresh.**

**Hard-won lessons to actively test for when reimplementing** (from MarketMind's own
DECISIONS.md/FIELD-NOTES.md — each cost them real debugging time):

1. A Deflated Sharpe with no sample-size term is just a sign test wearing a rigor costume —
   write a discriminative-validity test (does n=10 score differently from n=10,000 at equal
   Sharpe?) before trusting any DSR implementation.
2. Sharpe is order-invariant — permuting *order* of returns cannot test it; permutation nulls
   must destroy the specific structure the statistic actually depends on (sign-flip, not reorder).
3. Terminal return is also order-invariant — the same bug independently recurred in a second
   Monte Carlo module. Split resampling by statistic type: permutation for order-*dependent*
   stats (drawdown, ruin), bootstrap-with-replacement for order-*invariant* ones (terminal return).
4. A promotion gate must never re-optimize — if the gate itself searches parameters, it becomes
   an unpriced multiple-comparisons search, the exact failure mode it exists to catch. Keep
   parameter search in the interactive path; validate only fixed, already-chosen params in the gate.
5. Per-strategy Sharpe-stability across windows is a different, weaker question than whether the
   *same winning parameters* keep winning window to window — build both checks.
6. Grep for parameters declared in a function signature but silently ignored in the body — this
   exact bug class (a `cap`/rate argument accepted but hardcoded over) recurred at least twice.
7. Decide up front whether a refused/insufficient-sample attempt still counts toward the trial
   ledger — MarketMind left this an open exploit (free below-floor retries) because it's
   single-user; don't inherit that gap if this system is ever used by concurrent/automated search.
8. A near-FDR-boundary result is inherently unstable run-to-run — surface it as "borderline,"
   don't present a crisp binary pass/fail near the threshold.

---

## 17. Estimated implementation complexity

| Piece | Size | Why |
|---|---|---|
| Statistics core (DSR, BH-FDR, permutation, PBO/CSCV) | **Small** (1-2 days) | Near-verbatim port of pure-function code MarketMind already debugged |
| Candidate ledger | **Small** (1 day) | ~80-line SQLModel table + two query functions |
| Walk-forward runner against freqtrade's `Backtesting` | **Medium** (3-5 days) | New adapter code; freqtrade's `Backtesting` class isn't designed to be called in a tight per-window loop — needs a thin wrapper |
| Promotion gate CLI | **Small** (1-2 days) | Glue: run walk-forward → stats → ledger write → PASS/FAIL report |
| Regime classification (crypto) | **Medium** (3-4 days) | No equivalent to VIX for crypto off the shelf — needs its own rule cascade against BTC dominance/funding/realized vol, plus a data source |
| Cost-sensitivity stress sweep | **Small-Medium** (2-3 days) | Doesn't exist anywhere to copy; straightforward re-run-at-N-cost-multipliers harness |
| Live/paper edge monitoring | **Medium** (3-5 days) | Needs freqtrade's live Trade DB wired to a baseline comparison + significance test |
| Portfolio construction (correlation-aware) | **Medium** (3-5 days) | Defer until ≥2 validated strategies exist — nothing to construct a portfolio from yet |
| AI hypothesis generation | **Large** (open-ended) | Explicitly last per the user's own Phase 7 — defer until the gate exists and has rejected/accepted at least a few real candidates |

---

## 18. Recommended Phase 1 MVP

Build exactly enough to answer: *"is this strategy likely to have a real edge after costs and
after accounting for how much we searched?"* — nothing else yet.

1. New `research/` package alongside `freqtrade/` (same repo/venv, own SQLite DB).
2. Port `statistics.py` (DSR + BH-FDR + permutation) and `reality_check.py` (PBO/CSCV)
   near-verbatim; set `periods_per_year` for crypto.
3. Build one `WalkForwardRunner` that calls freqtrade's `Backtesting` class per rolling
   train/test window: grid-search params on train only, evaluate the winning fixed params on
   test only, deflate by the real grid size and real `n_obs`.
4. Candidate ledger table + `family_trial_count()`, so deflation compounds correctly across
   research sessions instead of resetting each run.
5. One CLI command — `research gate <strategy> --timerange ...` — chaining: freqtrade's own
   lookahead/recursive checks → walk-forward → DSR/BH/PBO → ledger write → PASS/FAIL report.
6. Output goes to the terminal first (matches Phase 14's "hide complexity, not evidence" — a
   plain-text summary plus a `--verbose` full-evidence dump is enough for v1; no UI yet).

Explicitly deferred to later phases, in this order: regime analysis → cost-sensitivity
sweep → live/paper edge monitoring → portfolio construction → AI hypothesis generation.
Each of those is independently useful but none of them matter if step 3-5 above doesn't
first prove out against one real freqtrade strategy.
