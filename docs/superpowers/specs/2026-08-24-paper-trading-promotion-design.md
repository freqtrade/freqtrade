# Paper-Trading Promotion — Design

## Context

Fourth sub-project from `CRYPTO_STRATEGY_DISCOVERY_PROPOSAL.md` (§17, paper-trading
promotion). `research/gate.py`'s `run_promotion_gate` already covers Backtest →
Validation → OOS (walk-forward evaluation, DSR/BH-FDR/PBO). This sub-project adds the
next stage the proposal requires before any strategy reaches LIVE: a tracked
**Paper-Trading** stage and an explicit, auditable **Live-eligibility** decision — never
an automatic one.

This is the first `research/` sub-project to read from freqtrade's own trade-persistence
layer (`freqtrade.persistence.Trade`) rather than only OHLCV price data
(`history.load_data`, used by `research/walkforward.py` and `research/regime.py`) or
values `research/gate.py` already computed in-process (`research/scoring.py`,
`research/cost_stress.py`). Grounded against real freqtrade internals, confirmed before
writing this spec:
- `freqtrade/freqtradebot.py:1045` stamps every opened trade with
  `strategy=self.strategy.get_strategy_name()` — the bot's own normal entry flow, not
  something a caller sets. Filtering a dry-run database's trades by strategy name is
  reliable.
- `freqtrade/constants.py:23`: `DEFAULT_DB_DRYRUN_URL = "sqlite:///tradesv3.dryrun.sqlite"`
  — freqtrade dry-run and live trades live in **separate database files** (selected by
  the bot's own `dry_run`/`db_url` config), not distinguished by a per-trade flag. A
  health evaluator must be pointed at a specific dry-run database file, not guess.

## What this is not

**Not an orchestrator.** This sub-project does not launch, manage, monitor, or configure
a running freqtrade dry-run process, does not touch exchange connectivity, and does not
generate freqtrade config files. Starting an actual paper-trading run (`freqtrade trade
--dry-run ...`) remains a manual, operational step outside this package — this is a
deployment/ops concern, not a research-package one, the same boundary this session has
already drawn around backtesting's own compute cost (see `research/cost_stress.py`'s
"NOT a slippage/market-impact model" and `research/regime.py`'s scope notes). What this
sub-project DOES do is track that a candidate has entered paper trading, and evaluate
its health once real dry-run trade data exists.

**Not automatic promotion to LIVE.** The proposal states this explicitly: "No strategy
should automatically go from BACKTEST to LIVE." Nothing in this module can transition a
`PromotionRecord` to `LIVE` — only a human, calling `promote_to_live` directly outside
any automated evaluation path. Automatic transitions exist only for `PAPER_TRADING` (an
explicit manual call, since starting to paper-trade is itself an operational decision
this package doesn't make) and `LIVE_ELIGIBLE`/`REJECTED` (the only states the health
evaluator can drive a candidate into, and even `LIVE_ELIGIBLE` is a recommendation, not
a live deployment).

**Not a new statistical test.** The health evaluator's degradation check reuses the
exact clipped-ratio pattern `research/scoring.py`'s `cost_sensitivity` already
established (fraction of a baseline Sharpe retained, clipped to `[0, 1]`, fail-closed on
a non-positive baseline) — consistent with this package's existing vocabulary rather
than inventing a new one.

**Not configurable thresholds.** `MIN_PAPER_TRADING_DAYS`, `MIN_PAPER_TRADES`, and
`MIN_DEGRADATION_RATIO` are fixed module-level constants, `ponytail:`-flagged starting
defaults — matching `research/scoring.py`'s `WEIGHTS`, `research/regime.py`'s
`trend_threshold`, `research/cost_stress.py`'s `DEFAULT_FEE_MULTIPLIERS`. Not a runtime
parameter until real usage says otherwise.

**Not per-trade-in-progress monitoring.** The health evaluator reads *closed* trades
only (a trade's outcome isn't known until it closes) — it is a point-in-time snapshot a
human runs on demand (or a future scheduler could run periodically — out of scope here),
not a live dashboard.

## Architecture

```
research/models.py  [extended]
  PromotionRecord  -- new table, one row per candidate's promotion lifecycle

research/promotion.py  [new]
  State machine (guarded transitions, ValueError on invalid current state):
    create_promotion_record(session, candidate_result_id) -> PromotionRecord
      PASSED_GATE  -- requires the referenced CandidateResult.survived is True
    start_paper_trading(session, promotion_id, dry_run_db_path, started_at=None) -> PromotionRecord
      PASSED_GATE -> PAPER_TRADING
    promote_to_live(session, promotion_id) -> PromotionRecord
      LIVE_ELIGIBLE -> LIVE   (manual only -- no automated path ever calls this)
    reject(session, promotion_id, reason) -> PromotionRecord
      PAPER_TRADING | LIVE_ELIGIBLE -> REJECTED   (manual override, always available)

  Health evaluation (pure computation, no state mutation):
    evaluate_paper_trading_health(session, promotion_id, dry_run_db_path=None,
                                   periods_per_year=365) -> dict
      Reads closed Trade rows from the dry-run database (a SEPARATE sqlite file/session
      from the research ledger's own `session`), computes days elapsed, trade count,
      paper Sharpe, and a degradation ratio against the original CandidateResult's
      oos_sharpe. Returns a verdict dict, does not touch the database.

  Applying an evaluation (the only place PAPER_TRADING can change state automatically):
    apply_health_evaluation(session, promotion_id, evaluation: dict) -> PromotionRecord
      PAPER_TRADING -> LIVE_ELIGIBLE   (evaluation["eligible"] is True)
      PAPER_TRADING -> PAPER_TRADING   (not enough evidence yet -- stays, no state change)
      PAPER_TRADING -> REJECTED        (enough evidence, but it failed the bar)
```

## Components

**`research/models.py`** (existing file, extended) — one new table:

```python
class PromotionRecord(Base):
    __tablename__ = "promotion_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_result_id: Mapped[int] = mapped_column(Integer, index=True)
    state: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime)
    paper_trading_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    paper_trading_db_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    resolution_reason: Mapped[str | None] = mapped_column(String, nullable=True)
```

`candidate_result_id` references `CandidateResult.id` (the specific gate-run row being
promoted — a strategy can have many `CandidateResult` rows across re-runs and parameter
sets; only one specific passing run is ever the thing under promotion). No SQLAlchemy
relationship/foreign-key constraint is declared — a plain indexed integer column, queried
by caller-side join when needed, matching `CandidateResult` itself having no relationships
to other tables today. `state` stores a `PromotionState` enum's `.value` (a plain string,
not a SQLAlchemy `Enum` column type) — consistent with how `CandidateResult.survived`
uses a plain `Boolean` rather than a richer type for a small fixed vocabulary.

**`research/promotion.py`** (new file):

```python
class PromotionState(str, Enum):
    PASSED_GATE = "passed_gate"
    PAPER_TRADING = "paper_trading"
    LIVE_ELIGIBLE = "live_eligible"
    LIVE = "live"
    REJECTED = "rejected"
```

Module constants (`ponytail:`-flagged starting defaults):

```python
MIN_PAPER_TRADING_DAYS = 14
MIN_PAPER_TRADES = 10
MIN_DEGRADATION_RATIO = 0.5
```

`create_promotion_record(session: Session, candidate_result_id: int) -> PromotionRecord`

Loads the referenced `CandidateResult` by id (`ValueError` if it doesn't exist — a
caller-contract violation, not a data problem, matching this package's established
fail-loudly convention). Raises `ValueError` if `CandidateResult.survived` is not `True`
— only a candidate that passed the gate can begin a promotion record; this is the one
enforcement point standing between an arbitrary backtest result and paper trading.
Creates and returns a new `PromotionRecord` in `PASSED_GATE` state,
`created_at=datetime.now(UTC)`.

`start_paper_trading(session: Session, promotion_id: int, dry_run_db_path: str,
started_at: datetime | None = None) -> PromotionRecord`

Loads the `PromotionRecord` by id (`ValueError` if missing). Raises `ValueError` if its
current state is not `PASSED_GATE` (guards against double-starting or starting a record
that's already further along). Sets `state = PAPER_TRADING`,
`paper_trading_db_path = dry_run_db_path`,
`paper_trading_started_at = started_at or datetime.now(UTC)`. This is the one manual,
human-initiated step this package requires before any evaluation can run — deliberately
manual, since actually starting a dry-run bot process is outside this package's scope
(see "What this is not").

`evaluate_paper_trading_health(session: Session, promotion_id: int, starting_balance:
float, dry_run_db_path: str | None = None, periods_per_year: int = 365) -> dict`

**`starting_balance` is required, not defaulted or derived (a real gap caught while
grounding the plan against `calculate_sharpe`'s actual signature, not something the first
draft of this spec accounted for).** `research/walkforward.py`'s existing
`calculate_sharpe` calls always pass `self.config["dry_run_wallet"]` — the same
backtest `config` dict already in scope there. This function has no `config` in scope
(only a `PromotionRecord` and a `CandidateResult` DB row), and `dry_run_wallet` isn't
persisted on either — a paper-trading bot's actual configured wallet size lives in that
bot's own config file, which this package doesn't read. Rather than guess, invent a
default, or reach into a config file this module has no other reason to touch, the
caller (a human who knows the paper-trading bot's own configured stake) supplies it
directly, the same way `research.gate.run_promotion_gate` takes its `config` from the
caller rather than discovering it.

Loads the `PromotionRecord` (raises `ValueError` if missing) and requires its state be
`PAPER_TRADING` (`ValueError` otherwise — evaluating a record that hasn't started paper
trading, or has already resolved, is a caller error). Loads the referenced
`CandidateResult` for `oos_sharpe` and `strategy_id`.

`dry_run_db_path` defaults to the record's own stored `paper_trading_db_path` when not
given — the override parameter exists for tests and for a caller who wants to evaluate
against a differently-located copy of the same database, not for routine use.

Opens a **fresh, independent** SQLAlchemy engine/session against
`dry_run_db_path` via freqtrade's own `freqtrade.persistence.init_db` and queries
`Trade` rows where `Trade.strategy == candidate.strategy_id`, `Trade.is_open is False`,
and `Trade.close_date >= paper_trading_started_at`. This session is separate from the
research ledger's own `session` argument — two independent database files, never the
same connection.

**Connection hygiene (verified via lmchatbot cross-check as a real, worth-specifying
concern):** the dry-run database file may belong to a currently-running freqtrade bot
process that could be writing to it concurrently. This function must not hold that
connection open longer than the single query needs: open the engine/session, run the
query, and close/dispose both in a `finally` block (or a context manager) before
returning — never leave a lingering handle on a file another process owns. Use a bounded
connection timeout (e.g. `create_engine(url, connect_args={"timeout": 30})`) rather than
the driver default, so a transient lock from the bot's own write contends gracefully
instead of hanging indefinitely. This function does not need write access to that
database — read-only intent should be clear from the code even though `init_db` itself
doesn't expose a strict read-only mode.

**One normalized instant per call, not two independent clock reads (verified via
lmchatbot cross-check):** capture `now = datetime.now(UTC)` exactly once at the top of
this function, and derive `days_elapsed` from that same captured value — never call
`datetime.now(UTC)` a second time later in the function body, which could let the SQL
query's implicit "as of" moment drift from the `days_elapsed` arithmetic's own "as of"
moment by however long the query took to run. Likewise, derive both the tz-aware value
used for `days_elapsed` arithmetic and the naive-UTC value used in the SQL filter (see
"Timezone-awareness resolution" below) from the same single normalized
`paper_trading_started_at` read, not two separate normalizations.

Computes:
- `days_elapsed = (datetime.now(UTC) - promotion.paper_trading_started_at).days`
- `n_trades = len(closed_trades)`
- `paper_sharpe`: via freqtrade's own `freqtrade.data.metrics.calculate_sharpe`, called
  the same way `research/walkforward.py` already calls it — `calculate_sharpe(trades_df,
  min_date, max_date, starting_balance)` — built from the queried `Trade` rows'
  `close_profit_abs`/`open_date`/`close_date` fields (matching how backtest results are
  scored elsewhere in this package, not a new sentinel/edge-case story to invent).
- `degradation_ratio`: `max(0.0, min(1.0, paper_sharpe / candidate.oos_sharpe))` if
  `candidate.oos_sharpe > 0`, else `0.0` — identical shape to
  `research/scoring.py`'s `cost_sensitivity` (a baseline-ratio, clipped, fail-closed on
  a non-positive baseline), reusing an established pattern rather than inventing a new
  one for the same kind of question ("how much of a reference Sharpe survived?").

  **Verified via lmchatbot cross-check: this is a coarse heuristic evidence gate, not a
  statistically rigorous comparison, and the spec says so explicitly rather than implying
  otherwise.** A `MIN_PAPER_TRADING_DAYS`/`MIN_PAPER_TRADES`-sized paper window is
  typically far shorter (days to weeks) than the OOS window a candidate was originally
  evaluated over (often months), so `paper_sharpe` carries materially more estimation
  noise than `candidate.oos_sharpe` — a real asymmetry inherent to comparing a young
  sample against a mature one, not a bug this module can fix by picking a different
  threshold number. `evaluate_paper_trading_health`'s docstring must say this plainly:
  the returned verdict is a first-pass filter for human judgment, not proof the strategy
  is fine.

Eligibility logic (not enough evidence vs. failed vs. passed — three distinct outcomes,
not a single boolean threshold):
- If `days_elapsed < MIN_PAPER_TRADING_DAYS` or `n_trades < MIN_PAPER_TRADES`:
  `eligible = False`, `enough_evidence = False`, reason names which threshold(s) weren't
  met yet.
- Elif `degradation_ratio < MIN_DEGRADATION_RATIO`: `eligible = False`,
  `enough_evidence = True`, reason names the degradation ratio and threshold.
- Else: `eligible = True`, `enough_evidence = True`.

Returns `{"eligible": bool, "enough_evidence": bool, "days_elapsed": int, "n_trades":
int, "paper_sharpe": float, "degradation_ratio": float, "reasons": list[str]}` — the
`enough_evidence` flag is what `apply_health_evaluation` uses to distinguish "stay in
PAPER_TRADING, ask again later" from "reject now, the evidence is in."

`apply_health_evaluation(session: Session, promotion_id: int, evaluation: dict) ->
PromotionRecord`

Loads the `PromotionRecord` (raises `ValueError` if missing or not in `PAPER_TRADING`
state — same guard as the evaluator, since this is meant to be called with that
function's own output). If `evaluation["eligible"]` is `True`: sets `state =
LIVE_ELIGIBLE`, `resolved_at = datetime.now(UTC)`, `resolution_reason` summarizing the
passing metrics. Elif `evaluation["enough_evidence"]` is `True` (eligible is False but
there's enough data to judge): sets `state = REJECTED`, `resolved_at`,
`resolution_reason = "; ".join(evaluation["reasons"])`. Else (not enough evidence yet):
no state change — the record stays `PAPER_TRADING` for a future re-evaluation, and this
function returns the unchanged record.

`promote_to_live(session: Session, promotion_id: int) -> PromotionRecord`

Loads the `PromotionRecord` (raises `ValueError` if missing). Raises `ValueError` if
state is not `LIVE_ELIGIBLE`. Sets `state = LIVE`, `resolved_at = datetime.now(UTC)`.
The only function in this module that produces a `LIVE` state — called only by a human,
directly, never from `apply_health_evaluation` or any other automated path.

`reject(session: Session, promotion_id: int, reason: str) -> PromotionRecord`

Loads the `PromotionRecord` (raises `ValueError` if missing). Raises `ValueError` if
current state is not `PAPER_TRADING` or `LIVE_ELIGIBLE` (a manual override available at
either post-gate stage, not from `PASSED_GATE` before paper trading has even started, and
not re-callable on an already-resolved record). Sets `state = REJECTED`, `resolved_at`,
`resolution_reason = reason`.

## Data flow

1. A candidate passes `run_promotion_gate` (existing, unchanged) → its `CandidateResult`
   row has `survived = True`.
2. A human calls `create_promotion_record(session, candidate_result.id)` →
   `PromotionRecord` in `PASSED_GATE`.
3. A human manually launches a freqtrade dry-run instance for that strategy (outside
   this package entirely), then calls `start_paper_trading(session, promotion.id,
   dry_run_db_path)` → `PAPER_TRADING`, recording where that instance's trade data lives.
4. Periodically (manually, or by a future scheduler — out of scope here), a human or
   script calls `evaluate_paper_trading_health(session, promotion.id)`, inspects the
   returned dict, and calls `apply_health_evaluation(session, promotion.id, evaluation)`
   to act on it → stays `PAPER_TRADING`, moves to `LIVE_ELIGIBLE`, or moves to
   `REJECTED`.
5. If `LIVE_ELIGIBLE`: a human, independently of anything this package computes, decides
   whether to actually deploy live and calls `promote_to_live(session, promotion.id)`.

## Timezone-awareness resolution

This session already hit a real bug in this exact shape earlier (`research/cli.py`'s
`--start`/`--end` parsing: `Backtesting.backtest()` compares against tz-aware (UTC)
pandas `Timestamp`s internally, and a naive `datetime` crashes deep inside it). Verified
directly against freqtrade's source before writing this section, not assumed:
`freqtrade/persistence/trade_model.py:1758` stores `open_date`/`close_date` as **naive**
datetimes on the raw ORM column (`default=datetime.now`, no `UTC`), but freqtrade's own
code never compares those raw columns directly for date arithmetic — it uses the
`open_date_utc`/`close_date_utc` **properties** (`trade_model.py:536,546`), each just
`self.open_date.replace(tzinfo=UTC)` / `self.close_date.replace(tzinfo=UTC)`. That is
the established freqtrade convention this module follows: query the raw
`Trade.close_date` column in the SQL `WHERE` clause (fine — comparing two naive UTC
values against each other in SQL is internally consistent, since freqtrade always writes
naive UTC), but do all **Python-side** date arithmetic (computing `days_elapsed`,
comparing against `paper_trading_started_at`) using `.close_date_utc`, matching how
freqtrade's own code does it.

`paper_trading_started_at` itself is written as `datetime.now(UTC)` (tz-aware) into a
plain SQLAlchemy `DateTime` column — the same pattern `research/ledger.py`'s
`CandidateResult.run_stamp` already uses (`run_stamp=run_stamp or datetime.now(UTC)`)
elsewhere in this package. Since SQLite round-tripping through SQLAlchemy's plain
`DateTime` can silently drop tzinfo, `evaluate_paper_trading_health` normalizes
defensively on read rather than assuming either behavior:
`started_at_aware = promotion.paper_trading_started_at; started_at_aware = started_at_aware.replace(tzinfo=UTC) if started_at_aware.tzinfo is None else started_at_aware`
— correct whether or not the round-trip preserved awareness, with no reliance on an
assumption neither this spec nor a quick source read can settle for SQLAlchemy's SQLite
dialect in general.

The `Trade` query's `Trade.close_date >= ...` filter compares against freqtrade's
**raw, naive-UTC** `close_date` column directly (not the `close_date_utc` property,
which isn't a queryable column) — so the bound value must itself be naive UTC:
`started_at_naive = started_at_aware.replace(tzinfo=None)`, derived from the same
`started_at_aware` computed above (see "One normalized instant per call" above — never
re-derive this independently). Python-side arithmetic (`days_elapsed`, and reading each
returned `Trade`'s own timestamp) uses the tz-aware `.close_date_utc`/`.open_date_utc`
properties and the tz-aware `started_at_aware`, consistently on the aware side of the
boundary; the SQL predicate is the one and only place the naive value is used.

## Error handling

- Every state-machine function raises `ValueError` on: a missing referenced row (record
  or candidate), or an invalid current-state-to-target-state transition. No silent
  no-ops on a caller error — this mirrors `research/regime.py`'s `ValueError` on an
  empty `windows` list and `research/scoring.py`'s implicit trust in well-formed inputs
  (a malformed call is a programming error to surface, not paper over).
- `evaluate_paper_trading_health` degrades gracefully (does not raise) when
  `n_trades == 0`: `paper_sharpe` is `0` (freqtrade's own `calculate_sharpe` already
  returns the sentinel `0` for zero trades — verified this session during the
  regime-breakdown sub-project, `freqtrade/data/metrics.py:466-467`), `degradation_ratio`
  computed normally from that `0`, and `enough_evidence` will be `False` regardless
  (since `n_trades < MIN_PAPER_TRADES` is trivially true at 0) — the eligibility logic
  naturally handles the empty case without a special branch.

## Testing

- `research/tests/test_promotion.py` (new): real SQLAlchemy session construction (an
  in-memory or `tmp_path` sqlite via `research.db.get_engine`/`get_session`, matching
  `research/tests/test_gate.py`'s own house style), real `CandidateResult`/
  `PromotionRecord` rows — no mocking of the state machine itself.
  1. `create_promotion_record` succeeds for a `survived=True` candidate, starts in
     `PASSED_GATE`.
  2. `create_promotion_record` raises `ValueError` for a `survived=False` candidate.
  3. `create_promotion_record` raises `ValueError` for a nonexistent `candidate_result_id`.
  4. `start_paper_trading` transitions `PASSED_GATE` → `PAPER_TRADING`, records
     `paper_trading_db_path`/`paper_trading_started_at` (default `started_at` is
     approximately "now").
  5. `start_paper_trading` raises `ValueError` when called on a record already in
     `PAPER_TRADING` (or any non-`PASSED_GATE` state).
  6. `promote_to_live` transitions `LIVE_ELIGIBLE` → `LIVE`.
  7. `promote_to_live` raises `ValueError` from any other state (including
     `PAPER_TRADING` directly — the point of the whole feature: you cannot skip
     `LIVE_ELIGIBLE`).
  8. `reject` transitions `PAPER_TRADING` → `REJECTED` and `LIVE_ELIGIBLE` → `REJECTED`,
     records the given reason.
  9. `reject` raises `ValueError` from `PASSED_GATE` (paper trading never started) and
     from `REJECTED`/`LIVE` (already resolved).
- `evaluate_paper_trading_health` / `apply_health_evaluation`: real freqtrade `Trade`
  rows inserted directly into a fresh dry-run-style sqlite database via
  `freqtrade.persistence.init_db` + direct `Trade(...)` construction (matching how
  `research/tests/test_regime.py`/`test_scoring.py` construct real dataclasses/rows
  directly rather than running a full backtest to get them) — not a mocked query result.
  10. Not-enough-evidence case: fewer than `MIN_PAPER_TRADES` closed trades inserted →
      `evaluate_paper_trading_health` returns `enough_evidence=False`, `eligible=False`;
      `apply_health_evaluation` leaves the record in `PAPER_TRADING`.
  11. Enough evidence, good degradation ratio: enough trades/days inserted with profit
      values chosen so the computed `paper_sharpe` clears `MIN_DEGRADATION_RATIO` of the
      candidate's `oos_sharpe` → `eligible=True`; `apply_health_evaluation` transitions
      to `LIVE_ELIGIBLE`.
  12. Enough evidence, bad degradation ratio: same trade/day counts but profit values
      chosen so `paper_sharpe` falls below the threshold → `eligible=False`,
      `enough_evidence=True`; `apply_health_evaluation` transitions to `REJECTED`.
  13. Zero-trades case: no trades inserted at all → `evaluate_paper_trading_health`
      does not raise, returns `n_trades=0`, `paper_sharpe=0`, `enough_evidence=False`.
  14. `degradation_ratio` edge case: `candidate.oos_sharpe <= 0` → `degradation_ratio`
      is exactly `0.0` regardless of `paper_sharpe`, mirroring
      `research/scoring.py`'s `cost_sensitivity` zero-baseline test.
  15. `evaluate_paper_trading_health`/`apply_health_evaluation` both raise `ValueError`
      when called on a record not in `PAPER_TRADING` (e.g. still `PASSED_GATE`, or
      already `LIVE_ELIGIBLE`).

## Open items resolved during brainstorming

- Scope: **state tracker + health evaluator only**, explicitly not orchestration/exchange
  connectivity/config generation (user-directed scope boundary, matches the pattern this
  session established for every prior sub-project's "what this is not").
- Storage: **a new `PromotionRecord` table alongside the existing `candidate_results`
  table**, referenced by `candidate_result_id` (a plain indexed column, no ORM
  relationship) rather than folding promotion state onto `CandidateResult` itself, since
  a candidate's gate-evaluation history and its promotion lifecycle are different
  concerns with different write patterns (one row per gate run vs. one row per promotion
  attempt, and a gate-passing candidate might never be promoted at all).
  **Ruling** (controller decision, not user-specified): kept these separate rather than
  adding promotion columns to `CandidateResult` directly, to avoid `NULL`-heavy columns
  on every gate-run row for the (likely common) case where most passing candidates are
  never promoted to paper trading at all.
- Degradation metric: **reuses `research/scoring.py`'s exact `cost_sensitivity`
  clipped-ratio shape** rather than a new formula, for internal consistency across the
  package (both ask "how much of a reference Sharpe survived a stress condition" — fee
  stress there, paper-trading reality here).
  **Ruling**: `MIN_PAPER_TRADING_DAYS=14`, `MIN_PAPER_TRADES=10`,
  `MIN_DEGRADATION_RATIO=0.5` are starting defaults, not derived from any real paper-
  trading history (none exists yet in this fork) — explicitly flagged as
  `ponytail:`-adjust-later constants, same treatment as every other threshold this
  session has introduced.
- Eligibility has **three outcomes, not two**: not-enough-evidence (stay), rejected
  (enough evidence, failed), eligible (enough evidence, passed) — a plain boolean
  "eligible or not" would conflate "we don't know yet" with "we know it's bad," which
  would either reject too early (premature judgment on a thin sample) or never resolve
  (if a caller only checked `eligible` and looped forever without ever reaching
  `REJECTED`). **Ruling**: this three-way split is load-bearing for
  `apply_health_evaluation`'s correctness, not an implementation detail — call this out
  explicitly to the plan's task reviewer so a "simplify to one boolean" suggestion
  during implementation gets rejected, not accepted.
