# Live Strategy Health — Design Spec

**Source:** `CRYPTO_STRATEGY_DISCOVERY_PROPOSAL.md` §18 ("Live strategy health"), the direct
sequel to §17 ("Paper-trading promotion", shipped in `research/promotion.py`, PR #8).

## Context

`research/promotion.py` already carries a `PromotionRecord` through
`PASSED_GATE -> PAPER_TRADING -> LIVE_ELIGIBLE -> LIVE` (or `REJECTED`), with `LIVE`
reachable only via a direct, manual `promote_to_live()` call. Once a record reaches `LIVE`,
nothing in this codebase watches it again. §18 asks for exactly that: "Once deployed,
continue evaluating whether the edge remains intact. Monitor expected vs. actual
expectancy, win rate, drawdown. States: HEALTHY, WATCH, DEGRADED, SUSPENDED. Do not
automatically assume every losing streak means the edge is dead -- use statistical
thresholds."

This spec reuses `evaluate_paper_trading_health`'s machinery almost entirely: real closed
`Trade` rows read from a real SQLite DB via freqtrade's own `init_db`, `calculate_sharpe`
for the realized statistic, and the identical degradation-ratio comparison against
`CandidateResult.oos_sharpe` that already exists and is already tested. The new part is
what happens with that ratio once a strategy has been live for a while: a 4-state ladder
instead of a binary eligible/not-eligible call, evaluated repeatedly over the strategy's
ongoing life rather than once.

**`freqtrade.persistence.init_db()`'s global `Trade.session` state, and the connection-
hygiene requirement it implies, are load-bearing constraints here too** -- see
`FIELD-NOTES.md`'s "freqtrade's `Trade.session` is GLOBAL class-level state" entry, and
`research/promotion.py:196-217`'s `try`/`finally` for the pattern to follow from the start.
The final review of the paper-trading-promotion branch found this missing on the first
pass; it must not be missing here on the first pass.

## What this is not

- **Not an automatic kill switch.** This module never stops a live freqtrade bot, never
  touches process control, never writes anything freqtrade itself reads. `SUSPENDED` is a
  recorded recommendation for a human to act on, exactly as `LIVE` itself is reachable only
  by a manual human call in `promotion.py`. There is no `apply_...` step that pulls a
  strategy from live trading -- that action, if it's ever built, is a human's job outside
  this module's scope.
- **Not a new statistical test.** The gating statistic is the same Sharpe degradation ratio
  `evaluate_paper_trading_health` already computes and this codebase already trusts. No new
  hypothesis test, no new distributional assumption.
- **Not a win-rate or drawdown threshold.** Both are computed and stored because they're
  free (same query, same closed-trade rows) and genuinely useful context for a human
  reading a health report, but neither gates the state. Inventing acceptance bands for
  them would mean fabricating thresholds with no statistical grounding behind them. (This
  is not a claim that Sharpe IS a drawdown metric -- it isn't, it measures mean return
  relative to return volatility, a related but distinct concept -- only that it's the one
  statistic this codebase already has a trusted, tested threshold methodology for, and
  extending that same methodology to two more metrics with no established acceptance
  bands would mean inventing numbers, not applying statistics.)
- **Not CLI-wired.** `research/promotion.py` itself shipped with no `research/cli.py`
  subcommand -- it's a library API, driven by whatever schedules it (a cron job, a
  notebook, a future orchestrator), all out of scope here. This module follows the same
  precedent: library + tests only.
- **Not configurable thresholds.** Like every other module in this package, the starting
  thresholds are fixed, `ponytail:`-flagged module constants, not runtime parameters.
- **Not safe for concurrent callers.** `record_health_check`'s read-then-write of "the
  latest row" is not transactionally isolated against a second concurrent caller
  evaluating the same `promotion_id` at the same time -- a real gap raised during this
  spec's lmchatbot cross-check, but this package has no scheduler or concurrent-caller
  infrastructure anywhere yet (every existing module assumes one researcher, one call,
  one process, matching `promotion.py`'s and `gate.py`'s own unstated but consistent
  assumption). Building real concurrency control belongs with whatever eventually
  schedules repeated health checks, not in this evaluation module itself.

## Data model

One new table in `research/models.py`, `HealthCheck` -- one row per evaluation, an audit
trail in the same spirit as `CandidateResult` (every trial logged, survivor or not):

```python
class HealthCheck(Base):
    """One row per live-health evaluation of a PromotionRecord already in LIVE state.
    The record's CURRENT health state is simply the latest row for its
    promotion_record_id, ordered by evaluated_at -- deliberately not a mutable field on
    PromotionRecord, to avoid a second source of truth that could drift out of sync."""

    __tablename__ = "health_checks"
    __table_args__ = (Index("ix_health_checks_promotion_id_evaluated_at",
                             "promotion_record_id", "evaluated_at", "id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    promotion_record_id: Mapped[int] = mapped_column(Integer, index=True)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime)
    state: Mapped[str] = mapped_column(String(20))
    enough_evidence: Mapped[bool] = mapped_column(Boolean)
    n_trades: Mapped[int] = mapped_column(Integer)
    live_sharpe: Mapped[float] = mapped_column(Float)
    degradation_ratio: Mapped[float] = mapped_column(Float)
    win_rate: Mapped[float] = mapped_column(Float)
    max_drawdown: Mapped[float] = mapped_column(Float)
    reasons_json: Mapped[str] = mapped_column(String, default="[]")
```

`promotion_record_id` is a loose reference (an indexed int, not a SQLAlchemy
`ForeignKey`), matching `PromotionRecord.candidate_result_id`'s own existing convention in
this codebase -- no cross-table FK constraints anywhere in `research/models.py` today. The
composite index (found worth adding during this spec's lmchatbot cross-check) covers the
"latest row for this promotion_id" query that `record_health_check` runs on every call --
`ORDER BY evaluated_at DESC, id DESC LIMIT 1` -- with a deterministic tie-breaker (`id`)
for the rare case two rows share a timestamp.

`reasons_json` follows `CandidateResult.evidence_json`'s existing string-column-holding-
JSON convention (a `list[str]` of reasons, JSON-encoded, the same shape `promotion.py`'s
own `reasons: list[str]` already produces internally -- just persisted here instead of
returned transiently).

**Persistence when `enough_evidence` is `False`:** `live_sharpe`, `degradation_ratio`,
`win_rate`, and `max_drawdown` are still computed and persisted using their normal
formulas (not `None`, not a sentinel) -- this matches `evaluate_paper_trading_health`'s
own established precedent of computing `paper_sharpe=0` even at `n_trades=0` rather than
leaving it undefined. The audit trail should show exactly what evidence existed, even
when there wasn't enough of it to act on; a `None` would hide that a real (if thin)
sample was measured. Columns stay non-nullable `float`, matching `PromotionRecord`'s own
convention of only making a column nullable when the domain genuinely has no value yet
(e.g. `resolved_at` before resolution) -- here a value always exists, it's just
low-confidence.

## States

```python
class HealthState(StrEnum):
    HEALTHY = "healthy"
    WATCH = "watch"
    DEGRADED = "degraded"
    SUSPENDED = "suspended"
```

Ordered worst-last: `HEALTHY, WATCH, DEGRADED, SUSPENDED`. A record with no `HealthCheck`
row yet has an implicit current state of `HEALTHY` -- a freshly-promoted `LIVE` strategy is
presumed healthy until shown otherwise; this is the only place an "innocent until proven
otherwise" default applies.

## Components

### `research/health.py`

```python
# ponytail: starting defaults, not derived from any real live-trading history (none
# exists yet in this fork) -- adjust based on real usage once this runs against real
# strategies. Deliberately mirrors promotion.py's MIN_PAPER_TRADING_DAYS/MIN_PAPER_TRADES
# shape, applied to a rolling window instead of a one-time cumulative check.
HEALTH_WINDOW_DAYS = 30
MIN_HEALTH_TRADES = 10
HEALTHY_THRESHOLD = 0.7    # degradation_ratio >= this -> target state HEALTHY
WATCH_THRESHOLD = 0.4      # >= this, < HEALTHY_THRESHOLD -> target state WATCH
DEGRADED_THRESHOLD = 0.15  # >= this, < WATCH_THRESHOLD -> target state DEGRADED
                           # < DEGRADED_THRESHOLD -> target state SUSPENDED
MIN_HEALTH_CHECK_INTERVAL_HOURS = 24  # a rung-move requires this much real time since
                                       # the last recorded evaluation -- see "Damping
                                       # requires new evidence, not just a new call" below


def evaluate_live_health(
    session: Session,
    promotion_id: int,
    starting_balance: float,
    live_db_path: str | None = None,
) -> dict:
    """Pure evaluation (no state mutation, no DB write) of a LIVE PromotionRecord's real
    trade history over the trailing HEALTH_WINDOW_DAYS. Returns a verdict dict; call
    record_health_check with the result to persist an audit row and (possibly) move the
    record's current health state.

    Raises ValueError if the record doesn't exist or isn't currently LIVE.

    `live_db_path` defaults to the record's own stored `paper_trading_db_path` when not
    given -- the same field promotion.py already uses to remember where a record's trade
    history lives (its name predates LIVE and is not renamed here; see "Naming" below).

    `starting_balance` is required, matching evaluate_paper_trading_health's own
    established contract -- there is no other way for this function to discover the
    live bot's configured wallet size.

    IMPORTANT: freqtrade.persistence.init_db() sets Trade.session as GLOBAL class-level
    state, not a scoped per-call connection. This function fully materializes its query
    results before returning and disposes the engine/session in a finally block before
    returning -- the same connection-hygiene requirement research/promotion.py's final
    review already established for this exact pattern, applied here from the start.

    Rolling window, not cumulative: uses only closed trades with
    close_date >= now - HEALTH_WINDOW_DAYS, not the strategy's full LIVE-to-date history.
    A cumulative statistic would be slow to reflect a real regime shift months into a
    strategy's live run; a rolling window stays responsive.

    Win rate and max drawdown are computed from the same window and returned for human
    context -- they do not affect target_state.

    Returns a dict with keys: enough_evidence (bool), n_trades (int), live_sharpe
    (float), degradation_ratio (float), win_rate (float), max_drawdown (float),
    target_state (str | None -- one of HealthState's values, derived from
    degradation_ratio against HEALTHY_THRESHOLD/WATCH_THRESHOLD/DEGRADED_THRESHOLD when
    enough_evidence is True; None when enough_evidence is False, since there is nothing
    to target yet), reasons (list[str]). The DECISION (which state the current window's
    evidence supports) lives here, in the evaluator -- matching
    evaluate_paper_trading_health's own precedent of computing "eligible" in the
    evaluator, not the applier. record_health_check only ever reads target_state; it
    never re-derives it from degradation_ratio itself.
    """


def record_health_check(
    session: Session,
    promotion_id: int,
    evaluation: dict,
    evaluated_at: datetime | None = None,
) -> HealthCheck:
    """Apply an evaluate_live_health() result: write one HealthCheck audit row, computing
    the record's new current state from its previous current state (the latest existing
    HealthCheck row for this promotion_id, or implicit HEALTHY if none exists yet).

    State-change rule: if evaluation["enough_evidence"] is False, the new state equals
    the previous state exactly (no change -- insufficient data this evaluation, not
    evidence of health); evaluation["target_state"] is ignored (it is None in this case
    anyway). Otherwise, look up the most recent PRIOR HealthCheck row's evaluated_at (or,
    if none exists, treat the interval requirement as already satisfied -- the first-ever
    check can move a rung immediately). If less than MIN_HEALTH_CHECK_INTERVAL_HOURS has
    elapsed since that prior row, the new state ALSO equals the previous state exactly --
    a rung-move requires both a passing target_state AND real elapsed time since the last
    recorded check, not just a function call. (An audit row is written either way, with a
    reason noting the interval wasn't met, so repeated manual invocations are still fully
    visible in the trail -- they just can't consume more than one rung's worth of movement
    per real evaluation period.) Otherwise, the new state moves AT MOST ONE STEP from the
    previous state toward evaluation["target_state"] (in either direction -- a recovering
    strategy climbs back up one step per evaluation too, exactly as a degrading one falls
    one step at a time).

    This two-part rule (evidence gate + time gate) is what actually satisfies "do not
    automatically assume every losing streak means the edge is dead": the one-rung cap
    alone is NOT sufficient by itself -- three evaluate+record calls made seconds apart
    against the identical unchanged rolling window would otherwise walk a record straight
    from HEALTHY to SUSPENDED with zero new trading evidence between them (a real gap
    caught during this spec's own lmchatbot cross-check). Requiring
    MIN_HEALTH_CHECK_INTERVAL_HOURS of real elapsed time between rung-moves means it takes
    that many hours of the strategy's ACTUAL live trading -- not merely repeated
    evaluation calls -- to walk from HEALTHY to SUSPENDED.

    Ties in evaluated_at when finding "the most recent prior row" break by `id DESC`
    (auto-increment, so a higher id is strictly later even if two rows share a
    timestamp).

    Raises ValueError if the record doesn't exist or isn't currently LIVE (same guard as
    evaluate_live_health -- this function does not re-derive it from evaluation, it
    re-checks the record directly, since evaluation may be stale by the time this is
    called).

    `evaluated_at` defaults to datetime.now(UTC) and is stamped onto the new HealthCheck
    row and used for the interval check above -- an explicit override exists so tests can
    construct a realistic multi-day sequence of evaluations without waiting real
    wall-clock hours between calls, the same reason start_paper_trading accepts an
    explicit started_at override.
    """
```

### `research/models.py`

Add `HealthCheck` as specified above. No changes to `PromotionRecord` or
`CandidateResult`.

## Naming

`PromotionRecord.paper_trading_db_path` and `paper_trading_started_at` keep their existing
names even though `evaluate_live_health` reads the same field for a `LIVE` record's real
trading database, not a paper one. Renaming those columns is a bigger, unrelated schema
change to an already-shipped table for a cosmetic improvement -- out of scope. A docstring
note on `evaluate_live_health` says so explicitly, so a future reader isn't confused by the
mismatch.

## Rolling window boundary

A trade's `close_date` (naive, freqtrade's raw column convention -- see
`research/promotion.py`'s own naive/aware split for the established pattern) is included
in the window when `close_date >= now_naive - timedelta(days=HEALTH_WINDOW_DAYS)`. No
upper bound beyond `now` is applied, matching `evaluate_paper_trading_health`'s existing
query shape (this was raised as a Low-severity, non-blocking robustness item on that
function during its own final review and stays out of scope here for the same reason).

## Degradation ratio

Identical formula to `evaluate_paper_trading_health`: `max(0.0, min(1.0, live_sharpe /
candidate.oos_sharpe))` when `candidate.oos_sharpe > 0`, else `0.0`. `live_sharpe` is `0`
when the window has zero closed trades (freqtrade's `calculate_sharpe` sentinel,
established in the previous sub-project).

## Testing

Real DB/dataclass construction throughout, no mocking of core logic, following
`research/tests/test_promotion.py`'s established house style (a file-scoped autouse
fixture resetting `Trade.session` after every test, a shared `_insert_...` trade-seeding
helper). Test cases:

1. `evaluate_live_health` raises `ValueError` when the record doesn't exist.
2. `evaluate_live_health` raises `ValueError` when the record isn't in `LIVE` state (e.g.
   still `PAPER_TRADING`).
3. Not enough evidence (window has fewer than `MIN_HEALTH_TRADES` closed trades) ->
   `enough_evidence` is `False`.
4. A trade older than `HEALTH_WINDOW_DAYS` is excluded from the window -- construct one
   trade just inside the boundary and one just outside, assert only the inside one is
   counted.
5. Hand-verified degradation ratio and Sharpe value for a "healthy" fixture (profits and
   an OOS baseline chosen so degradation_ratio clears `HEALTHY_THRESHOLD`), computed by
   hand against freqtrade's real `calculate_sharpe` formula the same way
   `test_promotion.py`'s fixtures were.
6. Hand-verified degradation ratio for a "degraded" fixture (clears `MIN_HEALTH_TRADES`
   but degradation_ratio lands in the `DEGRADED_THRESHOLD` band).
7. `record_health_check` with no prior `HealthCheck` row and a `HEALTHY`-target
   evaluation: new state is `HEALTHY` (matches the implicit starting state, no move
   needed).
8. `record_health_check` with no prior row and a `SUSPENDED`-target evaluation: new state
   is `WATCH`, not `SUSPENDED` -- proves the one-rung-per-evaluation damping from the
   implicit `HEALTHY` start.
9. `record_health_check` called three times in sequence with sustained `SUSPENDED`-target
   evaluations, each call's `evaluated_at` override spaced >= `MIN_HEALTH_CHECK_INTERVAL_
   HOURS` apart: state progresses `HEALTHY -> WATCH -> DEGRADED -> SUSPENDED` one call at
   a time, never skipping a rung.
10. The same three-call sequence as #9, but with `evaluated_at` overrides only MINUTES
    apart (no real new evidence period between calls): state stays `HEALTHY` after all
    three calls -- proves the interval gate, not just the one-rung cap, is what prevents
    a rapid-fire HEALTHY-to-SUSPENDED walk. This is the exact gap the lmchatbot design
    cross-check caught before implementation.
11. `record_health_check` recovering: starting from a `DEGRADED` current state (reached
    via prior calls spaced past the interval), a `HEALTHY`-target evaluation called past
    the interval moves the state to `WATCH`, not directly back to `HEALTHY`.
12. `record_health_check` with `enough_evidence=False`: new state equals the previous
    state exactly, regardless of what `degradation_ratio` happens to be in the returned
    (untrusted) evaluation dict, and regardless of how much time has passed.
13. `record_health_check` raises `ValueError` when the record isn't `LIVE` (checked
    independently of `evaluate_live_health`'s own guard, per this function's own
    docstring contract).
14. `win_rate` and `max_drawdown` are computed correctly against a hand-constructed trade
    sequence with a known win count and a known peak-to-trough drawdown.
15. Full-chain integration test: `create_promotion_record -> start_paper_trading ->
    evaluate_paper_trading_health -> apply_health_evaluation -> promote_to_live` (this
    prefix reuses `research/promotion.py`'s own already-shipped functions, by their real
    names, purely to reach a genuine `LIVE` record -- nothing new here) `->` insert real
    live trades `-> evaluate_live_health -> record_health_check` (these last two are the
    only new functions this sub-project adds), asserting the resulting `HealthCheck.state`
    and that it is queryable as "the latest row for this promotion_id."
16. `evaluate_live_health` on a record whose `candidate.oos_sharpe <= 0`:
    `degradation_ratio` is `0.0` regardless of `live_sharpe` (mirrors
    `test_evaluate_paper_trading_health_non_positive_oos_sharpe_is_zero_degradation`'s
    existing precedent). Note in the test: this candidate/state combination should be
    unreachable via the real pipeline in practice, since reaching `PAPER_TRADING` ->
    `LIVE_ELIGIBLE` already requires `evaluate_paper_trading_health` to have cleared
    `MIN_DEGRADATION_RATIO`, which is impossible when `oos_sharpe <= 0` -- tested anyway
    because the function's own contract shouldn't silently misbehave on a
    theoretically-excluded input, matching this codebase's existing practice of testing
    guarded-but-cheap-to-check edge cases.

## Open items resolved during brainstorming

- **Baseline for win_rate/max_drawdown**: considered persisting a baseline win rate and
  max drawdown on `CandidateResult` (mirroring `oos_sharpe`) to gate on them directly.
  Rejected: `CandidateResult`'s raw per-window `test_returns` are never persisted (only
  summary stats like `oos_sharpe` are), so a real baseline would require a schema change
  to an already-shipped table for two metrics this codebase has no established,
  statistically-grounded acceptance-band methodology for (unlike Sharpe degradation,
  which reuses `evaluate_paper_trading_health`'s already-tested approach). Reporting them
  as informational-only avoids both the schema change and inventing thresholds with no
  grounding behind them.
- **Damping needs a time gate, not just a rung cap** (added after this spec's lmchatbot
  cross-check): the original draft's one-rung-per-evaluation-call rule alone is
  defeatable by calling `evaluate_live_health` + `record_health_check` repeatedly against
  an unchanged rolling window -- three rapid calls would walk `HEALTHY` straight to
  `SUSPENDED` with zero new trading evidence. Fixed by requiring
  `MIN_HEALTH_CHECK_INTERVAL_HOURS` of real elapsed time (measured between the new
  evaluation and the most recent prior `HealthCheck` row) before a rung-move is allowed,
  in addition to the rung cap itself -- see `record_health_check`'s docstring. Considered
  and rejected: tying the gate to "new closed trades since the last check" instead of
  elapsed time -- more precise in principle, but requires tracking which specific trades
  a prior check already counted (a new field, more state to keep consistent), where a
  simple time gate reuses the exact `MIN_PAPER_TRADING_DAYS`-style elapsed-time pattern
  this codebase already established in `promotion.py`.
- **Concurrent/stale evaluations** (also raised during the cross-check): a real gap, ruled
  out of scope for the reason given in "What this is not" above -- this package has no
  concurrent-caller infrastructure anywhere yet, and building real transactional isolation
  belongs with whatever eventually schedules repeated health checks.
- **State storage**: considered a mutable `current_health_state` column on
  `PromotionRecord` instead of deriving it from the latest `HealthCheck` row. Rejected as
  a second source of truth that could drift from the audit trail (an `UPDATE` that fails
  to write, or a manual DB fix, would desync the two) -- deriving "current" from "latest
  row" makes the audit trail authoritative by construction.
- **Transition damping, stricter alternatives**: considered requiring N consecutive
  same-direction evaluations before moving a rung, instead of the rung-cap + time-gate
  combination adopted here. The adopted rule already satisfies "don't overreact to a
  single losing streak" (multiple evaluations spaced real hours apart are required to
  reach SUSPENDED from HEALTHY, each independently re-checking real trade evidence)
  without needing to inspect evaluation history beyond the single latest row -- YAGNI for
  a first version; a stricter rule is a natural, backward-compatible future tightening if
  real usage shows this version still reacts too fast.
