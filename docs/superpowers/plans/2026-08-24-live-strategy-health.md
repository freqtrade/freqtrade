# Live Strategy Health Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `research/health.py`, monitoring a `PromotionRecord` already in `LIVE`
state through a `HEALTHY -> WATCH -> DEGRADED -> SUSPENDED` ladder, driven by real
closed-trade evidence read from the live freqtrade database.

**Architecture:** Two functions mirroring `research/promotion.py`'s own pure-evaluator /
state-mutating-applier split: `evaluate_live_health` reads real `Trade` rows over a
rolling window and computes a verdict dict (no DB write of its own beyond the read);
`record_health_check` writes one audit row per call to a new `HealthCheck` table and
moves the record's current health state by at most one rung, gated on both fresh
evidence (enough trades) and fresh time (real hours elapsed since the last recorded
check).

**Tech Stack:** SQLAlchemy ORM (existing `research/models.py` `Base`), freqtrade's own
`init_db`/`Trade`/`calculate_sharpe`, pytest with real DB/dataclass construction (no
mocking of core logic).

**Spec:** `docs/superpowers/specs/2026-08-24-live-strategy-health-design.md`

## Global Constraints

- `research/health.py` defines `HealthState(StrEnum)` with values `HEALTHY="healthy"`,
  `WATCH="watch"`, `DEGRADED="degraded"`, `SUSPENDED="suspended"`, ordered worst-last.
- `evaluate_live_health(session, promotion_id, starting_balance, live_db_path=None) ->
  dict` is a PURE evaluator (no state mutation, no `HealthCheck` row written). Raises
  `ValueError` if the `PromotionRecord` doesn't exist or isn't currently `LIVE`. Returns
  a dict with keys `enough_evidence`, `n_trades`, `live_sharpe`, `degradation_ratio`,
  `win_rate`, `max_drawdown`, `target_state` (one of `HealthState`'s values, or `None`
  when `enough_evidence` is `False`), `reasons`.
- `record_health_check(session, promotion_id, evaluation, evaluated_at=None) ->
  HealthCheck` is the state-mutating applier. Raises `ValueError` if the record doesn't
  exist or isn't currently `LIVE` (re-checked independently of `evaluate_live_health`'s
  own guard). Writes exactly one `HealthCheck` row per call, always -- even when no rung
  moves.
- Damping is TWO gates, BOTH required before a rung moves: (1) `evaluation
  ["enough_evidence"]` must be `True`; (2) at least `MIN_HEALTH_CHECK_INTERVAL_HOURS`
  real hours must have elapsed since the most recent prior `HealthCheck` row for this
  `promotion_id` (no prior row = gate automatically satisfied). Failing either gate means
  the new state equals the previous state exactly, but the audit row is still written
  with a reason explaining why no move happened. When both gates pass, the state moves
  AT MOST ONE STEP from the previous state toward `evaluation["target_state"]`, in
  either direction.
- The record's CURRENT health state is never a mutable column -- it is always derived as
  "the latest `HealthCheck` row for this `promotion_id`, ordered by `evaluated_at` DESC,
  `id` DESC (deterministic tie-break)". A record with no `HealthCheck` row yet has an
  implicit current state of `HealthState.HEALTHY.value`.
- Module constants (all `ponytail:`-flagged, fixed, not runtime-configurable):
  `HEALTH_WINDOW_DAYS = 30`, `MIN_HEALTH_TRADES = 10`, `HEALTHY_THRESHOLD = 0.7`,
  `WATCH_THRESHOLD = 0.4`, `DEGRADED_THRESHOLD = 0.15`,
  `MIN_HEALTH_CHECK_INTERVAL_HOURS = 24`.
- `degradation_ratio` formula is IDENTICAL to `research/promotion.py`'s
  `evaluate_paper_trading_health`: `max(0.0, min(1.0, live_sharpe /
  candidate.oos_sharpe))` when `candidate.oos_sharpe > 0`, else `0.0`.
- **Connection hygiene is REQUIRED in the first implementation, not a later fix.**
  `evaluate_live_health` must wrap its `Trade.session.query(...).all()` call in
  `try`/`finally`, materializing the query inside `try` and, in `finally`, capturing the
  engine via `cast(Engine, Trade.session.get_bind())`, then `Trade.session.remove()`,
  then `engine.dispose()` -- copy this pattern verbatim from
  `research/promotion.py:196-217` (reproduced below in Task 1). This is a real
  freqtrade-internals fact (verified in the previous sub-project's final review, not
  guessed): `init_db()` always binds `Trade.session` to a real `Engine` via
  `sessionmaker(bind=engine)`, so `get_bind()`'s broader `Engine | Connection` return
  type is never actually a bare `Connection` here.
- `win_rate` and `max_drawdown` are computed and persisted for EVERY evaluation
  (including when `enough_evidence` is `False`) using their real formulas -- never
  `None`, never a sentinel. They do not affect `target_state`.
- No CLI wiring (`research/cli.py` is not touched by this plan). No configurable
  thresholds. No code path stops a live freqtrade bot or writes anything freqtrade
  itself reads.
- Timezone handling matches `research/promotion.py`'s established split exactly: a
  NAIVE-UTC value is used ONLY in the SQL filter against freqtrade's raw (naive)
  `Trade.close_date` column; this package's OWN `DateTime` columns (`HealthCheck
  .evaluated_at`, matching `PromotionRecord.created_at`/`paper_trading_started_at`
  /`resolved_at`) are written as tz-AWARE `datetime.now(UTC)` values and defensively
  re-awared (`.replace(tzinfo=UTC)` if `.tzinfo is None`) on every read back, since
  SQLite can silently drop tzinfo on round-trip.
- No mocking of core logic in tests. Real `Session`/dataclass construction, real
  `Trade` rows inserted into a real SQLite file, following
  `research/tests/test_promotion.py`'s established house style: a file-scoped autouse
  pytest fixture resets `Trade.session` to a fresh in-memory DB after every test in the
  file (protects any later test in the same pytest-xdist worker), and a shared
  `_insert_live_trades` helper releases its own `Trade.session` redirect immediately
  after inserting rows.

---

### Task 1: Data model + the pure evaluator

**Files:**
- Modify: `research/models.py` (add `Index` to the existing `sqlalchemy` import, add the
  `HealthCheck` class after `PromotionRecord`)
- Create: `research/health.py` (`HealthState`, module constants, `evaluate_live_health`)
- Test: `research/tests/test_health.py` (new file)

**Interfaces:**
- Consumes (from `research/promotion.py`, already shipped): `PromotionState` (enum,
  specifically `PromotionState.LIVE.value`), `_load_promotion_record(session,
  promotion_id) -> PromotionRecord` (raises `ValueError` if missing).
- Consumes (from `research/models.py`, already shipped): `CandidateResult`,
  `PromotionRecord` (specifically `.state`, `.candidate_result_id`,
  `.paper_trading_db_path`).
- Produces: `research.models.HealthCheck` (SQLAlchemy model, consumed by Task 2).
  `research.health.HealthState` (enum, consumed by Task 2).
  `research.health.evaluate_live_health(session, promotion_id, starting_balance,
  live_db_path=None) -> dict` (consumed by Task 2's integration test).
  `research.health._STATE_ORDER` (a `list[str]` of `HealthState` values in worst-last
  order, consumed by Task 2's `record_health_check`).

- [ ] **Step 1: Write the failing tests for the model and the evaluator's guard clauses**

Create `research/tests/test_health.py`:

```python
# research/tests/test_health.py
from datetime import UTC, datetime, timedelta

import pytest

from freqtrade.persistence import Trade, init_db
from research.db import get_engine, get_session
from research.health import (
    DEGRADED_THRESHOLD,
    HEALTH_WINDOW_DAYS,
    HEALTHY_THRESHOLD,
    MIN_HEALTH_TRADES,
    WATCH_THRESHOLD,
    HealthState,
    evaluate_live_health,
)
from research.models import CandidateResult
from research.promotion import (
    PromotionState,
    create_promotion_record,
    promote_to_live,
    start_paper_trading,
)


def _session(tmp_path):
    engine = get_engine(str(tmp_path / "research.sqlite"))
    return get_session(engine)


def _candidate(session, oos_sharpe=30.0):
    candidate = CandidateResult(
        run_stamp=datetime.now(UTC),
        strategy_id="TestStrategy",
        strategy_family="TestStrategy",
        params_json="{}",
        universe="BTC/USDT",
        timeframe="1h",
        discovery_start="2024-01-01",
        discovery_end="2024-06-01",
        n_trials_this_run=1,
        is_sharpe=1.0,
        oos_sharpe=oos_sharpe,
        deflated_sharpe=0.97,
        permutation_p=0.01,
        pbo=0.1,
        survived=True,
        evidence_json="{}",
    )
    session.add(candidate)
    session.flush()
    return candidate


def _live_record(session, tmp_path, oos_sharpe=30.0, db_name="live.sqlite"):
    """Reach a genuine LIVE PromotionRecord via research/promotion.py's own real
    functions -- no manual state injection."""
    candidate = _candidate(session, oos_sharpe=oos_sharpe)
    record = create_promotion_record(session, candidate.id)
    start_paper_trading(session, record.id, str(tmp_path / db_name))
    record.state = PromotionState.LIVE_ELIGIBLE.value
    session.flush()
    promote_to_live(session, record.id)
    return record, candidate


@pytest.fixture(autouse=True)
def _reset_trade_session_after_health_tests():
    """evaluate_live_health() calls freqtrade's own init_db(), which sets Trade.session
    as GLOBAL class-level state -- reset it to a fresh in-memory DB after every test in
    this file so no later test (in this file or elsewhere in the same pytest-xdist
    worker) can see Trade.session still pointed at one of this file's throwaway
    databases. See FIELD-NOTES.md."""
    yield
    init_db("sqlite://")


def _insert_live_trades(live_db_path, strategy_id, started_at, profits_abs):
    """Directly construct and insert closed Trade rows into a fresh live-style sqlite
    database -- one trade per entry in profits_abs, spaced evenly across the period from
    started_at to now. Mirrors research/tests/test_promotion.py's
    _insert_dry_run_trades helper exactly."""
    init_db(f"sqlite:///{live_db_path}")
    now = datetime.now(UTC).replace(tzinfo=None)
    started_naive = started_at.replace(tzinfo=None) if started_at.tzinfo else started_at
    span = now - started_naive
    step = span / max(1, len(profits_abs))
    for i, profit in enumerate(profits_abs):
        open_dt = started_naive + step * i
        close_dt = open_dt + timedelta(minutes=30)
        trade = Trade(
            pair="BTC/USDT",
            strategy=strategy_id,
            exchange="binance",
            is_open=False,
            open_date=open_dt,
            close_date=close_dt,
            close_profit_abs=float(profit),
            stake_amount=100.0,
            amount=1.0,
            open_rate=100.0,
            close_rate=100.0 + float(profit) / 100.0,
            fee_open=0.001,
            fee_close=0.001,
        )
        Trade.session.add(trade)
    Trade.session.flush()
    Trade.session.commit()
    init_db("sqlite://")  # release this function's own Trade.session redirect immediately


def test_evaluate_live_health_raises_for_missing_record(tmp_path):
    session = _session(tmp_path)

    with pytest.raises(ValueError, match="No PromotionRecord"):
        evaluate_live_health(session, 999, starting_balance=1000.0)


def test_evaluate_live_health_raises_when_not_live(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)
    start_paper_trading(session, record.id, str(tmp_path / "a.sqlite"))

    with pytest.raises(ValueError, match="cannot evaluate live health"):
        evaluate_live_health(session, record.id, starting_balance=1000.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd C:\dev\freqtrade && python -m pytest research/tests/test_health.py -v`
Expected: collection error / `ImportError` -- `research.health` doesn't exist yet.

- [ ] **Step 3: Add the `HealthCheck` model to `research/models.py`**

In `research/models.py`, change the import line:

```python
from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String
```

Then add, after the `PromotionRecord` class:

```python
class HealthCheck(Base):
    """One row per live-health evaluation of a PromotionRecord already in LIVE state.
    The record's CURRENT health state is simply the latest row for its
    promotion_record_id, ordered by evaluated_at -- deliberately not a mutable field on
    PromotionRecord, to avoid a second source of truth that could drift out of sync."""

    __tablename__ = "health_checks"
    __table_args__ = (
        Index(
            "ix_health_checks_promotion_id_evaluated_at",
            "promotion_record_id",
            "evaluated_at",
            "id",
        ),
    )

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

- [ ] **Step 4: Create `research/health.py` with the guard clauses only**

```python
# research/health.py
"""Live strategy health monitor: repeatedly evaluates a PromotionRecord already in LIVE
state against its own real trading history, classifying it into a HEALTHY -> WATCH ->
DEGRADED -> SUSPENDED ladder. Never stops a live bot -- SUSPENDED is a recorded
recommendation for a human, exactly as LIVE itself is reachable only by a manual human
call in research/promotion.py. See
docs/superpowers/specs/2026-08-24-live-strategy-health-design.md for the full design.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import cast

import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from freqtrade.data.metrics import calculate_sharpe
from freqtrade.persistence import Trade, init_db
from research.models import CandidateResult, HealthCheck
from research.promotion import PromotionState, _load_promotion_record


class HealthState(StrEnum):
    HEALTHY = "healthy"
    WATCH = "watch"
    DEGRADED = "degraded"
    SUSPENDED = "suspended"


_STATE_ORDER = [
    HealthState.HEALTHY.value,
    HealthState.WATCH.value,
    HealthState.DEGRADED.value,
    HealthState.SUSPENDED.value,
]

# ponytail: starting defaults, not derived from any real live-trading history (none
# exists yet in this fork) -- adjust based on real usage once this runs against real
# strategies. Deliberately mirrors promotion.py's MIN_PAPER_TRADING_DAYS/MIN_PAPER_TRADES
# shape, applied to a rolling window instead of a one-time cumulative check.
HEALTH_WINDOW_DAYS = 30
MIN_HEALTH_TRADES = 10
HEALTHY_THRESHOLD = 0.7
WATCH_THRESHOLD = 0.4
DEGRADED_THRESHOLD = 0.15
MIN_HEALTH_CHECK_INTERVAL_HOURS = 24


def evaluate_live_health(
    session: Session,
    promotion_id: int,
    starting_balance: float,
    live_db_path: str | None = None,
) -> dict:
    """Pure evaluation (no state mutation, no HealthCheck row written) of a LIVE
    PromotionRecord's real trade history over the trailing HEALTH_WINDOW_DAYS. Returns a
    verdict dict; call record_health_check with the result to persist an audit row and
    (possibly) move the record's current health state.

    `live_db_path` defaults to the record's own stored paper_trading_db_path -- the same
    field promotion.py uses to remember where a record's trade history lives (its name
    predates LIVE and is not renamed here; it holds a LIVE record's real trading
    database just as well as a PAPER_TRADING one's dry-run database).

    starting_balance is required, matching evaluate_paper_trading_health's own
    established contract -- there is no other way for this function to discover the
    live bot's configured wallet size.

    IMPORTANT: freqtrade.persistence.init_db() sets Trade.session as GLOBAL class-level
    state, not a scoped per-call connection. This function fully materializes its query
    results before returning and disposes the engine/session in a finally block before
    returning -- see FIELD-NOTES.md and research/promotion.py's own equivalent pattern.

    Rolling window, not cumulative: uses only closed trades with
    close_date >= now - HEALTH_WINDOW_DAYS, not the strategy's full LIVE-to-date
    history. Win rate and max drawdown are computed from the same window and returned
    for human context -- they do not affect target_state.
    """
    record = _load_promotion_record(session, promotion_id)
    if record.state != PromotionState.LIVE.value:
        raise ValueError(
            f"PromotionRecord {promotion_id} is in state {record.state!r}, not "
            f"{PromotionState.LIVE.value!r} -- cannot evaluate live health."
        )
    candidate = session.get(CandidateResult, record.candidate_result_id)
    if candidate is None:
        raise ValueError(f"No CandidateResult with id {record.candidate_result_id}")

    db_path = live_db_path or record.paper_trading_db_path
    if not db_path:
        raise ValueError(
            f"PromotionRecord {promotion_id} has no db path to evaluate -- pass "
            "live_db_path explicitly or ensure the record's stored path is set."
        )

    raise NotImplementedError  # placeholder -- filled in by Step 6
```

- [ ] **Step 5: Run the guard-clause tests to verify they pass**

Run: `cd C:\dev\freqtrade && python -m pytest research/tests/test_health.py -v`
Expected: `test_evaluate_live_health_raises_for_missing_record` and
`test_evaluate_live_health_raises_when_not_live` PASS. Any test exercising the real
query path would hit `NotImplementedError` -- none exist yet, so this is fine.

- [ ] **Step 6: Write the failing tests for the real evaluation logic**

Append to `research/tests/test_health.py`:

```python
def test_evaluate_live_health_not_enough_evidence(tmp_path):
    session = _session(tmp_path)
    record, candidate = _live_record(session, tmp_path)
    _insert_live_trades(
        tmp_path / "live.sqlite",
        candidate.strategy_id,
        datetime.now(UTC) - timedelta(days=HEALTH_WINDOW_DAYS),
        [8, 6, 7],
    )  # only 3 trades, MIN_HEALTH_TRADES is 10

    evaluation = evaluate_live_health(session, record.id, starting_balance=1000.0)

    assert evaluation["enough_evidence"] is False
    assert evaluation["n_trades"] == 3
    assert evaluation["target_state"] is None
    assert evaluation["win_rate"] == pytest.approx(1.0)  # all 3 profits are positive


def test_evaluate_live_health_excludes_trades_outside_the_window(tmp_path):
    session = _session(tmp_path)
    record, candidate = _live_record(session, tmp_path)
    init_db(f"sqlite:///{tmp_path / 'live.sqlite'}")
    now_naive = datetime.now(UTC).replace(tzinfo=None)
    inside = now_naive - timedelta(days=HEALTH_WINDOW_DAYS - 1)
    outside = now_naive - timedelta(days=HEALTH_WINDOW_DAYS + 1)
    for close_dt in (inside, outside):
        Trade.session.add(
            Trade(
                pair="BTC/USDT",
                strategy=candidate.strategy_id,
                exchange="binance",
                is_open=False,
                open_date=close_dt - timedelta(minutes=30),
                close_date=close_dt,
                close_profit_abs=10.0,
                stake_amount=100.0,
                amount=1.0,
                open_rate=100.0,
                close_rate=100.1,
                fee_open=0.001,
                fee_close=0.001,
            )
        )
    Trade.session.flush()
    Trade.session.commit()
    init_db("sqlite://")

    evaluation = evaluate_live_health(session, record.id, starting_balance=1000.0)

    assert evaluation["n_trades"] == 1


def test_evaluate_live_health_healthy_fixture_hand_verified_sharpe(tmp_path):
    session = _session(tmp_path)
    # oos_sharpe=30.0 chosen so degradation_ratio clips to exactly 1.0 (live_sharpe
    # 31.521597534248023 > oos_sharpe 30.0) -- matches HEALTHY_THRESHOLD=0.7 comfortably.
    record, candidate = _live_record(session, tmp_path, oos_sharpe=30.0)
    started_at = datetime.now(UTC) - timedelta(days=HEALTH_WINDOW_DAYS)
    _insert_live_trades(
        tmp_path / "live.sqlite",
        candidate.strategy_id,
        started_at,
        [8, 6, 7, 9, 5, 8, 6, 7, 9, 5],
    )

    evaluation = evaluate_live_health(session, record.id, starting_balance=1000.0)

    assert evaluation["enough_evidence"] is True
    assert evaluation["n_trades"] == 10
    # Hand-derived from freqtrade's real calculate_sharpe formula: total_return=0.07,
    # days_period=30 (HEALTH_WINDOW_DAYS), mean_daily_return=0.07/30, population stdev
    # of the per-trade normalized returns=0.0014142135623730951 (same fixture already
    # hand-verified in test_promotion.py over a 14-day window as 67.54628043053148;
    # Sharpe scales linearly with 1/days_period, so the 30-day value is
    # 67.54628043053148 * 14 / 30).
    assert evaluation["live_sharpe"] == pytest.approx(31.521597534248023)
    assert evaluation["degradation_ratio"] == pytest.approx(1.0)
    assert evaluation["target_state"] == HealthState.HEALTHY.value


def test_evaluate_live_health_degraded_fixture(tmp_path):
    session = _session(tmp_path)
    # Same trades/live_sharpe as the healthy fixture (31.521597534248023), but a much
    # larger oos_sharpe so degradation_ratio = 31.521597534248023 / 100.0 ~= 0.315,
    # landing inside [DEGRADED_THRESHOLD, WATCH_THRESHOLD) = [0.15, 0.4).
    record, candidate = _live_record(session, tmp_path, oos_sharpe=100.0)
    started_at = datetime.now(UTC) - timedelta(days=HEALTH_WINDOW_DAYS)
    _insert_live_trades(
        tmp_path / "live.sqlite",
        candidate.strategy_id,
        started_at,
        [8, 6, 7, 9, 5, 8, 6, 7, 9, 5],
    )

    evaluation = evaluate_live_health(session, record.id, starting_balance=1000.0)

    assert evaluation["degradation_ratio"] == pytest.approx(0.31521597534248023)
    assert evaluation["target_state"] == HealthState.DEGRADED.value


def test_evaluate_live_health_non_positive_oos_sharpe_is_zero_degradation(tmp_path):
    session = _session(tmp_path)
    record, candidate = _live_record(session, tmp_path, oos_sharpe=-0.5)
    started_at = datetime.now(UTC) - timedelta(days=HEALTH_WINDOW_DAYS)
    _insert_live_trades(
        tmp_path / "live.sqlite",
        candidate.strategy_id,
        started_at,
        [8, 6, 7, 9, 5, 8, 6, 7, 9, 5],
    )

    evaluation = evaluate_live_health(session, record.id, starting_balance=1000.0)

    assert evaluation["degradation_ratio"] == 0.0
    assert evaluation["target_state"] == HealthState.SUSPENDED.value
```

Note: `record, candidate = _live_record(...)` reaches `LIVE` in these tests via a
candidate whose `oos_sharpe` is only ever used by `evaluate_live_health` (not by
`start_paper_trading`/`promote_to_live`, which don't inspect it) -- `_live_record`'s
sequence is safe to use with any `oos_sharpe` value, including the non-positive one in
the last test above, even though that combination could never occur via the real
end-to-end pipeline (see the plan's Task 2 integration test, and the spec's own note on
this).

- [ ] **Step 7: Run the tests to verify they fail for the right reason**

Run: `cd C:\dev\freqtrade && python -m pytest research/tests/test_health.py -v`
Expected: the five new tests FAIL with `NotImplementedError` (from Step 4's
placeholder). The two guard-clause tests from Step 1 still PASS.

- [ ] **Step 8: Implement the real evaluation logic**

In `research/health.py`, replace the `raise NotImplementedError` line with:

```python
    now = datetime.now(UTC)
    window_start_aware = now - timedelta(days=HEALTH_WINDOW_DAYS)
    window_start_naive = window_start_aware.replace(tzinfo=None)

    init_db(f"sqlite:///{db_path}")
    try:
        closed_trades = (
            Trade.session.query(Trade)
            .filter(
                Trade.strategy == candidate.strategy_id,
                Trade.is_open.is_(False),
                Trade.close_date >= window_start_naive,
            )
            .all()
        )
    finally:
        # Connection hygiene: the live database file belongs to a currently-running
        # freqtrade bot process writing to it concurrently -- never leave a lingering
        # handle on it. Release the scoped session and dispose its engine before
        # returning, regardless of whether the query above succeeded. init_db() always
        # binds Trade.session via sessionmaker(bind=engine) with a real Engine (see
        # freqtrade/persistence/models.py's init_db) -- get_bind()'s broader
        # Engine | Connection return type is never a Connection here.
        engine = cast(Engine, Trade.session.get_bind())
        Trade.session.remove()
        engine.dispose()

    n_trades = len(closed_trades)

    if n_trades > 0:
        trades_df = pd.DataFrame({"profit_abs": [t.close_profit_abs for t in closed_trades]})
        live_sharpe = calculate_sharpe(trades_df, window_start_aware, now, starting_balance)
        win_rate = sum(1 for t in closed_trades if t.close_profit_abs > 0) / n_trades
        sorted_trades = sorted(closed_trades, key=lambda t: t.close_date)
        equity = starting_balance
        peak = starting_balance
        max_drawdown = 0.0
        for t in sorted_trades:
            equity += t.close_profit_abs
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)
    else:
        live_sharpe = 0
        win_rate = 0.0
        max_drawdown = 0.0

    if candidate.oos_sharpe > 0:
        degradation_ratio = max(0.0, min(1.0, live_sharpe / candidate.oos_sharpe))
    else:
        degradation_ratio = 0.0

    reasons: list[str] = []
    enough_evidence = n_trades >= MIN_HEALTH_TRADES
    if not enough_evidence:
        reasons.append(
            f"only {n_trades} trades in the last {HEALTH_WINDOW_DAYS} days, need >= "
            f"{MIN_HEALTH_TRADES}"
        )
        target_state = None
    elif degradation_ratio >= HEALTHY_THRESHOLD:
        target_state = HealthState.HEALTHY.value
    elif degradation_ratio >= WATCH_THRESHOLD:
        target_state = HealthState.WATCH.value
        reasons.append(
            f"degradation_ratio {degradation_ratio:.3f} below healthy threshold "
            f"{HEALTHY_THRESHOLD}"
        )
    elif degradation_ratio >= DEGRADED_THRESHOLD:
        target_state = HealthState.DEGRADED.value
        reasons.append(
            f"degradation_ratio {degradation_ratio:.3f} below watch threshold "
            f"{WATCH_THRESHOLD}"
        )
    else:
        target_state = HealthState.SUSPENDED.value
        reasons.append(
            f"degradation_ratio {degradation_ratio:.3f} below degraded threshold "
            f"{DEGRADED_THRESHOLD}"
        )

    return {
        "enough_evidence": enough_evidence,
        "n_trades": n_trades,
        "live_sharpe": live_sharpe,
        "degradation_ratio": degradation_ratio,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "target_state": target_state,
        "reasons": reasons,
    }
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `cd C:\dev\freqtrade && python -m pytest research/tests/test_health.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 10: Run the full research suite to confirm no breakage**

Run: `cd C:\dev\freqtrade && python -m pytest research/ -q`
Expected: 87 passed (80 existing + 7 new from this task; the remaining 9 of the spec's
16-test list arrive in Task 2).

- [ ] **Step 11: Lint and commit**

```bash
cd C:/dev/freqtrade
ruff check research/models.py research/health.py research/tests/test_health.py
ruff format research/models.py research/health.py research/tests/test_health.py
git add research/models.py research/health.py research/tests/test_health.py
git commit -m "feat(research): add HealthCheck model and evaluate_live_health() -- pure evaluator reading real LIVE trade history"
```

---

### Task 2: The state-mutating applier + full-chain integration

**Files:**
- Modify: `research/health.py` (add `record_health_check`)
- Test: `research/tests/test_health.py` (append)

**Interfaces:**
- Consumes: `research.health.HealthState`, `research.health._STATE_ORDER`,
  `research.health.MIN_HEALTH_CHECK_INTERVAL_HOURS`, `research.health
  .evaluate_live_health` (all from Task 1). `research.models.HealthCheck` (from Task 1).
  `research.promotion.PromotionState`, `research.promotion._load_promotion_record`,
  `research.promotion.create_promotion_record`, `.start_paper_trading`,
  `.evaluate_paper_trading_health`, `.apply_health_evaluation`, `.promote_to_live`
  (already shipped, reused by the integration test to reach a genuine `LIVE` record).
- Produces: `research.health.record_health_check(session, promotion_id, evaluation,
  evaluated_at=None) -> HealthCheck`.

- [ ] **Step 1: Write the failing tests**

Append to `research/tests/test_health.py`:

```python
import json

from research.health import (
    MIN_HEALTH_CHECK_INTERVAL_HOURS,
    record_health_check,
)
from research.promotion import apply_health_evaluation, evaluate_paper_trading_health


def _suspended_target_evaluation(n_trades=10):
    """A canned evaluation dict with target_state=SUSPENDED and enough_evidence=True --
    used by the damping tests below, which test record_health_check's own transition
    logic directly rather than re-deriving a real SUSPENDED-target evaluation from
    scratch each time."""
    return {
        "enough_evidence": True,
        "n_trades": n_trades,
        "live_sharpe": -5.0,
        "degradation_ratio": 0.05,
        "win_rate": 0.2,
        "max_drawdown": 0.5,
        "target_state": HealthState.SUSPENDED.value,
        "reasons": ["degradation_ratio 0.050 below degraded threshold 0.15"],
    }


def _healthy_target_evaluation(n_trades=10):
    return {
        "enough_evidence": True,
        "n_trades": n_trades,
        "live_sharpe": 40.0,
        "degradation_ratio": 1.0,
        "win_rate": 0.8,
        "max_drawdown": 0.05,
        "target_state": HealthState.HEALTHY.value,
        "reasons": [],
    }


def _not_enough_evidence_evaluation():
    return {
        "enough_evidence": False,
        "n_trades": 2,
        "live_sharpe": -999.0,  # deliberately alarming -- must be ignored
        "degradation_ratio": 0.0,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
        "target_state": None,
        "reasons": ["only 2 trades in the last 30 days, need >= 10"],
    }


def test_record_health_check_no_prior_row_healthy_target_stays_healthy(tmp_path):
    session = _session(tmp_path)
    record, _candidate = _live_record(session, tmp_path)

    check = record_health_check(session, record.id, _healthy_target_evaluation())

    assert check.state == HealthState.HEALTHY.value
    assert check.promotion_record_id == record.id


def test_record_health_check_no_prior_row_suspended_target_moves_one_rung(tmp_path):
    session = _session(tmp_path)
    record, _candidate = _live_record(session, tmp_path)

    check = record_health_check(session, record.id, _suspended_target_evaluation())

    assert check.state == HealthState.WATCH.value  # not SUSPENDED -- one rung from HEALTHY


def test_record_health_check_sustained_degradation_spaced_apart_reaches_suspended(tmp_path):
    session = _session(tmp_path)
    record, _candidate = _live_record(session, tmp_path)
    t0 = datetime.now(UTC)
    interval = timedelta(hours=MIN_HEALTH_CHECK_INTERVAL_HOURS + 1)

    c1 = record_health_check(session, record.id, _suspended_target_evaluation(), t0)
    c2 = record_health_check(session, record.id, _suspended_target_evaluation(), t0 + interval)
    c3 = record_health_check(
        session, record.id, _suspended_target_evaluation(), t0 + interval * 2
    )

    assert c1.state == HealthState.WATCH.value
    assert c2.state == HealthState.DEGRADED.value
    assert c3.state == HealthState.SUSPENDED.value


def test_record_health_check_rapid_calls_do_not_move_state(tmp_path):
    session = _session(tmp_path)
    record, _candidate = _live_record(session, tmp_path)
    t0 = datetime.now(UTC)
    minutes_apart = timedelta(minutes=5)

    c1 = record_health_check(session, record.id, _suspended_target_evaluation(), t0)
    c2 = record_health_check(
        session, record.id, _suspended_target_evaluation(), t0 + minutes_apart
    )
    c3 = record_health_check(
        session, record.id, _suspended_target_evaluation(), t0 + minutes_apart * 2
    )

    assert c1.state == HealthState.WATCH.value  # first call always allowed (no prior row)
    assert c2.state == HealthState.WATCH.value  # interval gate blocks the second move
    assert c3.state == HealthState.WATCH.value  # and the third
    assert any("since the last recorded check" in r for r in json.loads(c2.reasons_json))


def test_record_health_check_recovery_moves_one_rung_at_a_time(tmp_path):
    session = _session(tmp_path)
    record, _candidate = _live_record(session, tmp_path)
    t0 = datetime.now(UTC)
    interval = timedelta(hours=MIN_HEALTH_CHECK_INTERVAL_HOURS + 1)
    record_health_check(session, record.id, _suspended_target_evaluation(), t0)
    record_health_check(session, record.id, _suspended_target_evaluation(), t0 + interval)
    degraded_check = record_health_check(
        session, record.id, _suspended_target_evaluation(), t0 + interval * 2
    )
    assert degraded_check.state == HealthState.DEGRADED.value

    recovering = record_health_check(
        session, record.id, _healthy_target_evaluation(), t0 + interval * 3
    )

    assert recovering.state == HealthState.WATCH.value  # not straight back to HEALTHY


def test_record_health_check_not_enough_evidence_never_moves_state(tmp_path):
    session = _session(tmp_path)
    record, _candidate = _live_record(session, tmp_path)
    t0 = datetime.now(UTC)
    interval = timedelta(hours=MIN_HEALTH_CHECK_INTERVAL_HOURS + 1)
    record_health_check(session, record.id, _suspended_target_evaluation(), t0)  # -> WATCH

    check = record_health_check(
        session, record.id, _not_enough_evidence_evaluation(), t0 + interval
    )

    assert check.state == HealthState.WATCH.value  # unchanged despite the alarming dict


def test_record_health_check_raises_when_not_live(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)

    with pytest.raises(ValueError, match="cannot record a health check"):
        record_health_check(session, record.id, _healthy_target_evaluation())


def test_win_rate_and_max_drawdown_hand_verified(tmp_path):
    session = _session(tmp_path)
    record, candidate = _live_record(session, tmp_path)
    started_at = datetime.now(UTC) - timedelta(days=2)
    _insert_live_trades(
        tmp_path / "live.sqlite", candidate.strategy_id, started_at, [100, -50, 80, -120, 60]
    )

    evaluation = evaluate_live_health(session, record.id, starting_balance=1000.0)

    # win_rate: 3 of 5 trades positive (100, 80, 60) -> 0.6
    assert evaluation["win_rate"] == pytest.approx(0.6)
    # max_drawdown: equity 1000 -> 1100 -> 1050 -> 1130 -> 1010 -> 1070 (peak 1130 after
    # trade 3, low point 1010 after trade 4) -> (1130-1010)/1130
    assert evaluation["max_drawdown"] == pytest.approx(0.10619469026548672)


def test_full_promotion_and_health_chain_reaches_suspended(tmp_path):
    """Reuses research/promotion.py's own real functions to reach a genuine LIVE
    record (no new functions involved in this prefix), then exercises this task's two
    new functions against real inserted live trades."""
    session = _session(tmp_path)
    candidate = _candidate(session, oos_sharpe=2.0)
    record = create_promotion_record(session, candidate.id)
    paper_started = datetime.now(UTC) - timedelta(days=14)
    start_paper_trading(session, record.id, str(tmp_path / "paper.sqlite"), paper_started)
    _insert_live_trades(
        tmp_path / "paper.sqlite",
        candidate.strategy_id,
        paper_started,
        [8, 6, 7, 9, 5, 8, 6, 7, 9, 5],
    )
    paper_eval = evaluate_paper_trading_health(session, record.id, starting_balance=1000.0)
    apply_health_evaluation(session, record.id, paper_eval)
    assert record.state == PromotionState.LIVE_ELIGIBLE.value
    promote_to_live(session, record.id)
    assert record.state == PromotionState.LIVE.value

    live_started = datetime.now(UTC) - timedelta(days=HEALTH_WINDOW_DAYS)
    _insert_live_trades(
        tmp_path / "live.sqlite",
        candidate.strategy_id,
        live_started,
        [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3],  # sustained losses, well below oos_sharpe
    )
    t0 = datetime.now(UTC)
    interval = timedelta(hours=MIN_HEALTH_CHECK_INTERVAL_HOURS + 1)
    c1 = record_health_check(
        session,
        record.id,
        evaluate_live_health(session, record.id, starting_balance=1000.0, live_db_path=str(tmp_path / "live.sqlite")),
        t0,
    )
    c2 = record_health_check(
        session,
        record.id,
        evaluate_live_health(session, record.id, starting_balance=1000.0, live_db_path=str(tmp_path / "live.sqlite")),
        t0 + interval,
    )
    c3 = record_health_check(
        session,
        record.id,
        evaluate_live_health(session, record.id, starting_balance=1000.0, live_db_path=str(tmp_path / "live.sqlite")),
        t0 + interval * 2,
    )

    assert [c1.state, c2.state, c3.state] == [
        HealthState.WATCH.value,
        HealthState.DEGRADED.value,
        HealthState.SUSPENDED.value,
    ]
    # "current state" is queryable as the latest row for this promotion_id
    latest = (
        session.query(HealthCheck)
        .filter(HealthCheck.promotion_record_id == record.id)
        .order_by(HealthCheck.evaluated_at.desc(), HealthCheck.id.desc())
        .first()
    )
    assert latest.id == c3.id
```

Add `from research.models import HealthCheck` to the test file's existing
`from research.models import CandidateResult` import line (change it to `from
research.models import CandidateResult, HealthCheck`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd C:\dev\freqtrade && python -m pytest research/tests/test_health.py -v`
Expected: the 9 new tests FAIL with `ImportError: cannot import name 'record_health_check'`.

- [ ] **Step 3: Implement `record_health_check`**

Append to `research/health.py`:

```python
def record_health_check(
    session: Session,
    promotion_id: int,
    evaluation: dict,
    evaluated_at: datetime | None = None,
) -> HealthCheck:
    """Apply an evaluate_live_health() result: write one HealthCheck audit row, computing
    the record's new current state from its previous current state (the latest existing
    HealthCheck row for this promotion_id, or implicit HEALTHY if none exists yet).

    A rung-move requires BOTH evaluation["enough_evidence"] is True AND at least
    MIN_HEALTH_CHECK_INTERVAL_HOURS real hours elapsed since the most recent prior
    HealthCheck row (no prior row satisfies this automatically). Failing either gate
    means the new state equals the previous state exactly -- an audit row is still
    written either way, with a reason explaining why no move happened.

    Raises ValueError if the record doesn't exist or isn't currently LIVE (re-checked
    directly, independent of evaluate_live_health's own guard, since evaluation may be
    stale by the time this is called).
    """
    record = _load_promotion_record(session, promotion_id)
    if record.state != PromotionState.LIVE.value:
        raise ValueError(
            f"PromotionRecord {promotion_id} is in state {record.state!r}, not "
            f"{PromotionState.LIVE.value!r} -- cannot record a health check."
        )

    check_time = evaluated_at or datetime.now(UTC)
    if check_time.tzinfo is None:
        check_time = check_time.replace(tzinfo=UTC)

    prior = (
        session.query(HealthCheck)
        .filter(HealthCheck.promotion_record_id == promotion_id)
        .order_by(HealthCheck.evaluated_at.desc(), HealthCheck.id.desc())
        .first()
    )
    previous_state = prior.state if prior is not None else HealthState.HEALTHY.value

    reasons = list(evaluation["reasons"])
    if not evaluation["enough_evidence"]:
        new_state = previous_state
    else:
        interval_ok = True
        if prior is not None:
            prior_at = prior.evaluated_at
            if prior_at.tzinfo is None:
                prior_at = prior_at.replace(tzinfo=UTC)
            elapsed_hours = (check_time - prior_at).total_seconds() / 3600.0
            if elapsed_hours < MIN_HEALTH_CHECK_INTERVAL_HOURS:
                interval_ok = False
                reasons.append(
                    f"only {elapsed_hours:.1f}h since the last recorded check, need >= "
                    f"{MIN_HEALTH_CHECK_INTERVAL_HOURS}h before a state move"
                )
        if not interval_ok:
            new_state = previous_state
        else:
            cur_idx = _STATE_ORDER.index(previous_state)
            tgt_idx = _STATE_ORDER.index(evaluation["target_state"])
            if tgt_idx > cur_idx:
                new_idx = cur_idx + 1
            elif tgt_idx < cur_idx:
                new_idx = cur_idx - 1
            else:
                new_idx = cur_idx
            new_state = _STATE_ORDER[new_idx]

    check = HealthCheck(
        promotion_record_id=promotion_id,
        evaluated_at=check_time,
        state=new_state,
        enough_evidence=evaluation["enough_evidence"],
        n_trades=evaluation["n_trades"],
        live_sharpe=evaluation["live_sharpe"],
        degradation_ratio=evaluation["degradation_ratio"],
        win_rate=evaluation["win_rate"],
        max_drawdown=evaluation["max_drawdown"],
        reasons_json=json.dumps(reasons),
    )
    session.add(check)
    session.flush()
    return check
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd C:\dev\freqtrade && python -m pytest research/tests/test_health.py -v`
Expected: all 16 tests in the file PASS (7 from Task 1 + 9 from this task).

- [ ] **Step 5: Run the full research suite**

Run: `cd C:\dev\freqtrade && python -m pytest research/ -q`
Expected: all tests pass (96 = 80 existing + 16 new).

- [ ] **Step 6: Lint and commit**

```bash
cd C:/dev/freqtrade
ruff check research/health.py research/tests/test_health.py
ruff format research/health.py research/tests/test_health.py
git add research/health.py research/tests/test_health.py
git commit -m "feat(research): add record_health_check() -- damped state transitions for live strategy monitoring"
```
