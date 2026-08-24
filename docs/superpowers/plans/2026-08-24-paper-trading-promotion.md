# Paper-Trading Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Track a gate-passing candidate's promotion lifecycle (PASSED_GATE →
PAPER_TRADING → LIVE_ELIGIBLE → LIVE, REJECTED reachable from either post-gate stage)
with guarded state transitions, and a health evaluator that reads a real freqtrade
dry-run database to recommend (never automatically grant) live eligibility.

**Architecture:** One new table (`PromotionRecord`, in `research/models.py`) and one new
module (`research/promotion.py`) with two halves: a pure state machine (no freqtrade
dependency) and a health evaluator (the first `research/` code to read freqtrade's own
`Trade` persistence layer, guarded against a real global-state trap this session already
found once).

**Tech Stack:** Python, SQLAlchemy (research's own ledger DB), freqtrade's
`persistence.init_db`/`Trade` and `data.metrics.calculate_sharpe`, pandas, pytest with
real DB rows throughout (no mocking).

**Spec:** `docs/superpowers/specs/2026-08-24-paper-trading-promotion-design.md`

## Global Constraints

- `PromotionState` is a `class PromotionState(str, Enum)` with values `PASSED_GATE`,
  `PAPER_TRADING`, `LIVE_ELIGIBLE`, `LIVE`, `REJECTED` — stored on `PromotionRecord.state`
  as the enum's plain `.value` string, never a SQLAlchemy `Enum` column type.
- Every state-transition function raises `ValueError` (never a silent no-op) on: a
  missing referenced row, or an invalid current-state-to-target-state transition. Exact
  error message content is not load-bearing — tests match on a distinctive substring,
  not the full string — but every function must raise, not return `None` or an unchanged
  record, on a caller-contract violation.
- `MIN_PAPER_TRADING_DAYS = 14`, `MIN_PAPER_TRADES = 10`, `MIN_DEGRADATION_RATIO = 0.5`
  are fixed module-level constants in `research/promotion.py`, `ponytail:`-flagged
  starting defaults — not a runtime parameter, not derived from real paper-trading
  history (none exists yet in this fork).
- `LIVE` is reachable **only** via a direct `promote_to_live()` call. No other function
  in this module — especially not `apply_health_evaluation` — may ever set
  `state = PromotionState.LIVE.value`. This is the one non-negotiable constraint the
  whole sub-project exists to enforce; a task reviewer must treat any code path that
  violates it as Critical, full stop, no exceptions for "convenience."
- Eligibility from `evaluate_paper_trading_health` has **three** outcomes, not two:
  not-enough-evidence (`eligible=False, enough_evidence=False`), rejected
  (`eligible=False, enough_evidence=True`), eligible (`eligible=True,
  enough_evidence=True`). `apply_health_evaluation` maps these to: stay in
  `PAPER_TRADING` / transition to `REJECTED` / transition to `LIVE_ELIGIBLE`,
  respectively. Collapsing this to a single boolean is a plan-mandated defect if
  proposed during implementation or review — reject the simplification, keep the
  three-way split (see the spec's "Open items" section for why).
- `evaluate_paper_trading_health` requires `starting_balance: float` as an explicit,
  non-defaulted parameter (no config file is read to discover it — see the spec).
- **`freqtrade.persistence.init_db()` sets `Trade.session` as GLOBAL class-level state,
  not a scoped per-call connection** — the same class of bug this session already found
  and fixed once (`Trade.use_db`/`bt_trades` leaking across tests, PR #5). Task 2's
  production code must fully materialize (`.all()`) its query results before returning,
  and Task 2's test file must include a file-scoped autouse fixture that resets
  `Trade.session` to a fresh in-memory database after every test in that file, so this
  file's throwaway dry-run databases can never leak into any other test that happens to
  run afterward in the same pytest-xdist worker. This is not optional test hygiene —
  it is the direct, foreseeable recurrence of an already-fixed bug class if skipped.
- Timezone handling: `paper_trading_started_at` is written as `datetime.now(UTC)`
  (tz-aware) but SQLite round-tripping through SQLAlchemy's plain `DateTime` may drop
  tzinfo — `evaluate_paper_trading_health` normalizes defensively on read (adds `UTC` if
  naive) rather than assuming either behavior. The SQL filter against freqtrade's raw
  `Trade.close_date` column uses a **naive**-UTC value (freqtrade's own raw columns are
  naive); all Python-side arithmetic uses the tz-aware value. Both are derived from one
  single normalized read, not independently re-normalized in two places.
- One normalized `now = datetime.now(UTC)` captured once per `evaluate_paper_trading_health`
  call, reused for both `days_elapsed` arithmetic and (implicitly, as `calculate_sharpe`'s
  `max_date`) the Sharpe computation — never call `datetime.now(UTC)` a second time
  later in the same function body.
- No mocking of the state machine or the health evaluator's own logic in tests — real
  SQLAlchemy rows (research's own ledger DB) and real freqtrade `Trade` rows (a real,
  freshly-`init_db`'d sqlite file), matching every existing file in `research/tests/`.

---

### Task 1: `research/promotion.py` — state machine (no freqtrade Trade dependency)

**Files:**
- Modify: `research/models.py` (add `PromotionRecord`)
- Create: `research/promotion.py` (`PromotionState` enum, `create_promotion_record`,
  `start_paper_trading`, `promote_to_live`, `reject`)
- Test: `research/tests/test_promotion.py`

**Interfaces:**
- Consumes: `research.models.CandidateResult` (existing dataclass — see
  `research/models.py:11-37`), `research.db.get_engine`/`get_session` (existing, used by
  the test file only, matching `research/tests/test_gate.py`'s own house style).
- Produces: `PromotionState` enum, `create_promotion_record(session, candidate_result_id)
  -> PromotionRecord`, `start_paper_trading(session, promotion_id, dry_run_db_path,
  started_at=None) -> PromotionRecord`, `promote_to_live(session, promotion_id) ->
  PromotionRecord`, `reject(session, promotion_id, reason) -> PromotionRecord` — all
  consumed directly by Task 2 (which adds two more functions to the same module and file).

- [ ] **Step 1: Write the failing tests**

Create `research/tests/test_promotion.py`:

```python
# research/tests/test_promotion.py
from datetime import UTC, datetime

import pytest

from research.db import get_engine, get_session
from research.models import CandidateResult
from research.promotion import (
    PromotionState,
    create_promotion_record,
    promote_to_live,
    reject,
    start_paper_trading,
)


def _session(tmp_path):
    engine = get_engine(str(tmp_path / "research.sqlite"))
    return get_session(engine)


def _candidate(session, survived=True, oos_sharpe=1.5):
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
        survived=survived,
        evidence_json="{}",
    )
    session.add(candidate)
    session.flush()
    return candidate


def test_create_promotion_record_succeeds_for_a_passing_candidate(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session, survived=True)

    record = create_promotion_record(session, candidate.id)

    assert record.state == PromotionState.PASSED_GATE.value
    assert record.candidate_result_id == candidate.id


def test_create_promotion_record_raises_for_a_failing_candidate(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session, survived=False)

    with pytest.raises(ValueError, match="did not pass the gate"):
        create_promotion_record(session, candidate.id)


def test_create_promotion_record_raises_for_a_nonexistent_candidate(tmp_path):
    session = _session(tmp_path)

    with pytest.raises(ValueError, match="No CandidateResult"):
        create_promotion_record(session, 999)


def test_start_paper_trading_transitions_passed_gate_to_paper_trading(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)
    before = datetime.now(UTC)

    updated = start_paper_trading(session, record.id, "tradesv3.dryrun.sqlite")

    assert updated.state == PromotionState.PAPER_TRADING.value
    assert updated.paper_trading_db_path == "tradesv3.dryrun.sqlite"
    assert updated.paper_trading_started_at.replace(tzinfo=UTC) >= before


def test_start_paper_trading_raises_when_not_in_passed_gate(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)
    start_paper_trading(session, record.id, "tradesv3.dryrun.sqlite")

    with pytest.raises(ValueError, match="cannot start paper trading"):
        start_paper_trading(session, record.id, "tradesv3.dryrun.sqlite")


def test_promote_to_live_transitions_live_eligible_to_live(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)
    start_paper_trading(session, record.id, "tradesv3.dryrun.sqlite")
    record.state = PromotionState.LIVE_ELIGIBLE.value
    session.flush()

    updated = promote_to_live(session, record.id)

    assert updated.state == PromotionState.LIVE.value
    assert updated.resolved_at is not None


def test_promote_to_live_raises_from_paper_trading_directly(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)
    start_paper_trading(session, record.id, "tradesv3.dryrun.sqlite")

    with pytest.raises(ValueError, match="cannot promote to live"):
        promote_to_live(session, record.id)


def test_reject_transitions_paper_trading_and_live_eligible_to_rejected(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)

    record_a = create_promotion_record(session, candidate.id)
    start_paper_trading(session, record_a.id, "a.sqlite")
    rejected_a = reject(session, record_a.id, "degraded in paper trading")
    assert rejected_a.state == PromotionState.REJECTED.value
    assert rejected_a.resolution_reason == "degraded in paper trading"

    record_b = create_promotion_record(session, candidate.id)
    start_paper_trading(session, record_b.id, "b.sqlite")
    record_b.state = PromotionState.LIVE_ELIGIBLE.value
    session.flush()
    rejected_b = reject(session, record_b.id, "manual override")
    assert rejected_b.state == PromotionState.REJECTED.value


def test_reject_raises_from_passed_gate_and_already_resolved_states(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)

    with pytest.raises(ValueError, match=r"reject\(\) only applies"):
        reject(session, record.id, "too early")

    start_paper_trading(session, record.id, "a.sqlite")
    reject(session, record.id, "first rejection")
    with pytest.raises(ValueError, match=r"reject\(\) only applies"):
        reject(session, record.id, "second rejection")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest research/tests/test_promotion.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'research.promotion'`

- [ ] **Step 3: Add `PromotionRecord` to `research/models.py`**

Append to the end of `research/models.py`:

```python
class PromotionRecord(Base):
    """One row per promotion attempt for a specific passing CandidateResult -- tracks
    the Paper-Trading -> Live-eligibility lifecycle. A candidate can have many
    CandidateResult rows (re-runs, parameter sweeps); only specific passing runs ever
    get a PromotionRecord, and most never do."""

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

No new imports needed — `research/models.py` already imports `Boolean, DateTime, Float,
Integer, String` and `datetime` at the top of the file.

- [ ] **Step 4: Implement the state machine in `research/promotion.py`**

Create `research/promotion.py`:

```python
# research/promotion.py
"""Paper-trading promotion tracker: a state machine tracking a gate-passing candidate's
lifecycle from PASSED_GATE through PAPER_TRADING to a LIVE_ELIGIBLE recommendation or a
REJECTED verdict. LIVE is reachable only via a direct, manual promote_to_live() call --
never from any automated evaluation path (see evaluate_paper_trading_health /
apply_health_evaluation, added in Task 2 of this module's plan). See
docs/superpowers/specs/2026-08-24-paper-trading-promotion-design.md for the full design.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from sqlalchemy.orm import Session

from research.models import CandidateResult, PromotionRecord


class PromotionState(str, Enum):
    PASSED_GATE = "passed_gate"
    PAPER_TRADING = "paper_trading"
    LIVE_ELIGIBLE = "live_eligible"
    LIVE = "live"
    REJECTED = "rejected"


def _load_promotion_record(session: Session, promotion_id: int) -> PromotionRecord:
    record = session.get(PromotionRecord, promotion_id)
    if record is None:
        raise ValueError(f"No PromotionRecord with id {promotion_id}")
    return record


def create_promotion_record(session: Session, candidate_result_id: int) -> PromotionRecord:
    """Start a new promotion lifecycle for a candidate that already passed the gate.

    Raises ValueError if candidate_result_id doesn't exist, or if that candidate's
    CandidateResult.survived is not True -- only a passing gate result may be promoted.
    """
    candidate = session.get(CandidateResult, candidate_result_id)
    if candidate is None:
        raise ValueError(f"No CandidateResult with id {candidate_result_id}")
    if not candidate.survived:
        raise ValueError(
            f"CandidateResult {candidate_result_id} did not pass the gate "
            "(survived=False) -- only a passing candidate may be promoted."
        )
    record = PromotionRecord(
        candidate_result_id=candidate_result_id,
        state=PromotionState.PASSED_GATE.value,
        created_at=datetime.now(UTC),
    )
    session.add(record)
    session.flush()
    return record


def start_paper_trading(
    session: Session,
    promotion_id: int,
    dry_run_db_path: str,
    started_at: datetime | None = None,
) -> PromotionRecord:
    """Transition PASSED_GATE -> PAPER_TRADING.

    Raises ValueError if the record doesn't exist or isn't currently PASSED_GATE.
    """
    record = _load_promotion_record(session, promotion_id)
    if record.state != PromotionState.PASSED_GATE.value:
        raise ValueError(
            f"PromotionRecord {promotion_id} is in state {record.state!r}, not "
            f"{PromotionState.PASSED_GATE.value!r} -- cannot start paper trading."
        )
    record.state = PromotionState.PAPER_TRADING.value
    record.paper_trading_db_path = dry_run_db_path
    record.paper_trading_started_at = started_at or datetime.now(UTC)
    session.flush()
    return record


def promote_to_live(session: Session, promotion_id: int) -> PromotionRecord:
    """Transition LIVE_ELIGIBLE -> LIVE. The only function in this module that can
    produce a LIVE state -- called only by a human, directly, never from
    apply_health_evaluation or any other automated path.

    Raises ValueError if the record doesn't exist or isn't currently LIVE_ELIGIBLE.
    """
    record = _load_promotion_record(session, promotion_id)
    if record.state != PromotionState.LIVE_ELIGIBLE.value:
        raise ValueError(
            f"PromotionRecord {promotion_id} is in state {record.state!r}, not "
            f"{PromotionState.LIVE_ELIGIBLE.value!r} -- cannot promote to live."
        )
    record.state = PromotionState.LIVE.value
    record.resolved_at = datetime.now(UTC)
    session.flush()
    return record


def reject(session: Session, promotion_id: int, reason: str) -> PromotionRecord:
    """Manually transition PAPER_TRADING or LIVE_ELIGIBLE -> REJECTED.

    Raises ValueError if the record doesn't exist, is currently PASSED_GATE (paper
    trading never started), or is already resolved (REJECTED/LIVE).
    """
    record = _load_promotion_record(session, promotion_id)
    if record.state not in (
        PromotionState.PAPER_TRADING.value,
        PromotionState.LIVE_ELIGIBLE.value,
    ):
        raise ValueError(
            f"PromotionRecord {promotion_id} is in state {record.state!r} -- "
            f"reject() only applies from {PromotionState.PAPER_TRADING.value!r} or "
            f"{PromotionState.LIVE_ELIGIBLE.value!r}."
        )
    record.state = PromotionState.REJECTED.value
    record.resolved_at = datetime.now(UTC)
    record.resolution_reason = reason
    session.flush()
    return record
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest research/tests/test_promotion.py -v`
Expected: PASS (9 tests)

- [ ] **Step 6: Lint and format**

Run: `ruff check research/models.py research/promotion.py research/tests/test_promotion.py`
and `ruff format --check research/models.py research/promotion.py
research/tests/test_promotion.py`
Expected: no errors (fix with `ruff check --fix` / `ruff format` if needed, then re-run
Step 5 to confirm nothing broke)

- [ ] **Step 7: Commit**

```bash
git add research/models.py research/promotion.py research/tests/test_promotion.py
git commit -m "feat(research): add promotion.py state machine -- PASSED_GATE through LIVE"
```

---

### Task 2: health evaluator (freqtrade Trade-persistence integration)

**Files:**
- Modify: `research/promotion.py` (add `MIN_*` constants, `evaluate_paper_trading_health`,
  `apply_health_evaluation`)
- Test: `research/tests/test_promotion.py` (append)

**Interfaces:**
- Consumes: `PromotionState`, `_load_promotion_record`, `PromotionRecord`,
  `CandidateResult` (all from Task 1, already committed), `freqtrade.persistence.init_db`,
  `freqtrade.persistence.Trade`, `freqtrade.data.metrics.calculate_sharpe` (all existing,
  unmodified freqtrade APIs).
- Produces: `evaluate_paper_trading_health(session, promotion_id, starting_balance,
  dry_run_db_path=None, periods_per_year=365) -> dict`, `apply_health_evaluation(session,
  promotion_id, evaluation) -> PromotionRecord` — nothing further downstream; this is the
  final task in the plan.

- [ ] **Step 1: Write the failing tests**

Append to `research/tests/test_promotion.py`. First, add these imports at the top of the
file (alongside the existing ones):

```python
from datetime import timedelta

import pandas as pd
from freqtrade.persistence import Trade, init_db

from research.promotion import apply_health_evaluation, evaluate_paper_trading_health
```

Then append the following. A file-scoped autouse fixture first (this MUST be present
before any test in this file that calls `evaluate_paper_trading_health` runs, so add it
near the top of the new test code, not at the bottom):

```python
@pytest.fixture(autouse=True)
def _reset_trade_session_after_evaluation_tests():
    """evaluate_paper_trading_health() calls freqtrade's own init_db(), which sets
    Trade.session as GLOBAL class-level state (see the plan's Global Constraints for
    why) -- reset it to a fresh in-memory DB after every test in this file so no later
    test (in this file or elsewhere in the same pytest-xdist worker) can see
    Trade.session still pointed at one of this file's throwaway dry-run databases."""
    yield
    init_db("sqlite://")


def _insert_dry_run_trades(dry_run_db_path, strategy_id, started_at, profits_abs):
    """Directly construct and insert closed Trade rows into a fresh dry-run-style
    sqlite database -- one trade per entry in profits_abs, spaced evenly across the
    period from started_at to now, matching how research/tests/test_regime.py and
    test_scoring.py construct real rows directly rather than running a full backtest."""
    init_db(f"sqlite:///{dry_run_db_path}")
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


def test_evaluate_paper_trading_health_not_enough_evidence_when_too_few_trades(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session, oos_sharpe=2.0)
    record = create_promotion_record(session, candidate.id)
    started_at = datetime.now(UTC) - timedelta(days=14)
    start_paper_trading(session, record.id, str(tmp_path / "dryrun_thin.sqlite"), started_at)
    _insert_dry_run_trades(
        tmp_path / "dryrun_thin.sqlite", "TestStrategy", started_at, [8, 6, 7]
    )  # only 3 trades, MIN_PAPER_TRADES is 10

    evaluation = evaluate_paper_trading_health(session, record.id, starting_balance=1000.0)

    assert evaluation["enough_evidence"] is False
    assert evaluation["eligible"] is False
    assert evaluation["n_trades"] == 3

    updated = apply_health_evaluation(session, record.id, evaluation)
    assert updated.state == PromotionState.PAPER_TRADING.value


def test_evaluate_paper_trading_health_eligible_when_degradation_ratio_clears_bar(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session, oos_sharpe=2.0)
    record = create_promotion_record(session, candidate.id)
    started_at = datetime.now(UTC) - timedelta(days=14)
    start_paper_trading(session, record.id, str(tmp_path / "dryrun_good.sqlite"), started_at)
    _insert_dry_run_trades(
        tmp_path / "dryrun_good.sqlite",
        "TestStrategy",
        started_at,
        [8, 6, 7, 9, 5, 8, 6, 7, 9, 5],
    )

    evaluation = evaluate_paper_trading_health(session, record.id, starting_balance=1000.0)

    assert evaluation["enough_evidence"] is True
    assert evaluation["n_trades"] == 10
    assert evaluation["paper_sharpe"] == pytest.approx(67.54628043053148)
    assert evaluation["degradation_ratio"] == pytest.approx(1.0)
    assert evaluation["eligible"] is True

    updated = apply_health_evaluation(session, record.id, evaluation)
    assert updated.state == PromotionState.LIVE_ELIGIBLE.value
    assert updated.resolved_at is not None


def test_evaluate_paper_trading_health_rejected_when_degradation_ratio_fails_bar(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session, oos_sharpe=2.0)
    record = create_promotion_record(session, candidate.id)
    started_at = datetime.now(UTC) - timedelta(days=14)
    start_paper_trading(session, record.id, str(tmp_path / "dryrun_bad.sqlite"), started_at)
    _insert_dry_run_trades(
        tmp_path / "dryrun_bad.sqlite",
        "TestStrategy",
        started_at,
        [-3, 2, -4, 1, -2, 3, -5, 2, -3, 1],
    )

    evaluation = evaluate_paper_trading_health(session, record.id, starting_balance=1000.0)

    assert evaluation["enough_evidence"] is True
    assert evaluation["paper_sharpe"] == pytest.approx(-3.9705208944418664)
    assert evaluation["degradation_ratio"] == pytest.approx(0.0)
    assert evaluation["eligible"] is False

    updated = apply_health_evaluation(session, record.id, evaluation)
    assert updated.state == PromotionState.REJECTED.value
    assert updated.resolution_reason is not None


def test_evaluate_paper_trading_health_zero_trades_does_not_raise(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session, oos_sharpe=2.0)
    record = create_promotion_record(session, candidate.id)
    started_at = datetime.now(UTC) - timedelta(days=1)
    dry_run_path = tmp_path / "dryrun_empty.sqlite"
    start_paper_trading(session, record.id, str(dry_run_path), started_at)
    init_db(f"sqlite:///{dry_run_path}")  # create the (empty) schema, no trades inserted
    init_db("sqlite://")

    evaluation = evaluate_paper_trading_health(session, record.id, starting_balance=1000.0)

    assert evaluation["n_trades"] == 0
    assert evaluation["paper_sharpe"] == 0
    assert evaluation["enough_evidence"] is False


def test_evaluate_paper_trading_health_non_positive_oos_sharpe_is_zero_degradation(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session, oos_sharpe=-0.5)
    record = create_promotion_record(session, candidate.id)
    started_at = datetime.now(UTC) - timedelta(days=14)
    start_paper_trading(session, record.id, str(tmp_path / "dryrun_neg.sqlite"), started_at)
    _insert_dry_run_trades(
        tmp_path / "dryrun_neg.sqlite",
        "TestStrategy",
        started_at,
        [8, 6, 7, 9, 5, 8, 6, 7, 9, 5],
    )

    evaluation = evaluate_paper_trading_health(session, record.id, starting_balance=1000.0)

    assert evaluation["degradation_ratio"] == 0.0


def test_evaluate_paper_trading_health_raises_when_not_in_paper_trading(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)

    with pytest.raises(ValueError, match="cannot evaluate health"):
        evaluate_paper_trading_health(session, record.id, starting_balance=1000.0)


def test_apply_health_evaluation_raises_when_not_in_paper_trading(tmp_path):
    session = _session(tmp_path)
    candidate = _candidate(session)
    record = create_promotion_record(session, candidate.id)
    canned_evaluation = {
        "eligible": True,
        "enough_evidence": True,
        "days_elapsed": 30,
        "n_trades": 20,
        "paper_sharpe": 1.5,
        "degradation_ratio": 0.9,
        "reasons": [],
    }

    with pytest.raises(ValueError, match="cannot apply a health evaluation"):
        apply_health_evaluation(session, record.id, canned_evaluation)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest research/tests/test_promotion.py -v -k "health or apply"`
Expected: FAIL with `ImportError: cannot import name 'evaluate_paper_trading_health' from 'research.promotion'`

- [ ] **Step 3: Implement the health evaluator**

Append to `research/promotion.py`. First, update the imports at the top of the file to
add:

```python
import pandas as pd
from freqtrade.data.metrics import calculate_sharpe
from freqtrade.persistence import Trade, init_db
```

Then append the constants and both functions:

```python
# ponytail: starting defaults, not derived from any real paper-trading history (none
# exists yet in this fork) -- adjust based on real usage once this runs against real
# strategies.
MIN_PAPER_TRADING_DAYS = 14
MIN_PAPER_TRADES = 10
MIN_DEGRADATION_RATIO = 0.5


def evaluate_paper_trading_health(
    session: Session,
    promotion_id: int,
    starting_balance: float,
    dry_run_db_path: str | None = None,
    periods_per_year: int = 365,
) -> dict:
    """Pure evaluation (no state mutation) of a PAPER_TRADING record's real dry-run
    trade history. Returns a verdict dict; call apply_health_evaluation with the result
    to actually transition state.

    `starting_balance` is required -- the paper-trading bot's own configured wallet
    size, which this function has no other way to discover (see the spec for why
    run_promotion_gate's config isn't available here).

    IMPORTANT: freqtrade.persistence.init_db() sets Trade.session as GLOBAL class-level
    state, not a scoped per-call connection. This function fully materializes its query
    results before returning and must never be called concurrently with, or interleaved
    with, other in-process code relying on Trade.session pointing at a different
    database (e.g. a WalkForwardRunner/Backtesting run in the same process).

    Known limitation: the returned degradation_ratio is a coarse heuristic, not a
    statistically rigorous comparison -- a paper-trading window is typically far
    shorter than the OOS window a candidate was originally evaluated over, so
    paper_sharpe carries materially more estimation noise than the OOS baseline it's
    compared against. Treat this as a first-pass filter for human judgment, not proof.
    """
    record = _load_promotion_record(session, promotion_id)
    if record.state != PromotionState.PAPER_TRADING.value:
        raise ValueError(
            f"PromotionRecord {promotion_id} is in state {record.state!r}, not "
            f"{PromotionState.PAPER_TRADING.value!r} -- cannot evaluate health."
        )
    candidate = session.get(CandidateResult, record.candidate_result_id)
    if candidate is None:
        raise ValueError(f"No CandidateResult with id {record.candidate_result_id}")

    db_path = dry_run_db_path or record.paper_trading_db_path

    now = datetime.now(UTC)
    started_at_aware = record.paper_trading_started_at
    if started_at_aware.tzinfo is None:
        started_at_aware = started_at_aware.replace(tzinfo=UTC)
    started_at_naive = started_at_aware.replace(tzinfo=None)

    days_elapsed = (now - started_at_aware).days

    init_db(f"sqlite:///{db_path}")
    closed_trades = (
        Trade.session.query(Trade)
        .filter(
            Trade.strategy == candidate.strategy_id,
            Trade.is_open.is_(False),
            Trade.close_date >= started_at_naive,
        )
        .all()
    )
    n_trades = len(closed_trades)

    if n_trades > 0:
        trades_df = pd.DataFrame({"profit_abs": [t.close_profit_abs for t in closed_trades]})
        paper_sharpe = calculate_sharpe(trades_df, started_at_aware, now, starting_balance)
    else:
        paper_sharpe = 0

    if candidate.oos_sharpe > 0:
        degradation_ratio = max(0.0, min(1.0, paper_sharpe / candidate.oos_sharpe))
    else:
        degradation_ratio = 0.0

    reasons: list[str] = []
    enough_evidence = days_elapsed >= MIN_PAPER_TRADING_DAYS and n_trades >= MIN_PAPER_TRADES
    if not enough_evidence:
        eligible = False
        if days_elapsed < MIN_PAPER_TRADING_DAYS:
            reasons.append(
                f"only {days_elapsed} days elapsed, need >= {MIN_PAPER_TRADING_DAYS}"
            )
        if n_trades < MIN_PAPER_TRADES:
            reasons.append(f"only {n_trades} trades, need >= {MIN_PAPER_TRADES}")
    elif degradation_ratio < MIN_DEGRADATION_RATIO:
        eligible = False
        reasons.append(
            f"degradation_ratio {degradation_ratio:.3f} below threshold {MIN_DEGRADATION_RATIO}"
        )
    else:
        eligible = True

    return {
        "eligible": eligible,
        "enough_evidence": enough_evidence,
        "days_elapsed": days_elapsed,
        "n_trades": n_trades,
        "paper_sharpe": paper_sharpe,
        "degradation_ratio": degradation_ratio,
        "reasons": reasons,
    }


def apply_health_evaluation(
    session: Session, promotion_id: int, evaluation: dict
) -> PromotionRecord:
    """Apply an evaluate_paper_trading_health() result to the state machine.

    PAPER_TRADING -> LIVE_ELIGIBLE if eligible; PAPER_TRADING -> REJECTED if there's
    enough evidence but it failed the bar; otherwise no state change (stays
    PAPER_TRADING for a future re-evaluation).

    Raises ValueError if the record doesn't exist or isn't currently PAPER_TRADING.
    """
    record = _load_promotion_record(session, promotion_id)
    if record.state != PromotionState.PAPER_TRADING.value:
        raise ValueError(
            f"PromotionRecord {promotion_id} is in state {record.state!r}, not "
            f"{PromotionState.PAPER_TRADING.value!r} -- cannot apply a health evaluation."
        )
    if evaluation["eligible"]:
        record.state = PromotionState.LIVE_ELIGIBLE.value
        record.resolved_at = datetime.now(UTC)
        record.resolution_reason = (
            f"paper_sharpe={evaluation['paper_sharpe']:.3f}, "
            f"degradation_ratio={evaluation['degradation_ratio']:.3f}, "
            f"n_trades={evaluation['n_trades']}, days_elapsed={evaluation['days_elapsed']}"
        )
    elif evaluation["enough_evidence"]:
        record.state = PromotionState.REJECTED.value
        record.resolved_at = datetime.now(UTC)
        record.resolution_reason = "; ".join(evaluation["reasons"])
    session.flush()
    return record
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest research/tests/test_promotion.py -v`
Expected: PASS (16 tests: 9 from Task 1 + 7 new)

- [ ] **Step 5: Lint and format**

Run: `ruff check research/promotion.py research/tests/test_promotion.py` and
`ruff format --check research/promotion.py research/tests/test_promotion.py`
Expected: no errors (fix and re-run Step 4 if needed)

- [ ] **Step 6: Run the full research suite**

Run: `pytest research/ -v`
Expected: PASS (every test in `research/tests/`, confirming this module composes
cleanly with the rest of the package and doesn't leak `Trade.session` state into any
other file's tests)

- [ ] **Step 7: Commit**

```bash
git add research/promotion.py research/tests/test_promotion.py
git commit -m "feat(research): add paper-trading health evaluator reading real freqtrade dry-run data"
```
