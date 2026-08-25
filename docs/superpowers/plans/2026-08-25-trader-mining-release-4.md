# Trader/Wallet Mining Release 4 (Temporal Validation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chronologically split a wallet's `ReconstructedTrade` rows into TRAIN/VALIDATION/
TEST/FORWARD, report `compute_metrics` per period, and wire it into `trader-report`.

**Architecture:** Two new pure modules -- `splitting.py` (partition logic) and
`split_report.py` (per-period metrics + formatting) -- plus three new `trader-report` CLI
flags. Neither new module touches a `Session` or a database.

**Tech Stack:** Python stdlib only (`dataclasses`, `datetime`) -- no new dependency.

**Spec:** `docs/superpowers/specs/2026-08-25-trader-mining-release-4-design.md`

## Global Constraints

- `splitting.py` and `split_report.py` take/return plain values or `ReconstructedTrade`
  lists -- no `Session`, no DB import in either file.
- `assign_period`/`split_trades` are pure functions of a trade's own `entry_timestamp` and
  the configured `PeriodBoundaries` alone -- never a statistic computed across the full
  trade set (the `research/regime.py` anti-lookahead lesson).
- `compute_metrics` (from `research/trader_mining/metrics.py`) is imported and called per
  bucket, never modified, never reimplemented.
- Never the words "reject", "fail", or "threshold" describing TRAIN-to-VALIDATION
  degradation anywhere in `split_report.py` or its output -- diagnostic language only, per
  the spec's "What this is not."
- Every new dataclass/function must handle `trades=[]` and single-period-only histories
  without raising.
- Boundary comparisons are half-open: a boundary date itself belongs to the period that
  *starts* there (e.g. `train_end` itself is the first instant of VALIDATION, matching the
  proposal's own "TRAIN 2024 / VALIDATION 2025 H1" example where Jan 1 2025 starts
  VALIDATION, not the last instant of TRAIN).

---

### Task 1: `PeriodBoundaries` -- tz normalization + validation

**Files:**
- Create: `research/trader_mining/splitting.py`
- Test: `research/tests/trader_mining/test_splitting.py`

**Interfaces:**
- Produces: `research.trader_mining.splitting.PERIODS: tuple[str, str, str, str]`
- Produces: `research.trader_mining.splitting._to_naive_utc(dt: datetime) -> datetime`
- Produces: `research.trader_mining.splitting.PeriodBoundaries` (frozen dataclass:
  `train_end: datetime`, `validation_end: datetime`, `test_end: datetime`)

- [ ] **Step 1: Write the failing tests**

```python
# research/tests/trader_mining/test_splitting.py
from datetime import UTC, datetime, timedelta, timezone

import pytest

from research.trader_mining.splitting import PeriodBoundaries, _to_naive_utc


def test_to_naive_utc_strips_tzinfo_from_utc_datetime():
    assert _to_naive_utc(datetime(2025, 1, 1, tzinfo=UTC)) == datetime(2025, 1, 1)


def test_to_naive_utc_leaves_naive_input_untouched():
    """A naive datetime in this codebase always means 'already UTC' -- the ingestion
    invariant -- so it is never treated as a timezone-unknown error."""
    assert _to_naive_utc(datetime(2025, 1, 1)) == datetime(2025, 1, 1)


def test_to_naive_utc_converts_non_utc_offset_before_stripping():
    plus_five = timezone(timedelta(hours=5))
    dt = datetime(2025, 1, 1, 5, 0, tzinfo=plus_five)  # 2025-01-01T00:00 UTC
    assert _to_naive_utc(dt) == datetime(2025, 1, 1, 0, 0)


def test_boundaries_normalize_tz_aware_input_to_naive_utc():
    b = PeriodBoundaries(
        train_end=datetime(2025, 1, 1, tzinfo=UTC),
        validation_end=datetime(2025, 7, 1, tzinfo=UTC),
        test_end=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert b.train_end == datetime(2025, 1, 1)
    assert b.train_end.tzinfo is None
    assert b.validation_end == datetime(2025, 7, 1)
    assert b.test_end == datetime(2026, 1, 1)


def test_boundaries_treat_naive_input_as_already_utc():
    b = PeriodBoundaries(
        train_end=datetime(2025, 1, 1),
        validation_end=datetime(2025, 7, 1),
        test_end=datetime(2026, 1, 1),
    )

    assert b.train_end == datetime(2025, 1, 1)


def test_boundaries_reject_non_strictly_increasing_dates():
    with pytest.raises(ValueError, match="train_end < validation_end < test_end"):
        PeriodBoundaries(
            train_end=datetime(2025, 7, 1),
            validation_end=datetime(2025, 1, 1),  # before train_end
            test_end=datetime(2026, 1, 1),
        )


def test_boundaries_reject_equal_dates():
    """Equal boundaries would silently produce an empty period -- rejected, not tolerated."""
    with pytest.raises(ValueError, match="train_end < validation_end < test_end"):
        PeriodBoundaries(
            train_end=datetime(2025, 1, 1),
            validation_end=datetime(2025, 1, 1),
            test_end=datetime(2026, 1, 1),
        )
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest research/tests/trader_mining/test_splitting.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.trader_mining.splitting'`

- [ ] **Step 3: Implement `PeriodBoundaries` and `_to_naive_utc`**

```python
# research/trader_mining/splitting.py
"""Chronological TRAIN/VALIDATION/TEST/FORWARD split for a single wallet's reconstructed
trades -- Phase 6 of TRADER_WALLET_MINING_PROPOSAL.md, "the most important research
requirement." Pure and DB-free, mirroring research.trader_mining.metrics.compute_metrics'
own "pure function, no DB access" precedent.

A trade's period is a function of its own entry_timestamp and the configured
PeriodBoundaries alone -- never a statistic computed across the full trade set. See
research/regime.py's documented anti-lookahead trap for why: that classifier ranks each
window against a full-sample median computed across windows including future ones, safe
only because it's a post-hoc whole-run report. This module must not repeat that pattern.

Boundaries are compared timezone-naive throughout: a naive datetime in this codebase always
means "already UTC" (the ingestion invariant -- see research/trader_mining/ingestion.py),
which is also the only convention ReconstructedTrade timestamps are guaranteed to carry once
read back from SQLite (SQLite silently drops tzinfo on a fresh query even though fills are
written tz-aware UTC at ingestion time -- verified empirically, see the design doc). Tz-aware
input is converted to UTC before stripping, not just discarded, so a non-UTC offset still
normalizes correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

PERIODS: tuple[str, str, str, str] = ("TRAIN", "VALIDATION", "TEST", "FORWARD")


def _to_naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC)
    return dt.replace(tzinfo=None)


@dataclass(frozen=True)
class PeriodBoundaries:
    """The three cut points partitioning a wallet's trade history into four periods:
    TRAIN (< train_end), VALIDATION ([train_end, validation_end)), TEST
    ([validation_end, test_end)), FORWARD (>= test_end, open-ended). Normalized to naive
    UTC at construction; rejects non-strictly-increasing dates."""

    train_end: datetime
    validation_end: datetime
    test_end: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_end", _to_naive_utc(self.train_end))
        object.__setattr__(self, "validation_end", _to_naive_utc(self.validation_end))
        object.__setattr__(self, "test_end", _to_naive_utc(self.test_end))
        if not (self.train_end < self.validation_end < self.test_end):
            raise ValueError(
                "PeriodBoundaries requires train_end < validation_end < test_end, got "
                f"train_end={self.train_end}, validation_end={self.validation_end}, "
                f"test_end={self.test_end}"
            )
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest research/tests/trader_mining/test_splitting.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add research/trader_mining/splitting.py research/tests/trader_mining/test_splitting.py
git commit -m "feat(research): PeriodBoundaries with tz-normalization and boundary validation"
```

---

### Task 2: `assign_period` / `straddles_boundary` / `split_trades`

**Files:**
- Modify: `research/trader_mining/splitting.py`
- Test: `research/tests/trader_mining/test_splitting.py`

**Interfaces:**
- Consumes: `PeriodBoundaries` (Task 1)
- Produces: `research.trader_mining.splitting.assign_period(trade: ReconstructedTrade, boundaries: PeriodBoundaries) -> str`
- Produces: `research.trader_mining.splitting.straddles_boundary(trade: ReconstructedTrade, boundaries: PeriodBoundaries) -> bool`
- Produces: `research.trader_mining.splitting.SplitTrades` (frozen dataclass: `train: list[ReconstructedTrade]`, `validation: list[ReconstructedTrade]`, `test: list[ReconstructedTrade]`, `forward: list[ReconstructedTrade]`, `n_straddling: int`)
- Produces: `research.trader_mining.splitting.split_trades(trades: list[ReconstructedTrade], boundaries: PeriodBoundaries) -> SplitTrades`

- [ ] **Step 1: Write the failing tests**

```python
# research/tests/trader_mining/test_splitting.py -- add to the existing file
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from research.models import Base, ReconstructedTrade
from research.trader_mining.splitting import assign_period, split_trades, straddles_boundary

BOUNDARIES = PeriodBoundaries(
    train_end=datetime(2025, 1, 1, tzinfo=UTC),
    validation_end=datetime(2025, 7, 1, tzinfo=UTC),
    test_end=datetime(2026, 1, 1, tzinfo=UTC),
)


def _trade(entry_ts, exit_ts=None) -> ReconstructedTrade:
    exit_ts = exit_ts if exit_ts is not None else entry_ts
    return ReconstructedTrade(
        trader="0xAAA",
        symbol="BTC/USDC:USDC",
        direction="long",
        entry_timestamp=entry_ts,
        entry_price=100.0,
        exit_timestamp=exit_ts,
        exit_price=100.0,
        quantity=1.0,
        gross_pnl=10.0,
        fees=0.0,
        net_pnl=10.0,
        holding_time_seconds=3600.0,
        n_fills=2,
        is_truncated_start=False,
        was_liquidated=False,
    )


def test_assign_period_uses_entry_timestamp_not_exit_timestamp():
    """A trade entering in TRAIN but exiting in VALIDATION is still TRAIN -- assignment is
    entirely a function of the entry decision point, never the (only-knowable-at-exit)
    outcome timing."""
    t = _trade(entry_ts=datetime(2024, 6, 1, tzinfo=UTC), exit_ts=datetime(2025, 3, 1, tzinfo=UTC))

    assert assign_period(t, BOUNDARIES) == "TRAIN"


@pytest.mark.parametrize(
    "entry_ts,expected",
    [
        (datetime(2020, 1, 1, tzinfo=UTC), "TRAIN"),
        (datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC), "TRAIN"),
        (datetime(2025, 1, 1, tzinfo=UTC), "VALIDATION"),  # boundary starts the next period
        (datetime(2025, 6, 30, 23, 59, 59, tzinfo=UTC), "VALIDATION"),
        (datetime(2025, 7, 1, tzinfo=UTC), "TEST"),
        (datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC), "TEST"),
        (datetime(2026, 1, 1, tzinfo=UTC), "FORWARD"),
        (datetime(2030, 1, 1, tzinfo=UTC), "FORWARD"),  # FORWARD has no upper bound
    ],
)
def test_assign_period_boundary_semantics_are_half_open(entry_ts, expected):
    assert assign_period(_trade(entry_ts=entry_ts), BOUNDARIES) == expected


def test_assign_period_works_with_naive_entry_timestamp_and_tz_aware_boundaries():
    """Simulated version of the SQLite-naive-read landmine (see the real round-trip test
    below for the end-to-end proof): a tz-aware-boundary comparison against a naive
    (implicitly-UTC) trade timestamp must not raise TypeError."""
    t = _trade(entry_ts=datetime(2025, 3, 1))  # naive

    assert assign_period(t, BOUNDARIES) == "VALIDATION"


def test_straddles_boundary_true_when_entry_and_exit_periods_differ():
    t = _trade(entry_ts=datetime(2024, 12, 1, tzinfo=UTC), exit_ts=datetime(2025, 2, 1, tzinfo=UTC))

    assert straddles_boundary(t, BOUNDARIES) is True


def test_straddles_boundary_false_when_entry_and_exit_in_same_period():
    t = _trade(entry_ts=datetime(2025, 3, 1, tzinfo=UTC), exit_ts=datetime(2025, 4, 1, tzinfo=UTC))

    assert straddles_boundary(t, BOUNDARIES) is False


def test_split_trades_every_trade_appears_exactly_once():
    """No-overlap guarantee, enforced by test: partition 6 trades and confirm the four
    buckets are disjoint and their union recovers every input trade exactly once."""
    trades = [
        _trade(entry_ts=datetime(2024, 1, 1, tzinfo=UTC)),
        _trade(entry_ts=datetime(2024, 6, 1, tzinfo=UTC)),
        _trade(entry_ts=datetime(2025, 3, 1, tzinfo=UTC)),
        _trade(entry_ts=datetime(2025, 9, 1, tzinfo=UTC)),
        _trade(entry_ts=datetime(2026, 3, 1, tzinfo=UTC)),
        _trade(entry_ts=datetime(2025, 1, 1, tzinfo=UTC)),  # exactly on a boundary
    ]

    result = split_trades(trades, BOUNDARIES)

    all_bucketed = result.train + result.validation + result.test + result.forward
    assert len(all_bucketed) == len(trades)
    assert {id(t) for t in all_bucketed} == {id(t) for t in trades}
    assert len(result.train) == 2
    assert len(result.validation) == 2
    assert len(result.test) == 1
    assert len(result.forward) == 1


def test_split_trades_counts_straddling_trades():
    trades = [
        _trade(entry_ts=datetime(2024, 12, 1, tzinfo=UTC), exit_ts=datetime(2025, 2, 1, tzinfo=UTC)),
        _trade(entry_ts=datetime(2025, 3, 1, tzinfo=UTC), exit_ts=datetime(2025, 4, 1, tzinfo=UTC)),
    ]

    result = split_trades(trades, BOUNDARIES)

    assert result.n_straddling == 1


def test_split_trades_handles_empty_history_without_error():
    """Insufficient/empty-history handling: no trades is not an error condition."""
    result = split_trades([], BOUNDARIES)

    assert result.train == []
    assert result.validation == []
    assert result.test == []
    assert result.forward == []
    assert result.n_straddling == 0


def test_split_trades_handles_history_entirely_before_train_end():
    """All trades landing in one bucket, others legitimately empty -- not an error."""
    trades = [_trade(entry_ts=datetime(2024, 1, 1, tzinfo=UTC))]

    result = split_trades(trades, BOUNDARIES)

    assert len(result.train) == 1
    assert result.validation == [] and result.test == [] and result.forward == []


def test_assign_period_handles_real_sqlite_round_trip_naive_timestamps():
    """End-to-end proof of the documented landmine: entry_timestamp is written tz-aware UTC
    but comes back naive from a fresh query/session. A tz-aware PeriodBoundaries must still
    compare against it without raising TypeError."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    write_session = Session(engine)
    write_session.add(
        ReconstructedTrade(
            trader="0xAAA",
            symbol="BTC/USDC:USDC",
            direction="long",
            entry_timestamp=datetime(2025, 3, 1, tzinfo=UTC),
            entry_price=100.0,
            exit_timestamp=datetime(2025, 3, 1, tzinfo=UTC),
            exit_price=100.0,
            quantity=1.0,
            gross_pnl=10.0,
            fees=0.0,
            net_pnl=10.0,
            holding_time_seconds=3600.0,
            n_fills=2,
            is_truncated_start=False,
            was_liquidated=False,
        )
    )
    write_session.commit()
    write_session.close()

    read_session = Session(engine)
    trade = read_session.query(ReconstructedTrade).one()
    assert trade.entry_timestamp.tzinfo is None  # confirms the landmine actually reproduces

    assert assign_period(trade, BOUNDARIES) == "VALIDATION"  # must not raise TypeError
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest research/tests/trader_mining/test_splitting.py -v`
Expected: FAIL with `ImportError`/`AttributeError` for the new symbols

- [ ] **Step 3: Implement `assign_period`, `straddles_boundary`, `SplitTrades`, `split_trades`**

Add near the top of `splitting.py` (after the existing imports):

```python
from research.models import ReconstructedTrade
```

Add at the end of `splitting.py`:

```python
def _period_of(ts: datetime, boundaries: PeriodBoundaries) -> str:
    ts = _to_naive_utc(ts)
    if ts < boundaries.train_end:
        return "TRAIN"
    if ts < boundaries.validation_end:
        return "VALIDATION"
    if ts < boundaries.test_end:
        return "TEST"
    return "FORWARD"


def assign_period(trade: ReconstructedTrade, boundaries: PeriodBoundaries) -> str:
    """A trade's period is decided by its entry_timestamp alone -- never exit_timestamp,
    which would let the (only-knowable-at-exit) outcome influence which research phase the
    trade is scored in."""
    return _period_of(trade.entry_timestamp, boundaries)


def straddles_boundary(trade: ReconstructedTrade, boundaries: PeriodBoundaries) -> bool:
    """True when a trade's entry and exit fall in different periods. Diagnostic only --
    never changes assign_period's result or splits the trade itself."""
    return _period_of(trade.entry_timestamp, boundaries) != _period_of(
        trade.exit_timestamp, boundaries
    )


@dataclass(frozen=True)
class SplitTrades:
    train: list[ReconstructedTrade]
    validation: list[ReconstructedTrade]
    test: list[ReconstructedTrade]
    forward: list[ReconstructedTrade]
    n_straddling: int


def split_trades(trades: list[ReconstructedTrade], boundaries: PeriodBoundaries) -> SplitTrades:
    buckets: dict[str, list[ReconstructedTrade]] = {p: [] for p in PERIODS}
    n_straddling = 0
    for t in trades:
        buckets[assign_period(t, boundaries)].append(t)
        if straddles_boundary(t, boundaries):
            n_straddling += 1
    return SplitTrades(
        train=buckets["TRAIN"],
        validation=buckets["VALIDATION"],
        test=buckets["TEST"],
        forward=buckets["FORWARD"],
        n_straddling=n_straddling,
    )
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest research/tests/trader_mining/test_splitting.py -v`
Expected: PASS (18 tests total)

- [ ] **Step 5: Commit**

```bash
git add research/trader_mining/splitting.py research/tests/trader_mining/test_splitting.py
git commit -m "feat(research): assign_period/split_trades with straddle diagnostic and no-overlap guarantee"
```

---

### Task 3: `split_report.py` -- per-period metrics + report

**Files:**
- Create: `research/trader_mining/split_report.py`
- Test: `research/tests/trader_mining/test_split_report.py`

**Interfaces:**
- Consumes: `PeriodBoundaries`, `PERIODS`, `split_trades` (Tasks 1-2);
  `WalletMetrics`, `compute_metrics`, `format_report` (Release 3, unmodified)
- Produces: `research.trader_mining.split_report.PeriodSummary` (dataclass: `period: str`, `start: datetime | None`, `end: datetime | None`, `n_trades: int`, `metrics: WalletMetrics`)
- Produces: `research.trader_mining.split_report.SplitReport` (dataclass: `boundaries: PeriodBoundaries`, `periods: list[PeriodSummary]`, `n_straddling: int`, `whole_history: WalletMetrics`)
- Produces: `research.trader_mining.split_report.compute_split_report(trades: list[ReconstructedTrade], boundaries: PeriodBoundaries) -> SplitReport`
- Produces: `research.trader_mining.split_report.format_split_report(report: SplitReport, trader: str) -> str`

- [ ] **Step 1: Write the failing tests**

```python
# research/tests/trader_mining/test_split_report.py
from datetime import UTC, datetime

from research.models import ReconstructedTrade
from research.trader_mining.split_report import compute_split_report, format_split_report
from research.trader_mining.splitting import PeriodBoundaries

BOUNDARIES = PeriodBoundaries(
    train_end=datetime(2025, 1, 1, tzinfo=UTC),
    validation_end=datetime(2025, 7, 1, tzinfo=UTC),
    test_end=datetime(2026, 1, 1, tzinfo=UTC),
)


def _trade(net_pnl, entry_ts, exit_ts=None) -> ReconstructedTrade:
    exit_ts = exit_ts if exit_ts is not None else entry_ts
    return ReconstructedTrade(
        trader="0xAAA",
        symbol="BTC/USDC:USDC",
        direction="long",
        entry_timestamp=entry_ts,
        entry_price=100.0,
        exit_timestamp=exit_ts,
        exit_price=100.0,
        quantity=1.0,
        gross_pnl=net_pnl,
        fees=0.0,
        net_pnl=net_pnl,
        holding_time_seconds=3600.0,
        n_fills=2,
        is_truncated_start=False,
        was_liquidated=False,
    )


def test_compute_split_report_populates_all_four_periods_and_whole_history():
    trades = [
        _trade(10.0, datetime(2024, 6, 1, tzinfo=UTC)),
        _trade(20.0, datetime(2024, 8, 1, tzinfo=UTC)),
        _trade(-5.0, datetime(2025, 3, 1, tzinfo=UTC)),
        _trade(15.0, datetime(2025, 9, 1, tzinfo=UTC)),
    ]

    report = compute_split_report(trades, BOUNDARIES)

    assert [p.period for p in report.periods] == ["TRAIN", "VALIDATION", "TEST", "FORWARD"]
    train, validation, test, forward = report.periods
    assert train.n_trades == 2
    assert train.metrics.net_pnl == 30.0
    assert validation.n_trades == 1
    assert validation.metrics.net_pnl == -5.0
    assert test.n_trades == 1
    assert forward.n_trades == 0
    assert forward.metrics.trade_count == 0  # compute_metrics([]) -- reused, not reimplemented
    assert report.whole_history.trade_count == 4
    assert report.whole_history.net_pnl == 40.0
    assert report.n_straddling == 0


def test_compute_split_report_handles_all_periods_empty():
    """Insufficient history: an empty trade list must not raise, and every period's metrics
    must come back as compute_metrics([])'s own well-defined 'undefined' shape."""
    report = compute_split_report([], BOUNDARIES)

    assert all(p.n_trades == 0 for p in report.periods)
    assert all(p.metrics.win_rate is None for p in report.periods)
    assert report.whole_history.trade_count == 0


def test_period_summary_start_end_are_open_for_train_start_and_forward_end():
    report = compute_split_report([], BOUNDARIES)

    train, validation, test, forward = report.periods
    assert train.start is None
    assert train.end == BOUNDARIES.train_end
    assert validation.start == BOUNDARIES.train_end
    assert test.end == BOUNDARIES.test_end
    assert forward.start == BOUNDARIES.test_end
    assert forward.end is None


def test_format_split_report_labels_periods_and_shows_sample_counts_and_dates():
    trades = [_trade(10.0, datetime(2024, 6, 1, tzinfo=UTC)), _trade(-5.0, datetime(2025, 3, 1, tzinfo=UTC))]
    report = compute_split_report(trades, BOUNDARIES)

    text = format_split_report(report, "0xAAA")

    assert "TRAIN" in text and "VALIDATION" in text and "TEST" in text and "FORWARD" in text
    assert "n=1" in text  # each populated period's sample count appears
    assert "2025-01-01" in text  # a boundary date appears
    assert "0xAAA" in text


def test_format_split_report_shows_whole_history_as_a_distinct_labeled_section():
    trades = [_trade(10.0, datetime(2024, 6, 1, tzinfo=UTC))]
    report = compute_split_report(trades, BOUNDARIES)

    text = format_split_report(report, "0xAAA")

    assert "whole-history" in text.lower()
    assert "not out-of-sample" in text.lower()


def test_format_split_report_never_auto_rejects_on_expectancy_degradation():
    """Proposal review-notes correction: report TRAIN->VALIDATION expectancy degradation as
    a diagnostic only. No pass/fail verdict, no threshold language."""
    trades = [
        _trade(100.0, datetime(2024, 6, 1, tzinfo=UTC)),
        _trade(-90.0, datetime(2025, 3, 1, tzinfo=UTC)),  # huge TRAIN->VALIDATION degradation
    ]

    report = compute_split_report(trades, BOUNDARIES)
    text = format_split_report(report, "0xAAA")

    lowered = text.lower()
    assert "reject" not in lowered
    assert "fail" not in lowered
    assert "diagnostic" in lowered


def test_format_split_report_handles_empty_periods_without_crashing_or_printing_none():
    report = compute_split_report([], BOUNDARIES)

    text = format_split_report(report, "0xAAA")

    assert "None" not in text
    assert "n/a" in text
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest research/tests/trader_mining/test_split_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.trader_mining.split_report'`

- [ ] **Step 3: Implement `split_report.py`**

```python
# research/trader_mining/split_report.py
"""Per-period performance report for a chronologically split wallet -- computes
research.trader_mining.metrics.compute_metrics once per TRAIN/VALIDATION/TEST/FORWARD
bucket (unmodified) and formats the result alongside a distinctly labeled whole-history
reference section. See docs/superpowers/specs/2026-08-25-trader-mining-release-4-design.md.

TRAIN-to-VALIDATION expectancy change is reported as a diagnostic only -- the proposal's own
review-notes correction is explicit that an arbitrary percentage cutoff is unstable at
realistic sample sizes. There is no threshold, no pass/fail verdict, anywhere in this
module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from research.models import ReconstructedTrade
from research.trader_mining.metrics import WalletMetrics, compute_metrics, format_report
from research.trader_mining.splitting import PERIODS, PeriodBoundaries, split_trades


@dataclass
class PeriodSummary:
    period: str
    start: datetime | None
    end: datetime | None
    n_trades: int
    metrics: WalletMetrics


@dataclass
class SplitReport:
    boundaries: PeriodBoundaries
    periods: list[PeriodSummary]  # always length 4, PERIODS order
    n_straddling: int
    whole_history: WalletMetrics


def compute_split_report(
    trades: list[ReconstructedTrade], boundaries: PeriodBoundaries
) -> SplitReport:
    split = split_trades(trades, boundaries)
    bucketed = {
        "TRAIN": split.train,
        "VALIDATION": split.validation,
        "TEST": split.test,
        "FORWARD": split.forward,
    }
    bounds: dict[str, tuple[datetime | None, datetime | None]] = {
        "TRAIN": (None, boundaries.train_end),
        "VALIDATION": (boundaries.train_end, boundaries.validation_end),
        "TEST": (boundaries.validation_end, boundaries.test_end),
        "FORWARD": (boundaries.test_end, None),
    }
    periods = [
        PeriodSummary(
            period=p,
            start=bounds[p][0],
            end=bounds[p][1],
            n_trades=len(bucketed[p]),
            metrics=compute_metrics(bucketed[p]),
        )
        for p in PERIODS
    ]
    return SplitReport(
        boundaries=boundaries,
        periods=periods,
        n_straddling=split.n_straddling,
        whole_history=compute_metrics(trades),
    )


def format_split_report(report: SplitReport, trader: str) -> str:
    lines = [
        f"# Chronological Split Report: {trader}",
        "",
        f"Boundaries: TRAIN < {report.boundaries.train_end.date()} <= VALIDATION < "
        f"{report.boundaries.validation_end.date()} <= TEST < "
        f"{report.boundaries.test_end.date()} <= FORWARD",
        f"Trades spanning a period boundary (counted in their entry period): "
        f"{report.n_straddling}",
        "",
    ]
    for summary in report.periods:
        start = summary.start.date() if summary.start else "(start of history)"
        end = summary.end.date() if summary.end else "(ongoing)"
        lines.append(f"## {summary.period} [{start} - {end}), n={summary.n_trades}")
        lines.append("")
        lines.append(format_report(summary.metrics, trader))
        lines.append("")

    train_expectancy = report.periods[0].metrics.expectancy
    validation_expectancy = report.periods[1].metrics.expectancy
    if train_expectancy and validation_expectancy is not None:
        delta_pct = (validation_expectancy - train_expectancy) / abs(train_expectancy) * 100
        lines.append(
            f"Diagnostic: TRAIN->VALIDATION expectancy changed by {delta_pct:.1f}% "
            "(reported for awareness only -- no threshold, no automatic rejection)"
        )
    else:
        lines.append(
            "Diagnostic: TRAIN->VALIDATION expectancy change: n/a (insufficient data in "
            "TRAIN or VALIDATION)"
        )
    lines.append("")

    lines.append("## Whole-history (reference only -- NOT out-of-sample)")
    lines.append("")
    lines.append(format_report(report.whole_history, trader))

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest research/tests/trader_mining/test_split_report.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add research/trader_mining/split_report.py research/tests/trader_mining/test_split_report.py
git commit -m "feat(research): per-period metrics + chronological split report"
```

---

### Task 4: `trader-report` CLI wiring

**Files:**
- Modify: `research/cli.py`
- Test: `research/tests/test_cli.py`

**Interfaces:**
- Consumes: `PeriodBoundaries` (Task 1), `compute_split_report`, `format_split_report` (Task 3)

- [ ] **Step 1: Write the failing tests**

```python
# research/tests/test_cli.py -- add to the existing file
def test_trader_report_rejects_partial_split_flags(capsys):
    with pytest.raises(SystemExit):
        main(
            [
                "trader-report",
                "--trader",
                "0xAAA",
                "--train-end",
                "2025-01-01",
                "--validation-end",
                "2025-07-01",
                # --test-end deliberately omitted
            ]
        )

    assert "must be given together" in capsys.readouterr().err


def test_trader_report_prints_split_report_when_all_three_flags_given(mocker, capsys):
    from datetime import UTC, datetime

    from research.models import ReconstructedTrade

    trade = ReconstructedTrade(
        trader="0xAAA",
        symbol="BTC/USDC:USDC",
        direction="long",
        entry_timestamp=datetime(2024, 6, 1, tzinfo=UTC),
        entry_price=100.0,
        exit_timestamp=datetime(2024, 6, 1, tzinfo=UTC),
        exit_price=100.0,
        quantity=1.0,
        gross_pnl=10.0,
        fees=0.0,
        net_pnl=10.0,
        holding_time_seconds=3600.0,
        n_fills=2,
        is_truncated_start=False,
        was_liquidated=False,
    )
    mock_query = mocker.MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.all.return_value = [trade]
    mock_session = mocker.MagicMock()
    mock_session.query.return_value = mock_query
    mocker.patch("research.cli.get_engine")
    mocker.patch("research.cli.get_session", return_value=mock_session)

    exit_code = main(
        [
            "trader-report",
            "--trader",
            "0xAAA",
            "--train-end",
            "2025-01-01",
            "--validation-end",
            "2025-07-01",
            "--test-end",
            "2026-01-01",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "TRAIN" in captured.out and "VALIDATION" in captured.out
    assert "whole-history" in captured.out.lower()


def test_trader_report_unchanged_when_no_split_flags_given(mocker, capsys):
    """Regression guard: today's plain trader-report output must be the same code path
    (compute_metrics + format_report) when no split flags are passed."""
    mock_query = mocker.MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.all.return_value = []
    mock_session = mocker.MagicMock()
    mock_session.query.return_value = mock_query
    mocker.patch("research.cli.get_engine")
    mocker.patch("research.cli.get_session", return_value=mock_session)

    exit_code = main(["trader-report", "--trader", "0xAAA"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Chronological Split Report" not in captured.out
    assert "## Wallet Report: 0xAAA" in captured.out
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest research/tests/test_cli.py -v`
Expected: FAIL -- `error: unrecognized arguments: --train-end ...`

- [ ] **Step 3: Wire `cli.py`**

Add imports near the top of `cli.py`:

```python
from research.trader_mining.split_report import compute_split_report, format_split_report
from research.trader_mining.splitting import PeriodBoundaries
```

Add three new flags to the existing `trader_report` subparser, right after
`trader_report.add_argument("--db-path", default="user_data/research.sqlite")`:

```python
    trader_report.add_argument(
        "--train-end",
        help=(
            "YYYY-MM-DD, TRAIN/VALIDATION boundary (exclusive of VALIDATION). Requires "
            "--validation-end and --test-end."
        ),
    )
    trader_report.add_argument(
        "--validation-end",
        help=(
            "YYYY-MM-DD, VALIDATION/TEST boundary (exclusive of TEST). Requires "
            "--train-end and --test-end."
        ),
    )
    trader_report.add_argument(
        "--test-end",
        help=(
            "YYYY-MM-DD, TEST/FORWARD boundary (exclusive of FORWARD; FORWARD is "
            "open-ended). Requires --train-end and --validation-end."
        ),
    )
```

Replace the `trader-report` dispatch block with:

```python
    elif args.command == "trader-report":
        engine = get_engine(args.db_path)
        session = get_session(engine)
        query = session.query(ReconstructedTrade).filter(ReconstructedTrade.trader == args.trader)
        if args.symbol:
            query = query.filter(ReconstructedTrade.symbol == args.symbol)
        trades = query.all()

        split_flags = (args.train_end, args.validation_end, args.test_end)
        if any(split_flags) and not all(split_flags):
            trader_report.error(
                "--train-end, --validation-end, and --test-end must be given together "
                "(all three or none)"
            )

        if all(split_flags):
            boundaries = PeriodBoundaries(
                train_end=datetime.fromisoformat(args.train_end).replace(tzinfo=UTC),
                validation_end=datetime.fromisoformat(args.validation_end).replace(tzinfo=UTC),
                test_end=datetime.fromisoformat(args.test_end).replace(tzinfo=UTC),
            )
            split_report = compute_split_report(trades, boundaries)
            print(format_split_report(split_report, args.trader))
        else:
            metrics = compute_metrics(trades)
            print(format_report(metrics, args.trader))
        return 0
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest research/tests/test_cli.py -v`
Expected: PASS, all tests

- [ ] **Step 5: Run the full targeted suite, lint, typecheck**

Run: `pytest research/tests/trader_mining/ research/tests/test_models.py research/tests/test_cli.py -q`
Run: `ruff check research/ && ruff format --check research/ && mypy research/`
Expected: all clean, no regressions vs. the count going into this task

- [ ] **Step 6: Commit**

```bash
git add research/cli.py research/tests/test_cli.py
git commit -m "feat(research): trader-report --train-end/--validation-end/--test-end split output"
```

---

### Task 5: Real-data validation, external cross-check, code review, PR

- [ ] **Step 1: Real-data run**

Against `user_data/trader_mining_scratch.sqlite` -- a fixed, explicitly-named path, never
the default `user_data/research.sqlite` (which already holds real `gate` promotion/health
state) -- pick a wallet already ingested/reconstructed from an earlier release, then:

```bash
python -m research.cli trader-report --trader <WALLET> \
  --db-path user_data/trader_mining_scratch.sqlite \
  --train-end 2025-01-01 --validation-end 2025-07-01 --test-end 2026-01-01
```

(Already covered by `.gitignore`'s existing `*.sqlite`/`user_data/*` patterns -- no new
ignore rule needed. Named, not `<SCRATCH_DB>`, specifically so nobody has to invent a path
under time pressure and risk defaulting to the real one.)

Confirm: no crash; each period's sample count plus `n_straddling` reconciles against the
wallet's known total trade count; no `None` printed as the string `"None"`; boundary dates
print correctly. Then run the same command *without* the three flags and diff the output
against Release 3's previously-validated plain output to confirm it is unchanged.

- [ ] **Step 2: External cross-check (needs the user)**

Confirm the whole-history section's numbers still match the previously-validated Release 3
cross-check (this release must not have altered whole-history math), and sanity-check that
TRAIN+VALIDATION+TEST+FORWARD trade counts reconcile against the wallet's known total trade
count/date range.

- [ ] **Step 3: Code review**

Dispatch a code-review subagent against the full diff, following
`superpowers:requesting-code-review`. Ask it to independently re-verify: the half-open
boundary semantics at each of the 8 parametrized edge dates in Task 2's test; that
`assign_period` reads only `entry_timestamp` everywhere (grep for any stray
`exit_timestamp` use in period-assignment code paths); that `_to_naive_utc` converts-then-
strips rather than blindly discarding a non-UTC offset; that no auto-rejection/threshold/
pass-fail language exists anywhere in `split_report.py`'s output; and that the real-SQLite
round-trip test in Task 2 actually reproduces a naive read (not accidentally testing against
an already-naive in-memory object).

- [ ] **Step 4: Address findings, open PR**

Fix Critical/Important findings via TDD. PR body: Summary, Design (link this plan's spec),
real-data validation + cross-check outcome, code review findings addressed, Testing.

- [ ] **Step 5: Watch CI, merge**

Arm a Monitor on `gh pr checks`. Diagnose real-vs-known-flake before re-running (FIELD-NOTES.md's
documented flake category). Merge when green.

## Self-review notes (for the implementer)

- Every task's code blocks are complete and copy-pasteable, but double-check against the
  actual current file state before pasting, in case an earlier task's own file changed
  since this plan was written.
- `PeriodBoundaries` is `frozen=True` but `__post_init__` still normalizes its fields --
  this requires `object.__setattr__`, not plain attribute assignment (a frozen dataclass
  raises `FrozenInstanceError` on `self.train_end = ...`). This is intentional, not a bug
  to "fix" by removing `frozen=True`.
- `SplitReport.periods[0]`/`[1]` are always TRAIN/VALIDATION respectively because
  `compute_split_report` builds them in `PERIODS` order -- don't reorder that list without
  updating `format_split_report`'s degradation-diagnostic indexing.
