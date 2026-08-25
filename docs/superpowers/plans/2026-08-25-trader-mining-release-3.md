# Trader/Wallet Mining Release 3 (Performance Metrics + Report) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a wallet's `ReconstructedTrade` rows into a readable performance report --
`research/trader_mining/metrics.py` (new) plus a `trader-report WALLET` CLI subcommand.

**Architecture:** One new pure module. `compute_metrics(trades) -> WalletMetrics` is a
DB-free function (mirrors `reconstruct_trades`' own precedent); `research/cli.py`'s new
`trader-report` subcommand is the only place that touches the database.

**Tech Stack:** Python stdlib only (`statistics`, `math`) -- no new dependency.

**Spec:** `docs/superpowers/specs/2026-08-25-trader-mining-release-3-design.md`

## Global Constraints

- `compute_metrics` and every helper it calls take `list[ReconstructedTrade]` and return
  plain values -- no `Session`, no DB import anywhere in `metrics.py`.
- Every `Optional` field is `None`, never `0.0` or `NaN`, when undefined (0 trades, stdev
  undefined, division by zero) -- a `0.0` for an undefined risk-adjusted-shaped number reads
  as "bad", not "undefined", and would be actively misleading.
- Never use the words "Sharpe" or "risk-adjusted" for `trade_consistency_score`, or "Calmar"
  for `return_to_drawdown_ratio`, anywhere -- docstrings, report labels, variable names,
  commit messages included. See the spec's "What this is not" section for why.
- `max_drawdown` needs NO special-casing for 0 or 1 trades -- computed naturally from an
  initial peak of `0.0`: 0 trades -> loop never runs -> `0.0`; exactly 1 losing trade ->
  correctly produces a nonzero drawdown equal to that loss (verified by hand in Task 2 --
  the spec's own draft wording claiming "0.0 for zero or one trade" is corrected by this
  plan, not carried forward as written).

---

### Task 1: `WalletMetrics` dataclass + aggregate metrics

**Files:**
- Create: `research/trader_mining/metrics.py`
- Test: `research/tests/trader_mining/test_metrics.py`

**Interfaces:**
- Produces: `research.trader_mining.metrics.WalletMetrics` (dataclass, ALL fields declared
  in this task even though later tasks populate more of them -- see full field list below)
- Produces: `research.trader_mining.metrics.compute_metrics(trades: list[ReconstructedTrade]) -> WalletMetrics`

Full `WalletMetrics` field list (declared now, all populated by the end of Task 3):

```python
@dataclass
class WalletMetrics:
    trade_count: int
    total_volume: float
    gross_pnl: float
    fees: float
    net_pnl: float
    win_rate: float | None
    avg_win: float | None
    avg_loss: float | None
    profit_factor: float | None
    expectancy: float | None
    payoff_ratio: float | None
    median_trade_return: float | None
    avg_holding_period_seconds: float | None
    median_holding_period_seconds: float | None
    long_count: int
    short_count: int
    long_pct: float | None
    symbol_concentration: float | None
    max_drawdown: float
    max_losing_streak: int
    pnl_concentration_top_5: float | None
    trade_consistency_score: float | None
    return_to_drawdown_ratio: float | None
```

This task implements and tests: `trade_count`, `total_volume`, `gross_pnl`, `fees`,
`net_pnl`, `win_rate`, `avg_win`, `avg_loss`, `profit_factor`, `expectancy`,
`payoff_ratio`. Every other field is set to its "undefined" value (`None` for `Optional`
fields, `0` / `0.0` for `int`/`float` fields) in this task -- Tasks 2-3 replace those with
real computations.

- [ ] **Step 1: Write the failing tests**

```python
# research/tests/trader_mining/test_metrics.py
from datetime import UTC, datetime

from research.models import ReconstructedTrade
from research.trader_mining.metrics import compute_metrics


def _trade(
    net_pnl,
    gross_pnl=None,
    fees=0.0,
    entry_price=100.0,
    quantity=1.0,
    direction="long",
    symbol="BTC/USDC:USDC",
    exit_ts=datetime(2026, 1, 1, tzinfo=UTC),
    holding_seconds=3600.0,
) -> ReconstructedTrade:
    return ReconstructedTrade(
        trader="0xAAA",
        symbol=symbol,
        direction=direction,
        entry_timestamp=exit_ts,
        entry_price=entry_price,
        exit_timestamp=exit_ts,
        exit_price=entry_price,
        quantity=quantity,
        gross_pnl=gross_pnl if gross_pnl is not None else net_pnl + fees,
        fees=fees,
        net_pnl=net_pnl,
        holding_time_seconds=holding_seconds,
        n_fills=2,
        is_truncated_start=False,
        was_liquidated=False,
    )


def test_zero_trades_returns_all_undefined():
    m = compute_metrics([])

    assert m.trade_count == 0
    assert m.total_volume == 0.0
    assert m.gross_pnl == 0.0
    assert m.net_pnl == 0.0
    assert m.win_rate is None
    assert m.avg_win is None
    assert m.avg_loss is None
    assert m.profit_factor is None
    assert m.expectancy is None
    assert m.payoff_ratio is None


def test_aggregate_metrics_two_wins_one_loss():
    trades = [
        _trade(net_pnl=100.0, fees=1.0, entry_price=100.0, quantity=1.0),  # volume 100
        _trade(net_pnl=50.0, fees=0.5, entry_price=200.0, quantity=1.0),  # volume 200
        _trade(net_pnl=-30.0, fees=0.3, entry_price=100.0, quantity=1.0),  # volume 100
    ]

    m = compute_metrics(trades)

    assert m.trade_count == 3
    assert m.total_volume == 400.0  # 100 + 200 + 100
    assert m.net_pnl == 120.0  # 100 + 50 - 30
    assert m.fees == 1.8
    assert m.gross_pnl == 121.8  # net + fees, by _trade's own default
    assert m.win_rate == 2 / 3
    assert m.avg_win == 75.0  # (100 + 50) / 2
    assert m.avg_loss == -30.0  # only one loss
    assert m.profit_factor == 150.0 / 30.0  # 5.0 -- sum(wins)/abs(sum(losses))
    assert m.expectancy == 40.0  # 120 / 3
    assert m.payoff_ratio == 75.0 / 30.0  # 2.5 -- avg_win / abs(avg_loss)


def test_all_winning_has_no_avg_loss_or_profit_factor():
    trades = [_trade(net_pnl=10.0), _trade(net_pnl=20.0)]

    m = compute_metrics(trades)

    assert m.win_rate == 1.0
    assert m.avg_loss is None
    assert m.profit_factor is None  # no losses to divide by -- not infinity, not 0
    assert m.payoff_ratio is None


def test_all_losing_has_no_avg_win_or_payoff_ratio():
    trades = [_trade(net_pnl=-10.0), _trade(net_pnl=-20.0)]

    m = compute_metrics(trades)

    assert m.win_rate == 0.0
    assert m.avg_win is None
    assert m.profit_factor is None
    assert m.payoff_ratio is None


def test_breakeven_trade_counts_toward_trades_but_is_not_a_win_or_loss():
    trades = [_trade(net_pnl=10.0), _trade(net_pnl=0.0), _trade(net_pnl=-10.0)]

    m = compute_metrics(trades)

    assert m.trade_count == 3
    assert m.win_rate == 1 / 3  # only the +10 trade counts as a win
    assert m.avg_win == 10.0
    assert m.avg_loss == -10.0
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest research/tests/trader_mining/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.trader_mining.metrics'`

- [ ] **Step 3: Implement `WalletMetrics` and `compute_metrics`**

```python
# research/trader_mining/metrics.py
"""Performance metrics for a wallet's reconstructed trades. compute_metrics is pure and
DB-free, mirroring research.trader_mining.engine.reconstruct_trades' own "pure function,
no DB access" precedent -- research/cli.py's trader-report subcommand is the only caller
that touches a database.

trade_consistency_score is NOT a Sharpe ratio and NOT risk-adjusted -- ReconstructedTrade
carries no per-trade stop-distance/initial-risk data, so a true R-multiple-based measure
can't be computed honestly here. It's an SQN-shaped statistic over return-on-notional; see
its own docstring and docs/superpowers/specs/2026-08-25-trader-mining-release-3-design.md.
sqrt(N) gives it a t-statistic-like shape but is NOT a real significance test -- trade
outcomes can be autocorrelated or regime-dependent; real out-of-sample validation is
Release 4's job (the TRAIN/VALIDATION/TEST/FORWARD framework), not this module's.
"""

from __future__ import annotations

from dataclasses import dataclass

from research.models import ReconstructedTrade


@dataclass
class WalletMetrics:
    trade_count: int
    total_volume: float
    gross_pnl: float
    fees: float
    net_pnl: float
    win_rate: float | None
    avg_win: float | None
    avg_loss: float | None
    profit_factor: float | None
    expectancy: float | None
    payoff_ratio: float | None
    median_trade_return: float | None
    avg_holding_period_seconds: float | None
    median_holding_period_seconds: float | None
    long_count: int
    short_count: int
    long_pct: float | None
    symbol_concentration: float | None
    max_drawdown: float
    max_losing_streak: int
    pnl_concentration_top_5: float | None
    trade_consistency_score: float | None
    return_to_drawdown_ratio: float | None


def compute_metrics(trades: list[ReconstructedTrade]) -> WalletMetrics:
    n = len(trades)
    if n == 0:
        return WalletMetrics(
            trade_count=0,
            total_volume=0.0,
            gross_pnl=0.0,
            fees=0.0,
            net_pnl=0.0,
            win_rate=None,
            avg_win=None,
            avg_loss=None,
            profit_factor=None,
            expectancy=None,
            payoff_ratio=None,
            median_trade_return=None,
            avg_holding_period_seconds=None,
            median_holding_period_seconds=None,
            long_count=0,
            short_count=0,
            long_pct=None,
            symbol_concentration=None,
            max_drawdown=0.0,
            max_losing_streak=0,
            pnl_concentration_top_5=None,
            trade_consistency_score=None,
            return_to_drawdown_ratio=None,
        )

    total_volume = sum(t.entry_price * t.quantity for t in trades)
    gross_pnl = sum(t.gross_pnl for t in trades)
    fees = sum(t.fees for t in trades)
    net_pnl = sum(t.net_pnl for t in trades)

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl < 0]

    win_rate = len(wins) / n
    avg_win = (sum(t.net_pnl for t in wins) / len(wins)) if wins else None
    avg_loss = (sum(t.net_pnl for t in losses) / len(losses)) if losses else None
    profit_factor = (
        sum(t.net_pnl for t in wins) / abs(sum(t.net_pnl for t in losses)) if losses else None
    )
    expectancy = net_pnl / n
    payoff_ratio = (
        avg_win / abs(avg_loss) if (avg_win is not None and avg_loss is not None) else None
    )

    return WalletMetrics(
        trade_count=n,
        total_volume=total_volume,
        gross_pnl=gross_pnl,
        fees=fees,
        net_pnl=net_pnl,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        expectancy=expectancy,
        payoff_ratio=payoff_ratio,
        median_trade_return=None,
        avg_holding_period_seconds=None,
        median_holding_period_seconds=None,
        long_count=0,
        short_count=0,
        long_pct=None,
        symbol_concentration=None,
        max_drawdown=0.0,
        max_losing_streak=0,
        pnl_concentration_top_5=None,
        trade_consistency_score=None,
        return_to_drawdown_ratio=None,
    )
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest research/tests/trader_mining/test_metrics.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add research/trader_mining/metrics.py research/tests/trader_mining/test_metrics.py
git commit -m "feat(research): WalletMetrics + aggregate performance metrics"
```

---

### Task 2: Distribution + sequence-dependent metrics

**Files:**
- Modify: `research/trader_mining/metrics.py`
- Test: `research/tests/trader_mining/test_metrics.py`

**Interfaces:**
- Extends `compute_metrics` to populate: `median_trade_return`, `avg_holding_period_seconds`,
  `median_holding_period_seconds`, `long_count`, `short_count`, `long_pct`,
  `symbol_concentration`, `max_drawdown`, `max_losing_streak`, `pnl_concentration_top_5`.

- [ ] **Step 1: Write the failing tests**

```python
# research/tests/trader_mining/test_metrics.py -- add to the existing file
import statistics
from datetime import timedelta


def test_distribution_metrics():
    trades = [
        _trade(net_pnl=10.0, entry_price=100.0, quantity=1.0, direction="long", symbol="A"),
        _trade(net_pnl=20.0, entry_price=100.0, quantity=1.0, direction="short", symbol="A"),
        _trade(net_pnl=-5.0, entry_price=100.0, quantity=1.0, direction="long", symbol="B"),
        _trade(
            net_pnl=15.0,
            entry_price=100.0,
            quantity=1.0,
            direction="long",
            symbol="A",
            holding_seconds=7200.0,
        ),
    ]
    # per-trade returns (net_pnl / (entry_price*quantity)): 0.10, 0.20, -0.05, 0.15
    # median of [0.10, 0.20, -0.05, 0.15] sorted: [-0.05, 0.10, 0.15, 0.20] -> (0.10+0.15)/2

    m = compute_metrics(trades)

    assert m.median_trade_return == statistics.median([0.10, 0.20, -0.05, 0.15])
    assert m.avg_holding_period_seconds == (3600.0 * 3 + 7200.0) / 4
    assert m.median_holding_period_seconds == statistics.median(
        [3600.0, 3600.0, 3600.0, 7200.0]
    )
    assert m.long_count == 3
    assert m.short_count == 1
    assert m.long_pct == 0.75
    assert m.symbol_concentration == 0.75  # "A" appears in 3 of 4 trades


def test_max_drawdown_and_losing_streak_ordered_by_exit_timestamp():
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    trades = [
        _trade(net_pnl=100.0, exit_ts=t0),  # cumulative 100, peak 100
        _trade(net_pnl=-30.0, exit_ts=t0 + timedelta(hours=1)),  # cumulative 70, dd 30
        _trade(net_pnl=-20.0, exit_ts=t0 + timedelta(hours=2)),  # cumulative 50, dd 50
        _trade(net_pnl=60.0, exit_ts=t0 + timedelta(hours=3)),  # cumulative 110, new peak
    ]

    m = compute_metrics(trades)

    assert m.max_drawdown == 50.0  # peak 100 down to a low of 50
    assert m.max_losing_streak == 2  # the two consecutive losses


def test_single_losing_trade_produces_nonzero_drawdown():
    """max_drawdown needs no 0/1-trade special case -- a lone losing trade is itself a
    real drawdown from the implicit starting peak of 0.0."""
    m = compute_metrics([_trade(net_pnl=-40.0)])

    assert m.max_drawdown == 40.0


def test_single_winning_trade_has_zero_drawdown():
    m = compute_metrics([_trade(net_pnl=40.0)])

    assert m.max_drawdown == 0.0


def test_pnl_concentration_top_5_sums_largest_winners_over_total_net_pnl():
    trades = [_trade(net_pnl=v) for v in [50.0, 40.0, 30.0, 20.0, 10.0, 5.0, -20.0]]
    # total_net_pnl = 50+40+30+20+10+5-20 = 135
    # top 5 winners: 50+40+30+20+10 = 150

    m = compute_metrics(trades)

    assert m.pnl_concentration_top_5 == 150.0 / 135.0


def test_pnl_concentration_top_5_is_none_when_total_net_pnl_is_zero():
    trades = [_trade(net_pnl=50.0), _trade(net_pnl=-50.0)]

    m = compute_metrics(trades)

    assert m.pnl_concentration_top_5 is None
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest research/tests/trader_mining/test_metrics.py -v`
Expected: FAIL -- assertions against `None`/`0.0` placeholder values from Task 1

- [ ] **Step 3: Implement the distribution + sequence metrics**

Add near the top of `metrics.py`:

```python
import statistics
```

Replace the placeholder values in `compute_metrics`' return (the non-zero-trade branch)
with real computations, added just before the `return WalletMetrics(...)` call:

```python
    returns = [t.net_pnl / (t.entry_price * t.quantity) for t in trades]
    median_trade_return = statistics.median(returns)

    holding_periods = [t.holding_time_seconds for t in trades]
    avg_holding_period_seconds = sum(holding_periods) / n
    median_holding_period_seconds = statistics.median(holding_periods)

    long_count = sum(1 for t in trades if t.direction == "long")
    short_count = n - long_count
    long_pct = long_count / n

    symbol_counts: dict[str, int] = {}
    for t in trades:
        symbol_counts[t.symbol] = symbol_counts.get(t.symbol, 0) + 1
    symbol_concentration = max(symbol_counts.values()) / n

    ordered_by_exit = sorted(trades, key=lambda t: t.exit_timestamp)
    cumulative = 0.0
    peak = 0.0
    drawdown = 0.0
    losing_streak = 0
    longest_losing_streak = 0
    for t in ordered_by_exit:
        cumulative += t.net_pnl
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
        if t.net_pnl < 0:
            losing_streak += 1
            longest_losing_streak = max(longest_losing_streak, losing_streak)
        else:
            losing_streak = 0

    total_net_pnl_for_concentration = sum(t.net_pnl for t in trades)
    winners_sorted = sorted((t.net_pnl for t in trades if t.net_pnl > 0), reverse=True)
    pnl_concentration_top_5 = (
        sum(winners_sorted[:5]) / total_net_pnl_for_concentration
        if total_net_pnl_for_concentration != 0
        else None
    )
```

Then update the `WalletMetrics(...)` construction to use these instead of the Task 1
placeholders (`median_trade_return=median_trade_return`,
`avg_holding_period_seconds=avg_holding_period_seconds`,
`median_holding_period_seconds=median_holding_period_seconds`, `long_count=long_count`,
`short_count=short_count`, `long_pct=long_pct`,
`symbol_concentration=symbol_concentration`, `max_drawdown=drawdown`,
`max_losing_streak=longest_losing_streak`,
`pnl_concentration_top_5=pnl_concentration_top_5`).

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest research/tests/trader_mining/test_metrics.py -v`
Expected: PASS (12 tests total)

- [ ] **Step 5: Commit**

```bash
git add research/trader_mining/metrics.py research/tests/trader_mining/test_metrics.py
git commit -m "feat(research): distribution and sequence-dependent wallet metrics"
```

---

### Task 3: `trade_consistency_score` + `return_to_drawdown_ratio`

**Files:**
- Modify: `research/trader_mining/metrics.py`
- Test: `research/tests/trader_mining/test_metrics.py`

**Interfaces:**
- Extends `compute_metrics` to populate: `trade_consistency_score`,
  `return_to_drawdown_ratio`.

This is the statistically sensitive part of this release -- every edge case below must be
tested, not assumed.

- [ ] **Step 1: Write the failing tests**

```python
# research/tests/trader_mining/test_metrics.py -- add to the existing file
import math


def test_trade_consistency_score_is_none_for_zero_or_one_trade():
    assert compute_metrics([]).trade_consistency_score is None
    assert compute_metrics([_trade(net_pnl=10.0)]).trade_consistency_score is None


def test_trade_consistency_score_is_none_when_all_returns_identical():
    """Zero-variance guard -- every trade has the exact same return-on-notional. A real,
    if unusual, case (e.g. a market maker collecting an identical spread every trade), not
    just a theoretical one. Must not raise ZeroDivisionError/StatisticsError."""
    trades = [
        _trade(net_pnl=10.0, entry_price=100.0, quantity=1.0),
        _trade(net_pnl=10.0, entry_price=100.0, quantity=1.0),
        _trade(net_pnl=10.0, entry_price=100.0, quantity=1.0),
    ]

    m = compute_metrics(trades)

    assert m.trade_consistency_score is None


def test_trade_consistency_score_hand_computed():
    # returns (net_pnl / (entry_price*quantity)): 0.10, 0.20, -0.05
    trades = [
        _trade(net_pnl=10.0, entry_price=100.0, quantity=1.0),
        _trade(net_pnl=20.0, entry_price=100.0, quantity=1.0),
        _trade(net_pnl=-5.0, entry_price=100.0, quantity=1.0),
    ]
    returns = [0.10, 0.20, -0.05]
    expected_mean = statistics.mean(returns)
    expected_stdev = statistics.stdev(returns)  # sample stdev, N-1 denominator
    expected = math.sqrt(3) * expected_mean / expected_stdev

    m = compute_metrics(trades)

    assert m.trade_consistency_score == pytest.approx(expected)


def test_return_to_drawdown_ratio_is_none_when_no_drawdown():
    m = compute_metrics([_trade(net_pnl=10.0)])  # single winner -- max_drawdown is 0.0

    assert m.max_drawdown == 0.0
    assert m.return_to_drawdown_ratio is None


def test_return_to_drawdown_ratio_hand_computed():
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    trades = [
        _trade(net_pnl=100.0, exit_ts=t0),
        _trade(net_pnl=-40.0, exit_ts=t0 + timedelta(hours=1)),
    ]
    # net_pnl total = 60.0, max_drawdown = 40.0 (peak 100 down to a low of 60)

    m = compute_metrics(trades)

    assert m.max_drawdown == 40.0
    assert m.return_to_drawdown_ratio == pytest.approx(60.0 / 40.0)
```

(add `import pytest` and `import math` to the test file if not already present)

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest research/tests/trader_mining/test_metrics.py -v`
Expected: FAIL -- `trade_consistency_score`/`return_to_drawdown_ratio` still `None` from
the Task 1/2 placeholders in the non-zero branch

- [ ] **Step 3: Implement**

Add `import math` near the top of `metrics.py`. Add, just before the final
`WalletMetrics(...)` return in `compute_metrics`:

```python
    if n < 2:
        trade_consistency_score = None
    else:
        stdev_r = statistics.stdev(returns)
        trade_consistency_score = (
            None if stdev_r == 0 else math.sqrt(n) * statistics.mean(returns) / stdev_r
        )

    return_to_drawdown_ratio = net_pnl / drawdown if drawdown > 0 else None
```

Update the `WalletMetrics(...)` construction:
`trade_consistency_score=trade_consistency_score`,
`return_to_drawdown_ratio=return_to_drawdown_ratio`.

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest research/tests/trader_mining/test_metrics.py -v`
Expected: PASS (17 tests total)

- [ ] **Step 5: Commit**

```bash
git add research/trader_mining/metrics.py research/tests/trader_mining/test_metrics.py
git commit -m "feat(research): trade_consistency_score and return_to_drawdown_ratio"
```

---

### Task 4: `format_report`

**Files:**
- Modify: `research/trader_mining/metrics.py`
- Test: `research/tests/trader_mining/test_metrics.py`

**Interfaces:**
- Produces: `research.trader_mining.metrics.format_report(metrics: WalletMetrics, trader: str) -> str`

- [ ] **Step 1: Write the failing tests**

```python
# research/tests/trader_mining/test_metrics.py -- add to the existing file
from research.trader_mining.metrics import format_report


def test_format_report_includes_trader_and_key_numbers():
    trades = [_trade(net_pnl=100.0), _trade(net_pnl=-30.0)]
    m = compute_metrics(trades)

    report = format_report(m, "0xAAA")

    assert "0xAAA" in report
    assert "2" in report  # trade count appears somewhere
    assert "70.0" in report or "70.00" in report  # net_pnl = 70


def test_format_report_prints_na_for_undefined_metrics_not_zero():
    m = compute_metrics([])  # zero trades -- everything Optional is None

    report = format_report(m, "0xAAA")

    assert "n/a" in report
    assert "None" not in report  # never leak Python's None repr into the report


def test_format_report_labels_trade_consistency_score_honestly():
    """The report text itself, not just docstrings, must not call this Sharpe or
    risk-adjusted -- someone reading only the printed report needs the caveat too."""
    m = compute_metrics([_trade(net_pnl=10.0), _trade(net_pnl=20.0), _trade(net_pnl=-5.0)])

    report = format_report(m, "0xAAA")

    assert "sharpe" not in report.lower()
    assert "risk-adjusted" not in report.lower()
    assert "consistency" in report.lower()


def test_format_report_labels_drawdown_and_ratio_honestly():
    m = compute_metrics([_trade(net_pnl=10.0), _trade(net_pnl=-5.0)])

    report = format_report(m, "0xAAA")

    assert "calmar" not in report.lower()
    assert "mark-to-market" in report.lower() or "closed-trade" in report.lower()
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest research/tests/trader_mining/test_metrics.py -k format_report -v`
Expected: FAIL with `ImportError: cannot import name 'format_report'`

- [ ] **Step 3: Implement `format_report`**

```python
# research/trader_mining/metrics.py -- add at the end of the file

def _fmt(value: float | None, spec: str = "{:.4f}") -> str:
    return "n/a" if value is None else spec.format(value)


def format_report(metrics: WalletMetrics, trader: str) -> str:
    """Plain console/Markdown text, per the proposal review notes' explicit
    simplification (no separate reports.py module, no templating engine)."""
    lines = [
        f"## Wallet Report: {trader}",
        "",
        f"- Trades: {metrics.trade_count}",
        f"- Total volume: {_fmt(metrics.total_volume, '{:.2f}')}",
        f"- Gross P&L: {_fmt(metrics.gross_pnl, '{:.2f}')}",
        f"- Fees: {_fmt(metrics.fees, '{:.2f}')}",
        f"- Net P&L: {_fmt(metrics.net_pnl, '{:.2f}')}",
        f"- Win rate: {_fmt(metrics.win_rate, '{:.1%}')}",
        f"- Avg win: {_fmt(metrics.avg_win, '{:.2f}')}",
        f"- Avg loss: {_fmt(metrics.avg_loss, '{:.2f}')}",
        f"- Profit factor: {_fmt(metrics.profit_factor)}",
        f"- Expectancy: {_fmt(metrics.expectancy, '{:.2f}')}",
        f"- Payoff ratio: {_fmt(metrics.payoff_ratio)}",
        f"- Median trade return: {_fmt(metrics.median_trade_return, '{:.2%}')}",
        f"- Avg holding period: {_fmt(metrics.avg_holding_period_seconds, '{:.0f}')}s",
        f"- Median holding period: {_fmt(metrics.median_holding_period_seconds, '{:.0f}')}s",
        f"- Long/short: {metrics.long_count}/{metrics.short_count}",
        f"- Symbol concentration: {_fmt(metrics.symbol_concentration, '{:.1%}')}",
        f"- Max losing streak: {metrics.max_losing_streak}",
        f"- P&L concentration (top 5 winners): {_fmt(metrics.pnl_concentration_top_5, '{:.1%}')}",
        f"- Closed-trade P&L drawdown: {_fmt(metrics.max_drawdown, '{:.2f}')} "
        "(absolute $, NOT mark-to-market portfolio drawdown -- overlapping/open positions "
        "aren't captured)",
        f"- Return/drawdown ratio: {_fmt(metrics.return_to_drawdown_ratio)} "
        "(not annualized -- not Calmar)",
        f"- Trade consistency score: {_fmt(metrics.trade_consistency_score)} "
        "(NOT Sharpe/risk-adjusted -- sqrt(N)*mean(r)/stdev(r) over return-on-notional; "
        "not a real significance test)",
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest research/tests/trader_mining/test_metrics.py -v`
Expected: PASS, all tests (21 total)

- [ ] **Step 5: Commit**

```bash
git add research/trader_mining/metrics.py research/tests/trader_mining/test_metrics.py
git commit -m "feat(research): format_report for wallet metrics"
```

---

### Task 5: `trader-report` CLI subcommand

**Files:**
- Modify: `research/cli.py`
- Test: `research/tests/test_cli.py`

**Interfaces:**
- Consumes: `compute_metrics`, `format_report` (Tasks 1-4), `ReconstructedTrade`

- [ ] **Step 1: Write the failing test**

```python
# research/tests/test_cli.py -- add to the existing file
def test_trader_report_command_prints_formatted_report(mocker, capsys):
    from research.trader_mining.metrics import WalletMetrics

    canned = WalletMetrics(
        trade_count=2,
        total_volume=300.0,
        gross_pnl=71.0,
        fees=1.0,
        net_pnl=70.0,
        win_rate=0.5,
        avg_win=100.0,
        avg_loss=-30.0,
        profit_factor=100.0 / 30.0,
        expectancy=35.0,
        payoff_ratio=100.0 / 30.0,
        median_trade_return=0.15,
        avg_holding_period_seconds=3600.0,
        median_holding_period_seconds=3600.0,
        long_count=2,
        short_count=0,
        long_pct=1.0,
        symbol_concentration=1.0,
        max_drawdown=30.0,
        max_losing_streak=1,
        pnl_concentration_top_5=1.0,
        trade_consistency_score=1.2,
        return_to_drawdown_ratio=70.0 / 30.0,
    )
    mock_query = mocker.MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.all.return_value = []  # trades themselves are irrelevant -- compute_metrics is mocked
    mock_session = mocker.MagicMock()
    mock_session.query.return_value = mock_query
    mocker.patch("research.cli.get_engine")
    mocker.patch("research.cli.get_session", return_value=mock_session)
    mocker.patch("research.cli.compute_metrics", return_value=canned)

    exit_code = main(
        ["trader-report", "--trader", "0x0000000000000000000000000000000000000000"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "0x0000000000000000000000000000000000000000" in captured.out
    assert "70.00" in captured.out
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest research/tests/test_cli.py::test_trader_report_command_prints_formatted_report -v`
Expected: FAIL -- `error: argument command: invalid choice: 'trader-report'`

- [ ] **Step 3: Wire `cli.py`**

```python
# research/cli.py -- add import
from research.trader_mining.metrics import compute_metrics, format_report
```

Add a new subparser alongside `trader_analyze`:

```python
    trader_report = sub.add_parser(
        "trader-report", help="Print a performance report for a wallet's reconstructed trades"
    )
    trader_report.add_argument("--trader", required=True, help="Wallet address")
    trader_report.add_argument("--symbol", help="Limit to one symbol (default: all)")
    trader_report.add_argument("--db-path", default="user_data/research.sqlite")
```

Add a new branch in `main`'s dispatch, after the `trader-analyze` branch:

```python
    elif args.command == "trader-report":
        engine = get_engine(args.db_path)
        session = get_session(engine)
        query = session.query(ReconstructedTrade).filter(ReconstructedTrade.trader == args.trader)
        if args.symbol:
            query = query.filter(ReconstructedTrade.symbol == args.symbol)
        trades = query.all()
        metrics = compute_metrics(trades)
        print(format_report(metrics, args.trader))
        return 0
```

`ReconstructedTrade` needs importing in `cli.py` if not already (it isn't -- `cli.py`
currently only imports `reconstruct_and_persist_trades`, not the model itself):

```python
from research.models import ReconstructedTrade
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest research/tests/test_cli.py -v`
Expected: PASS, all tests

- [ ] **Step 5: Run the full targeted suite, lint, typecheck**

Run: `pytest research/tests/trader_mining/ research/tests/test_models.py research/tests/test_cli.py -q`
Run: `ruff check research/ && ruff format --check research/ && mypy research/`
Expected: all clean, no regressions vs. the count going into this task

- [ ] **Step 6: Commit**

```bash
git add research/cli.py research/tests/test_cli.py
git commit -m "feat(research): add trader-report CLI subcommand"
```

---

### Task 6: Real-data validation, external cross-check, code review, PR

- [ ] **Step 1: Real-data run**

Against a scratch db path (not the repo's own `user_data/research.sqlite`), pick a real
wallet already ingested/reconstructed in Release 2 or the ledger-reconciliation work (or
run `trader-import`/`trader-analyze` fresh for one), then:

```bash
python -m research.cli trader-report --trader <WALLET> --db-path <SCRATCH_DB>
```

Sanity-check the printed numbers are plausible (no crash, no `None` printed as the string
`"None"`, win rate between 0-100%, etc.) -- this is a supplementary integration check, not
a substitute for Tasks 1-4's hand-computed unit tests.

- [ ] **Step 2: External cross-check (needs the user)**

Ask the user for independent, externally-sourced reference numbers for the chosen wallet
(Hyperliquid's leaderboard-displayed PNL/ROI/Volume at minimum). Compare against this
release's computed `net_pnl`/`total_volume`. Do not expect exact equality -- reconcile any
delta explicitly against the known, legitimate sources of divergence (unrealized P&L on
open positions, funding payments, truncated/unreconciled ingestion history) before
concluding anything is a bug. If the delta isn't explained by one of those, stop and treat
it as a real bug -- don't guess past it.

- [ ] **Step 3: Code review**

Dispatch a code-review subagent against the full diff (base = `develop` at this branch's
fork point, head = current tip), following `superpowers:requesting-code-review`. Ask it to
independently re-verify at least: the `trade_consistency_score` formula and its zero-
variance/N<2 guards (hand-derive an example, don't trust the tests' own claimed expected
values), the `max_drawdown`/`max_losing_streak` sequencing logic against a hand-traced
example, and whether any report label still implies "Sharpe"/"risk-adjusted"/"Calmar"
despite the explicit ban on those words.

- [ ] **Step 4: Address findings, open PR**

Fix Critical/Important findings via TDD, same as every prior release. Open a PR the same
way #14/#15/#16/#17/#18 were structured -- Summary, Design, real-data validation
(including the external cross-check's outcome), code review findings addressed, Testing.

- [ ] **Step 5: Watch CI, merge**

Arm a Monitor on `gh pr checks`. Diagnose any failure as real-vs-known-flake before
chasing it (FIELD-NOTES.md's documented `--random-order -n auto` category). Merge when
green or only a documented pre-existing flake remains -- no branch protection blocks this.

## Self-review notes (for the implementer)

- Every task's code blocks are complete and copy-pasteable, but double-check against the
  actual current file state before pasting, in case an earlier task's own file changed
  since this plan was written.
- Task 1's `WalletMetrics` dataclass declares all 23 fields up front, but only ~11 are
  populated with real logic by the end of Task 1 -- the rest stay at their explicit
  "undefined" placeholder value until Tasks 2-3 replace them. This is normal incremental
  build-up, not a plan placeholder violation (every VALUE used in Task 1's own tests is
  concrete, hand-computed, and asserted).
