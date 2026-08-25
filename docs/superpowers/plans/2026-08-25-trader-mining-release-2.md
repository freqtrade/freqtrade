# Trader/Wallet Mining Release 2 (Trade Reconstruction) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Group a wallet's already-ingested Hyperliquid fills (Release 1's `NormalizedFill`
rows) into logical `ReconstructedTrade` records, following the exchange's own position
transitions rather than an imposed lot-accounting convention.

**Architecture:** One new module, `research/trader_mining/engine.py`, with a pure
`reconstruct_trades` function (the core zero-crossing/reversal algorithm, no DB access) and
a thin `reconstruct_and_persist_trades` orchestration function (queries fills, calls the
pure function per symbol, deletes+reinserts `ReconstructedTrade` rows). `research/models.py`
gains the `ReconstructedTrade` table; `research/cli.py` gains a `trader-analyze` subcommand.

**Tech Stack:** Python, `decimal.Decimal` for all position/price/pnl/fee arithmetic (never
float, for exact zero-boundary detection), SQLAlchemy 2.0 declarative (existing shared
`Base`), pytest with hand-built `NormalizedFill` fixtures (no DB for the core algorithm's
own tests).

**Spec:** `docs/superpowers/specs/2026-08-25-trader-mining-release-2-design.md`

## Global Constraints

- `reconstruct_trades(trader: str, symbol: str, fills: list[NormalizedFill]) ->
  list[ReconstructedTrade]` -- pure function. `fills` must already be sorted by
  `(timestamp, tid)` by the caller; not re-sorted internally.
- All position/price/quantity/pnl/fee values are converted to `Decimal(str(x))` at the
  point of use -- never raw float arithmetic. This matters specifically because the
  zero-crossing check (`running_position == Decimal("0")`) is exact-equality-sensitive.
- Trade boundary: **zero-crossing** (flat-to-flat). A trade opens when position moves from
  zero to non-zero, closes when it returns to exactly zero.
- **Bootstrap**: `running_position` starts from `fills[0].position` (Hyperliquid's own
  `startPosition`), not assumed zero. If non-zero, that trade gets
  `is_truncated_start=True`.
- **Reversal** (a single fill whose signed quantity crosses zero and lands on the opposite
  sign): split into a closing leg (`quantity = abs(running_position)`, gets **100% of the
  fill's `closed_pnl`**, per the spec's corrected-via-external-review rule -- NOT a
  proportional split) and an opening leg (`quantity = fill.quantity - abs(running_position)`,
  `closed_pnl = 0`). **Fees** on a reversal fill DO split proportionally by quantity between
  the two legs (unlike `closed_pnl`).
- Within one trade: `entry_price`/`exit_price` are quantity-weighted averages across every
  *observed* entry-like / exit-like fill (classified per-fill by whether `abs(end_position)`
  increased or decreased relative to `abs(running_position)` at that point in the
  sequence) -- this is what makes scale-in/scale-out work correctly.
- Truncated-start fallback: if a trade has zero observed entry-like fills before it closes,
  `entry_price`/`entry_timestamp` fall back to the fill that started the trade's own
  price/timestamp (an explicit approximation, not a real entry -- `is_truncated_start=True`
  is the caller's signal to treat it with that skepticism).
- `quantity <= 0` on any fill raises `ValueError` -- fail loudly on malformed data.
- A trade left open at the end of the fill sequence (position never returns to zero) is
  **not** emitted -- only fully closed trades appear in the returned list.
- Liquidation flag: any fill in a trade whose `direction` field contains the substring
  `"Liquidat"` sets that trade's `was_liquidated = True`.
- `ReconstructedTrade` fields: `trader`, `symbol`, `direction` (`"long"`/`"short"`),
  `entry_timestamp`, `entry_price`, `exit_timestamp`, `exit_price`, `quantity` (total
  exited, or total entered if exit_qty is somehow smaller -- see Task 2), `gross_pnl`,
  `fees`, `net_pnl` (`gross_pnl - fees`), `holding_time_seconds`, `n_fills`,
  `is_truncated_start`, `was_liquidated`.
- `reconstruct_and_persist_trades(session, trader, symbol=None) -> ReconstructResult`
  (`n_trades: int`, `symbols: list[str]`) -- recomputes from scratch every run: deletes
  existing `ReconstructedTrade` rows for the scope, re-derives from every currently-stored
  `NormalizedFill`, re-inserts, one transaction.
- `research/cli.py` gains `trader-analyze`: `--trader` (required), `--symbol` (optional),
  `--db-path` (default matching existing subcommands).

---

### Task 1: `research/models.py` — `ReconstructedTrade` table

**Files:**
- Modify: `research/models.py` (append one new class)
- Test: `research/tests/test_models.py` (extend)

**Interfaces:**
- Consumes: the existing `Base`.
- Produces: `ReconstructedTrade` ORM class, consumed by Task 2 and Task 3.

- [ ] **Step 1: Write the failing test**

Append to `research/tests/test_models.py`:

```python
def test_reconstructed_trade_round_trips():
    session = _memory_session()
    session.add(
        ReconstructedTrade(
            trader="0xAAA",
            symbol="BTC/USDC:USDC",
            direction="long",
            entry_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            entry_price=100.0,
            exit_timestamp=datetime(2026, 1, 2, tzinfo=UTC),
            exit_price=110.0,
            quantity=5.0,
            gross_pnl=50.0,
            fees=1.0,
            net_pnl=49.0,
            holding_time_seconds=86400.0,
            n_fills=3,
            is_truncated_start=False,
            was_liquidated=False,
        )
    )
    session.commit()

    row = session.query(ReconstructedTrade).one()
    assert row.trader == "0xAAA"
    assert row.symbol == "BTC/USDC:USDC"
    assert row.direction == "long"
    assert row.entry_price == 100.0
    assert row.exit_price == 110.0
    assert row.quantity == 5.0
    assert row.net_pnl == 49.0
    assert row.n_fills == 3
    assert row.is_truncated_start is False
    assert row.was_liquidated is False
```

Add `ReconstructedTrade` to the existing `from research.models import ...` line at the top
of the file.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest research/tests/test_models.py -v -k reconstructed_trade`
Expected: FAIL with `ImportError: cannot import name 'ReconstructedTrade' from 'research.models'`

- [ ] **Step 3: Implement `ReconstructedTrade`**

Append to `research/models.py` (after the existing `NormalizedFill` class):

```python
class ReconstructedTrade(Base):
    """One row per logical trade, grouped from NormalizedFill rows by
    research.trader_mining.engine.reconstruct_trades -- zero-to-zero position spans, not
    an imposed lot-accounting convention. Recomputed from scratch on every
    reconstruct_and_persist_trades run, not incrementally patched."""

    __tablename__ = "reconstructed_trades"
    __table_args__ = (Index("ix_reconstructed_trades_trader_symbol", "trader", "symbol"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trader: Mapped[str] = mapped_column(String(120))
    symbol: Mapped[str] = mapped_column(String(80))
    direction: Mapped[str] = mapped_column(String(10))
    entry_timestamp: Mapped[datetime] = mapped_column(DateTime)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_timestamp: Mapped[datetime] = mapped_column(DateTime)
    exit_price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    gross_pnl: Mapped[float] = mapped_column(Float)
    fees: Mapped[float] = mapped_column(Float)
    net_pnl: Mapped[float] = mapped_column(Float)
    holding_time_seconds: Mapped[float] = mapped_column(Float)
    n_fills: Mapped[int] = mapped_column(Integer)
    is_truncated_start: Mapped[bool] = mapped_column(Boolean)
    was_liquidated: Mapped[bool] = mapped_column(Boolean)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest research/tests/test_models.py -v`
Expected: PASS (all tests in the file, including the new one)

- [ ] **Step 5: Lint and format**

Run: `ruff check research/models.py research/tests/test_models.py` and
`ruff format --check research/models.py research/tests/test_models.py`

- [ ] **Step 6: Commit**

```bash
git add research/models.py research/tests/test_models.py
git commit -m "feat(research): add ReconstructedTrade table"
```

---

### Task 2: `research/trader_mining/engine.py` — `reconstruct_trades` (core algorithm)

**Files:**
- Create: `research/trader_mining/engine.py`
- Test: `research/tests/trader_mining/test_engine.py`

**Interfaces:**
- Consumes: `research.models.NormalizedFill`, `ReconstructedTrade` (Task 1).
- Produces: `reconstruct_trades(trader, symbol, fills) -> list[ReconstructedTrade]`,
  consumed by Task 3.

- [ ] **Step 1: Write the failing tests**

Create `research/tests/trader_mining/test_engine.py`:

```python
# research/tests/trader_mining/test_engine.py
from datetime import UTC, datetime, timedelta

import pytest

from research.models import NormalizedFill
from research.trader_mining.engine import reconstruct_trades


TRADER = "0xAAA"
SYMBOL = "BTC/USDC:USDC"
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _fill(
    tid, side, price, qty, position, ts=T0, closed_pnl=0.0, fee=0.1, direction="Open Long"
):
    return NormalizedFill(
        trader=TRADER,
        tid=tid,
        timestamp=ts,
        symbol=SYMBOL,
        side=side,
        price=price,
        quantity=qty,
        notional=price * qty,
        position=position,
        closed_pnl=closed_pnl,
        direction=direction,
        crossed=True,
        fee=fee,
        fee_currency="USDC",
        order_id=str(tid),
    )


def test_one_entry_one_exit():
    fills = [
        _fill(1, "buy", 100.0, 5.0, position=0.0, ts=T0),
        _fill(
            2,
            "sell",
            110.0,
            5.0,
            position=5.0,
            ts=T0 + timedelta(hours=1),
            closed_pnl=50.0,
            direction="Close Long",
        ),
    ]

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    assert len(trades) == 1
    t = trades[0]
    assert t.direction == "long"
    assert t.entry_price == pytest.approx(100.0)
    assert t.exit_price == pytest.approx(110.0)
    assert t.quantity == pytest.approx(5.0)
    assert t.gross_pnl == pytest.approx(50.0)
    assert t.fees == pytest.approx(0.2)
    assert t.net_pnl == pytest.approx(49.8)
    assert t.n_fills == 2
    assert t.is_truncated_start is False
    assert t.was_liquidated is False
    assert t.holding_time_seconds == pytest.approx(3600.0)


def test_scale_in_weighted_average_entry_price():
    fills = [
        _fill(1, "buy", 100.0, 3.0, position=0.0, ts=T0),
        _fill(2, "buy", 106.0, 2.0, position=3.0, ts=T0 + timedelta(minutes=1)),
        _fill(
            3,
            "sell",
            120.0,
            5.0,
            position=5.0,
            ts=T0 + timedelta(hours=1),
            closed_pnl=88.0,
            direction="Close Long",
        ),
    ]

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    assert len(trades) == 1
    # (100*3 + 106*2) / 5 = 102.4
    assert trades[0].entry_price == pytest.approx(102.4)
    assert trades[0].quantity == pytest.approx(5.0)


def test_scale_out_weighted_average_exit_price():
    fills = [
        _fill(1, "buy", 100.0, 5.0, position=0.0, ts=T0),
        _fill(
            2,
            "sell",
            108.0,
            2.0,
            position=5.0,
            ts=T0 + timedelta(minutes=1),
            closed_pnl=16.0,
            direction="Close Long",
        ),
        _fill(
            3,
            "sell",
            112.0,
            3.0,
            position=3.0,
            ts=T0 + timedelta(hours=1),
            closed_pnl=36.0,
            direction="Close Long",
        ),
    ]

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    assert len(trades) == 1
    # (108*2 + 112*3) / 5 = 110.4
    assert trades[0].exit_price == pytest.approx(110.4)
    assert trades[0].gross_pnl == pytest.approx(52.0)


def test_reversal_with_followup_close_produces_two_trades():
    fills = [
        _fill(1, "buy", 100.0, 5.0, position=0.0, ts=T0),
        # reversal: +5 -> sell 8 -> -3. Closing leg qty=5 (full closed_pnl=40, fee split
        # 5/8), opening leg qty=3 (closed_pnl=0, fee split 3/8).
        _fill(
            2,
            "sell",
            120.0,
            8.0,
            position=5.0,
            ts=T0 + timedelta(hours=1),
            closed_pnl=40.0,
            fee=0.8,
            direction="Close Long",
        ),
        # closes the new -3 short
        _fill(
            3,
            "buy",
            115.0,
            3.0,
            position=-3.0,
            ts=T0 + timedelta(hours=2),
            closed_pnl=15.0,
            direction="Close Short",
        ),
    ]

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    assert len(trades) == 2
    closed_long, closed_short = trades[0], trades[1]
    assert closed_long.direction == "long"
    assert closed_long.quantity == pytest.approx(5.0)
    assert closed_long.gross_pnl == pytest.approx(40.0)
    assert closed_long.fees == pytest.approx(0.1 + 0.8 * (5 / 8))
    assert closed_long.n_fills == 2

    assert closed_short.direction == "short"
    assert closed_short.quantity == pytest.approx(3.0)
    assert closed_short.entry_price == pytest.approx(120.0)
    assert closed_short.gross_pnl == pytest.approx(15.0)  # 0 from reversal + 15 from close
    assert closed_short.fees == pytest.approx(0.8 * (3 / 8) + 0.1)
    assert closed_short.n_fills == 2


def test_reversal_with_no_followup_produces_only_the_closing_trade():
    fills = [
        _fill(1, "buy", 100.0, 5.0, position=0.0, ts=T0),
        _fill(
            2,
            "sell",
            120.0,
            8.0,
            position=5.0,
            ts=T0 + timedelta(hours=1),
            closed_pnl=40.0,
            direction="Close Long",
        ),
    ]

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    # the newly-opened -3 short is still open -- not emitted
    assert len(trades) == 1
    assert trades[0].direction == "long"
    assert trades[0].quantity == pytest.approx(5.0)


def test_truncated_start_with_no_observed_entry_falls_back_to_first_fill():
    fills = [
        _fill(
            1,
            "sell",
            110.0,
            5.0,
            position=5.0,  # non-zero on the very first fill -- predates history
            ts=T0,
            closed_pnl=25.0,
            direction="Close Long",
        ),
    ]

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    assert len(trades) == 1
    t = trades[0]
    assert t.is_truncated_start is True
    assert t.direction == "long"
    # no entry-like fill was observed -- entry falls back to this fill's own price/time
    assert t.entry_price == pytest.approx(110.0)
    assert t.entry_timestamp == T0
    assert t.exit_price == pytest.approx(110.0)
    assert t.gross_pnl == pytest.approx(25.0)


def test_truncated_start_with_one_observed_entry_fill():
    fills = [
        _fill(1, "buy", 100.0, 2.0, position=3.0, ts=T0),  # 3 -> 5, scale-in observed
        _fill(
            2,
            "sell",
            120.0,
            5.0,
            position=5.0,
            ts=T0 + timedelta(hours=1),
            closed_pnl=60.0,
            direction="Close Long",
        ),
    ]

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    assert len(trades) == 1
    t = trades[0]
    assert t.is_truncated_start is True
    # entry_price reflects ONLY the observed 2.0-qty entry fill, not the unobserved 3.0
    assert t.entry_price == pytest.approx(100.0)
    assert t.quantity == pytest.approx(5.0)


def test_liquidation_fill_flags_the_trade():
    fills = [
        _fill(1, "buy", 100.0, 5.0, position=0.0, ts=T0),
        _fill(
            2,
            "sell",
            80.0,
            5.0,
            position=5.0,
            ts=T0 + timedelta(hours=1),
            closed_pnl=-100.0,
            direction="Liquidated Long",
        ),
    ]

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    assert trades[0].was_liquidated is True


def test_nonpositive_quantity_raises():
    fills = [_fill(1, "buy", 100.0, 0.0, position=0.0, ts=T0)]

    with pytest.raises(ValueError, match="tid=1"):
        reconstruct_trades(TRADER, SYMBOL, fills)


def test_unclosed_trailing_position_is_not_emitted():
    fills = [_fill(1, "buy", 100.0, 5.0, position=0.0, ts=T0)]

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    assert trades == []


def test_empty_fills_returns_empty_list():
    assert reconstruct_trades(TRADER, SYMBOL, []) == []


def test_long_fill_sequence_scaling_repeatedly_closes_exactly():
    """Regression guard for float-drift risk -- Decimal arithmetic must close this
    exactly even after 40 scale-in/scale-out fills that would accumulate float error if
    summed naively."""
    fills = []
    position = 0.0
    tid = 1
    ts = T0
    for i in range(20):
        fills.append(_fill(tid, "buy", 100.0 + i, 0.1, position=position, ts=ts))
        position += 0.1
        tid += 1
        ts += timedelta(minutes=1)
    for i in range(19):
        fills.append(
            _fill(tid, "sell", 100.0 + i, 0.1, position=position, ts=ts, closed_pnl=0.01)
        )
        position -= 0.1
        tid += 1
        ts += timedelta(minutes=1)
    # final fill fully closes the remaining 0.1
    fills.append(
        _fill(tid, "sell", 105.0, 0.1, position=position, ts=ts, closed_pnl=0.01)
    )

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    assert len(trades) == 1
    assert trades[0].quantity == pytest.approx(2.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest research/tests/trader_mining/test_engine.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'research.trader_mining.engine'`

- [ ] **Step 3: Implement `reconstruct_trades`**

Create `research/trader_mining/engine.py`:

```python
# research/trader_mining/engine.py
"""Trade reconstruction: groups a (trader, symbol)'s already-normalized fills into
logical ReconstructedTrade records, following Hyperliquid's own position transitions
(startPosition + signed quantity) rather than an imposed lot-accounting convention like
FIFO. See docs/superpowers/specs/2026-08-25-trader-mining-release-2-design.md for the
full algorithm writeup, including why reversal closed_pnl goes 100% to the closing leg
(corrected via external review, not the original proposal draft's proportional-split
guess) and why position tracking uses Decimal, not float.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from research.models import NormalizedFill, ReconstructedTrade


@dataclass
class _TradeState:
    is_truncated_start: bool
    fallback_price: Decimal
    fallback_timestamp: datetime
    direction: str
    entry_notional: Decimal = field(default_factory=lambda: Decimal(0))
    entry_qty: Decimal = field(default_factory=lambda: Decimal(0))
    entry_timestamp: datetime | None = None
    exit_notional: Decimal = field(default_factory=lambda: Decimal(0))
    exit_qty: Decimal = field(default_factory=lambda: Decimal(0))
    exit_timestamp: datetime | None = None
    gross_pnl: Decimal = field(default_factory=lambda: Decimal(0))
    fees: Decimal = field(default_factory=lambda: Decimal(0))
    n_fills: int = 0
    was_liquidated: bool = False

    def add_entry(self, fill: NormalizedFill, qty: Decimal, fee: Decimal) -> None:
        if self.entry_timestamp is None:
            self.entry_timestamp = fill.timestamp
        self.entry_notional += Decimal(str(fill.price)) * qty
        self.entry_qty += qty
        self.fees += fee
        self.n_fills += 1
        if "Liquidat" in fill.direction:
            self.was_liquidated = True

    def add_exit(
        self, fill: NormalizedFill, qty: Decimal, closed_pnl: Decimal, fee: Decimal
    ) -> None:
        self.exit_timestamp = fill.timestamp
        self.exit_notional += Decimal(str(fill.price)) * qty
        self.exit_qty += qty
        self.gross_pnl += closed_pnl
        self.fees += fee
        self.n_fills += 1
        if "Liquidat" in fill.direction:
            self.was_liquidated = True

    def finalize(self, trader: str, symbol: str) -> ReconstructedTrade:
        entry_price = (
            float(self.entry_notional / self.entry_qty)
            if self.entry_qty
            else float(self.fallback_price)
        )
        entry_timestamp = self.entry_timestamp or self.fallback_timestamp
        exit_price = (
            float(self.exit_notional / self.exit_qty)
            if self.exit_qty
            else float(self.fallback_price)
        )
        exit_timestamp = self.exit_timestamp or self.fallback_timestamp
        quantity = float(self.exit_qty) if self.exit_qty else float(self.entry_qty)
        return ReconstructedTrade(
            trader=trader,
            symbol=symbol,
            direction=self.direction,
            entry_timestamp=entry_timestamp,
            entry_price=entry_price,
            exit_timestamp=exit_timestamp,
            exit_price=exit_price,
            quantity=quantity,
            gross_pnl=float(self.gross_pnl),
            fees=float(self.fees),
            net_pnl=float(self.gross_pnl - self.fees),
            holding_time_seconds=(exit_timestamp - entry_timestamp).total_seconds(),
            n_fills=self.n_fills,
            is_truncated_start=self.is_truncated_start,
            was_liquidated=self.was_liquidated,
        )


def reconstruct_trades(
    trader: str, symbol: str, fills: list[NormalizedFill]
) -> list[ReconstructedTrade]:
    """fills must already be sorted by (timestamp, tid) -- not re-sorted here."""
    if not fills:
        return []

    for fill in fills:
        if fill.quantity <= 0:
            raise ValueError(f"fill tid={fill.tid} has non-positive quantity {fill.quantity}")

    trades: list[ReconstructedTrade] = []
    running_position = Decimal(str(fills[0].position))
    trade: _TradeState | None = None

    if running_position != 0:
        trade = _TradeState(
            is_truncated_start=True,
            fallback_price=Decimal(str(fills[0].price)),
            fallback_timestamp=fills[0].timestamp,
            direction="long" if running_position > 0 else "short",
        )

    for fill in fills:
        qty = Decimal(str(fill.quantity))
        signed_qty = qty if fill.side == "buy" else -qty
        end_position = running_position + signed_qty

        if trade is None:
            trade = _TradeState(
                is_truncated_start=False,
                fallback_price=Decimal(str(fill.price)),
                fallback_timestamp=fill.timestamp,
                direction="long" if end_position > 0 else "short",
            )

        is_reversal = (
            running_position != 0
            and end_position != 0
            and (end_position > 0) != (running_position > 0)
        )

        if is_reversal:
            close_qty = abs(running_position)
            open_qty = qty - close_qty
            fee = Decimal(str(fill.fee))
            close_fee = fee * (close_qty / qty)
            trade.add_exit(fill, close_qty, Decimal(str(fill.closed_pnl)), close_fee)
            trades.append(trade.finalize(trader, symbol))

            trade = _TradeState(
                is_truncated_start=False,
                fallback_price=Decimal(str(fill.price)),
                fallback_timestamp=fill.timestamp,
                direction="long" if end_position > 0 else "short",
            )
            trade.add_entry(fill, open_qty, fee - close_fee)
            running_position = end_position
            continue

        if abs(end_position) > abs(running_position):
            trade.add_entry(fill, qty, Decimal(str(fill.fee)))
        else:
            trade.add_exit(fill, qty, Decimal(str(fill.closed_pnl)), Decimal(str(fill.fee)))

        running_position = end_position

        if running_position == 0:
            trades.append(trade.finalize(trader, symbol))
            trade = None

    return trades
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest research/tests/trader_mining/test_engine.py -v`
Expected: PASS (13 tests). If any fail, trust the tests (they were traced by hand against
the algorithm before this plan was written) over the implementation -- fix the code, not
the test, unless you find an actual error in the hand-traced expected values.

- [ ] **Step 5: Lint and format**

Run: `ruff check research/trader_mining/engine.py research/tests/trader_mining/test_engine.py`
and `ruff format --check research/trader_mining/engine.py research/tests/trader_mining/test_engine.py`

- [ ] **Step 6: Commit**

```bash
git add research/trader_mining/engine.py research/tests/trader_mining/test_engine.py
git commit -m "feat(research): add reconstruct_trades -- zero-crossing trade boundary algorithm"
```

---

### Task 3: `research/trader_mining/engine.py` — `reconstruct_and_persist_trades` (orchestration)

**Files:**
- Modify: `research/trader_mining/engine.py`
- Test: `research/tests/trader_mining/test_engine.py` (extend)

**Interfaces:**
- Consumes: `reconstruct_trades` (Task 2), `research.models.NormalizedFill`,
  `ReconstructedTrade` (Task 1).
- Produces: `ReconstructResult`, `reconstruct_and_persist_trades(session, trader,
  symbol=None)` -- consumed by Task 4.

- [ ] **Step 1: Write the failing tests**

Append to `research/tests/trader_mining/test_engine.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from research.models import Base
from research.trader_mining.engine import reconstruct_and_persist_trades


def _memory_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _add_normalized_fill(session, **kwargs):
    defaults = dict(
        trader=TRADER,
        symbol=SYMBOL,
        side="buy",
        price=100.0,
        quantity=5.0,
        notional=500.0,
        position=0.0,
        closed_pnl=0.0,
        direction="Open Long",
        crossed=True,
        fee=0.1,
        fee_currency="USDC",
        order_id="1",
    )
    defaults.update(kwargs)
    session.add(NormalizedFill(**defaults))


def test_persists_reconstructed_trades_from_stored_fills():
    session = _memory_session()
    _add_normalized_fill(session, tid=1, side="buy", position=0.0, timestamp=T0)
    _add_normalized_fill(
        session,
        tid=2,
        side="sell",
        position=5.0,
        timestamp=T0 + timedelta(hours=1),
        closed_pnl=50.0,
        direction="Close Long",
    )
    session.commit()

    result = reconstruct_and_persist_trades(session, TRADER)

    assert result.n_trades == 1
    assert result.symbols == [SYMBOL]
    assert session.query(ReconstructedTrade).count() == 1


def test_rerunning_after_new_fills_replaces_not_duplicates():
    session = _memory_session()
    _add_normalized_fill(session, tid=1, side="buy", position=0.0, timestamp=T0)
    _add_normalized_fill(
        session,
        tid=2,
        side="sell",
        position=5.0,
        timestamp=T0 + timedelta(hours=1),
        closed_pnl=50.0,
        direction="Close Long",
    )
    session.commit()
    reconstruct_and_persist_trades(session, TRADER)

    # a second, independent round-trip trade for the same trader/symbol
    _add_normalized_fill(session, tid=3, side="buy", position=0.0, timestamp=T0 + timedelta(hours=2))
    _add_normalized_fill(
        session,
        tid=4,
        side="sell",
        position=5.0,
        timestamp=T0 + timedelta(hours=3),
        closed_pnl=20.0,
        direction="Close Long",
    )
    session.commit()

    result = reconstruct_and_persist_trades(session, TRADER)

    assert result.n_trades == 2
    assert session.query(ReconstructedTrade).count() == 2


def test_scoping_to_one_symbol_leaves_other_symbols_untouched():
    session = _memory_session()
    _add_normalized_fill(session, tid=1, symbol="BTC/USDC:USDC", side="buy", position=0.0, timestamp=T0)
    _add_normalized_fill(
        session,
        tid=2,
        symbol="BTC/USDC:USDC",
        side="sell",
        position=5.0,
        timestamp=T0 + timedelta(hours=1),
        closed_pnl=50.0,
        direction="Close Long",
    )
    _add_normalized_fill(session, tid=3, symbol="ETH/USDC:USDC", side="buy", position=0.0, timestamp=T0)
    _add_normalized_fill(
        session,
        tid=4,
        symbol="ETH/USDC:USDC",
        side="sell",
        position=5.0,
        timestamp=T0 + timedelta(hours=1),
        closed_pnl=10.0,
        direction="Close Long",
    )
    session.commit()

    result = reconstruct_and_persist_trades(session, TRADER, symbol="BTC/USDC:USDC")

    assert result.n_trades == 1
    assert result.symbols == ["BTC/USDC:USDC"]
    assert session.query(ReconstructedTrade).count() == 1
```

Add `timedelta` to the existing `from datetime import ...` line if not already imported
(it already is, from Task 2's own tests).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest research/tests/trader_mining/test_engine.py -v -k persist`
Expected: FAIL with `ImportError: cannot import name 'reconstruct_and_persist_trades'`

- [ ] **Step 3: Implement `reconstruct_and_persist_trades`**

Append to `research/trader_mining/engine.py`, below the existing `reconstruct_trades`
function (`dataclass` is already imported at the top of the file from Task 2 -- no new
import needed for the dataclass itself):

```python
@dataclass
class ReconstructResult:
    n_trades: int
    symbols: list[str]


def reconstruct_and_persist_trades(
    session: Session, trader: str, symbol: str | None = None
) -> ReconstructResult:
    """Recompute (trader, symbol)'s trades from scratch -- deletes existing
    ReconstructedTrade rows for the scope and re-derives from every currently-stored
    NormalizedFill, rather than incrementally patching. Simpler and more correct: a
    later trader-import run backfilling older history could retroactively change
    earlier trade boundaries, which incremental reconciliation can't handle cleanly."""
    symbols_query = session.query(NormalizedFill.symbol).filter(NormalizedFill.trader == trader)
    if symbol is not None:
        symbols_query = symbols_query.filter(NormalizedFill.symbol == symbol)
    symbols = sorted({s for (s,) in symbols_query.distinct().all()})

    total_trades = 0
    for sym in symbols:
        fills = (
            session.query(NormalizedFill)
            .filter(NormalizedFill.trader == trader, NormalizedFill.symbol == sym)
            .order_by(NormalizedFill.timestamp, NormalizedFill.tid)
            .all()
        )
        session.query(ReconstructedTrade).filter(
            ReconstructedTrade.trader == trader, ReconstructedTrade.symbol == sym
        ).delete()

        new_trades = reconstruct_trades(trader, sym, fills)
        for t in new_trades:
            session.add(t)
        total_trades += len(new_trades)

    session.commit()
    return ReconstructResult(n_trades=total_trades, symbols=symbols)
```

Also add the needed imports at the top of `research/trader_mining/engine.py` (alongside
the existing ones):

```python
from sqlalchemy.orm import Session
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest research/tests/trader_mining/test_engine.py -v`
Expected: PASS (all tests in the file, 16 total)

- [ ] **Step 5: Lint and format**

Run: `ruff check research/trader_mining/engine.py research/tests/trader_mining/test_engine.py`
and `ruff format --check research/trader_mining/engine.py research/tests/trader_mining/test_engine.py`

- [ ] **Step 6: Commit**

```bash
git add research/trader_mining/engine.py research/tests/trader_mining/test_engine.py
git commit -m "feat(research): add reconstruct_and_persist_trades -- recompute-from-scratch orchestration"
```

---

### Task 4: `research/cli.py` — `trader-analyze` subcommand

**Files:**
- Modify: `research/cli.py`
- Test: `research/tests/test_cli.py`

**Interfaces:**
- Consumes: `research.trader_mining.engine.reconstruct_and_persist_trades`,
  `ReconstructResult` (Task 3).
- Produces: nothing further downstream -- final task in this plan.

- [ ] **Step 1: Write the failing test**

Append to `research/tests/test_cli.py`:

```python
def test_trader_analyze_command_forwards_args_and_prints_result(mocker, capsys):
    from research.trader_mining.engine import ReconstructResult

    mock_reconstruct = mocker.patch(
        "research.cli.reconstruct_and_persist_trades",
        return_value=ReconstructResult(n_trades=7, symbols=["BTC/USDC:USDC", "ETH/USDC:USDC"]),
    )
    mocker.patch("research.cli.get_engine")
    mocker.patch("research.cli.get_session")

    exit_code = main(
        [
            "trader-analyze",
            "--trader",
            "0x0000000000000000000000000000000000000000",
            "--symbol",
            "BTC/USDC:USDC",
            "--db-path",
            "user_data/research.sqlite",
        ]
    )

    _, kwargs = mock_reconstruct.call_args
    assert kwargs["trader"] == "0x0000000000000000000000000000000000000000"
    assert kwargs["symbol"] == "BTC/USDC:USDC"

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "n_trades: 7" in captured.out
    assert "BTC/USDC:USDC" in captured.out
    assert "ETH/USDC:USDC" in captured.out
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest research/tests/test_cli.py -v -k trader_analyze`
Expected: FAIL with `error: argument command: invalid choice: 'trader-analyze'`

- [ ] **Step 3: Add the `trader-analyze` subcommand**

In `research/cli.py`, add the import (alongside the existing `research.trader_mining...`
import):

```python
from research.trader_mining.engine import reconstruct_and_persist_trades
from research.trader_mining.ingestion import ingest_hyperliquid_fills
```

After the existing `trader_import = sub.add_parser(...)` block, before
`args = parser.parse_args(argv)`, add:

```python
    trader_analyze = sub.add_parser(
        "trader-analyze", help="Reconstruct trades from a wallet's ingested fills"
    )
    trader_analyze.add_argument("--trader", required=True, help="Wallet address")
    trader_analyze.add_argument("--symbol", help="Limit to one symbol (default: all)")
    trader_analyze.add_argument("--db-path", default="user_data/research.sqlite")
```

Add the new dispatch branch after the existing `elif args.command == "trader-import":`
block (as another `elif`):

```python
    elif args.command == "trader-analyze":
        engine = get_engine(args.db_path)
        session = get_session(engine)
        result = reconstruct_and_persist_trades(session, trader=args.trader, symbol=args.symbol)
        print(f"n_trades: {result.n_trades}")
        print(f"symbols: {', '.join(result.symbols)}")
        return 0
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest research/tests/test_cli.py -v`
Expected: PASS (all tests in the file, including the new one)

- [ ] **Step 5: Lint and format**

Run: `ruff check research/cli.py research/tests/test_cli.py` and
`ruff format --check research/cli.py research/tests/test_cli.py`

- [ ] **Step 6: Run the full targeted test set (models, trader_mining, cli)**

Run: `pytest research/tests/trader_mining/ research/tests/test_models.py research/tests/test_cli.py -v`
Expected: PASS (every test across all 4 tasks, confirming they compose cleanly)

- [ ] **Step 7: Commit**

```bash
git add research/cli.py research/tests/test_cli.py
git commit -m "feat(research): add trader-analyze CLI subcommand"
```
