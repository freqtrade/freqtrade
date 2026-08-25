# research/tests/trader_mining/test_engine.py
from datetime import UTC, datetime, timedelta

import pytest

from research.models import NormalizedFill
from research.trader_mining.engine import reconstruct_trades


TRADER = "0xAAA"
SYMBOL = "BTC/USDC:USDC"
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _fill(tid, side, price, qty, position, ts=T0, closed_pnl=0.0, fee=0.1, direction="Open Long"):
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
        fills.append(_fill(tid, "sell", 100.0 + i, 0.1, position=position, ts=ts, closed_pnl=0.01))
        position -= 0.1
        tid += 1
        ts += timedelta(minutes=1)
    # final fill fully closes the remaining 0.1
    fills.append(_fill(tid, "sell", 105.0, 0.1, position=position, ts=ts, closed_pnl=0.01))

    trades = reconstruct_trades(TRADER, SYMBOL, fills)

    assert len(trades) == 1
    assert trades[0].quantity == pytest.approx(2.0)
