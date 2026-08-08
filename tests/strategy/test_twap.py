# tests/strategy/twapstrategytest.py
from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from freqtrade.persistence import Trade
from freqtrade.util import dt_now

from .strats.twapstrategytest import TWAPStrategyTest


class FakeOrder:
    """Minimal stand-in for a filled Order"""

    def __init__(self, stake_amount_filled: float, order_filled_utc):
        self.stake_amount_filled = stake_amount_filled
        self.order_filled_utc = order_filled_utc


def _make_trade(default_conf, is_short: bool, stake_amount: float, open_date_utc):
    trade = MagicMock(spec=Trade)
    trade.has_open_orders = False
    trade.pair = "ETH/BTC"
    trade.is_short = is_short
    trade.entry_side = "sell" if is_short else "buy"
    trade.exit_side = "buy" if is_short else "sell"
    trade.open_date_utc = open_date_utc
    trade.stake_amount = stake_amount
    return trade


@pytest.mark.parametrize("is_short", [False, True])
def test_twap_entry_slicing_fills_all_slices(default_conf, is_short):
    strategy = TWAPStrategyTest(default_conf)
    strategy.ome_populate_exit_trend = MagicMock(return_value=False)

    num_slices = strategy.twap_num_slices
    total_stake = 100.0
    open_date = dt_now()
    trade = _make_trade(default_conf, is_short, total_stake, open_date)

    first_stake = strategy.custom_stake_amount(
        pair=trade.pair,
        current_time=open_date,
        current_rate=1.0,
        proposed_stake=total_stake,
        min_stake=None,
        max_stake=1000.0,
        leverage=1.0,
        entry_tag=None,
        side="short" if is_short else "long",
    )
    filled_entries = [FakeOrder(first_stake, open_date)]
    filled_exits: list = []

    def _select_filled_orders(side):
        return filled_entries if side == trade.entry_side else filled_exits

    trade.select_filled_orders.side_effect = _select_filled_orders

    total_filled = first_stake
    current_time = open_date

    for expected_slice_no in range(2, num_slices + 1):
        # Too early since the last fill -> no new slice yet
        early_time = filled_entries[-1].order_filled_utc + timedelta(seconds=1)
        assert (
            strategy.adjust_trade_position(
                trade, early_time, 1.0, 0.0, None, 1000.0, 1.0, 1.0, 0.0, 0.0
            )
            is None
        )

        current_time = filled_entries[-1].order_filled_utc + timedelta(
            minutes=strategy.twap_interval_minutes
        )
        result = strategy.adjust_trade_position(
            trade, current_time, 1.0, 0.0, None, 1000.0, 1.0, 1.0, 0.0, 0.0
        )
        assert result is not None
        stake, tag = result
        assert stake == pytest.approx(first_stake, rel=1e-9)
        assert tag == f"twap_entry_{expected_slice_no}_of_{num_slices}"

        total_filled += stake
        filled_entries.append(FakeOrder(stake, current_time))

    assert len(filled_entries) == num_slices
    assert total_filled == pytest.approx(total_stake, rel=1e-9)

    # Fully entered no exit signal adjust need to return None
    result = strategy.adjust_trade_position(
        trade,
        current_time + timedelta(minutes=strategy.twap_interval_minutes),
        1.0,
        0.0,
        None,
        1000.0,
        1.0,
        1.0,
        0.0,
        0.0,
    )
    assert result is None


@pytest.mark.parametrize("is_short", [False, True])
def test_twap_exit_slicing_fills_all_slices(default_conf, is_short):
    strategy = TWAPStrategyTest(default_conf)
    strategy.ome_populate_exit_trend = MagicMock(return_value=True)

    num_slices = strategy.twap_num_slices
    total_stake = 100.0
    open_date = dt_now()
    trade = _make_trade(default_conf, is_short, total_stake, open_date)

    # Entry phase already fully filled
    filled_entries = [
        FakeOrder(
            total_stake / num_slices,
            open_date + timedelta(minutes=i * strategy.twap_interval_minutes),
        )
        for i in range(num_slices)
    ]
    filled_exits: list = []

    def _select_filled_orders(side):
        return filled_entries if side == trade.entry_side else filled_exits

    trade.select_filled_orders.side_effect = _select_filled_orders

    current_time = filled_entries[-1].order_filled_utc
    total_exited = 0.0

    for _ in range(num_slices):
        if filled_exits:
            early_time = filled_exits[-1].order_filled_utc + timedelta(seconds=1)
            assert (
                strategy.adjust_trade_position(
                    trade, early_time, 1.0, 0.0, None, 1000.0, 1.0, 1.0, 0.0, 0.0
                )
                is None
            )
            current_time = filled_exits[-1].order_filled_utc + timedelta(
                minutes=strategy.twap_interval_minutes
            )

        result = strategy.adjust_trade_position(
            trade, current_time, 1.0, 0.0, None, 1000.0, 1.0, 1.0, 0.0, 0.0
        )
        assert result is not None
        stake = result
        assert stake < 0

        total_exited += stake
        filled_exits.append(FakeOrder(-stake, current_time))
        trade.stake_amount += stake

    assert len(filled_exits) == num_slices
    assert total_exited == pytest.approx(-total_stake, rel=1e-9)
    assert trade.stake_amount == pytest.approx(0.0, abs=1e-6)

    # Fully exited adjust need to return None
    result = strategy.adjust_trade_position(
        trade,
        current_time + timedelta(minutes=strategy.twap_interval_minutes),
        1.0,
        0.0,
        None,
        1000.0,
        1.0,
        1.0,
        0.0,
        0.0,
    )
    assert result is None
