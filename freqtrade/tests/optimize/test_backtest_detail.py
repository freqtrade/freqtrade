# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, C0330, unused-argument
import logging
from unittest.mock import MagicMock

from pandas import DataFrame
import pytest


from freqtrade.optimize import get_timeframe
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.strategy.interface import SellType
from freqtrade.tests.optimize import (BTrade, BTContainer, _build_backtest_dataframe,
                                      _get_frame_time_from_offset, tests_ticker_interval)
from freqtrade.tests.conftest import patch_exchange


# Test 1 Minus 8% Close
# Test with Stop-loss at 1%
# TC1: Stop-Loss Triggered 1% loss
tc1 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # enter trade (signal on last candle)
    [2, 4987, 5012, 4600, 4600, 6172, 0, 0],  # exit with stoploss hit
    [3, 4975, 5000, 4980, 4977, 6172, 0, 0],
    [4, 4977, 4987, 4977, 4995, 6172, 0, 0],
    [5, 4995, 4995, 4995, 4950, 6172, 0, 0]],
    stop_loss=-0.01, roi=1, profit_perc=-0.01,
    trades=[BTrade(sell_reason=SellType.STOP_LOSS, open_tick=1, close_tick=2)]
)


# Test 2 Minus 4% Low, minus 1% close
# Test with Stop-Loss at 3%
# TC2: Stop-Loss Triggered 3% Loss
tc2 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # enter trade (signal on last candle)
    [2, 4987, 5012, 4962, 4975, 6172, 0, 0],
    [3, 4975, 5000, 4800, 4962, 6172, 0, 0],  # exit with stoploss hit
    [4, 4962, 4987, 4937, 4950, 6172, 0, 0],
    [5, 4950, 4975, 4925, 4950, 6172, 0, 0]],
    stop_loss=-0.03, roi=1, profit_perc=-0.03,
    trades=[BTrade(sell_reason=SellType.STOP_LOSS, open_tick=1, close_tick=3)]
)


# Test 3 Candle drops 4%, Recovers 1%.
#               Entry Criteria Met
# 	            Candle drops 20%
# Candle Data for test 3
# Test with Stop-Loss at 2%
# TC3: Trade-A: Stop-Loss Triggered 2% Loss
#          Trade-B: Stop-Loss Triggered 2% Loss
tc3 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # enter trade (signal on last candle)
    [2, 4987, 5012, 4800, 4975, 6172, 0, 0],  # exit with stoploss hit
    [3, 4975, 5000, 4950, 4962, 6172, 1, 0],
    [4, 4975, 5000, 4950, 4962, 6172, 0, 0],  # enter trade 2 (signal on last candle)
    [5, 4962, 4987, 4000, 4000, 6172, 0, 0],  # exit with stoploss hit
    [6, 4950, 4975, 4975, 4950, 6172, 0, 0]],
    stop_loss=-0.02, roi=1, profit_perc=-0.04,
    trades=[BTrade(sell_reason=SellType.STOP_LOSS, open_tick=1, close_tick=2),
            BTrade(sell_reason=SellType.STOP_LOSS, open_tick=4, close_tick=5)]
)

# Test 4 Minus 3% / recovery +15%
# Candle Data for test 3 – Candle drops 3% Closed 15% up
# Test with Stop-loss at 2% ROI 6%
# TC4: Stop-Loss Triggered 2% Loss
tc4 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # enter trade (signal on last candle)
    [2, 4987, 5750, 4850, 5750, 6172, 0, 0],  # Exit with stoploss hit
    [3, 4975, 5000, 4950, 4962, 6172, 0, 0],
    [4, 4962, 4987, 4937, 4950, 6172, 0, 0],
    [5, 4950, 4975, 4925, 4950, 6172, 0, 0]],
    stop_loss=-0.02, roi=0.06, profit_perc=-0.02,
    trades=[BTrade(sell_reason=SellType.STOP_LOSS, open_tick=1, close_tick=2)]
)

# Test 5 / Drops 0.5% Closes +20%
# Set stop-loss at 1% ROI 3%
# TC5: ROI triggers 3% Gain
tc5 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4980, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4980, 4987, 6172, 0, 0],  # enter trade (signal on last candle)
    [2, 4987, 5025, 4975, 4987, 6172, 0, 0],
    [3, 4975, 6000, 4975, 6000, 6172, 0, 0],  # ROI
    [4, 4962, 4987, 4972, 4950, 6172, 0, 0],
    [5, 4950, 4975, 4925, 4950, 6172, 0, 0]],
    stop_loss=-0.01, roi=0.03, profit_perc=0.03,
    trades=[BTrade(sell_reason=SellType.ROI, open_tick=1, close_tick=3)]
)

# Test 6 / Drops 3% / Recovers 6% Positive / Closes 1% positve
# Candle Data for test 6
# Set stop-loss at 2% ROI at 5%
# TC6: Stop-Loss triggers 2% Loss
tc6 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # enter trade (signal on last candle)
    [2, 4987, 5300, 4850, 5050, 6172, 0, 0],  # Exit with stoploss
    [3, 4975, 5000, 4950, 4962, 6172, 0, 0],
    [4, 4962, 4987, 4972, 4950, 6172, 0, 0],
    [5, 4950, 4975, 4925, 4950, 6172, 0, 0]],
    stop_loss=-0.02, roi=0.05, profit_perc=-0.02,
    trades=[BTrade(sell_reason=SellType.STOP_LOSS, open_tick=1, close_tick=2)]
)

# Test 7 - 6% Positive / 1% Negative / Close 1% Positve
# Candle Data for test 7
# Set stop-loss at 2% ROI at 3%
# TC7: ROI Triggers 3% Gain
tc7 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],
    [2, 4987, 5300, 4950, 5050, 6172, 0, 0],
    [3, 4975, 5000, 4950, 4962, 6172, 0, 0],
    [4, 4962, 4987, 4972, 4950, 6172, 0, 0],
    [5, 4950, 4975, 4925, 4950, 6172, 0, 0]],
    stop_loss=-0.02, roi=0.03, profit_perc=0.03,
    trades=[BTrade(sell_reason=SellType.ROI, open_tick=1, close_tick=2)]
)

TESTS = [
    tc1,
    tc2,
    tc3,
    tc4,
    tc5,
    tc6,
    tc7,
]


@pytest.mark.parametrize("data", TESTS)
def test_backtest_results(default_conf, fee, mocker, caplog, data) -> None:
    """
    run functional tests
    """
    default_conf["stoploss"] = data.stop_loss
    default_conf["minimal_roi"] = {"0": data.roi}
    default_conf['ticker_interval'] = tests_ticker_interval
    mocker.patch('freqtrade.exchange.Exchange.get_fee', MagicMock(return_value=0.0))
    patch_exchange(mocker)
    frame = _build_backtest_dataframe(data.data)
    backtesting = Backtesting(default_conf)
    backtesting.advise_buy = lambda a, m: frame
    backtesting.advise_sell = lambda a, m: frame
    caplog.set_level(logging.DEBUG)

    pair = 'UNITTEST/BTC'
    # Dummy data as we mock the analyze functions
    data_processed = {pair: DataFrame()}
    min_date, max_date = get_timeframe({pair: frame})
    results = backtesting.backtest(
        {
            'stake_amount': default_conf['stake_amount'],
            'processed': data_processed,
            'max_open_trades': 10,
            'start_date': min_date,
            'end_date': max_date,
        }
    )
    print(results.T)

    assert len(results) == len(data.trades)
    assert round(results["profit_percent"].sum(), 3) == round(data.profit_perc, 3)

    for c, trade in enumerate(data.trades):
        res = results.iloc[c]
        assert res.sell_reason == trade.sell_reason
        assert res.open_time == _get_frame_time_from_offset(trade.open_tick)
        assert res.close_time == _get_frame_time_from_offset(trade.close_tick)
