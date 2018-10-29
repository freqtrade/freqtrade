# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, unused-argument
import logging
from unittest.mock import MagicMock
from typing import NamedTuple, List

from pandas import DataFrame
import pytest
import arrow


from freqtrade.optimize.backtesting import Backtesting
from freqtrade.strategy.interface import SellType
from freqtrade.tests.conftest import patch_exchange, log_has


ticker_start_time = arrow.get(2018, 10, 3)
ticker_interval_in_minute = 60


class BTContainer(NamedTuple):
    """
    NamedTuple Defining BacktestResults inputs.
    """
    data: List[float]
    stop_loss: float
    roi: float
    trades: int
    profit_perc: float
    sell_r: SellType


def _build_dataframe(ticker_with_signals):
    columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'buy', 'sell']

    frame = DataFrame.from_records(ticker_with_signals, columns=columns)
    frame['date'] = frame['date'].apply(lambda x: ticker_start_time.shift(
                    minutes=(x * ticker_interval_in_minute)).datetime)
    # Ensure floats are in place
    for column in ['open', 'high', 'low', 'close', 'volume']:
        frame[column] = frame[column].astype('float64')
    return frame


data_profit = [
    [0, 0.0009910, 0.001011, 0.00098618, 0.001000, 12345, 1, 0],
    [1, 0.001000, 0.001010, 0.0009900, 0.0009900, 12345, 0, 0],
    [2, 0.0009900, 0.001011, 0.00091618, 0.0009900, 12345, 0, 0],
    [3, 0.001000, 0.001011, 0.00098618, 0.001100, 12345, 0, 1],
    [4, 0.001000, 0.001011, 0.00098618, 0.0009900, 12345, 0, 0]]

tc_profit1 = BTContainer(data=data_profit, stop_loss=-0.01, roi=1, trades=1,
                         profit_perc=0.10557, sell_r=SellType.STOP_LOSS)  # should be stoploss - drops 8%
tc_profit2 = BTContainer(data=data_profit, stop_loss=-0.10, roi=1,
                         trades=1, profit_perc=0.10557, sell_r=SellType.STOP_LOSS)


tc_loss0 = BTContainer(data=[
    [0, 0.0009910, 0.001011, 0.00098618, 0.001000, 12345, 1, 0],
    [1, 0.001000, 0.001010, 0.0009900, 0.001000, 12345, 0, 0],
    [2, 0.001000, 0.001011, 0.0010618, 0.00091618, 12345, 0, 0],
    [3, 0.001000, 0.001011, 0.00098618, 0.00091618, 12345, 0, 0],
    [4, 0.001000, 0.001011, 0.00098618, 0.00091618, 12345, 0, 0]],
    stop_loss=-0.05, roi=1, trades=1, profit_perc=-0.08839, sell_r=SellType.STOP_LOSS)


# Test 1 Minus 8% Close
# Candle Data for test 1 – close at -8% (9200)
# Test with Stop-loss at 1%
# TC1: Stop-Loss Triggered 1% loss
tc1 = BTContainer(data=[
    [0, 10000.0, 10050, 9950, 9975, 12345, 1, 0],
    [1, 10000, 10050, 9950, 9975, 12345, 0, 0],  # enter trade (signal on last candle)
    [2, 9975, 10025, 9200, 9200, 12345, 0, 0],  # Exit with stoploss hit
    [3, 9950, 10000, 9960, 9955, 12345, 0, 0],
    [4, 9955, 9975, 9955, 9990, 12345, 0, 0],
    [5, 9990, 9990, 9990, 9900, 12345, 0, 0]],
    stop_loss=-0.01, roi=1, trades=1, profit_perc=-0.01, sell_r=SellType.STOP_LOSS)


# Test 2 Minus 4% Low, minus 1% close
# Candle Data for test 2
# Test with Stop-Loss at 3%
# TC2: Stop-Loss Triggered 3% Loss
tc2 = BTContainer(data=[
    [0, 10000, 10050, 9950, 9975, 12345, 1, 0],
    [1, 10000, 10050, 9950, 9975, 12345, 0, 0],  # enter trade (signal on last candle)
    [2, 9975, 10025, 9925, 9950, 12345, 0, 0],
    [3, 9950, 10000, 9600, 9925, 12345, 0, 0],  # Exit with stoploss hit
    [4, 9925, 9975, 9875, 9900, 12345, 0, 0],
    [5, 9900, 9950, 9850, 9900, 12345, 0, 0]],
    stop_loss=-0.03, roi=1, trades=1, profit_perc=-0.03, sell_r=SellType.STOP_LOSS)


# Test 3 Candle drops 4%, Recovers 1%.
#               Entry Criteria Met
# 	            Candle drops 20%
# Candle Data for test 3
# Test with Stop-Loss at 2%
# TC3: Trade-A: Stop-Loss Triggered 2% Loss
#          Trade-B: Stop-Loss Triggered 2% Loss
tc3 = BTContainer(data=[
    [0, 10000, 10050, 9950, 9975, 12345, 1, 0],
    [1, 10000, 10050, 9950, 9975, 12345, 0, 0],  # enter trade (signal on last candle)
    [2, 9975, 10025, 9600, 9950, 12345, 0, 0],
    [3, 9950, 10000, 9900, 9925, 12345, 1, 0],  # enter trade 2 (signal on last candle)
    [4, 9950, 10000, 9900, 9925, 12345, 0, 0],
    [5, 9925, 9975, 8000, 8000, 12345, 0, 0],
    [6, 9900, 9950, 9950, 9900, 12345, 0, 0]],
    stop_loss=-0.02, roi=1, trades=2, profit_perc=-0.04, sell_r=SellType.STOP_LOSS)

# Test 4 Minus 3% / recovery +15%
# Candle Data for test 4 – Candle drops 3% Closed 15% up
# Test with Stop-loss at 2% ROI 6%
# TC4: Stop-Loss Triggered 2% Loss
tc4 = BTContainer(data=[
    [0, 10000, 10050, 9950, 9975, 12345, 1, 0],
    [1, 10000, 10050, 9950, 9975, 12345, 0, 0],  # enter trade (signal on last candle)
    [2, 9975, 11500, 9700, 11500, 12345, 0, 0],
    [3, 9950, 10000, 9900, 9925, 12345, 0, 0],
    [4, 9925, 9975, 9875, 9900, 12345, 0, 0],  # Exit with stoploss hit
    [5, 9900, 9950, 9850, 9900, 12345, 0, 0]],
    stop_loss=-0.02, roi=0.06, trades=1, profit_perc=-0.02, sell_r=SellType.STOP_LOSS)

# Test 5 / Drops 0.5% Closes +20%
# Candle Data for test 5
# Set stop-loss at 1% ROI 3%
# TC5: ROI triggers 3% Gain
tc5 = BTContainer(data=[
    [0, 10000, 10050, 9960, 9975, 12345, 1, 0],
    [1, 10000, 10050, 9960, 9975, 12345, 0, 0],  # enter trade (signal on last candle)
    [2, 9975, 10050, 9950, 9975, 12345, 0, 0],
    [3, 9950, 12000, 9950, 12000, 12345, 0, 0],  # ROI
    [4, 9925, 9975, 9945, 9900, 12345, 0, 0],
    [5, 9900, 9950, 9850, 9900, 12345, 0, 0]],
    stop_loss=-0.01, roi=0.03, trades=1, profit_perc=0.03, sell_r=SellType.ROI)

# Test 6 / Drops 3% / Recovers 6% Positive / Closes 1% positve
# Candle Data for test 6
# Set stop-loss at 2% ROI at 5%
# TC6: Stop-Loss triggers 2% Loss
tc6 = BTContainer(data=[
    [0, 10000, 10050, 9950, 9975, 12345, 1, 0],
    [1, 10000, 10050, 9950, 9975, 12345, 0, 0],  # enter trade (signal on last candle)
    [2, 9975, 10600, 9700, 10100, 12345, 0, 0],  # Exit with stoploss
    [3, 9950, 10000, 9900, 9925, 12345, 0, 0],
    [4, 9925, 9975, 9945, 9900, 12345, 0, 0],
    [5, 9900, 9950, 9850, 9900, 12345, 0, 0]],
    stop_loss=-0.02, roi=0.05, trades=1, profit_perc=-0.02, sell_r=SellType.STOP_LOSS)

# Test 7 - 6% Positive / 1% Negative / Close 1% Positve
# Candle Data for test 7
# Set stop-loss at 2% ROI at 3%
# TC7: ROI Triggers 3% Gain
tc7 = BTContainer(data=[
    [0, 10000, 10050, 9950, 9975, 12345, 1, 0],
    [1, 10000, 10050, 9950, 9975, 12345, 0, 0],  # enter trade (signal on last candle)
    [2, 9975, 10600, 9900, 10100, 12345, 0, 0],  # ROI
    [3, 9950, 10000, 9900, 9925, 12345, 0, 0],
    [4, 9925, 9975, 9945, 9900, 12345, 0, 0],
    [5, 9900, 9950, 9850, 9900, 12345, 0, 0]],
    stop_loss=-0.02, roi=0.03, trades=1, profit_perc=0.03, sell_r=SellType.ROI)

TESTS = [
    # tc_profit1,
    # tc_profit2,
    # tc_loss0,
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
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    # TODO: don't Mock fee to for now
    mocker.patch('freqtrade.exchange.Exchange.get_fee', MagicMock(return_value=0.0))
    patch_exchange(mocker)
    frame = _build_dataframe(data.data)
    backtesting = Backtesting(default_conf)
    backtesting.advise_buy = lambda a, m: frame
    backtesting.advise_sell = lambda a, m: frame
    caplog.set_level(logging.DEBUG)

    pair = 'UNITTEST/BTC'
    # Dummy data as we mock the analyze functions
    data_processed = {pair: DataFrame()}
    results = backtesting.backtest(
        {
            'stake_amount': default_conf['stake_amount'],
            'processed': data_processed,
            'max_open_trades': 10,
        }
    )
    print(results.T)

    assert len(results) == data.trades
    assert round(results["profit_percent"].sum(), 3) == round(data.profit_perc, 3)
    if data.sell_r == SellType.STOP_LOSS:
        assert log_has("Stop loss hit.", caplog.record_tuples)
    else:
        assert not log_has("Stop loss hit.", caplog.record_tuples)
    log_test = (f'Force_selling still open trade UNITTEST/BTC with '
                f'{results.iloc[-1].profit_percent} perc - {results.iloc[-1].profit_abs}')
    if data.sell_r == SellType.FORCE_SELL:
        assert log_has(log_test,
                       caplog.record_tuples)
    else:
        assert not log_has(log_test,
                           caplog.record_tuples)
