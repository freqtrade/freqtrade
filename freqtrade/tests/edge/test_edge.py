# pragma pylint: disable=missing-docstring, C0103, C0330
# pragma pylint: disable=protected-access, too-many-lines, invalid-name, too-many-arguments

import pytest
import logging
from freqtrade.tests.conftest import get_patched_freqtradebot
from freqtrade.edge import Edge, PairInfo
from pandas import DataFrame, to_datetime
from freqtrade.strategy.interface import SellType
from freqtrade.tests.optimize import (BTrade, BTContainer, _build_backtest_dataframe,
                                      _get_frame_time_from_offset)
import arrow
import numpy as np
import math

from unittest.mock import MagicMock

# Cases to be tested:
# 1) Open trade should be removed from the end
# 2) Two complete trades within dataframe (with sell hit for all)
# 3) Entered, sl 1%, candle drops 8% => Trade closed, 1% loss
# 4) Entered, sl 3%, candle drops 4%, recovers to 1% => Trade closed, 3% loss
# 5) Stoploss and sell are hit. should sell on stoploss
####################################################################

ticker_start_time = arrow.get(2018, 10, 3)
ticker_interval_in_minute = 60
_ohlc = {'date': 0, 'buy': 1, 'open': 2, 'high': 3, 'low': 4, 'close': 5, 'sell': 6, 'volume': 7}


# Open trade should be removed from the end
tc0 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 1]],  # enter trade (signal on last candle)
    stop_loss=-0.99, roi=float('inf'), profit_perc=0.00,
    trades=[]
)

# Two complete trades within dataframe(with sell hit for all)
tc1 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 1],  # enter trade (signal on last candle)
    [2, 5000, 5025, 4975, 4987, 6172, 0, 0],  # exit at open
    [3, 5000, 5025, 4975, 4987, 6172, 1, 0],  # no action
    [4, 5000, 5025, 4975, 4987, 6172, 0, 0],  # should enter the trade
    [5, 5000, 5025, 4975, 4987, 6172, 0, 1],  # no action
    [6, 5000, 5025, 4975, 4987, 6172, 0, 0],  # should sell
],
    stop_loss=-0.99, roi=float('inf'), profit_perc=0.00,
    trades=[BTrade(sell_reason=SellType.SELL_SIGNAL, open_tick=1, close_tick=2),
            BTrade(sell_reason=SellType.SELL_SIGNAL, open_tick=4, close_tick=6)]
)

# 3) Entered, sl 1%, candle drops 8% => Trade closed, 1% loss
tc2 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4600, 4987, 6172, 0, 0],  # enter trade, stoploss hit
    [2, 5000, 5025, 4975, 4987, 6172, 0, 0],
],
    stop_loss=-0.01, roi=float('inf'), profit_perc=-0.01,
    trades=[BTrade(sell_reason=SellType.STOP_LOSS, open_tick=1, close_tick=1)]
)

# 4) Entered, sl 3 %, candle drops 4%, recovers to 1 % = > Trade closed, 3 % loss
tc3 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4800, 4987, 6172, 0, 0],  # enter trade, stoploss hit
    [2, 5000, 5025, 4975, 4987, 6172, 0, 0],
],
    stop_loss=-0.03, roi=float('inf'), profit_perc=-0.03,
    trades=[BTrade(sell_reason=SellType.STOP_LOSS, open_tick=1, close_tick=1)]
)

# 5) Stoploss and sell are hit. should sell on stoploss
tc4 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4800, 4987, 6172, 0, 1],  # enter trade, stoploss hit, sell signal
    [2, 5000, 5025, 4975, 4987, 6172, 0, 0],
],
    stop_loss=-0.03, roi=float('inf'), profit_perc=-0.03,
    trades=[BTrade(sell_reason=SellType.STOP_LOSS, open_tick=1, close_tick=1)]
)

TESTS = [
    tc0,
    tc1,
    tc2,
    tc3,
    tc4
]


@pytest.mark.parametrize("data", TESTS)
def test_edge_results(edge_conf, mocker, caplog, data) -> None:
    """
    run functional tests
    """
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    frame = _build_backtest_dataframe(data.data)
    caplog.set_level(logging.DEBUG)
    edge.fee = 0

    trades = edge._find_trades_for_stoploss_range(frame, 'TEST/BTC', [data.stop_loss])
    results = edge._fill_calculable_fields(DataFrame(trades)) if trades else DataFrame()

    print(results)

    assert len(trades) == len(data.trades)

    if not results.empty:
        assert round(results["profit_percent"].sum(), 3) == round(data.profit_perc, 3)

    for c, trade in enumerate(data.trades):
        res = results.iloc[c]
        assert res.exit_type == trade.sell_reason
        assert res.open_time == _get_frame_time_from_offset(trade.open_tick)
        assert res.close_time == _get_frame_time_from_offset(trade.close_tick)


def test_adjust(mocker, default_conf):
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    edge = Edge(default_conf, freqtrade.exchange, freqtrade.strategy)
    mocker.patch('freqtrade.edge.Edge._cached_pairs', mocker.PropertyMock(
        return_value={
            'E/F': PairInfo(-0.01, 0.66, 3.71, 0.50, 1.71, 10, 60),
            'C/D': PairInfo(-0.01, 0.66, 3.71, 0.50, 1.71, 10, 60),
            'N/O': PairInfo(-0.01, 0.66, 3.71, 0.50, 1.71, 10, 60)
        }
    ))

    pairs = ['A/B', 'C/D', 'E/F', 'G/H']
    assert(edge.adjust(pairs) == ['E/F', 'C/D'])


def test_stoploss(mocker, default_conf):
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    edge = Edge(default_conf, freqtrade.exchange, freqtrade.strategy)
    mocker.patch('freqtrade.edge.Edge._cached_pairs', mocker.PropertyMock(
        return_value={
            'E/F': PairInfo(-0.01, 0.66, 3.71, 0.50, 1.71, 10, 60),
            'C/D': PairInfo(-0.01, 0.66, 3.71, 0.50, 1.71, 10, 60),
            'N/O': PairInfo(-0.01, 0.66, 3.71, 0.50, 1.71, 10, 60)
        }
    ))

    assert edge.stoploss('E/F') == -0.01


def _validate_ohlc(buy_ohlc_sell_matrice):
    for index, ohlc in enumerate(buy_ohlc_sell_matrice):
        # if not high < open < low or not high < close < low
        if not ohlc[3] >= ohlc[2] >= ohlc[4] or not ohlc[3] >= ohlc[5] >= ohlc[4]:
            raise Exception('Line ' + str(index + 1) + ' of ohlc has invalid values!')
    return True


def _build_dataframe(buy_ohlc_sell_matrice):
    _validate_ohlc(buy_ohlc_sell_matrice)
    tickers = []
    for ohlc in buy_ohlc_sell_matrice:
        ticker = {
            'date': ticker_start_time.shift(
                minutes=(
                    ohlc[0] *
                    ticker_interval_in_minute)).timestamp *
            1000,
            'buy': ohlc[1],
            'open': ohlc[2],
            'high': ohlc[3],
            'low': ohlc[4],
            'close': ohlc[5],
            'sell': ohlc[6]}
        tickers.append(ticker)

    frame = DataFrame(tickers)
    frame['date'] = to_datetime(frame['date'],
                                unit='ms',
                                utc=True,
                                infer_datetime_format=True)

    return frame


def _time_on_candle(number):
    return np.datetime64(ticker_start_time.shift(
        minutes=(number * ticker_interval_in_minute)).timestamp * 1000, 'ms')


def test_edge_heartbeat_calculate(mocker, edge_conf):
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    heartbeat = edge_conf['edge']['process_throttle_secs']

    # should not recalculate if heartbeat not reached
    edge._last_updated = arrow.utcnow().timestamp - heartbeat + 1

    assert edge.calculate() is False


def mocked_load_data(datadir, pairs=[], ticker_interval='0m', refresh_pairs=False,
                     timerange=None, exchange=None):
    hz = 0.1
    base = 0.001

    ETHBTC = [
        [
            ticker_start_time.shift(minutes=(x * ticker_interval_in_minute)).timestamp * 1000,
            math.sin(x * hz) / 1000 + base,
            math.sin(x * hz) / 1000 + base + 0.0001,
            math.sin(x * hz) / 1000 + base - 0.0001,
            math.sin(x * hz) / 1000 + base,
            123.45
        ] for x in range(0, 500)]

    hz = 0.2
    base = 0.002
    LTCBTC = [
        [
            ticker_start_time.shift(minutes=(x * ticker_interval_in_minute)).timestamp * 1000,
            math.sin(x * hz) / 1000 + base,
            math.sin(x * hz) / 1000 + base + 0.0001,
            math.sin(x * hz) / 1000 + base - 0.0001,
            math.sin(x * hz) / 1000 + base,
            123.45
        ] for x in range(0, 500)]

    pairdata = {'NEO/BTC': ETHBTC, 'LTC/BTC': LTCBTC}
    return pairdata


def test_edge_process_downloaded_data(mocker, default_conf):
    default_conf['datadir'] = None
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.get_fee', MagicMock(return_value=0.001))
    mocker.patch('freqtrade.optimize.load_data', mocked_load_data)
    edge = Edge(default_conf, freqtrade.exchange, freqtrade.strategy)

    assert edge.calculate()
    assert len(edge._cached_pairs) == 2
    assert edge._last_updated <= arrow.utcnow().timestamp + 2


def test_process_expectancy(mocker, edge_conf):
    edge_conf['edge']['min_trade_number'] = 2
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)

    def get_fee():
        return 0.001

    freqtrade.exchange.get_fee = get_fee
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)

    trades = [
        {'pair': 'TEST/BTC',
         'stoploss': -0.9,
         'profit_percent': '',
         'profit_abs': '',
         'open_time': np.datetime64('2018-10-03T00:05:00.000000000'),
         'close_time': np.datetime64('2018-10-03T00:10:00.000000000'),
         'open_index': 1,
         'close_index': 1,
         'trade_duration': '',
         'open_rate': 17,
         'close_rate': 17,
         'exit_type': 'sell_signal'},

        {'pair': 'TEST/BTC',
         'stoploss': -0.9,
         'profit_percent': '',
         'profit_abs': '',
         'open_time': np.datetime64('2018-10-03T00:20:00.000000000'),
         'close_time': np.datetime64('2018-10-03T00:25:00.000000000'),
         'open_index': 4,
         'close_index': 4,
         'trade_duration': '',
         'open_rate': 20,
         'close_rate': 20,
         'exit_type': 'sell_signal'},

        {'pair': 'TEST/BTC',
         'stoploss': -0.9,
         'profit_percent': '',
         'profit_abs': '',
         'open_time': np.datetime64('2018-10-03T00:30:00.000000000'),
         'close_time': np.datetime64('2018-10-03T00:40:00.000000000'),
         'open_index': 6,
         'close_index': 7,
         'trade_duration': '',
         'open_rate': 26,
         'close_rate': 34,
         'exit_type': 'sell_signal'}
    ]

    trades_df = DataFrame(trades)
    trades_df = edge._fill_calculable_fields(trades_df)
    final = edge._process_expectancy(trades_df)
    assert len(final) == 1

    assert 'TEST/BTC' in final
    assert final['TEST/BTC'].stoploss == -0.9
    assert round(final['TEST/BTC'].winrate, 10) == 0.3333333333
    assert round(final['TEST/BTC'].risk_reward_ratio, 10) == 306.5384615384
    assert round(final['TEST/BTC'].required_risk_reward, 10) == 2.0
    assert round(final['TEST/BTC'].expectancy, 10) == 101.5128205128
