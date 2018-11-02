from freqtrade.tests.conftest import get_patched_exchange
from freqtrade.edge import Edge
from pandas import DataFrame, to_datetime
from freqtrade.strategy.interface import SellType
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


def test_filter(mocker, default_conf):
    exchange = get_patched_exchange(mocker, default_conf)
    edge = Edge(default_conf, exchange)
    mocker.patch('freqtrade.edge.Edge._cached_pairs', mocker.PropertyMock(
        return_value=[
            ['E/F', -0.01, 0.66, 3.71, 0.50, 1.71],
            ['C/D', -0.01, 0.66, 3.71, 0.50, 1.71],
            ['N/O', -0.01, 0.66, 3.71, 0.50, 1.71]
        ]
    ))

    pairs = ['A/B', 'C/D', 'E/F', 'G/H']
    assert(edge.filter(pairs) == ['E/F', 'C/D'])


def test_stoploss(mocker, default_conf):
    exchange = get_patched_exchange(mocker, default_conf)
    edge = Edge(default_conf, exchange)
    mocker.patch('freqtrade.edge.Edge._cached_pairs', mocker.PropertyMock(
        return_value=[
            ['E/F', -0.01, 0.66, 3.71, 0.50, 1.71],
            ['C/D', -0.01, 0.66, 3.71, 0.50, 1.71],
            ['N/O', -0.01, 0.66, 3.71, 0.50, 1.71]
        ]
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


def test_edge_heartbeat_calculate(mocker, default_conf):
    exchange = get_patched_exchange(mocker, default_conf)
    edge = Edge(default_conf, exchange)
    heartbeat = default_conf['edge']['process_throttle_secs']

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
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch('freqtrade.exchange.Exchange.get_fee', MagicMock(return_value=0.001))
    mocker.patch('freqtrade.optimize.load_data', mocked_load_data)
    mocker.patch('freqtrade.exchange.Exchange.refresh_tickers', MagicMock())
    edge = Edge(default_conf, exchange)

    assert edge.calculate()
    assert len(edge._cached_pairs) == 2
    assert edge._last_updated <= arrow.utcnow().timestamp + 2


def test_process_expectancy(mocker, default_conf):
    default_conf['edge']['min_trade_number'] = 2
    exchange = get_patched_exchange(mocker, default_conf)

    def get_fee():
        return 0.001

    exchange.get_fee = get_fee
    edge = Edge(default_conf, exchange)

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

    # TODO: check expectancy + win rate etc


def test_remove_open_trade_at_the_end(mocker, default_conf):
    exchange = get_patched_exchange(mocker, default_conf)
    edge = Edge(default_conf, exchange)

    stoploss = -0.99  # we don't want stoploss to be hit in this test
    ticker = [
        # D=Date, B=Buy,  O=Open,  H=High,  L=Low,  C=Close, S=Sell
        # D, B,  O,  H,  L,  C, S
        [3, 1, 12, 25, 11, 20, 0],  # ->
        [4, 0, 20, 30, 19, 25, 1],  # -> should enter the trade
    ]

    ticker_df = _build_dataframe(ticker)
    trades = edge._find_trades_for_stoploss_range(ticker_df, 'TEST/BTC', [stoploss])

    # No trade should be found
    assert len(trades) == 0


def test_two_complete_trades(mocker, default_conf):
    exchange = get_patched_exchange(mocker, default_conf)
    edge = Edge(default_conf, exchange)

    stoploss = -0.99  # we don't want stoploss to be hit in this test
    ticker = [
        # D=Date, B=Buy,  O=Open,  H=High,  L=Low,  C=Close, S=Sell
        # D, B,  O,  H,  L,  C, S
        [0, 1, 15, 20, 12, 17, 0],  # -> no action
        [1, 0, 17, 18, 13, 14, 1],  # -> should enter the trade as B signal recieved on last candle
        [2, 0, 14, 15, 11, 12, 0],  # -> exit the trade as the sell signal recieved on last candle
        [3, 1, 12, 25, 11, 20, 0],  # -> no action
        [4, 0, 20, 30, 19, 25, 0],  # -> should enter the trade
        [5, 0, 25, 27, 22, 26, 1],  # -> no action
        [6, 0, 26, 36, 25, 35, 0],  # -> should sell
    ]

    ticker_df = _build_dataframe(ticker)
    trades = edge._find_trades_for_stoploss_range(ticker_df, 'TEST/BTC', [stoploss])

    # Two trades must have occured
    assert len(trades) == 2

    # First trade check
    assert trades[0]['open_time'] == _time_on_candle(1)
    assert trades[0]['close_time'] == _time_on_candle(2)
    assert trades[0]['open_rate'] == ticker[1][_ohlc['open']]
    assert trades[0]['close_rate'] == ticker[2][_ohlc['open']]
    assert trades[0]['exit_type'] == SellType.SELL_SIGNAL
    ##############################################################

    # Second trade check
    assert trades[1]['open_time'] == _time_on_candle(4)
    assert trades[1]['close_time'] == _time_on_candle(6)
    assert trades[1]['open_rate'] == ticker[4][_ohlc['open']]
    assert trades[1]['close_rate'] == ticker[6][_ohlc['open']]
    assert trades[1]['exit_type'] == SellType.SELL_SIGNAL
    ##############################################################


# 3) Entered, sl 1%, candle drops 8% => Trade closed, 1% loss
def test_case_3(mocker, default_conf):
    exchange = get_patched_exchange(mocker, default_conf)
    edge = Edge(default_conf, exchange)

    stoploss = -0.01  # we don't want stoploss to be hit in this test
    ticker = [
        # D=Date, B=Buy,  O=Open,  H=High,  L=Low,  C=Close, S=Sell
        # D, B,  O,  H,  L,  C, S
        [0, 1, 15, 20, 12, 17, 0],  # -> no action
        [1, 0, 14, 15, 11, 12, 0],  # -> enter to trade, stoploss hit
        [2, 1, 12, 25, 11, 20, 0],  # -> no action
    ]

    ticker_df = _build_dataframe(ticker)
    trades = edge._find_trades_for_stoploss_range(ticker_df, 'TEST/BTC', [stoploss])

    # Two trades must have occured
    assert len(trades) == 1

    # First trade check
    assert trades[0]['open_time'] == _time_on_candle(1)
    assert trades[0]['close_time'] == _time_on_candle(1)
    assert trades[0]['open_rate'] == ticker[1][_ohlc['open']]
    assert trades[0]['close_rate'] == (stoploss + 1) * trades[0]['open_rate']
    assert trades[0]['exit_type'] == SellType.STOP_LOSS
    ##############################################################


# 4) Entered, sl 3 %, candle drops 4%, recovers to 1 % = > Trade closed, 3 % loss
def test_case_4(mocker, default_conf):
    exchange = get_patched_exchange(mocker, default_conf)
    edge = Edge(default_conf, exchange)

    stoploss = -0.03  # we don't want stoploss to be hit in this test
    ticker = [
        # D=Date, B=Buy,  O=Open,  H=High,  L=Low,  C=Close, S=Sell
        # D, B,  O,  H,  L,   C,    S
        [0, 1, 15, 20, 12, 17, 0],  # -> no action
        [1, 0, 17, 22, 16.90, 17, 0],  # -> enter to trade
        [2, 0, 16, 17, 14.4, 15.5, 0],  # -> stoploss hit
        [3, 0, 17, 25, 16.9, 22, 0],  # -> no action
    ]

    ticker_df = _build_dataframe(ticker)
    trades = edge._find_trades_for_stoploss_range(ticker_df, 'TEST/BTC', [stoploss])

    # Two trades must have occured
    assert len(trades) == 1

    # First trade check
    assert trades[0]['open_time'] == _time_on_candle(1)
    assert trades[0]['close_time'] == _time_on_candle(2)
    assert trades[0]['open_rate'] == ticker[1][_ohlc['open']]
    assert trades[0]['close_rate'] == (stoploss + 1) * trades[0]['open_rate']
    assert trades[0]['exit_type'] == SellType.STOP_LOSS
    ##############################################################


# 5) Stoploss and sell are hit. should sell on stoploss
def test_case_5(mocker, default_conf):
    exchange = get_patched_exchange(mocker, default_conf)
    edge = Edge(default_conf, exchange)

    exchange = get_patched_exchange(mocker, default_conf)
    edge = Edge(default_conf, exchange)

    stoploss = -0.03  # we don't want stoploss to be hit in this test
    ticker = [
        # D=Date, B=Buy,  O=Open,  H=High,  L=Low,  C=Close, S=Sell
        # D, B,  O,  H,  L,   C,    S
        [0, 1, 15, 20, 12, 17, 0],  # -> no action
        [1, 0, 17, 22, 16.90, 17, 0],  # -> enter to trade
        [2, 0, 16, 17, 14.4, 15.5, 1],  # -> stoploss hit and also sell signal
        [3, 0, 17, 25, 16.9, 22, 0],  # -> no action
    ]

    ticker_df = _build_dataframe(ticker)
    trades = edge._find_trades_for_stoploss_range(ticker_df, 'TEST/BTC', [stoploss])

    # Two trades must have occured
    assert len(trades) == 1

    # First trade check
    assert trades[0]['open_time'] == _time_on_candle(1)
    assert trades[0]['close_time'] == _time_on_candle(2)
    assert trades[0]['open_rate'] == ticker[1][_ohlc['open']]
    assert trades[0]['close_rate'] == (stoploss + 1) * trades[0]['open_rate']
    assert trades[0]['exit_type'] == SellType.STOP_LOSS
    ##############################################################
