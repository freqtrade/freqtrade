from freqtrade.tests.conftest import get_patched_exchange
from freqtrade.edge import Edge
from pandas import DataFrame, to_datetime
import arrow
import numpy as np


# Cases to be tested:
# SELL POINTS:
# 1) Three complete trades within dataframe (with sell hit for all)
# 2) Two complete trades but one without sell hit (remains open)
# 3) Two complete trades and one buy signal while one trade is open
# 4) Two complete trades with buy=1 on the last frame
###################################################################
# STOPLOSS:
# 5) Candle drops 8%, stoploss at 1%: Trade closed, 1% loss
# 6) Candle drops 4% but recovers to 1% loss, stoploss at 3%: Trade closed, 3% loss
# 7) Candle drops 4% recovers to 1% entry criteria are met, candle drops
#    20%, stoploss at 2%: Trade 1 closed, Loss 2%, Trade 2 opened, Trade 2 closed, Loss 2%
####################################################################
# PRIORITY TO STOPLOSS:
# 8) Stoploss and sell are hit. should sell on stoploss
####################################################################

ticker_start_time = arrow.get(2018, 10, 3)
ticker_interval_in_minute = 5


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

    pairs = ['A/B', 'C/D', 'E/F', 'G/H']
    assert edge.stoploss('E/F') == -0.01


def _validate_ohlc(buy_ohlc_sell_matrice):
    for index, ohlc in enumerate(buy_ohlc_sell_matrice):
        # if not high < open < low or not high < close < low
        if not ohlc[3] > ohlc[2] > ohlc[4] or not ohlc[3] > ohlc[5] > ohlc[4]:
            raise Exception('Line ' + str(index + 1) + ' of ohlc has invalid values!')
    return True


def _build_dataframe(buy_ohlc_sell_matrice):
    _validate_ohlc(buy_ohlc_sell_matrice)
    tickers = []
    for ohlc in buy_ohlc_sell_matrice:
        ticker = {
            # ticker every 5 min
            'date': ticker_start_time.shift(minutes=(ohlc[0] * 5)).timestamp * 1000,
            'buy': ohlc[1],
            'open': ohlc[2],
            'high': ohlc[3],
            'low': ohlc[4],
            'close': ohlc[5],
            'sell': ohlc[6]
        }
        tickers.append(ticker)

    frame = DataFrame(tickers)
    frame['date'] = to_datetime(frame['date'],
                                unit='ms',
                                utc=True,
                                infer_datetime_format=True)

    return frame


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
         'exit_type': 'sell_signal'},  # sdfsdf

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


def test_three_complete_trades(mocker, default_conf):
    exchange = get_patched_exchange(mocker, default_conf)
    edge = Edge(default_conf, exchange)

    stoploss = -0.90  # we don't want stoploss to be hit in this test
    three_sell_points_hit = [
        # Date, Buy, O, H, L, C, Sell
        [1, 1, 15, 20, 12, 17, 0],  # -> should enter the trade
        [2, 0, 17, 18, 13, 14, 1],  # -> should sell (trade 1 completed)
        [3, 0, 14, 15, 11, 12, 0],  # -> no action
        [4, 1, 12, 25, 11, 20, 0],  # -> should enter the trade
        [5, 0, 20, 30, 19, 25, 1],  # -> should sell (trade 2 completed)
        [6, 1, 25, 27, 22, 26, 1],  # -> buy and sell, should enter the trade
        [7, 0, 26, 36, 25, 35, 1],  # -> should sell (trade 3 completed)
    ]

    ticker_df = _build_dataframe(three_sell_points_hit)
    trades = edge._find_trades_for_stoploss_range(ticker_df, 'TEST/BTC', [stoploss])

    # Three trades must have occured
    assert len(trades) == 3

    # First trade check
    # open time should be on line 1
    assert trades[0]['open_time'] == np.datetime64(ticker_start_time.shift(
        minutes=(1 * ticker_interval_in_minute)).timestamp * 1000, 'ms')

    # close time should be on line 2
    assert trades[0]['close_time'] == np.datetime64(ticker_start_time.shift(
        minutes=(2 * ticker_interval_in_minute)).timestamp * 1000, 'ms')

    # Second trade check
    # open time should be on line 4
    assert trades[1]['open_time'] == np.datetime64(ticker_start_time.shift(
        minutes=(4 * ticker_interval_in_minute)).timestamp * 1000, 'ms')

    # close time should be on line 5
    assert trades[1]['close_time'] == np.datetime64(ticker_start_time.shift(
        minutes=(5 * ticker_interval_in_minute)).timestamp * 1000, 'ms')

    # Third trade check
    # open time should be on line 6
    assert trades[2]['open_time'] == np.datetime64(ticker_start_time.shift(
        minutes=(6 * ticker_interval_in_minute)).timestamp * 1000, 'ms')

    # close time should be on line 7
    assert trades[2]['close_time'] == np.datetime64(ticker_start_time.shift(
        minutes=(7 * ticker_interval_in_minute)).timestamp * 1000, 'ms')
