from freqtrade.tests.conftest import get_patched_exchange
from freqtrade.edge import Edge
from pandas import DataFrame


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
            'date': ohlc[0],
            'buy': ohlc[1],
            'open': ohlc[2],
            'high': ohlc[3],
            'low': ohlc[4],
            'close': ohlc[5],
            'sell': ohlc[6]
        }
        tickers.append(ticker)
    return DataFrame(tickers)


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

    # Three trades must have happened
    assert len(trades) == 3

    # First trade check
    assert trades[0]['open_time'] == 1
    assert trades[0]['close_time'] == 2

    # Second trade check
    assert trades[1]['open_time'] == 4
    assert trades[1]['close_time'] == 5

    # Third trade check
    assert trades[2]['open_time'] == 6
    assert trades[2]['close_time'] == 7
