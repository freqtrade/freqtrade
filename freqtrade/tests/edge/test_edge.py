from freqtrade.tests.conftest import log_has, get_patched_exchange
from freqtrade.edge import Edge


# Cases to be tested:
############################### SELL POINTS #####################################
# 1) Three complete trades within dataframe (with sell hit for all)
# 2) Two complete trades but one without sell hit (remains open)
# 3) Two complete trades and one buy signal while one trade is open
# 4) Two complete trades with buy=1 on the last frame
################################# STOPLOSS ######################################
# 5) Candle drops 8%, stoploss at 1%: Trade closed, 1% loss
# 6) Candle drops 4% but recovers to 1% loss, stoploss at 3%: Trade closed, 3% loss
# 7) Candle drops 4% recovers to 1% entry criteria are met, candle drops
#    20%, stoploss at 2%: Trade 1 closed, Loss 2%, Trade 2 opened, Trade 2 closed, Loss 2%
############################ PRIORITY TO STOPLOSS ################################
# 8) Stoploss and sell are hit. should sell on stoploss

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


def test_three_complete_trades():
    stoploss = -0.01
    three_sell_points_hit = [
        # Buy, O, H, L, C, Sell
        [1, 15, 20, 12, 17, 0],  # -> should enter the trade
        [1, 17, 18, 13, 14, 1],  # -> should sell (trade 1 completed)
        [0, 14, 15, 11, 12, 0],  # -> no action
        [1, 12, 25, 11, 20, 0],  # -> should enter the trade
        [0, 20, 30, 21, 25, 1],  # -> should sell (trade 2 completed)
        [1, 25, 27, 22, 26, 1],  # -> buy and sell, should enter the trade
        [0, 26, 36, 25, 35, 1],  # -> should sell (trade 3 completed)
    ]
