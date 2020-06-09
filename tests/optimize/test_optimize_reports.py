from pathlib import Path

import pandas as pd
import pytest
from arrow import Arrow

from freqtrade.edge import PairInfo
from freqtrade.optimize.optimize_reports import (
    generate_pair_metrics, generate_edge_table, generate_sell_reason_stats,
    text_table_bt_results, text_table_sell_reason, generate_strategy_metrics,
    text_table_strategy, store_backtest_result)
from freqtrade.strategy.interface import SellType
from tests.conftest import patch_exchange


def test_text_table_bt_results(default_conf, mocker):

    results = pd.DataFrame(
        {
            'pair': ['ETH/BTC', 'ETH/BTC'],
            'profit_percent': [0.1, 0.2],
            'profit_abs': [0.2, 0.4],
            'trade_duration': [10, 30],
            'wins': [2, 0],
            'draws': [0, 0],
            'losses': [0, 0]
        }
    )

    result_str = (
        '|    Pair |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit BTC |'
        '   Tot Profit % |   Avg Duration |   Wins |   Draws |   Losses |\n'
        '|---------+--------+----------------+----------------+------------------+'
        '----------------+----------------+--------+---------+----------|\n'
        '| ETH/BTC |      2 |          15.00 |          30.00 |       0.60000000 |'
        '          15.00 |        0:20:00 |      2 |       0 |        0 |\n'
        '|   TOTAL |      2 |          15.00 |          30.00 |       0.60000000 |'
        '          15.00 |        0:20:00 |      2 |       0 |        0 |'
    )

    pair_results = generate_pair_metrics(data={'ETH/BTC': {}}, stake_currency='BTC',
                                         max_open_trades=2, results=results)
    assert text_table_bt_results(pair_results, stake_currency='BTC') == result_str


def test_generate_pair_metrics(default_conf, mocker):

    results = pd.DataFrame(
        {
            'pair': ['ETH/BTC', 'ETH/BTC'],
            'profit_percent': [0.1, 0.2],
            'profit_abs': [0.2, 0.4],
            'trade_duration': [10, 30],
            'wins': [2, 0],
            'draws': [0, 0],
            'losses': [0, 0]
        }
    )

    pair_results = generate_pair_metrics(data={'ETH/BTC': {}}, stake_currency='BTC',
                                         max_open_trades=2, results=results)
    assert isinstance(pair_results, list)
    assert len(pair_results) == 2
    assert pair_results[-1]['key'] == 'TOTAL'
    assert (
        pytest.approx(pair_results[-1]['profit_mean_pct']) == pair_results[-1]['profit_mean'] * 100)
    assert (
        pytest.approx(pair_results[-1]['profit_sum_pct']) == pair_results[-1]['profit_sum'] * 100)


def test_text_table_sell_reason(default_conf):

    results = pd.DataFrame(
        {
            'pair': ['ETH/BTC', 'ETH/BTC', 'ETH/BTC'],
            'profit_percent': [0.1, 0.2, -0.1],
            'profit_abs': [0.2, 0.4, -0.2],
            'trade_duration': [10, 30, 10],
            'wins': [2, 0, 0],
            'draws': [0, 0, 0],
            'losses': [0, 0, 1],
            'sell_reason': [SellType.ROI, SellType.ROI, SellType.STOP_LOSS]
        }
    )

    result_str = (
        '|   Sell Reason |   Sells |   Wins |   Draws |   Losses |'
        '   Avg Profit % |   Cum Profit % |   Tot Profit BTC |   Tot Profit % |\n'
        '|---------------+---------+--------+---------+----------+'
        '----------------+----------------+------------------+----------------|\n'
        '|           roi |       2 |      2 |       0 |        0 |'
        '             15 |             30 |              0.6 |             15 |\n'
        '|     stop_loss |       1 |      0 |       0 |        1 |'
        '            -10 |            -10 |             -0.2 |             -5 |'
    )

    sell_reason_stats = generate_sell_reason_stats(max_open_trades=2,
                                                   results=results)
    assert text_table_sell_reason(sell_reason_stats=sell_reason_stats,
                                  stake_currency='BTC') == result_str


def test_generate_sell_reason_stats(default_conf):

    results = pd.DataFrame(
        {
            'pair': ['ETH/BTC', 'ETH/BTC', 'ETH/BTC'],
            'profit_percent': [0.1, 0.2, -0.1],
            'profit_abs': [0.2, 0.4, -0.2],
            'trade_duration': [10, 30, 10],
            'wins': [2, 0, 0],
            'draws': [0, 0, 0],
            'losses': [0, 0, 1],
            'sell_reason': [SellType.ROI, SellType.ROI, SellType.STOP_LOSS]
        }
    )

    sell_reason_stats = generate_sell_reason_stats(max_open_trades=2,
                                                   results=results)
    roi_result = sell_reason_stats[0]
    assert roi_result['sell_reason'] == 'roi'
    assert roi_result['trades'] == 2
    assert pytest.approx(roi_result['profit_mean']) == 0.15
    assert roi_result['profit_mean_pct'] == round(roi_result['profit_mean'] * 100, 2)
    assert pytest.approx(roi_result['profit_mean']) == 0.15
    assert roi_result['profit_mean_pct'] == round(roi_result['profit_mean'] * 100, 2)

    stop_result = sell_reason_stats[1]

    assert stop_result['sell_reason'] == 'stop_loss'
    assert stop_result['trades'] == 1
    assert pytest.approx(stop_result['profit_mean']) == -0.1
    assert stop_result['profit_mean_pct'] == round(stop_result['profit_mean'] * 100, 2)
    assert pytest.approx(stop_result['profit_mean']) == -0.1
    assert stop_result['profit_mean_pct'] == round(stop_result['profit_mean'] * 100, 2)


def test_text_table_strategy(default_conf, mocker):
    results = {}
    results['TestStrategy1'] = pd.DataFrame(
        {
            'pair': ['ETH/BTC', 'ETH/BTC', 'ETH/BTC'],
            'profit_percent': [0.1, 0.2, 0.3],
            'profit_abs': [0.2, 0.4, 0.5],
            'trade_duration': [10, 30, 10],
            'wins': [2, 0, 0],
            'draws': [0, 0, 0],
            'losses': [0, 0, 1],
            'sell_reason': [SellType.ROI, SellType.ROI, SellType.STOP_LOSS]
        }
    )
    results['TestStrategy2'] = pd.DataFrame(
        {
            'pair': ['LTC/BTC', 'LTC/BTC', 'LTC/BTC'],
            'profit_percent': [0.4, 0.2, 0.3],
            'profit_abs': [0.4, 0.4, 0.5],
            'trade_duration': [15, 30, 15],
            'wins': [4, 1, 0],
            'draws': [0, 0, 0],
            'losses': [0, 0, 1],
            'sell_reason': [SellType.ROI, SellType.ROI, SellType.STOP_LOSS]
        }
    )

    result_str = (
        '|      Strategy |   Buys |   Avg Profit % |   Cum Profit % |   Tot'
        ' Profit BTC |   Tot Profit % |   Avg Duration |   Wins |   Draws |   Losses |\n'
        '|---------------+--------+----------------+----------------+------------------+'
        '----------------+----------------+--------+---------+----------|\n'
        '| TestStrategy1 |      3 |          20.00 |          60.00 |       1.10000000 |'
        '          30.00 |        0:17:00 |      3 |       0 |        0 |\n'
        '| TestStrategy2 |      3 |          30.00 |          90.00 |       1.30000000 |'
        '          45.00 |        0:20:00 |      3 |       0 |        0 |'
    )

    strategy_results = generate_strategy_metrics(stake_currency='BTC',
                                                 max_open_trades=2,
                                                 all_results=results)

    assert text_table_strategy(strategy_results, 'BTC') == result_str


def test_generate_edge_table(edge_conf, mocker):

    results = {}
    results['ETH/BTC'] = PairInfo(-0.01, 0.60, 2, 1, 3, 10, 60)
    assert generate_edge_table(results).count('+') == 7
    assert generate_edge_table(results).count('| ETH/BTC |') == 1
    assert generate_edge_table(results).count(
        '|   Risk Reward Ratio |   Required Risk Reward |   Expectancy |') == 1


def test_backtest_record(default_conf, fee, mocker):
    names = []
    records = []
    patch_exchange(mocker)
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    mocker.patch(
        'freqtrade.optimize.optimize_reports.file_dump_json',
        new=lambda n, r: (names.append(n), records.append(r))
    )

    results = {'DefStrat': pd.DataFrame({"pair": ["UNITTEST/BTC", "UNITTEST/BTC",
                                                  "UNITTEST/BTC", "UNITTEST/BTC"],
                                         "profit_percent": [0.003312, 0.010801, 0.013803, 0.002780],
                                         "profit_abs": [0.000003, 0.000011, 0.000014, 0.000003],
                                         "open_time": [Arrow(2017, 11, 14, 19, 32, 00).datetime,
                                                       Arrow(2017, 11, 14, 21, 36, 00).datetime,
                                                       Arrow(2017, 11, 14, 22, 12, 00).datetime,
                                                       Arrow(2017, 11, 14, 22, 44, 00).datetime],
                                         "close_time": [Arrow(2017, 11, 14, 21, 35, 00).datetime,
                                                        Arrow(2017, 11, 14, 22, 10, 00).datetime,
                                                        Arrow(2017, 11, 14, 22, 43, 00).datetime,
                                                        Arrow(2017, 11, 14, 22, 58, 00).datetime],
                                         "open_rate": [0.002543, 0.003003, 0.003089, 0.003214],
                                         "close_rate": [0.002546, 0.003014, 0.003103, 0.003217],
                                         "open_index": [1, 119, 153, 185],
                                         "close_index": [118, 151, 184, 199],
                                         "trade_duration": [123, 34, 31, 14],
                                         "open_at_end": [False, False, False, True],
                                         "sell_reason": [SellType.ROI, SellType.STOP_LOSS,
                                                         SellType.ROI, SellType.FORCE_SELL]
                                         })}
    store_backtest_result(Path("backtest-result.json"), results)
    # Assert file_dump_json was only called once
    assert names == [Path('backtest-result.json')]
    records = records[0]
    # Ensure records are of correct type
    assert len(records) == 4

    # reset test to test with strategy name
    names = []
    records = []
    results['Strat'] = results['DefStrat']
    results['Strat2'] = results['DefStrat']
    store_backtest_result(Path("backtest-result.json"), results)
    assert names == [
        Path('backtest-result-DefStrat.json'),
        Path('backtest-result-Strat.json'),
        Path('backtest-result-Strat2.json'),
    ]
    records = records[0]
    # Ensure records are of correct type
    assert len(records) == 4

    # ('UNITTEST/BTC', 0.00331158, '1510684320', '1510691700', 0, 117)
    # Below follows just a typecheck of the schema/type of trade-records
    oix = None
    for (pair, profit, date_buy, date_sell, buy_index, dur,
         openr, closer, open_at_end, sell_reason) in records:
        assert pair == 'UNITTEST/BTC'
        assert isinstance(profit, float)
        # FIX: buy/sell should be converted to ints
        assert isinstance(date_buy, float)
        assert isinstance(date_sell, float)
        assert isinstance(openr, float)
        assert isinstance(closer, float)
        assert isinstance(open_at_end, bool)
        assert isinstance(sell_reason, str)
        isinstance(buy_index, pd._libs.tslib.Timestamp)
        if oix:
            assert buy_index > oix
        oix = buy_index
        assert dur > 0
