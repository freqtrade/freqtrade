import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from arrow import Arrow

from freqtrade.configuration import TimeRange
from freqtrade.constants import LAST_BT_RESULT_FN
from freqtrade.data import history
from freqtrade.data.btanalysis import get_latest_backtest_filename, load_backtest_data
from freqtrade.edge import PairInfo
from freqtrade.optimize.optimize_reports import (generate_backtest_stats, generate_daily_stats,
                                                 generate_edge_table, generate_pair_metrics,
                                                 generate_sell_reason_stats,
                                                 generate_strategy_metrics, store_backtest_stats,
                                                 text_table_bt_results, text_table_sell_reason,
                                                 text_table_strategy)
from freqtrade.resolvers.strategy_resolver import StrategyResolver
from freqtrade.strategy.interface import SellType
from tests.data.test_history import _backup_file, _clean_test_file


def test_text_table_bt_results():

    results = pd.DataFrame(
        {
            'pair': ['ETH/BTC', 'ETH/BTC'],
            'profit_ratio': [0.1, 0.2],
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
                                         starting_balance=4, results=results)
    assert text_table_bt_results(pair_results, stake_currency='BTC') == result_str


def test_generate_backtest_stats(default_conf, testdatadir):
    default_conf.update({'strategy': 'DefaultStrategy'})
    StrategyResolver.load_strategy(default_conf)

    results = {'DefStrat': {
        'results': pd.DataFrame({"pair": ["UNITTEST/BTC", "UNITTEST/BTC",
                                          "UNITTEST/BTC", "UNITTEST/BTC"],
                                 "profit_ratio": [0.003312, 0.010801, 0.013803, 0.002780],
                                 "profit_abs": [0.000003, 0.000011, 0.000014, 0.000003],
                                 "open_date": [Arrow(2017, 11, 14, 19, 32, 00).datetime,
                                               Arrow(2017, 11, 14, 21, 36, 00).datetime,
                                               Arrow(2017, 11, 14, 22, 12, 00).datetime,
                                               Arrow(2017, 11, 14, 22, 44, 00).datetime],
                                 "close_date": [Arrow(2017, 11, 14, 21, 35, 00).datetime,
                                                Arrow(2017, 11, 14, 22, 10, 00).datetime,
                                                Arrow(2017, 11, 14, 22, 43, 00).datetime,
                                                Arrow(2017, 11, 14, 22, 58, 00).datetime],
                                 "open_rate": [0.002543, 0.003003, 0.003089, 0.003214],
                                 "close_rate": [0.002546, 0.003014, 0.003103, 0.003217],
                                 "trade_duration": [123, 34, 31, 14],
                                 "is_open": [False, False, False, True],
                                 "stake_amount": [0.01, 0.01, 0.01, 0.01],
                                 "sell_reason": [SellType.ROI, SellType.STOP_LOSS,
                                                 SellType.ROI, SellType.FORCE_SELL]
                                 }),
        'config': default_conf,
        'locks': [],
        'final_balance': 1000.02,
        'backtest_start_time': Arrow.utcnow().int_timestamp,
        'backtest_end_time': Arrow.utcnow().int_timestamp,
        }
        }
    timerange = TimeRange.parse_timerange('1510688220-1510700340')
    min_date = Arrow.fromtimestamp(1510688220)
    max_date = Arrow.fromtimestamp(1510700340)
    btdata = history.load_data(testdatadir, '1m', ['UNITTEST/BTC'], timerange=timerange,
                               fill_up_missing=True)

    stats = generate_backtest_stats(btdata, results, min_date, max_date)
    assert isinstance(stats, dict)
    assert 'strategy' in stats
    assert 'DefStrat' in stats['strategy']
    assert 'strategy_comparison' in stats
    strat_stats = stats['strategy']['DefStrat']
    assert strat_stats['backtest_start'] == min_date.datetime
    assert strat_stats['backtest_end'] == max_date.datetime
    assert strat_stats['total_trades'] == len(results['DefStrat']['results'])
    # Above sample had no loosing trade
    assert strat_stats['max_drawdown'] == 0.0

    # Retry with losing trade
    results = {'DefStrat': {
        'results': pd.DataFrame(
            {"pair": ["UNITTEST/BTC", "UNITTEST/BTC", "UNITTEST/BTC", "UNITTEST/BTC"],
             "profit_ratio": [0.003312, 0.010801, -0.013803, 0.002780],
             "profit_abs": [0.000003, 0.000011, -0.000014, 0.000003],
             "open_date": [Arrow(2017, 11, 14, 19, 32, 00).datetime,
                           Arrow(2017, 11, 14, 21, 36, 00).datetime,
                           Arrow(2017, 11, 14, 22, 12, 00).datetime,
                           Arrow(2017, 11, 14, 22, 44, 00).datetime],
             "close_date": [Arrow(2017, 11, 14, 21, 35, 00).datetime,
                            Arrow(2017, 11, 14, 22, 10, 00).datetime,
                            Arrow(2017, 11, 14, 22, 43, 00).datetime,
                            Arrow(2017, 11, 14, 22, 58, 00).datetime],
             "open_rate": [0.002543, 0.003003, 0.003089, 0.003214],
             "close_rate": [0.002546, 0.003014, 0.0032903, 0.003217],
             "trade_duration": [123, 34, 31, 14],
             "is_open": [False, False, False, True],
             "stake_amount": [0.01, 0.01, 0.01, 0.01],
             "sell_reason": [SellType.ROI, SellType.ROI,
                             SellType.STOP_LOSS, SellType.FORCE_SELL]
             }),
        'config': default_conf,
        'locks': [],
        'final_balance': 1000.02,
        'backtest_start_time': Arrow.utcnow().int_timestamp,
        'backtest_end_time': Arrow.utcnow().int_timestamp,
        }
    }

    stats = generate_backtest_stats(btdata, results, min_date, max_date)
    assert isinstance(stats, dict)
    assert 'strategy' in stats
    assert 'DefStrat' in stats['strategy']
    assert 'strategy_comparison' in stats
    strat_stats = stats['strategy']['DefStrat']

    assert strat_stats['max_drawdown'] == 0.013803
    assert strat_stats['drawdown_start'] == datetime(2017, 11, 14, 22, 10, tzinfo=timezone.utc)
    assert strat_stats['drawdown_end'] == datetime(2017, 11, 14, 22, 43, tzinfo=timezone.utc)
    assert strat_stats['drawdown_end_ts'] == 1510699380000
    assert strat_stats['drawdown_start_ts'] == 1510697400000
    assert strat_stats['pairlist'] == ['UNITTEST/BTC']

    # Test storing stats
    filename = Path(testdatadir / 'btresult.json')
    filename_last = Path(testdatadir / LAST_BT_RESULT_FN)
    _backup_file(filename_last, copy_file=True)
    assert not filename.is_file()

    store_backtest_stats(filename, stats)

    # get real Filename (it's btresult-<date>.json)
    last_fn = get_latest_backtest_filename(filename_last.parent)
    assert re.match(r"btresult-.*\.json", last_fn)

    filename1 = (testdatadir / last_fn)
    assert filename1.is_file()
    content = filename1.read_text()
    assert 'max_drawdown' in content
    assert 'strategy' in content
    assert 'pairlist' in content

    assert filename_last.is_file()

    _clean_test_file(filename_last)
    filename1.unlink()


def test_store_backtest_stats(testdatadir, mocker):

    dump_mock = mocker.patch('freqtrade.optimize.optimize_reports.file_dump_json')

    store_backtest_stats(testdatadir, {})

    assert dump_mock.call_count == 2
    assert isinstance(dump_mock.call_args_list[0][0][0], Path)
    assert str(dump_mock.call_args_list[0][0][0]).startswith(str(testdatadir/'backtest-result'))

    dump_mock.reset_mock()
    filename = testdatadir / 'testresult.json'
    store_backtest_stats(filename, {})
    assert dump_mock.call_count == 2
    assert isinstance(dump_mock.call_args_list[0][0][0], Path)
    # result will be testdatadir / testresult-<timestamp>.json
    assert str(dump_mock.call_args_list[0][0][0]).startswith(str(testdatadir / 'testresult'))


def test_generate_pair_metrics():

    results = pd.DataFrame(
        {
            'pair': ['ETH/BTC', 'ETH/BTC'],
            'profit_ratio': [0.1, 0.2],
            'profit_abs': [0.2, 0.4],
            'trade_duration': [10, 30],
            'wins': [2, 0],
            'draws': [0, 0],
            'losses': [0, 0]
        }
    )

    pair_results = generate_pair_metrics(data={'ETH/BTC': {}}, stake_currency='BTC',
                                         starting_balance=2, results=results)
    assert isinstance(pair_results, list)
    assert len(pair_results) == 2
    assert pair_results[-1]['key'] == 'TOTAL'
    assert (
        pytest.approx(pair_results[-1]['profit_mean_pct']) == pair_results[-1]['profit_mean'] * 100)
    assert (
        pytest.approx(pair_results[-1]['profit_sum_pct']) == pair_results[-1]['profit_sum'] * 100)


def test_generate_daily_stats(testdatadir):

    filename = testdatadir / "backtest-result_new.json"
    bt_data = load_backtest_data(filename)
    res = generate_daily_stats(bt_data)
    assert isinstance(res, dict)
    assert round(res['backtest_best_day'], 4) == 0.1796
    assert round(res['backtest_worst_day'], 4) == -0.1468
    assert res['winning_days'] == 14
    assert res['draw_days'] == 4
    assert res['losing_days'] == 3
    assert res['winner_holding_avg'] == timedelta(seconds=1440)
    assert res['loser_holding_avg'] == timedelta(days=1, seconds=21420)

    # Select empty dataframe!
    res = generate_daily_stats(bt_data.loc[bt_data['open_date'] == '2000-01-01', :])
    assert isinstance(res, dict)
    assert round(res['backtest_best_day'], 4) == 0.0
    assert res['winning_days'] == 0
    assert res['draw_days'] == 0
    assert res['losing_days'] == 0


def test_text_table_sell_reason():

    results = pd.DataFrame(
        {
            'pair': ['ETH/BTC', 'ETH/BTC', 'ETH/BTC'],
            'profit_ratio': [0.1, 0.2, -0.1],
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


def test_generate_sell_reason_stats():

    results = pd.DataFrame(
        {
            'pair': ['ETH/BTC', 'ETH/BTC', 'ETH/BTC'],
            'profit_ratio': [0.1, 0.2, -0.1],
            'profit_abs': [0.2, 0.4, -0.2],
            'trade_duration': [10, 30, 10],
            'wins': [2, 0, 0],
            'draws': [0, 0, 0],
            'losses': [0, 0, 1],
            'sell_reason': [SellType.ROI.value, SellType.ROI.value, SellType.STOP_LOSS.value]
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


def test_text_table_strategy(default_conf):
    default_conf['max_open_trades'] = 2
    default_conf['dry_run_wallet'] = 3
    results = {}
    results['TestStrategy1'] = {'results': pd.DataFrame(
        {
            'pair': ['ETH/BTC', 'ETH/BTC', 'ETH/BTC'],
            'profit_ratio': [0.1, 0.2, 0.3],
            'profit_abs': [0.2, 0.4, 0.5],
            'trade_duration': [10, 30, 10],
            'wins': [2, 0, 0],
            'draws': [0, 0, 0],
            'losses': [0, 0, 1],
            'sell_reason': [SellType.ROI, SellType.ROI, SellType.STOP_LOSS]
        }
    ), 'config': default_conf}
    results['TestStrategy2'] = {'results': pd.DataFrame(
        {
            'pair': ['LTC/BTC', 'LTC/BTC', 'LTC/BTC'],
            'profit_ratio': [0.4, 0.2, 0.3],
            'profit_abs': [0.4, 0.4, 0.5],
            'trade_duration': [15, 30, 15],
            'wins': [4, 1, 0],
            'draws': [0, 0, 0],
            'losses': [0, 0, 1],
            'sell_reason': [SellType.ROI, SellType.ROI, SellType.STOP_LOSS]
        }
    ), 'config': default_conf}

    result_str = (
        '|      Strategy |   Buys |   Avg Profit % |   Cum Profit % |   Tot'
        ' Profit BTC |   Tot Profit % |   Avg Duration |   Wins |   Draws |   Losses |\n'
        '|---------------+--------+----------------+----------------+------------------+'
        '----------------+----------------+--------+---------+----------|\n'
        '| TestStrategy1 |      3 |          20.00 |          60.00 |       1.10000000 |'
        '          36.67 |        0:17:00 |      3 |       0 |        0 |\n'
        '| TestStrategy2 |      3 |          30.00 |          90.00 |       1.30000000 |'
        '          43.33 |        0:20:00 |      3 |       0 |        0 |'
    )

    strategy_results = generate_strategy_metrics(all_results=results)

    assert text_table_strategy(strategy_results, 'BTC') == result_str


def test_generate_edge_table():

    results = {}
    results['ETH/BTC'] = PairInfo(-0.01, 0.60, 2, 1, 3, 10, 60)
    assert generate_edge_table(results).count('+') == 7
    assert generate_edge_table(results).count('| ETH/BTC |') == 1
    assert generate_edge_table(results).count(
        '|   Risk Reward Ratio |   Required Risk Reward |   Expectancy |') == 1
