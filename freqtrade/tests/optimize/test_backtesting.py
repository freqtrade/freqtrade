# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, unused-argument

import json
import math
import random
from copy import deepcopy
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from arrow import Arrow

from freqtrade import DependencyException, constants, optimize
from freqtrade.arguments import Arguments, TimeRange
from freqtrade.optimize.backtesting import (Backtesting, setup_configuration,
                                            start)
from freqtrade.tests.conftest import log_has, patch_exchange
from freqtrade.strategy.interface import SellType
from freqtrade.strategy.default_strategy import DefaultStrategy


def get_args(args) -> List[str]:
    return Arguments(args, '').get_parsed_arg()


def trim_dictlist(dict_list, num):
    new = {}
    for pair, pair_data in dict_list.items():
        new[pair] = pair_data[num:]
    return new


def load_data_test(what):
    timerange = TimeRange(None, 'line', 0, -101)
    data = optimize.load_data(None, ticker_interval='1m',
                              pairs=['UNITTEST/BTC'], timerange=timerange)
    pair = data['UNITTEST/BTC']
    datalen = len(pair)
    # Depending on the what parameter we now adjust the
    # loaded data looks:
    # pair :: [[    1509836520000,   unix timestamp in ms
    #               0.00162008,      open
    #               0.00162008,      high
    #               0.00162008,      low
    #               0.00162008,      close
    #               108.14853839     base volume
    #           ]]
    base = 0.001
    if what == 'raise':
        return {'UNITTEST/BTC': [
            [
                pair[x][0],  # Keep old dates
                x * base,  # But replace O,H,L,C
                x * base + 0.0001,
                x * base - 0.0001,
                x * base,
                pair[x][5],  # Keep old volume
            ] for x in range(0, datalen)
        ]}
    if what == 'lower':
        return {'UNITTEST/BTC': [
            [
                pair[x][0],  # Keep old dates
                1 - x * base,  # But replace O,H,L,C
                1 - x * base + 0.0001,
                1 - x * base - 0.0001,
                1 - x * base,
                pair[x][5]  # Keep old volume
            ] for x in range(0, datalen)
        ]}
    if what == 'sine':
        hz = 0.1  # frequency
        return {'UNITTEST/BTC': [
            [
                pair[x][0],  # Keep old dates
                math.sin(x * hz) / 1000 + base,  # But replace O,H,L,C
                math.sin(x * hz) / 1000 + base + 0.0001,
                math.sin(x * hz) / 1000 + base - 0.0001,
                math.sin(x * hz) / 1000 + base,
                pair[x][5]  # Keep old volume
            ] for x in range(0, datalen)
        ]}
    return data


def simple_backtest(config, contour, num_results, mocker) -> None:
    patch_exchange(mocker)
    backtesting = Backtesting(config)

    data = load_data_test(contour)
    processed = backtesting.tickerdata_to_dataframe(data)
    assert isinstance(processed, dict)
    results = backtesting.backtest(
        {
            'stake_amount': config['stake_amount'],
            'processed': processed,
            'max_open_trades': 1,
            'realistic': True
        }
    )
    # results :: <class 'pandas.core.frame.DataFrame'>
    assert len(results) == num_results


def mocked_load_data(datadir, pairs=[], ticker_interval='0m', refresh_pairs=False,
                     timerange=None, exchange=None):
    tickerdata = optimize.load_tickerdata_file(datadir, 'UNITTEST/BTC', '1m', timerange=timerange)
    pairdata = {'UNITTEST/BTC': tickerdata}
    return pairdata


# use for mock freqtrade.exchange.get_ticker_history'
def _load_pair_as_ticks(pair, tickfreq):
    ticks = optimize.load_data(None, ticker_interval=tickfreq, pairs=[pair])
    ticks = trim_dictlist(ticks, -201)
    return ticks[pair]


# FIX: fixturize this?
def _make_backtest_conf(mocker, conf=None, pair='UNITTEST/BTC', record=None):
    data = optimize.load_data(None, ticker_interval='8m', pairs=[pair])
    data = trim_dictlist(data, -201)
    patch_exchange(mocker)
    backtesting = Backtesting(conf)
    return {
        'stake_amount': conf['stake_amount'],
        'processed': backtesting.tickerdata_to_dataframe(data),
        'max_open_trades': 10,
        'realistic': True,
        'record': record
    }


def _trend(signals, buy_value, sell_value):
    n = len(signals['low'])
    buy = np.zeros(n)
    sell = np.zeros(n)
    for i in range(0, len(signals['buy'])):
        if random.random() > 0.5:  # Both buy and sell signals at same timeframe
            buy[i] = buy_value
            sell[i] = sell_value
    signals['buy'] = buy
    signals['sell'] = sell
    return signals


def _trend_alternate(dataframe=None):
    signals = dataframe
    low = signals['low']
    n = len(low)
    buy = np.zeros(n)
    sell = np.zeros(n)
    for i in range(0, len(buy)):
        if i % 2 == 0:
            buy[i] = 1
        else:
            sell[i] = 1
    signals['buy'] = buy
    signals['sell'] = sell
    return dataframe


# Unit tests
def test_setup_configuration_without_arguments(mocker, default_conf, caplog) -> None:
    """
    Test setup_configuration() function
    """
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))

    args = [
        '--config', 'config.json',
        '--strategy', 'DefaultStrategy',
        'backtesting'
    ]

    config = setup_configuration(get_args(args))
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert log_has(
        'Using data folder: {} ...'.format(config['datadir']),
        caplog.record_tuples
    )
    assert 'ticker_interval' in config
    assert not log_has('Parameter -i/--ticker-interval detected ...', caplog.record_tuples)

    assert 'live' not in config
    assert not log_has('Parameter -l/--live detected ...', caplog.record_tuples)

    assert 'realistic_simulation' not in config
    assert not log_has('Parameter --realistic-simulation detected ...', caplog.record_tuples)

    assert 'refresh_pairs' not in config
    assert not log_has('Parameter -r/--refresh-pairs-cached detected ...', caplog.record_tuples)

    assert 'timerange' not in config
    assert 'export' not in config


def test_setup_configuration_with_arguments(mocker, default_conf, caplog) -> None:
    """
    Test setup_configuration() function
    """
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))

    args = [
        '--config', 'config.json',
        '--strategy', 'DefaultStrategy',
        '--datadir', '/foo/bar',
        'backtesting',
        '--ticker-interval', '1m',
        '--live',
        '--realistic-simulation',
        '--refresh-pairs-cached',
        '--timerange', ':100',
        '--export', '/bar/foo',
        '--export-filename', 'foo_bar.json'
    ]

    config = setup_configuration(get_args(args))
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert log_has(
        'Using data folder: {} ...'.format(config['datadir']),
        caplog.record_tuples
    )
    assert 'ticker_interval' in config
    assert log_has('Parameter -i/--ticker-interval detected ...', caplog.record_tuples)
    assert log_has(
        'Using ticker_interval: 1m ...',
        caplog.record_tuples
    )

    assert 'live' in config
    assert log_has('Parameter -l/--live detected ...', caplog.record_tuples)

    assert 'realistic_simulation' in config
    assert log_has('Parameter --realistic-simulation detected ...', caplog.record_tuples)
    assert log_has('Using max_open_trades: 1 ...', caplog.record_tuples)

    assert 'refresh_pairs' in config
    assert log_has('Parameter -r/--refresh-pairs-cached detected ...', caplog.record_tuples)
    assert 'timerange' in config
    assert log_has(
        'Parameter --timerange detected: {} ...'.format(config['timerange']),
        caplog.record_tuples
    )

    assert 'export' in config
    assert log_has(
        'Parameter --export detected: {} ...'.format(config['export']),
        caplog.record_tuples
    )
    assert 'exportfilename' in config
    assert log_has(
        'Storing backtest results to {} ...'.format(config['exportfilename']),
        caplog.record_tuples
    )


def test_setup_configuration_unlimited_stake_amount(mocker, default_conf, caplog) -> None:
    """
    Test setup_configuration() function
    """

    conf = deepcopy(default_conf)
    conf['stake_amount'] = constants.UNLIMITED_STAKE_AMOUNT

    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(conf)
    ))

    args = [
        '--config', 'config.json',
        '--strategy', 'DefaultStrategy',
        'backtesting'
    ]

    with pytest.raises(DependencyException, match=r'.*stake amount.*'):
        setup_configuration(get_args(args))


def test_start(mocker, fee, default_conf, caplog) -> None:
    """
    Test start() function
    """
    start_mock = MagicMock()
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.start', start_mock)
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))
    args = [
        '--config', 'config.json',
        '--strategy', 'DefaultStrategy',
        'backtesting'
    ]
    args = get_args(args)
    start(args)
    assert log_has(
        'Starting freqtrade in Backtesting mode',
        caplog.record_tuples
    )
    assert start_mock.call_count == 1


def test_backtesting_init(mocker, default_conf) -> None:
    """
    Test Backtesting._init() method
    """
    patch_exchange(mocker)
    get_fee = mocker.patch('freqtrade.exchange.Exchange.get_fee', MagicMock(return_value=0.5))
    backtesting = Backtesting(default_conf)
    assert backtesting.config == default_conf
    assert backtesting.ticker_interval == '5m'
    assert callable(backtesting.tickerdata_to_dataframe)
    assert callable(backtesting.populate_buy_trend)
    assert callable(backtesting.populate_sell_trend)
    get_fee.assert_called()
    assert backtesting.fee == 0.5


def test_tickerdata_to_dataframe(default_conf, mocker) -> None:
    """
    Test Backtesting.tickerdata_to_dataframe() method
    """
    patch_exchange(mocker)
    timerange = TimeRange(None, 'line', 0, -100)
    tick = optimize.load_tickerdata_file(None, 'UNITTEST/BTC', '1m', timerange=timerange)
    tickerlist = {'UNITTEST/BTC': tick}

    backtesting = Backtesting(default_conf)
    data = backtesting.tickerdata_to_dataframe(tickerlist)
    assert len(data['UNITTEST/BTC']) == 99

    # Load strategy to compare the result between Backtesting function and strategy are the same
    strategy = DefaultStrategy(default_conf)
    data2 = strategy.tickerdata_to_dataframe(tickerlist)
    assert data['UNITTEST/BTC'].equals(data2['UNITTEST/BTC'])


def test_get_timeframe(default_conf, mocker) -> None:
    """
    Test Backtesting.get_timeframe() method
    """
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)

    data = backtesting.tickerdata_to_dataframe(
        optimize.load_data(
            None,
            ticker_interval='1m',
            pairs=['UNITTEST/BTC']
        )
    )
    min_date, max_date = backtesting.get_timeframe(data)
    assert min_date.isoformat() == '2017-11-04T23:02:00+00:00'
    assert max_date.isoformat() == '2017-11-14T22:58:00+00:00'


def test_generate_text_table(default_conf, mocker):
    """
    Test Backtesting.generate_text_table() method
    """
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)

    results = pd.DataFrame(
        {
            'pair': ['ETH/BTC', 'ETH/BTC'],
            'profit_percent': [0.1, 0.2],
            'profit_abs': [0.2, 0.4],
            'trade_duration': [10, 30],
            'profit': [2, 0],
            'loss': [0, 0]
        }
    )

    result_str = (
        '| pair    |   buy count |   avg profit % |   cum profit % |   '
        'total profit BTC |   avg duration |   profit |   loss |\n'
        '|:--------|------------:|---------------:|---------------:|'
        '-------------------:|---------------:|---------:|-------:|\n'
        '| ETH/BTC |           2 |          15.00 |          30.00 |         '
        '0.60000000 |           20.0 |        2 |      0 |\n'
        '| TOTAL   |           2 |          15.00 |          30.00 |         '
        '0.60000000 |           20.0 |        2 |      0 |'
    )
    print(result_str)
    assert backtesting._generate_text_table(data={'ETH/BTC': {}}, results=results) == result_str


def test_backtesting_start(default_conf, mocker, caplog) -> None:
    """
    Test Backtesting.start() method
    """

    def get_timeframe(input1, input2):
        return Arrow(2017, 11, 14, 21, 17), Arrow(2017, 11, 14, 22, 59)

    mocker.patch('freqtrade.optimize.load_data', mocked_load_data)
    mocker.patch('freqtrade.exchange.Exchange.get_ticker_history')
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.optimize.backtesting.Backtesting',
        backtest=MagicMock(),
        _generate_text_table=MagicMock(return_value='1'),
        get_timeframe=get_timeframe,
    )

    conf = deepcopy(default_conf)
    conf['exchange']['pair_whitelist'] = ['UNITTEST/BTC']
    conf['ticker_interval'] = 1
    conf['live'] = False
    conf['datadir'] = None
    conf['export'] = None
    conf['timerange'] = '-100'

    backtesting = Backtesting(conf)
    backtesting.start()
    # check the logs, that will contain the backtest result
    exists = [
        'Using local backtesting data (using whitelist in given config) ...',
        'Using stake_currency: BTC ...',
        'Using stake_amount: 0.001 ...',
        'Measuring data from 2017-11-14T21:17:00+00:00 '
        'up to 2017-11-14T22:59:00+00:00 (0 days)..'
    ]
    for line in exists:
        assert log_has(line, caplog.record_tuples)


def test_backtesting_start_no_data(default_conf, mocker, caplog) -> None:
    """
    Test Backtesting.start() method if no data is found
    """

    def get_timeframe(input1, input2):
        return Arrow(2017, 11, 14, 21, 17), Arrow(2017, 11, 14, 22, 59)

    mocker.patch('freqtrade.optimize.load_data', MagicMock(return_value={}))
    mocker.patch('freqtrade.exchange.Exchange.get_ticker_history')
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.optimize.backtesting.Backtesting',
        backtest=MagicMock(),
        _generate_text_table=MagicMock(return_value='1'),
        get_timeframe=get_timeframe,
    )

    conf = deepcopy(default_conf)
    conf['exchange']['pair_whitelist'] = ['UNITTEST/BTC']
    conf['ticker_interval'] = "1m"
    conf['live'] = False
    conf['datadir'] = None
    conf['export'] = None
    conf['timerange'] = '20180101-20180102'

    backtesting = Backtesting(conf)
    backtesting.start()
    # check the logs, that will contain the backtest result

    assert log_has('No data found. Terminating.', caplog.record_tuples)


def test_backtest(default_conf, fee, mocker) -> None:
    """
    Test Backtesting.backtest() method
    """
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)
    pair = 'UNITTEST/BTC'
    data = optimize.load_data(None, ticker_interval='5m', pairs=['UNITTEST/BTC'])
    data = trim_dictlist(data, -200)
    data_processed = backtesting.tickerdata_to_dataframe(data)
    results = backtesting.backtest(
        {
            'stake_amount': default_conf['stake_amount'],
            'processed': data_processed,
            'max_open_trades': 10,
            'realistic': True
        }
    )
    assert not results.empty
    assert len(results) == 2

    expected = pd.DataFrame(
        {'pair': [pair, pair],
         'profit_percent': [0.00029975, 0.00056708],
         'profit_abs': [1.49e-06, 7.6e-07],
         'open_time': [Arrow(2018, 1, 29, 18, 40, 0).datetime,
                       Arrow(2018, 1, 30, 3, 30, 0).datetime],
         'close_time': [Arrow(2018, 1, 29, 22, 40, 0).datetime,
                        Arrow(2018, 1, 30, 4, 20, 0).datetime],
         'open_index': [77, 183],
         'close_index': [125, 193],
         'trade_duration': [240, 50],
         'open_at_end': [False, False],
         'open_rate': [0.104445, 0.10302485],
         'close_rate': [0.105, 0.10359999],
         'sell_reason': [SellType.ROI, SellType.ROI]
         })
    pd.testing.assert_frame_equal(results, expected)
    data_pair = data_processed[pair]
    for _, t in results.iterrows():
        ln = data_pair.loc[data_pair["date"] == t["open_time"]]
        # Check open trade rate alignes to open rate
        assert ln is not None
        assert round(ln.iloc[0]["open"], 6) == round(t["open_rate"], 6)
        # check close trade rate alignes to close rate
        ln = data_pair.loc[data_pair["date"] == t["close_time"]]
        assert round(ln.iloc[0]["open"], 6) == round(t["close_rate"], 6)


def test_backtest_1min_ticker_interval(default_conf, fee, mocker) -> None:
    """
    Test Backtesting.backtest() method with 1 min ticker
    """
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)

    # Run a backtesting for an exiting 5min ticker_interval
    data = optimize.load_data(None, ticker_interval='1m', pairs=['UNITTEST/BTC'])
    data = trim_dictlist(data, -200)
    results = backtesting.backtest(
        {
            'stake_amount': default_conf['stake_amount'],
            'processed': backtesting.tickerdata_to_dataframe(data),
            'max_open_trades': 1,
            'realistic': True
        }
    )
    assert not results.empty
    assert len(results) == 1


def test_processed(default_conf, mocker) -> None:
    """
    Test Backtesting.backtest() method with offline data
    """
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)

    dict_of_tickerrows = load_data_test('raise')
    dataframes = backtesting.tickerdata_to_dataframe(dict_of_tickerrows)
    dataframe = dataframes['UNITTEST/BTC']
    cols = dataframe.columns
    # assert the dataframe got some of the indicator columns
    for col in ['close', 'high', 'low', 'open', 'date',
                'ema50', 'ao', 'macd', 'plus_dm']:
        assert col in cols


def test_backtest_pricecontours(default_conf, fee, mocker) -> None:
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    tests = [['raise', 18], ['lower', 0], ['sine', 16]]
    for [contour, numres] in tests:
        simple_backtest(default_conf, contour, numres, mocker)


# Test backtest using offline data (testdata directory)
def test_backtest_ticks(default_conf, fee, mocker):
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    patch_exchange(mocker)
    ticks = [1, 5]
    fun = Backtesting(default_conf).populate_buy_trend
    for _ in ticks:
        backtest_conf = _make_backtest_conf(mocker, conf=default_conf)
        backtesting = Backtesting(default_conf)
        backtesting.populate_buy_trend = fun  # Override
        backtesting.populate_sell_trend = fun  # Override
        results = backtesting.backtest(backtest_conf)
        assert not results.empty


def test_backtest_clash_buy_sell(mocker, default_conf):
    # Override the default buy trend function in our default_strategy
    def fun(dataframe=None):
        buy_value = 1
        sell_value = 1
        return _trend(dataframe, buy_value, sell_value)

    backtest_conf = _make_backtest_conf(mocker, conf=default_conf)
    backtesting = Backtesting(default_conf)
    backtesting.populate_buy_trend = fun  # Override
    backtesting.populate_sell_trend = fun  # Override
    results = backtesting.backtest(backtest_conf)
    assert results.empty


def test_backtest_only_sell(mocker, default_conf):
    # Override the default buy trend function in our default_strategy
    def fun(dataframe=None):
        buy_value = 0
        sell_value = 1
        return _trend(dataframe, buy_value, sell_value)

    backtest_conf = _make_backtest_conf(mocker, conf=default_conf)
    backtesting = Backtesting(default_conf)
    backtesting.populate_buy_trend = fun  # Override
    backtesting.populate_sell_trend = fun  # Override
    results = backtesting.backtest(backtest_conf)
    assert results.empty


def test_backtest_alternate_buy_sell(default_conf, fee, mocker):
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    backtest_conf = _make_backtest_conf(mocker, conf=default_conf, pair='UNITTEST/BTC')
    backtesting = Backtesting(default_conf)
    backtesting.populate_buy_trend = _trend_alternate  # Override
    backtesting.populate_sell_trend = _trend_alternate  # Override
    results = backtesting.backtest(backtest_conf)
    backtesting._store_backtest_result("test_.json", results)
    assert len(results) == 4
    # One trade was force-closed at the end
    assert len(results.loc[results.open_at_end]) == 1


def test_backtest_record(default_conf, fee, mocker):
    names = []
    records = []
    patch_exchange(mocker)
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    mocker.patch(
        'freqtrade.optimize.backtesting.file_dump_json',
        new=lambda n, r: (names.append(n), records.append(r))
    )

    backtesting = Backtesting(default_conf)
    results = pd.DataFrame({"pair": ["UNITTEST/BTC", "UNITTEST/BTC",
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
                            })
    backtesting._store_backtest_result("backtest-result.json", results)
    assert len(results) == 4
    # Assert file_dump_json was only called once
    assert names == ['backtest-result.json']
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


def test_backtest_start_live(default_conf, mocker, caplog):
    conf = deepcopy(default_conf)
    conf['exchange']['pair_whitelist'] = ['UNITTEST/BTC']
    mocker.patch('freqtrade.exchange.Exchange.get_ticker_history',
                 new=lambda s, n, i: _load_pair_as_ticks(n, i))
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest', MagicMock())
    mocker.patch('freqtrade.optimize.backtesting.Backtesting._generate_text_table', MagicMock())
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(conf)
    ))

    args = MagicMock()
    args.ticker_interval = 1
    args.level = 10
    args.live = True
    args.datadir = None
    args.export = None
    args.strategy = 'DefaultStrategy'
    args.timerange = '-100'  # needed due to MagicMock malleability

    args = [
        '--config', 'config.json',
        '--strategy', 'DefaultStrategy',
        '--datadir', 'freqtrade/tests/testdata',
        'backtesting',
        '--ticker-interval', '1m',
        '--live',
        '--timerange', '-100',
        '--realistic-simulation'
    ]
    args = get_args(args)
    start(args)
    # check the logs, that will contain the backtest result
    exists = [
        'Parameter -i/--ticker-interval detected ...',
        'Using ticker_interval: 1m ...',
        'Parameter -l/--live detected ...',
        'Using max_open_trades: 1 ...',
        'Parameter --timerange detected: -100 ...',
        'Using data folder: freqtrade/tests/testdata ...',
        'Using stake_currency: BTC ...',
        'Using stake_amount: 0.001 ...',
        'Downloading data for all pairs in whitelist ...',
        'Measuring data from 2017-11-14T19:31:00+00:00 up to 2017-11-14T22:58:00+00:00 (0 days)..',
        'Parameter --realistic-simulation detected ...'
    ]

    for line in exists:
        assert log_has(line, caplog.record_tuples)
