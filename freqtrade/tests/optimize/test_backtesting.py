# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, unused-argument

import json
import math
from typing import List
from copy import deepcopy
from unittest.mock import MagicMock
import pandas as pd
from freqtrade import optimize
from freqtrade.optimize.backtesting import Backtesting, start, setup_configuration
from freqtrade.arguments import Arguments
from freqtrade.analyze import Analyze
import freqtrade.tests.conftest as tt  # test tools


# Avoid to reinit the same object again and again
_BACKTESTING = Backtesting(tt.default_conf())


def get_args(args) -> List[str]:
    return Arguments(args, '').get_parsed_arg()


def trim_dictlist(dict_list, num):
    new = {}
    for pair, pair_data in dict_list.items():
        new[pair] = pair_data[num:]
    return new


def load_data_test(what):
    timerange = ((None, 'line'), None, -100)
    data = optimize.load_data(None, ticker_interval=1, pairs=['BTC_UNITEST'], timerange=timerange)
    pair = data['BTC_UNITEST']
    datalen = len(pair)
    # Depending on the what parameter we now adjust the
    # loaded data looks:
    # pair :: [{'O': 0.123, 'H': 0.123, 'L': 0.123,
    #           'C': 0.123, 'V': 123.123,
    #           'T': '2017-11-04T23:02:00', 'BV': 0.123}]
    base = 0.001
    if what == 'raise':
        return {'BTC_UNITEST':
                [{'T': pair[x]['T'],  # Keep old dates
                  'V': pair[x]['V'],  # Keep old volume
                  'BV': pair[x]['BV'],  # keep too
                  'O': x * base,        # But replace O,H,L,C
                  'H': x * base + 0.0001,
                  'L': x * base - 0.0001,
                  'C': x * base} for x in range(0, datalen)]}
    if what == 'lower':
        return {'BTC_UNITEST':
                [{'T': pair[x]['T'],  # Keep old dates
                  'V': pair[x]['V'],  # Keep old volume
                  'BV': pair[x]['BV'],  # keep too
                  'O': 1 - x * base,        # But replace O,H,L,C
                  'H': 1 - x * base + 0.0001,
                  'L': 1 - x * base - 0.0001,
                  'C': 1 - x * base} for x in range(0, datalen)]}
    if what == 'sine':
        hz = 0.1  # frequency
        return {'BTC_UNITEST':
                [{'T': pair[x]['T'],  # Keep old dates
                  'V': pair[x]['V'],  # Keep old volume
                  'BV': pair[x]['BV'],  # keep too
                  # But replace O,H,L,C
                  'O': math.sin(x * hz) / 1000 + base,
                  'H': math.sin(x * hz) / 1000 + base + 0.0001,
                  'L': math.sin(x * hz) / 1000 + base - 0.0001,
                  'C': math.sin(x * hz) / 1000 + base} for x in range(0, datalen)]}
    return data


def simple_backtest(config, contour, num_results) -> None:
    backtesting = _BACKTESTING

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


def mocked_load_data(datadir, pairs=[], ticker_interval=0, refresh_pairs=False, timerange=None):
    tickerdata = optimize.load_tickerdata_file(datadir, 'BTC_UNITEST', 1, timerange=timerange)
    pairdata = {'BTC_UNITEST': tickerdata}
    return pairdata


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
        '--strategy', 'default_strategy',
        'backtesting'
    ]

    config = setup_configuration(get_args(args))
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert tt.log_has(
        'Parameter --datadir detected: {} ...'.format(config['datadir']),
        caplog.record_tuples
    )
    assert 'ticker_interval' in config
    assert not tt.log_has('Parameter -i/--ticker-interval detected ...', caplog.record_tuples)

    assert 'live' not in config
    assert not tt.log_has('Parameter -l/--live detected ...', caplog.record_tuples)

    assert 'realistic_simulation' not in config
    assert not tt.log_has('Parameter --realistic-simulation detected ...', caplog.record_tuples)

    assert 'refresh_pairs' not in config
    assert not tt.log_has('Parameter -r/--refresh-pairs-cached detected ...', caplog.record_tuples)

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
        '--strategy', 'default_strategy',
        '--datadir', '/foo/bar',
        'backtesting',
        '--ticker-interval', '1',
        '--live',
        '--realistic-simulation',
        '--refresh-pairs-cached',
        '--timerange', ':100',
        '--export', '/bar/foo'
    ]

    config = setup_configuration(get_args(args))
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert tt.log_has(
        'Parameter --datadir detected: {} ...'.format(config['datadir']),
        caplog.record_tuples
    )
    assert 'ticker_interval' in config
    assert tt.log_has('Parameter -i/--ticker-interval detected ...', caplog.record_tuples)
    assert tt.log_has(
        'Using ticker_interval: 1 ...',
        caplog.record_tuples
    )

    assert 'live' in config
    assert tt.log_has('Parameter -l/--live detected ...', caplog.record_tuples)

    assert 'realistic_simulation'in config
    assert tt.log_has('Parameter --realistic-simulation detected ...', caplog.record_tuples)
    assert tt.log_has('Using max_open_trades: 1 ...', caplog.record_tuples)

    assert 'refresh_pairs'in config
    assert tt.log_has('Parameter -r/--refresh-pairs-cached detected ...', caplog.record_tuples)

    import pprint
    pprint.pprint(caplog.record_tuples)
    pprint.pprint(config['timerange'])
    assert 'timerange' in config
    assert tt.log_has(
        'Parameter --timerange detected: {} ...'.format(config['timerange']),
        caplog.record_tuples
    )

    assert 'export' in config
    assert tt.log_has(
        'Parameter --export detected: {} ...'.format(config['export']),
        caplog.record_tuples
    )


def test_start(mocker, default_conf, caplog) -> None:
    """
    Test start() function
    """
    start_mock = MagicMock()
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.start', start_mock)
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))
    args = [
        '--config', 'config.json',
        '--strategy', 'default_strategy',
        'backtesting'
    ]
    args = get_args(args)
    start(args)
    assert tt.log_has(
        'Starting freqtrade in Backtesting mode',
        caplog.record_tuples
    )
    assert start_mock.call_count == 1


def test_backtesting__init__(mocker, default_conf) -> None:
    """
    Test Backtesting.__init__() method
    """
    init_mock = MagicMock()
    mocker.patch('freqtrade.optimize.backtesting.Backtesting._init', init_mock)

    backtesting = Backtesting(default_conf)
    assert backtesting.config == default_conf
    assert backtesting.analyze is None
    assert backtesting.ticker_interval is None
    assert backtesting.tickerdata_to_dataframe is None
    assert backtesting.populate_buy_trend is None
    assert backtesting.populate_sell_trend is None
    assert init_mock.call_count == 1


def test_backtesting_init(default_conf) -> None:
    """
    Test Backtesting._init() method
    """
    backtesting = Backtesting(default_conf)
    assert backtesting.config == default_conf
    assert isinstance(backtesting.analyze, Analyze)
    assert backtesting.ticker_interval == 5
    assert callable(backtesting.tickerdata_to_dataframe)
    assert callable(backtesting.populate_buy_trend)
    assert callable(backtesting.populate_sell_trend)


def test_get_timeframe() -> None:
    """
    Test Backtesting.get_timeframe() method
    """
    backtesting = _BACKTESTING

    data = backtesting.tickerdata_to_dataframe(
        optimize.load_data(
            None,
            ticker_interval=1,
            pairs=['BTC_UNITEST']
        )
    )
    min_date, max_date = backtesting.get_timeframe(data)
    assert min_date.isoformat() == '2017-11-04T23:02:00+00:00'
    assert max_date.isoformat() == '2017-11-14T22:59:00+00:00'


def test_generate_text_table():
    """
    Test Backtesting.generate_text_table() method
    """
    backtesting = _BACKTESTING

    results = pd.DataFrame(
        {
            'currency': ['BTC_ETH', 'BTC_ETH'],
            'profit_percent': [0.1, 0.2],
            'profit_BTC': [0.2, 0.4],
            'duration': [10, 30],
            'profit': [2, 0],
            'loss': [0, 0]
        }
    )

    result_str = (
        'pair       buy count    avg profit %    '
        'total profit BTC    avg duration    profit    loss\n'
        '-------  -----------  --------------  '
        '------------------  --------------  --------  ------\n'
        'BTC_ETH            2           15.00          '
        '0.60000000           100.0         2       0\n'
        'TOTAL              2           15.00          '
        '0.60000000           100.0         2       0'
    )

    assert backtesting._generate_text_table(data={'BTC_ETH': {}}, results=results) == result_str


def test_backtesting_start(default_conf, mocker, caplog) -> None:
    """
    Test Backtesting.start() method
    """
    mocker.patch.multiple('freqtrade.optimize', load_data=mocked_load_data)
    mocker.patch('freqtrade.exchange.get_ticker_history', MagicMock)

    conf = deepcopy(default_conf)
    conf['exchange']['pair_whitelist'] = ['BTC_UNITEST']
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
        assert tt.log_has(line, caplog.record_tuples)


def test_backtest(default_conf) -> None:
    """
    Test Backtesting.backtest() method
    """
    backtesting = _BACKTESTING

    data = optimize.load_data(None, ticker_interval=5, pairs=['BTC_ETH'])
    data = trim_dictlist(data, -200)
    results = backtesting.backtest(
        {
            'stake_amount': default_conf['stake_amount'],
            'processed': backtesting.tickerdata_to_dataframe(data),
            'max_open_trades': 10,
            'realistic': True
        }
    )
    assert not results.empty


def test_backtest_1min_ticker_interval(default_conf) -> None:
    """
    Test Backtesting.backtest() method with 1 min ticker
    """
    backtesting = _BACKTESTING

    # Run a backtesting for an exiting 5min ticker_interval
    data = optimize.load_data(None, ticker_interval=1, pairs=['BTC_UNITEST'])
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


def test_processed() -> None:
    """
    Test Backtesting.backtest() method with offline data
    """
    backtesting = _BACKTESTING

    dict_of_tickerrows = load_data_test('raise')
    dataframes = backtesting.tickerdata_to_dataframe(dict_of_tickerrows)
    dataframe = dataframes['BTC_UNITEST']
    cols = dataframe.columns
    # assert the dataframe got some of the indicator columns
    for col in ['close', 'high', 'low', 'open', 'date',
                'ema50', 'ao', 'macd', 'plus_dm']:
        assert col in cols


def test_backtest_pricecontours(default_conf) -> None:
    tests = [['raise', 17], ['lower', 0], ['sine', 17]]
    for [contour, numres] in tests:
        simple_backtest(default_conf, contour, numres)
