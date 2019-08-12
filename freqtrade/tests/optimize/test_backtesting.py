# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, unused-argument

import math
import random
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from arrow import Arrow

from freqtrade import DependencyException, OperationalException, constants
from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.btanalysis import evaluate_result_multi
from freqtrade.data.converter import parse_ticker_dataframe
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.history import get_timeframe
from freqtrade.optimize import setup_configuration, start_backtesting
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.state import RunMode
from freqtrade.strategy.default_strategy import DefaultStrategy
from freqtrade.strategy.interface import SellType
from freqtrade.tests.conftest import (get_args, log_has, log_has_re,
                                      patch_exchange,
                                      patched_configuration_load_config_file)


def trim_dictlist(dict_list, num):
    new = {}
    for pair, pair_data in dict_list.items():
        new[pair] = pair_data[num:].reset_index()
    return new


def load_data_test(what):
    timerange = TimeRange(None, 'line', 0, -101)
    pair = history.load_tickerdata_file(None, ticker_interval='1m',
                                        pair='UNITTEST/BTC', timerange=timerange)
    datalen = len(pair)

    base = 0.001
    if what == 'raise':
        data = [
            [
                pair[x][0],  # Keep old dates
                x * base,  # But replace O,H,L,C
                x * base + 0.0001,
                x * base - 0.0001,
                x * base,
                pair[x][5],  # Keep old volume
            ] for x in range(0, datalen)
        ]
    if what == 'lower':
        data = [
            [
                pair[x][0],  # Keep old dates
                1 - x * base,  # But replace O,H,L,C
                1 - x * base + 0.0001,
                1 - x * base - 0.0001,
                1 - x * base,
                pair[x][5]  # Keep old volume
            ] for x in range(0, datalen)
        ]
    if what == 'sine':
        hz = 0.1  # frequency
        data = [
            [
                pair[x][0],  # Keep old dates
                math.sin(x * hz) / 1000 + base,  # But replace O,H,L,C
                math.sin(x * hz) / 1000 + base + 0.0001,
                math.sin(x * hz) / 1000 + base - 0.0001,
                math.sin(x * hz) / 1000 + base,
                pair[x][5]  # Keep old volume
            ] for x in range(0, datalen)
        ]
    return {'UNITTEST/BTC': parse_ticker_dataframe(data, '1m', pair="UNITTEST/BTC",
                                                   fill_missing=True)}


def simple_backtest(config, contour, num_results, mocker) -> None:
    patch_exchange(mocker)
    config['ticker_interval'] = '1m'
    backtesting = Backtesting(config)

    data = load_data_test(contour)
    processed = backtesting.strategy.tickerdata_to_dataframe(data)
    min_date, max_date = get_timeframe(processed)
    assert isinstance(processed, dict)
    results = backtesting.backtest(
        {
            'stake_amount': config['stake_amount'],
            'processed': processed,
            'max_open_trades': 1,
            'position_stacking': False,
            'start_date': min_date,
            'end_date': max_date,
        }
    )
    # results :: <class 'pandas.core.frame.DataFrame'>
    assert len(results) == num_results


def mocked_load_data(datadir, pairs=[], ticker_interval='0m', refresh_pairs=False,
                     timerange=None, exchange=None, live=False):
    tickerdata = history.load_tickerdata_file(datadir, 'UNITTEST/BTC', '1m', timerange=timerange)
    pairdata = {'UNITTEST/BTC': parse_ticker_dataframe(tickerdata, '1m', pair="UNITTEST/BTC",
                                                       fill_missing=True)}
    return pairdata


# use for mock ccxt.fetch_ohlvc'
def _load_pair_as_ticks(pair, tickfreq):
    ticks = history.load_tickerdata_file(None, ticker_interval=tickfreq, pair=pair)
    ticks = ticks[-201:]
    return ticks


# FIX: fixturize this?
def _make_backtest_conf(mocker, conf=None, pair='UNITTEST/BTC', record=None):
    data = history.load_data(datadir=None, ticker_interval='1m', pairs=[pair])
    data = trim_dictlist(data, -201)
    patch_exchange(mocker)
    backtesting = Backtesting(conf)
    processed = backtesting.strategy.tickerdata_to_dataframe(data)
    min_date, max_date = get_timeframe(processed)
    return {
        'stake_amount': conf['stake_amount'],
        'processed': processed,
        'max_open_trades': 10,
        'position_stacking': False,
        'record': record,
        'start_date': min_date,
        'end_date': max_date,
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


def _trend_alternate(dataframe=None, metadata=None):
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
    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        '--config', 'config.json',
        '--strategy', 'DefaultStrategy',
        'backtesting'
    ]

    config = setup_configuration(get_args(args), RunMode.BACKTEST)
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert log_has('Using data directory: {} ...'.format(config['datadir']), caplog)
    assert 'ticker_interval' in config
    assert not log_has_re('Parameter -i/--ticker-interval detected .*', caplog)

    assert 'live' not in config
    assert not log_has('Parameter -l/--live detected ...', caplog)

    assert 'position_stacking' not in config
    assert not log_has('Parameter --enable-position-stacking detected ...', caplog)

    assert 'refresh_pairs' not in config
    assert not log_has('Parameter -r/--refresh-pairs-cached detected ...', caplog)

    assert 'timerange' not in config
    assert 'export' not in config
    assert 'runmode' in config
    assert config['runmode'] == RunMode.BACKTEST


@pytest.mark.filterwarnings("ignore:DEPRECATED")
def test_setup_bt_configuration_with_arguments(mocker, default_conf, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch(
        'freqtrade.configuration.configuration.create_datadir',
        lambda c, x: x
    )

    args = [
        '--config', 'config.json',
        '--strategy', 'DefaultStrategy',
        '--datadir', '/foo/bar',
        'backtesting',
        '--ticker-interval', '1m',
        '--live',
        '--enable-position-stacking',
        '--disable-max-market-positions',
        '--refresh-pairs-cached',
        '--timerange', ':100',
        '--export', '/bar/foo',
        '--export-filename', 'foo_bar.json'
    ]

    config = setup_configuration(get_args(args), RunMode.BACKTEST)
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert config['runmode'] == RunMode.BACKTEST

    assert log_has('Using data directory: {} ...'.format(config['datadir']), caplog)
    assert 'ticker_interval' in config
    assert log_has('Parameter -i/--ticker-interval detected ... Using ticker_interval: 1m ...',
                   caplog)

    assert 'live' in config
    assert log_has('Parameter -l/--live detected ...', caplog)

    assert 'position_stacking' in config
    assert log_has('Parameter --enable-position-stacking detected ...', caplog)

    assert 'use_max_market_positions' in config
    assert log_has('Parameter --disable-max-market-positions detected ...', caplog)
    assert log_has('max_open_trades set to unlimited ...', caplog)

    assert 'refresh_pairs' in config
    assert log_has('Parameter -r/--refresh-pairs-cached detected ...', caplog)

    assert 'timerange' in config
    assert log_has('Parameter --timerange detected: {} ...'.format(config['timerange']), caplog)

    assert 'export' in config
    assert log_has('Parameter --export detected: {} ...'.format(config['export']), caplog)
    assert 'exportfilename' in config
    assert log_has('Storing backtest results to {} ...'.format(config['exportfilename']), caplog)


def test_setup_configuration_unlimited_stake_amount(mocker, default_conf, caplog) -> None:
    default_conf['stake_amount'] = constants.UNLIMITED_STAKE_AMOUNT

    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        '--config', 'config.json',
        '--strategy', 'DefaultStrategy',
        'backtesting'
    ]

    with pytest.raises(DependencyException, match=r'.*stake amount.*'):
        setup_configuration(get_args(args), RunMode.BACKTEST)


def test_start(mocker, fee, default_conf, caplog) -> None:
    start_mock = MagicMock()
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.start', start_mock)
    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        '--config', 'config.json',
        '--strategy', 'DefaultStrategy',
        'backtesting'
    ]
    args = get_args(args)
    start_backtesting(args)
    assert log_has('Starting freqtrade in Backtesting mode', caplog)
    assert start_mock.call_count == 1


ORDER_TYPES = [
    {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    },
    {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True
    }]


@pytest.mark.parametrize("order_types", ORDER_TYPES)
def test_backtesting_init(mocker, default_conf, order_types) -> None:
    """
    Check that stoploss_on_exchange is set to False while backtesting
    since backtesting assumes a perfect stoploss anyway.
    """
    default_conf["order_types"] = order_types
    patch_exchange(mocker)
    get_fee = mocker.patch('freqtrade.exchange.Exchange.get_fee', MagicMock(return_value=0.5))
    backtesting = Backtesting(default_conf)
    assert backtesting.config == default_conf
    assert backtesting.ticker_interval == '5m'
    assert callable(backtesting.strategy.tickerdata_to_dataframe)
    assert callable(backtesting.advise_buy)
    assert callable(backtesting.advise_sell)
    assert isinstance(backtesting.strategy.dp, DataProvider)
    get_fee.assert_called()
    assert backtesting.fee == 0.5
    assert not backtesting.strategy.order_types["stoploss_on_exchange"]


def test_backtesting_init_no_ticker_interval(mocker, default_conf, caplog) -> None:
    """
    Check that stoploss_on_exchange is set to False while backtesting
    since backtesting assumes a perfect stoploss anyway.
    """
    patch_exchange(mocker)
    del default_conf['ticker_interval']
    default_conf['strategy_list'] = ['DefaultStrategy',
                                     'TestStrategy']

    mocker.patch('freqtrade.exchange.Exchange.get_fee', MagicMock(return_value=0.5))
    with pytest.raises(OperationalException):
        Backtesting(default_conf)
    log_has("Ticker-interval needs to be set in either configuration "
            "or as cli argument `--ticker-interval 5m`", caplog)


def test_tickerdata_to_dataframe_bt(default_conf, mocker) -> None:
    patch_exchange(mocker)
    timerange = TimeRange(None, 'line', 0, -100)
    tick = history.load_tickerdata_file(None, 'UNITTEST/BTC', '1m', timerange=timerange)
    tickerlist = {'UNITTEST/BTC': parse_ticker_dataframe(tick, '1m', pair="UNITTEST/BTC",
                                                         fill_missing=True)}

    backtesting = Backtesting(default_conf)
    data = backtesting.strategy.tickerdata_to_dataframe(tickerlist)
    assert len(data['UNITTEST/BTC']) == 102

    # Load strategy to compare the result between Backtesting function and strategy are the same
    strategy = DefaultStrategy(default_conf)
    data2 = strategy.tickerdata_to_dataframe(tickerlist)
    assert data['UNITTEST/BTC'].equals(data2['UNITTEST/BTC'])


def test_generate_text_table(default_conf, mocker):
    patch_exchange(mocker)
    default_conf['max_open_trades'] = 2
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
        'tot profit BTC |   tot profit % | avg duration   |   profit |   loss |\n'
        '|:--------|------------:|---------------:|---------------:|'
        '-----------------:|---------------:|:---------------|---------:|-------:|\n'
        '| ETH/BTC |           2 |          15.00 |          30.00 |       '
        '0.60000000 |          15.00 | 0:20:00        |        2 |      0 |\n'
        '| TOTAL   |           2 |          15.00 |          30.00 |       '
        '0.60000000 |          15.00 | 0:20:00        |        2 |      0 |'
    )
    assert backtesting._generate_text_table(data={'ETH/BTC': {}}, results=results) == result_str


def test_generate_text_table_sell_reason(default_conf, mocker):
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)

    results = pd.DataFrame(
        {
            'pair': ['ETH/BTC', 'ETH/BTC', 'ETH/BTC'],
            'profit_percent': [0.1, 0.2, 0.3],
            'profit_abs': [0.2, 0.4, 0.5],
            'trade_duration': [10, 30, 10],
            'profit': [2, 0, 0],
            'loss': [0, 0, 1],
            'sell_reason': [SellType.ROI, SellType.ROI, SellType.STOP_LOSS]
        }
    )

    result_str = (
        '| Sell Reason   |   Count |\n'
        '|:--------------|--------:|\n'
        '| roi           |       2 |\n'
        '| stop_loss     |       1 |'
    )
    assert backtesting._generate_text_table_sell_reason(
        data={'ETH/BTC': {}}, results=results) == result_str


def test_generate_text_table_strategyn(default_conf, mocker):
    """
    Test Backtesting.generate_text_table_sell_reason() method
    """
    patch_exchange(mocker)
    default_conf['max_open_trades'] = 2
    backtesting = Backtesting(default_conf)
    results = {}
    results['ETH/BTC'] = pd.DataFrame(
        {
            'pair': ['ETH/BTC', 'ETH/BTC', 'ETH/BTC'],
            'profit_percent': [0.1, 0.2, 0.3],
            'profit_abs': [0.2, 0.4, 0.5],
            'trade_duration': [10, 30, 10],
            'profit': [2, 0, 0],
            'loss': [0, 0, 1],
            'sell_reason': [SellType.ROI, SellType.ROI, SellType.STOP_LOSS]
        }
    )
    results['LTC/BTC'] = pd.DataFrame(
        {
            'pair': ['LTC/BTC', 'LTC/BTC', 'LTC/BTC'],
            'profit_percent': [0.4, 0.2, 0.3],
            'profit_abs': [0.4, 0.4, 0.5],
            'trade_duration': [15, 30, 15],
            'profit': [4, 1, 0],
            'loss': [0, 0, 1],
            'sell_reason': [SellType.ROI, SellType.ROI, SellType.STOP_LOSS]
        }
    )

    result_str = (
        '| Strategy   |   buy count |   avg profit % |   cum profit % '
        '|   tot profit BTC |   tot profit % | avg duration   |   profit |   loss |\n'
        '|:-----------|------------:|---------------:|---------------:'
        '|-----------------:|---------------:|:---------------|---------:|-------:|\n'
        '| ETH/BTC    |           3 |          20.00 |          60.00 '
        '|       1.10000000 |          30.00 | 0:17:00        |        3 |      0 |\n'
        '| LTC/BTC    |           3 |          30.00 |          90.00 '
        '|       1.30000000 |          45.00 | 0:20:00        |        3 |      0 |'
    )
    print(backtesting._generate_text_table_strategy(all_results=results))
    assert backtesting._generate_text_table_strategy(all_results=results) == result_str


def test_backtesting_start(default_conf, mocker, caplog) -> None:
    def get_timeframe(input1):
        return Arrow(2017, 11, 14, 21, 17), Arrow(2017, 11, 14, 22, 59)

    mocker.patch('freqtrade.data.history.load_data', mocked_load_data)
    mocker.patch('freqtrade.data.history.get_timeframe', get_timeframe)
    mocker.patch('freqtrade.exchange.Exchange.refresh_latest_ohlcv', MagicMock())
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.optimize.backtesting.Backtesting',
        backtest=MagicMock(),
        _generate_text_table=MagicMock(return_value='1'),
    )

    default_conf['exchange']['pair_whitelist'] = ['UNITTEST/BTC']
    default_conf['ticker_interval'] = '1m'
    default_conf['live'] = False
    default_conf['datadir'] = None
    default_conf['export'] = None
    default_conf['timerange'] = '-100'

    backtesting = Backtesting(default_conf)
    backtesting.start()
    # check the logs, that will contain the backtest result
    exists = [
        'Using stake_currency: BTC ...',
        'Using stake_amount: 0.001 ...',
        'Backtesting with data from 2017-11-14T21:17:00+00:00 '
        'up to 2017-11-14T22:59:00+00:00 (0 days)..'
    ]
    for line in exists:
        assert log_has(line, caplog)


def test_backtesting_start_no_data(default_conf, mocker, caplog) -> None:
    def get_timeframe(input1):
        return Arrow(2017, 11, 14, 21, 17), Arrow(2017, 11, 14, 22, 59)

    mocker.patch('freqtrade.data.history.load_data', MagicMock(return_value={}))
    mocker.patch('freqtrade.data.history.get_timeframe', get_timeframe)
    mocker.patch('freqtrade.exchange.Exchange.refresh_latest_ohlcv', MagicMock())
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.optimize.backtesting.Backtesting',
        backtest=MagicMock(),
        _generate_text_table=MagicMock(return_value='1'),
    )

    default_conf['exchange']['pair_whitelist'] = ['UNITTEST/BTC']
    default_conf['ticker_interval'] = "1m"
    default_conf['live'] = False
    default_conf['datadir'] = None
    default_conf['export'] = None
    default_conf['timerange'] = '20180101-20180102'

    backtesting = Backtesting(default_conf)
    backtesting.start()
    # check the logs, that will contain the backtest result

    assert log_has('No data found. Terminating.', caplog)


def test_backtest(default_conf, fee, mocker) -> None:
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)
    pair = 'UNITTEST/BTC'
    timerange = TimeRange(None, 'line', 0, -201)
    data = history.load_data(datadir=None, ticker_interval='5m', pairs=['UNITTEST/BTC'],
                             timerange=timerange)
    data_processed = backtesting.strategy.tickerdata_to_dataframe(data)
    min_date, max_date = get_timeframe(data_processed)
    results = backtesting.backtest(
        {
            'stake_amount': default_conf['stake_amount'],
            'processed': data_processed,
            'max_open_trades': 10,
            'position_stacking': False,
            'start_date': min_date,
            'end_date': max_date,
        }
    )
    assert not results.empty
    assert len(results) == 2

    expected = pd.DataFrame(
        {'pair': [pair, pair],
         'profit_percent': [0.0, 0.0],
         'profit_abs': [0.0, 0.0],
         'open_time': pd.to_datetime([Arrow(2018, 1, 29, 18, 40, 0).datetime,
                                      Arrow(2018, 1, 30, 3, 30, 0).datetime], utc=True
                                     ),
         'close_time': pd.to_datetime([Arrow(2018, 1, 29, 22, 35, 0).datetime,
                                       Arrow(2018, 1, 30, 4, 10, 0).datetime], utc=True),
         'open_index': [78, 184],
         'close_index': [125, 192],
         'trade_duration': [235, 40],
         'open_at_end': [False, False],
         'open_rate': [0.104445, 0.10302485],
         'close_rate': [0.104969, 0.103541],
         'sell_reason': [SellType.ROI, SellType.ROI]
         })
    pd.testing.assert_frame_equal(results, expected)
    data_pair = data_processed[pair]
    for _, t in results.iterrows():
        ln = data_pair.loc[data_pair["date"] == t["open_time"]]
        # Check open trade rate alignes to open rate
        assert ln is not None
        assert round(ln.iloc[0]["open"], 6) == round(t["open_rate"], 6)
        # check close trade rate alignes to close rate or is between high and low
        ln = data_pair.loc[data_pair["date"] == t["close_time"]]
        assert (round(ln.iloc[0]["open"], 6) == round(t["close_rate"], 6) or
                round(ln.iloc[0]["low"], 6) < round(
                t["close_rate"], 6) < round(ln.iloc[0]["high"], 6))


def test_backtest_1min_ticker_interval(default_conf, fee, mocker) -> None:
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)

    # Run a backtesting for an exiting 1min ticker_interval
    timerange = TimeRange(None, 'line', 0, -200)
    data = history.load_data(datadir=None, ticker_interval='1m', pairs=['UNITTEST/BTC'],
                             timerange=timerange)
    processed = backtesting.strategy.tickerdata_to_dataframe(data)
    min_date, max_date = get_timeframe(processed)
    results = backtesting.backtest(
        {
            'stake_amount': default_conf['stake_amount'],
            'processed': processed,
            'max_open_trades': 1,
            'position_stacking': False,
            'start_date': min_date,
            'end_date': max_date,
        }
    )
    assert not results.empty
    assert len(results) == 1


def test_processed(default_conf, mocker) -> None:
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)

    dict_of_tickerrows = load_data_test('raise')
    dataframes = backtesting.strategy.tickerdata_to_dataframe(dict_of_tickerrows)
    dataframe = dataframes['UNITTEST/BTC']
    cols = dataframe.columns
    # assert the dataframe got some of the indicator columns
    for col in ['close', 'high', 'low', 'open', 'date',
                'ema50', 'ao', 'macd', 'plus_dm']:
        assert col in cols


def test_backtest_pricecontours(default_conf, fee, mocker) -> None:
    # TODO: Evaluate usefullness of this, the patterns and buy-signls are unrealistic
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    tests = [['raise', 19], ['lower', 0], ['sine', 35]]
    # We need to enable sell-signal - otherwise it sells on ROI!!
    default_conf['experimental'] = {"use_sell_signal": True}

    for [contour, numres] in tests:
        simple_backtest(default_conf, contour, numres, mocker)


def test_backtest_clash_buy_sell(mocker, default_conf):
    # Override the default buy trend function in our default_strategy
    def fun(dataframe=None, pair=None):
        buy_value = 1
        sell_value = 1
        return _trend(dataframe, buy_value, sell_value)

    backtest_conf = _make_backtest_conf(mocker, conf=default_conf)
    backtesting = Backtesting(default_conf)
    backtesting.advise_buy = fun  # Override
    backtesting.advise_sell = fun  # Override
    results = backtesting.backtest(backtest_conf)
    assert results.empty


def test_backtest_only_sell(mocker, default_conf):
    # Override the default buy trend function in our default_strategy
    def fun(dataframe=None, pair=None):
        buy_value = 0
        sell_value = 1
        return _trend(dataframe, buy_value, sell_value)

    backtest_conf = _make_backtest_conf(mocker, conf=default_conf)
    backtesting = Backtesting(default_conf)
    backtesting.advise_buy = fun  # Override
    backtesting.advise_sell = fun  # Override
    results = backtesting.backtest(backtest_conf)
    assert results.empty


def test_backtest_alternate_buy_sell(default_conf, fee, mocker):
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    mocker.patch('freqtrade.optimize.backtesting.file_dump_json', MagicMock())
    backtest_conf = _make_backtest_conf(mocker, conf=default_conf, pair='UNITTEST/BTC')
    # We need to enable sell-signal - otherwise it sells on ROI!!
    default_conf['experimental'] = {"use_sell_signal": True}
    default_conf['ticker_interval'] = '1m'
    backtesting = Backtesting(default_conf)
    backtesting.advise_buy = _trend_alternate  # Override
    backtesting.advise_sell = _trend_alternate  # Override
    results = backtesting.backtest(backtest_conf)
    backtesting._store_backtest_result("test_.json", results)
    # 200 candles in backtest data
    # won't buy on first (shifted by 1)
    # 100 buys signals
    assert len(results) == 100
    # One trade was force-closed at the end
    assert len(results.loc[results.open_at_end]) == 0


@pytest.mark.parametrize("pair", ['ADA/BTC', 'LTC/BTC'])
@pytest.mark.parametrize("tres", [0, 20, 30])
def test_backtest_multi_pair(default_conf, fee, mocker, tres, pair):

    def _trend_alternate_hold(dataframe=None, metadata=None):
        """
        Buy every xth candle - sell every other xth -2 (hold on to pairs a bit)
        """
        if metadata['pair'] in('ETH/BTC', 'LTC/BTC'):
            multi = 20
        else:
            multi = 18
        dataframe['buy'] = np.where(dataframe.index % multi == 0, 1, 0)
        dataframe['sell'] = np.where((dataframe.index + multi - 2) % multi == 0, 1, 0)
        return dataframe

    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    patch_exchange(mocker)

    pairs = ['ADA/BTC', 'DASH/BTC', 'ETH/BTC', 'LTC/BTC', 'NXT/BTC']
    data = history.load_data(datadir=None, ticker_interval='5m', pairs=pairs)
    # Only use 500 lines to increase performance
    data = trim_dictlist(data, -500)

    # Remove data for one pair from the beginning of the data
    data[pair] = data[pair][tres:].reset_index()
    # We need to enable sell-signal - otherwise it sells on ROI!!
    default_conf['experimental'] = {"use_sell_signal": True}
    default_conf['ticker_interval'] = '5m'

    backtesting = Backtesting(default_conf)
    backtesting.advise_buy = _trend_alternate_hold  # Override
    backtesting.advise_sell = _trend_alternate_hold  # Override

    data_processed = backtesting.strategy.tickerdata_to_dataframe(data)
    min_date, max_date = get_timeframe(data_processed)
    backtest_conf = {
        'stake_amount': default_conf['stake_amount'],
        'processed': data_processed,
        'max_open_trades': 3,
        'position_stacking': False,
        'start_date': min_date,
        'end_date': max_date,
    }

    results = backtesting.backtest(backtest_conf)

    # Make sure we have parallel trades
    assert len(evaluate_result_multi(results, '5min', 2)) > 0
    # make sure we don't have trades with more than configured max_open_trades
    assert len(evaluate_result_multi(results, '5min', 3)) == 0

    backtest_conf = {
        'stake_amount': default_conf['stake_amount'],
        'processed': data_processed,
        'max_open_trades': 1,
        'position_stacking': False,
        'start_date': min_date,
        'end_date': max_date,
    }
    results = backtesting.backtest(backtest_conf)
    assert len(evaluate_result_multi(results, '5min', 1)) == 0


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

    # reset test to test with strategy name
    names = []
    records = []
    backtesting._store_backtest_result("backtest-result.json", results, "DefStrat")
    assert len(results) == 4
    # Assert file_dump_json was only called once
    assert names == ['backtest-result-DefStrat.json']
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


@pytest.mark.filterwarnings("ignore:DEPRECATED")
def test_backtest_start_live(default_conf, mocker, caplog):
    default_conf['exchange']['pair_whitelist'] = ['UNITTEST/BTC']

    async def load_pairs(pair, timeframe, since):
        return _load_pair_as_ticks(pair, timeframe)

    api_mock = MagicMock()
    api_mock.fetch_ohlcv = load_pairs

    patch_exchange(mocker, api_mock)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest', MagicMock())
    mocker.patch('freqtrade.optimize.backtesting.Backtesting._generate_text_table', MagicMock())
    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        '--config', 'config.json',
        '--strategy', 'DefaultStrategy',
        '--datadir', 'freqtrade/tests/testdata',
        'backtesting',
        '--ticker-interval', '1m',
        '--live',
        '--timerange', '-100',
        '--enable-position-stacking',
        '--disable-max-market-positions'
    ]
    args = get_args(args)
    start_backtesting(args)
    # check the logs, that will contain the backtest result
    exists = [
        'Parameter -i/--ticker-interval detected ... Using ticker_interval: 1m ...',
        'Parameter -l/--live detected ...',
        'Ignoring max_open_trades (--disable-max-market-positions was used) ...',
        'Parameter --timerange detected: -100 ...',
        'Using data directory: freqtrade/tests/testdata ...',
        'Using stake_currency: BTC ...',
        'Using stake_amount: 0.001 ...',
        'Live: Downloading data for all defined pairs ...',
        'Backtesting with data from 2017-11-14T19:31:00+00:00 '
        'up to 2017-11-14T22:58:00+00:00 (0 days)..',
        'Parameter --enable-position-stacking detected ...'
    ]

    for line in exists:
        assert log_has(line, caplog)


@pytest.mark.filterwarnings("ignore:DEPRECATED")
def test_backtest_start_multi_strat(default_conf, mocker, caplog):
    default_conf['exchange']['pair_whitelist'] = ['UNITTEST/BTC']

    async def load_pairs(pair, timeframe, since):
        return _load_pair_as_ticks(pair, timeframe)
    api_mock = MagicMock()
    api_mock.fetch_ohlcv = load_pairs

    patch_exchange(mocker, api_mock)
    backtestmock = MagicMock()
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest', backtestmock)
    gen_table_mock = MagicMock()
    mocker.patch('freqtrade.optimize.backtesting.Backtesting._generate_text_table', gen_table_mock)
    gen_strattable_mock = MagicMock()
    mocker.patch('freqtrade.optimize.backtesting.Backtesting._generate_text_table_strategy',
                 gen_strattable_mock)
    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        '--config', 'config.json',
        '--datadir', 'freqtrade/tests/testdata',
        'backtesting',
        '--ticker-interval', '1m',
        '--live',
        '--timerange', '-100',
        '--enable-position-stacking',
        '--disable-max-market-positions',
        '--strategy-list',
        'DefaultStrategy',
        'TestStrategy',
    ]
    args = get_args(args)
    start_backtesting(args)
    # 2 backtests, 4 tables
    assert backtestmock.call_count == 2
    assert gen_table_mock.call_count == 4
    assert gen_strattable_mock.call_count == 1

    # check the logs, that will contain the backtest result
    exists = [
        'Parameter -i/--ticker-interval detected ... Using ticker_interval: 1m ...',
        'Parameter -l/--live detected ...',
        'Ignoring max_open_trades (--disable-max-market-positions was used) ...',
        'Parameter --timerange detected: -100 ...',
        'Using data directory: freqtrade/tests/testdata ...',
        'Using stake_currency: BTC ...',
        'Using stake_amount: 0.001 ...',
        'Live: Downloading data for all defined pairs ...',
        'Backtesting with data from 2017-11-14T19:31:00+00:00 '
        'up to 2017-11-14T22:58:00+00:00 (0 days)..',
        'Parameter --enable-position-stacking detected ...',
        'Running backtesting for Strategy DefaultStrategy',
        'Running backtesting for Strategy TestStrategy',
    ]

    for line in exists:
        assert log_has(line, caplog)
