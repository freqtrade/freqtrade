# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103
import random
import logging
import math
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from freqtrade import exchange, optimize
from freqtrade.exchange import Bittrex
from freqtrade.optimize import preprocess
from freqtrade.optimize.backtesting import backtest, generate_text_table, get_timeframe
import freqtrade.optimize.backtesting as backtesting


def trim_dictlist(dict_list, num):
    new = {}
    for pair, pair_data in dict_list.items():
        new[pair] = pair_data[num:]
    return new


# use for mock freqtrade.exchange.get_ticker_history'
def _load_pair_as_ticks(pair, tickfreq):
    ticks = optimize.load_data(None, ticker_interval=8, pairs=[pair])
    ticks = trim_dictlist(ticks, -200)
    return ticks[pair]


# FIX: fixturize this?
def _make_backtest_conf(conf=None,
                        pair='BTC_UNITEST',
                        record=None):
    data = optimize.load_data(None, ticker_interval=8, pairs=[pair])
    data = trim_dictlist(data, -200)
    return {'stake_amount': conf['stake_amount'],
            'processed': optimize.preprocess(data),
            'max_open_trades': 10,
            'realistic': True,
            'record': record}


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


def _run_backtest_1(strategy, fun, backtest_conf):
    # strategy is a global (hidden as a singleton), so we
    # emulate strategy being pure, by override/restore here
    # if we dont do this, the override in strategy will carry over
    # to other tests
    old_buy = strategy.populate_buy_trend
    old_sell = strategy.populate_sell_trend
    strategy.populate_buy_trend = fun  # Override
    strategy.populate_sell_trend = fun  # Override
    results = backtest(backtest_conf)
    strategy.populate_buy_trend = old_buy  # restore override
    strategy.populate_sell_trend = old_sell  # restore override
    return results


def test_generate_text_table():
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
    print(generate_text_table({'BTC_ETH': {}}, results, 'BTC'))
    assert generate_text_table({'BTC_ETH': {}}, results, 'BTC') == (
        'pair       buy count    avg profit %    total profit BTC    avg duration    profit    loss\n'  # noqa
        '-------  -----------  --------------  ------------------  --------------  --------  ------\n'  # noqa
        'BTC_ETH            2           15.00          0.60000000            20.0         2       0\n'  # noqa
        'TOTAL              2           15.00          0.60000000            20.0         2       0')  # noqa


def test_get_timeframe(default_strategy):
    data = preprocess(optimize.load_data(
        None, ticker_interval=1, pairs=['BTC_UNITEST']))
    min_date, max_date = get_timeframe(data)
    assert min_date.isoformat() == '2017-11-04T23:02:00+00:00'
    assert max_date.isoformat() == '2017-11-14T22:59:00+00:00'


def test_backtest(default_strategy, default_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    exchange._API = Bittrex({'key': '', 'secret': ''})

    data = optimize.load_data(None, ticker_interval=5, pairs=['BTC_ETH'])
    data = trim_dictlist(data, -200)
    results = backtest({'stake_amount': default_conf['stake_amount'],
                        'processed': optimize.preprocess(data),
                        'max_open_trades': 10,
                        'realistic': True})
    assert not results.empty


def test_backtest_1min_ticker_interval(default_strategy, default_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    exchange._API = Bittrex({'key': '', 'secret': ''})

    # Run a backtesting for an exiting 5min ticker_interval
    data = optimize.load_data(None, ticker_interval=1, pairs=['BTC_UNITEST'])
    data = trim_dictlist(data, -200)
    results = backtest({'stake_amount': default_conf['stake_amount'],
                        'processed': optimize.preprocess(data),
                        'max_open_trades': 1,
                        'realistic': True})
    assert not results.empty


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


def simple_backtest(config, contour, num_results):
    data = load_data_test(contour)
    processed = optimize.preprocess(data)
    assert isinstance(processed, dict)
    results = backtest({'stake_amount': config['stake_amount'],
                        'processed': processed,
                        'max_open_trades': 1,
                        'realistic': True})
    # results :: <class 'pandas.core.frame.DataFrame'>
    assert len(results) == num_results


# Test backtest using offline data (testdata directory)


def test_backtest_ticks(default_conf, mocker, default_strategy):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    ticks = [1, 5]
    fun = default_strategy.populate_buy_trend
    for tick in ticks:
        backtest_conf = _make_backtest_conf(conf=default_conf)
        results = _run_backtest_1(default_strategy, fun, backtest_conf)
        assert not results.empty


def test_backtest_clash_buy_sell(default_conf, mocker, default_strategy):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)

    # Override the default buy trend function in our default_strategy
    def fun(dataframe=None):
        buy_value = 1
        sell_value = 1
        return _trend(dataframe, buy_value, sell_value)

    backtest_conf = _make_backtest_conf(conf=default_conf)
    results = _run_backtest_1(default_strategy, fun, backtest_conf)
    assert results.empty


def test_backtest_only_sell(default_conf, mocker, default_strategy):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)

    # Override the default buy trend function in our default_strategy
    def fun(dataframe=None):
        buy_value = 0
        sell_value = 1
        return _trend(dataframe, buy_value, sell_value)

    backtest_conf = _make_backtest_conf(conf=default_conf)
    results = _run_backtest_1(default_strategy, fun, backtest_conf)
    assert results.empty


def test_backtest_alternate_buy_sell(default_conf, mocker, default_strategy):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    backtest_conf = _make_backtest_conf(conf=default_conf, pair='BTC_UNITEST')
    results = _run_backtest_1(default_strategy, _trend_alternate,
                              backtest_conf)
    assert len(results) == 3


def test_backtest_record(default_conf, mocker, default_strategy):
    names = []
    records = []
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.misc.file_dump_json',
                 new=lambda n, r: (names.append(n), records.append(r)))
    backtest_conf = _make_backtest_conf(
        conf=default_conf,
        pair='BTC_UNITEST',
        record="trades"
    )
    results = _run_backtest_1(default_strategy, _trend_alternate,
                              backtest_conf)
    assert len(results) == 3
    # Assert file_dump_json was only called once
    assert names == ['backtest-result.json']
    records = records[0]
    # Ensure records are of correct type
    assert len(records) == 3
    # ('BTC_UNITEST', 0.00331158, '1510684320', '1510691700', 0, 117)
    # Below follows just a typecheck of the schema/type of trade-records
    oix = None
    for (pair, profit, date_buy, date_sell, buy_index, dur) in records:
        assert pair == 'BTC_UNITEST'
        isinstance(profit, float)
        # FIX: buy/sell should be converted to ints
        isinstance(date_buy, str)
        isinstance(date_sell, str)
        isinstance(buy_index, pd._libs.tslib.Timestamp)
        if oix:
            assert buy_index > oix
        oix = buy_index
        assert dur > 0


def test_processed(default_conf, mocker, default_strategy):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    dict_of_tickerrows = load_data_test('raise')
    dataframes = optimize.preprocess(dict_of_tickerrows)
    dataframe = dataframes['BTC_UNITEST']
    cols = dataframe.columns
    # assert the dataframe got some of the indicator columns
    for col in ['close', 'high', 'low', 'open', 'date',
                'ema50', 'ao', 'macd', 'plus_dm']:
        assert col in cols


def test_backtest_pricecontours(default_conf, mocker, default_strategy):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    tests = [['raise', 17], ['lower', 0], ['sine', 17]]
    for [contour, numres] in tests:
        simple_backtest(default_conf, contour, numres)


def mocked_load_data(datadir, pairs=[], ticker_interval=0, refresh_pairs=False, timerange=None):
    tickerdata = optimize.load_tickerdata_file(datadir, 'BTC_UNITEST', 1, timerange=timerange)
    pairdata = {'BTC_UNITEST': tickerdata}
    return pairdata


def test_backtest_start(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    default_conf['exchange']['pair_whitelist'] = ['BTC_UNITEST']
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.misc.load_config', new=lambda s: default_conf)
    mocker.patch.multiple('freqtrade.optimize',
                          load_data=mocked_load_data)
    args = MagicMock()
    args.ticker_interval = 1
    args.level = 10
    args.live = False
    args.datadir = None
    args.export = None
    args.timerange = '-100'  # needed due to MagicMock malleability
    backtesting.start(args)
    # check the logs, that will contain the backtest result
    exists = ['Using max_open_trades: 1 ...',
              'Using stake_amount: 0.001 ...',
              'Measuring data from 2017-11-14T21:17:00+00:00 '
              'up to 2017-11-14T22:59:00+00:00 (0 days)..']
    for line in exists:
        assert ('freqtrade.optimize.backtesting',
                logging.INFO,
                line) in caplog.record_tuples


def test_backtest_start_live(default_strategy, default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    default_conf['exchange']['pair_whitelist'] = ['BTC_UNITEST']
    mocker.patch('freqtrade.exchange.get_ticker_history',
                 new=lambda n, i: _load_pair_as_ticks(n, i))
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.misc.load_config', new=lambda s: default_conf)
    args = MagicMock()
    args.ticker_interval = 1
    args.level = 10
    args.live = True
    args.datadir = None
    args.export = None
    args.timerange = '-100'  # needed due to MagicMock malleability
    backtesting.start(args)
    # check the logs, that will contain the backtest result
    exists = ['Using max_open_trades: 1 ...',
              'Using stake_amount: 0.001 ...',
              'Measuring data from 2017-11-14T19:32:00+00:00 '
              'up to 2017-11-14T22:59:00+00:00 (0 days)..']
    for line in exists:
        assert ('freqtrade.optimize.backtesting',
                logging.INFO,
                line) in caplog.record_tuples
