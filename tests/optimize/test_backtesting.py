# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, unused-argument

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pandas as pd
import pytest
from arrow import Arrow

from freqtrade.commands.optimize_commands import setup_optimize_configuration, start_backtesting
from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.btanalysis import BT_DATA_COLUMNS, evaluate_result_multi
from freqtrade.data.converter import clean_ohlcv_dataframe
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.history import get_timerange
from freqtrade.enums import RunMode, SellType
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.persistence import LocalTrade
from freqtrade.resolvers import StrategyResolver
from tests.conftest import (get_args, log_has, log_has_re, patch_exchange,
                            patched_configuration_load_config_file)


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


def trim_dictlist(dict_list, num):
    new = {}
    for pair, pair_data in dict_list.items():
        new[pair] = pair_data[num:].reset_index()
    return new


def load_data_test(what, testdatadir):
    timerange = TimeRange.parse_timerange('1510694220-1510700340')
    data = history.load_pair_history(pair='UNITTEST/BTC', datadir=testdatadir,
                                     timeframe='1m', timerange=timerange,
                                     drop_incomplete=False,
                                     fill_up_missing=False)

    base = 0.001
    if what == 'raise':
        data.loc[:, 'open'] = data.index * base
        data.loc[:, 'high'] = data.index * base + 0.0001
        data.loc[:, 'low'] = data.index * base - 0.0001
        data.loc[:, 'close'] = data.index * base

    if what == 'lower':
        data.loc[:, 'open'] = 1 - data.index * base
        data.loc[:, 'high'] = 1 - data.index * base + 0.0001
        data.loc[:, 'low'] = 1 - data.index * base - 0.0001
        data.loc[:, 'close'] = 1 - data.index * base

    if what == 'sine':
        hz = 0.1  # frequency
        data.loc[:, 'open'] = np.sin(data.index * hz) / 1000 + base
        data.loc[:, 'high'] = np.sin(data.index * hz) / 1000 + base + 0.0001
        data.loc[:, 'low'] = np.sin(data.index * hz) / 1000 + base - 0.0001
        data.loc[:, 'close'] = np.sin(data.index * hz) / 1000 + base

    return {'UNITTEST/BTC': clean_ohlcv_dataframe(data, timeframe='1m', pair='UNITTEST/BTC',
                                                  fill_missing=True)}


def simple_backtest(config, contour, mocker, testdatadir) -> None:
    patch_exchange(mocker)
    config['timeframe'] = '1m'
    backtesting = Backtesting(config)
    backtesting._set_strategy(backtesting.strategylist[0])

    data = load_data_test(contour, testdatadir)
    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)
    assert isinstance(processed, dict)
    results = backtesting.backtest(
        processed=processed,
        start_date=min_date,
        end_date=max_date,
        max_open_trades=1,
        position_stacking=False,
        enable_protections=config.get('enable_protections', False),
    )
    # results :: <class 'pandas.core.frame.DataFrame'>
    return results


# FIX: fixturize this?
def _make_backtest_conf(mocker, datadir, conf=None, pair='UNITTEST/BTC'):
    data = history.load_data(datadir=datadir, timeframe='1m', pairs=[pair])
    data = trim_dictlist(data, -201)
    patch_exchange(mocker)
    backtesting = Backtesting(conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)
    return {
        'processed': processed,
        'start_date': min_date,
        'end_date': max_date,
        'max_open_trades': 10,
        'position_stacking': False,
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
def test_setup_optimize_configuration_without_arguments(mocker, default_conf, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        'backtesting',
        '--config', 'config.json',
        '--strategy', 'StrategyTestV2',
        '--export', 'none'
    ]

    config = setup_optimize_configuration(get_args(args), RunMode.BACKTEST)
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert log_has('Using data directory: {} ...'.format(config['datadir']), caplog)
    assert 'timeframe' in config
    assert not log_has_re('Parameter -i/--ticker-interval detected .*', caplog)

    assert 'position_stacking' not in config
    assert not log_has('Parameter --enable-position-stacking detected ...', caplog)

    assert 'timerange' not in config
    assert 'export' in config
    assert config['export'] == 'none'
    assert 'runmode' in config
    assert config['runmode'] == RunMode.BACKTEST


def test_setup_bt_configuration_with_arguments(mocker, default_conf, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch(
        'freqtrade.configuration.configuration.create_datadir',
        lambda c, x: x
    )

    args = [
        'backtesting',
        '--config', 'config.json',
        '--strategy', 'StrategyTestV2',
        '--datadir', '/foo/bar',
        '--timeframe', '1m',
        '--enable-position-stacking',
        '--disable-max-market-positions',
        '--timerange', ':100',
        '--export-filename', 'foo_bar.json',
        '--fee', '0',
    ]

    config = setup_optimize_configuration(get_args(args), RunMode.BACKTEST)
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert config['runmode'] == RunMode.BACKTEST

    assert log_has('Using data directory: {} ...'.format(config['datadir']), caplog)
    assert 'timeframe' in config
    assert log_has('Parameter -i/--timeframe detected ... Using timeframe: 1m ...',
                   caplog)

    assert 'position_stacking' in config
    assert log_has('Parameter --enable-position-stacking detected ...', caplog)

    assert 'use_max_market_positions' in config
    assert log_has('Parameter --disable-max-market-positions detected ...', caplog)
    assert log_has('max_open_trades set to unlimited ...', caplog)

    assert 'timerange' in config
    assert log_has('Parameter --timerange detected: {} ...'.format(config['timerange']), caplog)

    assert 'export' in config
    assert 'exportfilename' in config
    assert isinstance(config['exportfilename'], Path)
    assert log_has('Storing backtest results to {} ...'.format(config['exportfilename']), caplog)

    assert 'fee' in config
    assert log_has('Parameter --fee detected, setting fee to: {} ...'.format(config['fee']), caplog)


def test_setup_optimize_configuration_stake_amount(mocker, default_conf, caplog) -> None:

    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        'backtesting',
        '--config', 'config.json',
        '--strategy', 'StrategyTestV2',
        '--stake-amount', '1',
        '--starting-balance', '2'
    ]

    conf = setup_optimize_configuration(get_args(args), RunMode.BACKTEST)
    assert isinstance(conf, dict)

    args = [
        'backtesting',
        '--config', 'config.json',
        '--strategy', 'StrategyTestV2',
        '--stake-amount', '1',
        '--starting-balance', '0.5'
    ]
    with pytest.raises(OperationalException, match=r"Starting balance .* smaller .*"):
        setup_optimize_configuration(get_args(args), RunMode.BACKTEST)


def test_start(mocker, fee, default_conf, caplog) -> None:
    start_mock = MagicMock()
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.start', start_mock)
    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        'backtesting',
        '--config', 'config.json',
        '--strategy', 'StrategyTestV2',
    ]
    pargs = get_args(args)
    start_backtesting(pargs)
    assert log_has('Starting freqtrade in Backtesting mode', caplog)
    assert start_mock.call_count == 1


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
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.config == default_conf
    assert backtesting.timeframe == '5m'
    assert callable(backtesting.strategy.advise_all_indicators)
    assert callable(backtesting.strategy.advise_buy)
    assert callable(backtesting.strategy.advise_sell)
    assert isinstance(backtesting.strategy.dp, DataProvider)
    get_fee.assert_called()
    assert backtesting.fee == 0.5
    assert not backtesting.strategy.order_types["stoploss_on_exchange"]


def test_backtesting_init_no_timeframe(mocker, default_conf, caplog) -> None:
    patch_exchange(mocker)
    del default_conf['timeframe']
    default_conf['strategy_list'] = ['StrategyTestV2',
                                     'SampleStrategy']

    mocker.patch('freqtrade.exchange.Exchange.get_fee', MagicMock(return_value=0.5))
    with pytest.raises(OperationalException):
        Backtesting(default_conf)
    log_has("Ticker-interval needs to be set in either configuration "
            "or as cli argument `--ticker-interval 5m`", caplog)


def test_data_with_fee(default_conf, mocker, testdatadir) -> None:
    patch_exchange(mocker)
    default_conf['fee'] = 0.1234

    fee_mock = mocker.patch('freqtrade.exchange.Exchange.get_fee', MagicMock(return_value=0.5))
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.fee == 0.1234
    assert fee_mock.call_count == 0

    default_conf['fee'] = 0.0
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.fee == 0.0
    assert fee_mock.call_count == 0


def test_data_to_dataframe_bt(default_conf, mocker, testdatadir) -> None:
    patch_exchange(mocker)
    timerange = TimeRange.parse_timerange('1510694220-1510700340')
    data = history.load_data(testdatadir, '1m', ['UNITTEST/BTC'], timerange=timerange,
                             fill_up_missing=True)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    processed = backtesting.strategy.advise_all_indicators(data)
    assert len(processed['UNITTEST/BTC']) == 102

    # Load strategy to compare the result between Backtesting function and strategy are the same
    default_conf.update({'strategy': 'StrategyTestV2'})
    strategy = StrategyResolver.load_strategy(default_conf)

    processed2 = strategy.advise_all_indicators(data)
    assert processed['UNITTEST/BTC'].equals(processed2['UNITTEST/BTC'])


def test_backtest_abort(default_conf, mocker, testdatadir) -> None:
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)
    backtesting.check_abort()

    backtesting.abort = True

    with pytest.raises(DependencyException, match="Stop requested"):
        backtesting.check_abort()
    # abort flag resets
    assert backtesting.abort is False
    assert backtesting.progress.progress == 0


def test_backtesting_start(default_conf, mocker, testdatadir, caplog) -> None:
    def get_timerange(input1):
        return Arrow(2017, 11, 14, 21, 17), Arrow(2017, 11, 14, 22, 59)

    mocker.patch('freqtrade.data.history.get_timerange', get_timerange)
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest')
    mocker.patch('freqtrade.optimize.backtesting.generate_backtest_stats')
    mocker.patch('freqtrade.optimize.backtesting.show_backtest_results')
    sbs = mocker.patch('freqtrade.optimize.backtesting.store_backtest_stats')
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['UNITTEST/BTC']))

    default_conf['timeframe'] = '1m'
    default_conf['datadir'] = testdatadir
    default_conf['export'] = 'trades'
    default_conf['exportfilename'] = 'export.txt'
    default_conf['timerange'] = '-1510694220'

    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.bot_loop_start = MagicMock()
    backtesting.start()
    # check the logs, that will contain the backtest result
    exists = [
        'Backtesting with data from 2017-11-14 21:17:00 '
        'up to 2017-11-14 22:59:00 (0 days).'
    ]
    for line in exists:
        assert log_has(line, caplog)
    assert backtesting.strategy.dp._pairlists is not None
    assert backtesting.strategy.bot_loop_start.call_count == 1
    assert sbs.call_count == 1


def test_backtesting_start_no_data(default_conf, mocker, caplog, testdatadir) -> None:
    def get_timerange(input1):
        return Arrow(2017, 11, 14, 21, 17), Arrow(2017, 11, 14, 22, 59)

    mocker.patch('freqtrade.data.history.history_utils.load_pair_history',
                 MagicMock(return_value=pd.DataFrame()))
    mocker.patch('freqtrade.data.history.get_timerange', get_timerange)
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest')
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['UNITTEST/BTC']))

    default_conf['timeframe'] = "1m"
    default_conf['datadir'] = testdatadir
    default_conf['export'] = 'none'
    default_conf['timerange'] = '20180101-20180102'

    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    with pytest.raises(OperationalException, match='No data found. Terminating.'):
        backtesting.start()


def test_backtesting_no_pair_left(default_conf, mocker, caplog, testdatadir) -> None:
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    mocker.patch('freqtrade.data.history.history_utils.load_pair_history',
                 MagicMock(return_value=pd.DataFrame()))
    mocker.patch('freqtrade.data.history.get_timerange', get_timerange)
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest')
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=[]))

    default_conf['timeframe'] = "1m"
    default_conf['datadir'] = testdatadir
    default_conf['export'] = 'none'
    default_conf['timerange'] = '20180101-20180102'

    with pytest.raises(OperationalException, match='No pair in whitelist.'):
        Backtesting(default_conf)

    default_conf['pairlists'] = [{"method": "VolumePairList", "number_assets": 5}]
    with pytest.raises(OperationalException, match='VolumePairList not allowed for backtesting.'):
        Backtesting(default_conf)

    default_conf.update({
        'pairlists': [{"method": "StaticPairList"}],
        'timeframe_detail': '1d',
    })

    with pytest.raises(OperationalException,
                       match='Detail timeframe must be smaller than strategy timeframe.'):
        Backtesting(default_conf)


def test_backtesting_pairlist_list(default_conf, mocker, caplog, testdatadir, tickers) -> None:
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    mocker.patch('freqtrade.exchange.Exchange.get_tickers', tickers)
    mocker.patch('freqtrade.exchange.Exchange.price_to_precision', lambda s, x, y: y)
    mocker.patch('freqtrade.data.history.get_timerange', get_timerange)
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest')
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['XRP/BTC']))
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.refresh_pairlist')

    default_conf['ticker_interval'] = "1m"
    default_conf['datadir'] = testdatadir
    default_conf['export'] = 'none'
    # Use stoploss from strategy
    del default_conf['stoploss']
    default_conf['timerange'] = '20180101-20180102'

    default_conf['pairlists'] = [{"method": "VolumePairList", "number_assets": 5}]
    with pytest.raises(OperationalException, match='VolumePairList not allowed for backtesting.'):
        Backtesting(default_conf)

    default_conf['pairlists'] = [{"method": "StaticPairList"}, {"method": "PerformanceFilter"}]
    with pytest.raises(OperationalException,
                       match='PerformanceFilter not allowed for backtesting.'):
        Backtesting(default_conf)

    default_conf['pairlists'] = [{"method": "StaticPairList"}, {"method": "PrecisionFilter"}, ]
    Backtesting(default_conf)

    # Multiple strategies
    default_conf['strategy_list'] = ['StrategyTestV2', 'TestStrategyLegacyV1']
    with pytest.raises(OperationalException,
                       match='PrecisionFilter not allowed for backtesting multiple strategies.'):
        Backtesting(default_conf)


def test_backtest__enter_trade(default_conf, fee, mocker) -> None:
    default_conf['use_sell_signal'] = False
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    mocker.patch("freqtrade.exchange.Exchange.get_min_pair_stake_amount", return_value=0.00001)
    patch_exchange(mocker)
    default_conf['stake_amount'] = 'unlimited'
    default_conf['max_open_trades'] = 2
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    pair = 'UNITTEST/BTC'
    row = [
        pd.Timestamp(year=2020, month=1, day=1, hour=5, minute=0),
        1,  # Buy
        0.001,  # Open
        0.0011,  # Close
        0,  # Sell
        0.00099,  # Low
        0.0012,  # High
        '',  # Buy Signal Name
    ]
    trade = backtesting._enter_trade(pair, row=row)
    assert isinstance(trade, LocalTrade)
    assert trade.stake_amount == 495

    # Fake 2 trades, so there's not enough amount for the next trade left.
    LocalTrade.trades_open.append(trade)
    LocalTrade.trades_open.append(trade)
    trade = backtesting._enter_trade(pair, row=row)
    assert trade is None
    LocalTrade.trades_open.pop()
    trade = backtesting._enter_trade(pair, row=row)
    assert trade is not None

    backtesting.strategy.custom_stake_amount = lambda **kwargs: 123.5
    trade = backtesting._enter_trade(pair, row=row)
    assert trade
    assert trade.stake_amount == 123.5

    # In case of error - use proposed stake
    backtesting.strategy.custom_stake_amount = lambda **kwargs: 20 / 0
    trade = backtesting._enter_trade(pair, row=row)
    assert trade
    assert trade.stake_amount == 495

    # Stake-amount too high!
    mocker.patch("freqtrade.exchange.Exchange.get_min_pair_stake_amount", return_value=600.0)

    trade = backtesting._enter_trade(pair, row=row)
    assert trade is None

    # Stake-amount throwing error
    mocker.patch("freqtrade.wallets.Wallets.get_trade_stake_amount",
                 side_effect=DependencyException)

    trade = backtesting._enter_trade(pair, row=row)
    assert trade is None

    backtesting.cleanup()


def test_backtest__get_sell_trade_entry(default_conf, fee, mocker) -> None:
    default_conf['use_sell_signal'] = False
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    mocker.patch("freqtrade.exchange.Exchange.get_min_pair_stake_amount", return_value=0.00001)
    patch_exchange(mocker)
    default_conf['timeframe_detail'] = '1m'
    default_conf['max_open_trades'] = 2
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    pair = 'UNITTEST/BTC'
    row = [
        pd.Timestamp(year=2020, month=1, day=1, hour=4, minute=55, tzinfo=timezone.utc),
        1,  # Buy
        200,  # Open
        201,  # Close
        0,  # Sell
        195,  # Low
        201.5,  # High
        '',  # Buy Signal Name
    ]

    trade = backtesting._enter_trade(pair, row=row)
    assert isinstance(trade, LocalTrade)

    row_sell = [
        pd.Timestamp(year=2020, month=1, day=1, hour=5, minute=0, tzinfo=timezone.utc),
        0,  # Buy
        200,  # Open
        201,  # Close
        0,  # Sell
        195,  # Low
        210.5,  # High
        '',  # Buy Signal Name
    ]
    row_detail = pd.DataFrame(
        [
            [
                pd.Timestamp(year=2020, month=1, day=1, hour=5, minute=0, tzinfo=timezone.utc),
                1, 200, 199, 0, 197, 200.1, '',
            ], [
                pd.Timestamp(year=2020, month=1, day=1, hour=5, minute=1, tzinfo=timezone.utc),
                0, 199, 199.5, 0, 199, 199.7, '',
            ], [
                pd.Timestamp(year=2020, month=1, day=1, hour=5, minute=2, tzinfo=timezone.utc),
                0, 199.5, 200.5, 0, 199, 200.8, '',
            ], [
                pd.Timestamp(year=2020, month=1, day=1, hour=5, minute=3, tzinfo=timezone.utc),
                0, 200.5, 210.5, 0, 193, 210.5, '',  # ROI sell (?)
            ], [
                pd.Timestamp(year=2020, month=1, day=1, hour=5, minute=4, tzinfo=timezone.utc),
                0, 200, 199, 0, 193, 200.1, '',
            ],
        ], columns=["date", "buy", "open", "close", "sell", "low", "high", "buy_tag"]
    )

    # No data available.
    res = backtesting._get_sell_trade_entry(trade, row_sell)
    assert res is not None
    assert res.sell_reason == SellType.ROI.value
    assert res.close_date_utc == datetime(2020, 1, 1, 5, 0, tzinfo=timezone.utc)

    # Enter new trade
    trade = backtesting._enter_trade(pair, row=row)
    assert isinstance(trade, LocalTrade)
    # Assign empty ... no result.
    backtesting.detail_data[pair] = pd.DataFrame(
        [], columns=["date", "buy", "open", "close", "sell", "low", "high", "buy_tag"])

    res = backtesting._get_sell_trade_entry(trade, row)
    assert res is None

    # Assign backtest-detail data
    backtesting.detail_data[pair] = row_detail

    res = backtesting._get_sell_trade_entry(trade, row_sell)
    assert res is not None
    assert res.sell_reason == SellType.ROI.value
    # Sell at minute 3 (not available above!)
    assert res.close_date_utc == datetime(2020, 1, 1, 5, 3, tzinfo=timezone.utc)
    assert round(res.close_rate, 3) == round(209.0225, 3)


def test_backtest_one(default_conf, fee, mocker, testdatadir) -> None:
    default_conf['use_sell_signal'] = False
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    mocker.patch("freqtrade.exchange.Exchange.get_min_pair_stake_amount", return_value=0.00001)
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    pair = 'UNITTEST/BTC'
    timerange = TimeRange('date', None, 1517227800, 0)
    data = history.load_data(datadir=testdatadir, timeframe='5m', pairs=['UNITTEST/BTC'],
                             timerange=timerange)
    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)
    result = backtesting.backtest(
        processed=processed,
        start_date=min_date,
        end_date=max_date,
        max_open_trades=10,
        position_stacking=False,
    )
    results = result['results']
    assert not results.empty
    assert len(results) == 2

    expected = pd.DataFrame(
        {'pair': [pair, pair],
         'stake_amount': [0.001, 0.001],
         'amount': [0.00957442, 0.0097064],
         'open_date': pd.to_datetime([Arrow(2018, 1, 29, 18, 40, 0).datetime,
                                      Arrow(2018, 1, 30, 3, 30, 0).datetime], utc=True
                                     ),
         'close_date': pd.to_datetime([Arrow(2018, 1, 29, 22, 35, 0).datetime,
                                       Arrow(2018, 1, 30, 4, 10, 0).datetime], utc=True),
         'open_rate': [0.104445, 0.10302485],
         'close_rate': [0.104969, 0.103541],
         'fee_open': [0.0025, 0.0025],
         'fee_close': [0.0025, 0.0025],
         'trade_duration': [235, 40],
         'profit_ratio': [0.0, 0.0],
         'profit_abs': [0.0, 0.0],
         'sell_reason': [SellType.ROI.value, SellType.ROI.value],
         'initial_stop_loss_abs': [0.0940005, 0.09272236],
         'initial_stop_loss_ratio': [-0.1, -0.1],
         'stop_loss_abs': [0.0940005, 0.09272236],
         'stop_loss_ratio': [-0.1, -0.1],
         'min_rate': [0.10370188, 0.10300000000000001],
         'max_rate': [0.10501, 0.1038888],
         'is_open': [False, False],
         'buy_tag': [None, None],
         })
    pd.testing.assert_frame_equal(results, expected)
    data_pair = processed[pair]
    for _, t in results.iterrows():
        ln = data_pair.loc[data_pair["date"] == t["open_date"]]
        # Check open trade rate alignes to open rate
        assert ln is not None
        assert round(ln.iloc[0]["open"], 6) == round(t["open_rate"], 6)
        # check close trade rate alignes to close rate or is between high and low
        ln = data_pair.loc[data_pair["date"] == t["close_date"]]
        assert (round(ln.iloc[0]["open"], 6) == round(t["close_rate"], 6) or
                round(ln.iloc[0]["low"], 6) < round(
                t["close_rate"], 6) < round(ln.iloc[0]["high"], 6))


def test_backtest_1min_timeframe(default_conf, fee, mocker, testdatadir) -> None:
    default_conf['use_sell_signal'] = False
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    mocker.patch("freqtrade.exchange.Exchange.get_min_pair_stake_amount", return_value=0.00001)
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])

    # Run a backtesting for an exiting 1min timeframe
    timerange = TimeRange.parse_timerange('1510688220-1510700340')
    data = history.load_data(datadir=testdatadir, timeframe='1m', pairs=['UNITTEST/BTC'],
                             timerange=timerange)
    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)
    results = backtesting.backtest(
        processed=processed,
        start_date=min_date,
        end_date=max_date,
        max_open_trades=1,
        position_stacking=False,
    )
    assert not results['results'].empty
    assert len(results['results']) == 1


def test_processed(default_conf, mocker, testdatadir) -> None:
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])

    dict_of_tickerrows = load_data_test('raise', testdatadir)
    dataframes = backtesting.strategy.advise_all_indicators(dict_of_tickerrows)
    dataframe = dataframes['UNITTEST/BTC']
    cols = dataframe.columns
    # assert the dataframe got some of the indicator columns
    for col in ['close', 'high', 'low', 'open', 'date',
                'ema10', 'rsi', 'fastd', 'plus_di']:
        assert col in cols


def test_backtest_pricecontours_protections(default_conf, fee, mocker, testdatadir) -> None:
    # While this test IS a copy of test_backtest_pricecontours, it's needed to ensure
    # results do not carry-over to the next run, which is not given by using parametrize.
    default_conf['protections'] = [
        {
            "method": "CooldownPeriod",
            "stop_duration": 3,
        }]

    default_conf['enable_protections'] = True
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    mocker.patch("freqtrade.exchange.Exchange.get_min_pair_stake_amount", return_value=0.00001)
    tests = [
        ['sine', 9],
        ['raise', 10],
        ['lower', 0],
        ['sine', 9],
        ['raise', 10],
    ]
    # While buy-signals are unrealistic, running backtesting
    # over and over again should not cause different results
    for [contour, numres] in tests:
        assert len(simple_backtest(default_conf, contour, mocker, testdatadir)['results']) == numres


@pytest.mark.parametrize('protections,contour,expected', [
    (None, 'sine', 35),
    (None, 'raise', 19),
    (None, 'lower', 0),
    (None, 'sine', 35),
    (None, 'raise', 19),
    ([{"method": "CooldownPeriod", "stop_duration": 3}], 'sine', 9),
    ([{"method": "CooldownPeriod", "stop_duration": 3}], 'raise', 10),
    ([{"method": "CooldownPeriod", "stop_duration": 3}], 'lower', 0),
    ([{"method": "CooldownPeriod", "stop_duration": 3}], 'sine', 9),
    ([{"method": "CooldownPeriod", "stop_duration": 3}], 'raise', 10),
])
def test_backtest_pricecontours(default_conf, fee, mocker, testdatadir,
                                protections, contour, expected) -> None:
    if protections:
        default_conf['protections'] = protections
        default_conf['enable_protections'] = True

    mocker.patch("freqtrade.exchange.Exchange.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    # While buy-signals are unrealistic, running backtesting
    # over and over again should not cause different results
    assert len(simple_backtest(default_conf, contour, mocker, testdatadir)['results']) == expected


def test_backtest_clash_buy_sell(mocker, default_conf, testdatadir):
    # Override the default buy trend function in our StrategyTestV2
    def fun(dataframe=None, pair=None):
        buy_value = 1
        sell_value = 1
        return _trend(dataframe, buy_value, sell_value)

    backtest_conf = _make_backtest_conf(mocker, conf=default_conf, datadir=testdatadir)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.advise_buy = fun  # Override
    backtesting.strategy.advise_sell = fun  # Override
    result = backtesting.backtest(**backtest_conf)
    assert result['results'].empty


def test_backtest_only_sell(mocker, default_conf, testdatadir):
    # Override the default buy trend function in our StrategyTestV2
    def fun(dataframe=None, pair=None):
        buy_value = 0
        sell_value = 1
        return _trend(dataframe, buy_value, sell_value)

    backtest_conf = _make_backtest_conf(mocker, conf=default_conf, datadir=testdatadir)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.advise_buy = fun  # Override
    backtesting.strategy.advise_sell = fun  # Override
    result = backtesting.backtest(**backtest_conf)
    assert result['results'].empty


def test_backtest_alternate_buy_sell(default_conf, fee, mocker, testdatadir):
    mocker.patch("freqtrade.exchange.Exchange.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    backtest_conf = _make_backtest_conf(mocker, conf=default_conf,
                                        pair='UNITTEST/BTC', datadir=testdatadir)
    default_conf['timeframe'] = '1m'
    backtesting = Backtesting(default_conf)
    backtesting.required_startup = 0
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.advise_buy = _trend_alternate  # Override
    backtesting.strategy.advise_sell = _trend_alternate  # Override
    result = backtesting.backtest(**backtest_conf)
    # 200 candles in backtest data
    # won't buy on first (shifted by 1)
    # 100 buys signals
    results = result['results']
    assert len(results) == 100
    # Cached data should be 200
    analyzed_df = backtesting.dataprovider.get_analyzed_dataframe('UNITTEST/BTC', '1m')[0]
    assert len(analyzed_df) == 200
    # Expect last candle to be 1 below end date (as the last candle is assumed as "incomplete"
    # during backtesting)
    expected_last_candle_date = backtest_conf['end_date'] - timedelta(minutes=1)
    assert analyzed_df.iloc[-1]['date'].to_pydatetime() == expected_last_candle_date

    # One trade was force-closed at the end
    assert len(results.loc[results['is_open']]) == 0


@pytest.mark.parametrize("pair", ['ADA/BTC', 'LTC/BTC'])
@pytest.mark.parametrize("tres", [0, 20, 30])
def test_backtest_multi_pair(default_conf, fee, mocker, tres, pair, testdatadir):

    def _trend_alternate_hold(dataframe=None, metadata=None):
        """
        Buy every xth candle - sell every other xth -2 (hold on to pairs a bit)
        """
        if metadata['pair'] in ('ETH/BTC', 'LTC/BTC'):
            multi = 20
        else:
            multi = 18
        dataframe['buy'] = np.where(dataframe.index % multi == 0, 1, 0)
        dataframe['sell'] = np.where((dataframe.index + multi - 2) % multi == 0, 1, 0)
        return dataframe

    mocker.patch("freqtrade.exchange.Exchange.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    patch_exchange(mocker)

    pairs = ['ADA/BTC', 'DASH/BTC', 'ETH/BTC', 'LTC/BTC', 'NXT/BTC']
    data = history.load_data(datadir=testdatadir, timeframe='5m', pairs=pairs)
    # Only use 500 lines to increase performance
    data = trim_dictlist(data, -500)

    # Remove data for one pair from the beginning of the data
    if tres > 0:
        data[pair] = data[pair][tres:].reset_index()
    default_conf['timeframe'] = '5m'

    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.advise_buy = _trend_alternate_hold  # Override
    backtesting.strategy.advise_sell = _trend_alternate_hold  # Override

    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)
    backtest_conf = {
        'processed': processed,
        'start_date': min_date,
        'end_date': max_date,
        'max_open_trades': 3,
        'position_stacking': False,
    }

    results = backtesting.backtest(**backtest_conf)

    # Make sure we have parallel trades
    assert len(evaluate_result_multi(results['results'], '5m', 2)) > 0
    # make sure we don't have trades with more than configured max_open_trades
    assert len(evaluate_result_multi(results['results'], '5m', 3)) == 0

    # Cached data correctly removed amounts
    offset = 1 if tres == 0 else 0
    removed_candles = len(data[pair]) - offset - backtesting.strategy.startup_candle_count
    assert len(backtesting.dataprovider.get_analyzed_dataframe(pair, '5m')[0]) == removed_candles
    assert len(backtesting.dataprovider.get_analyzed_dataframe(
        'NXT/BTC', '5m')[0]) == len(data['NXT/BTC']) - 1 - backtesting.strategy.startup_candle_count

    backtest_conf = {
        'processed': processed,
        'start_date': min_date,
        'end_date': max_date,
        'max_open_trades': 1,
        'position_stacking': False,
    }
    results = backtesting.backtest(**backtest_conf)
    assert len(evaluate_result_multi(results['results'], '5m', 1)) == 0


def test_backtest_start_timerange(default_conf, mocker, caplog, testdatadir):

    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest')
    mocker.patch('freqtrade.optimize.backtesting.generate_backtest_stats')
    mocker.patch('freqtrade.optimize.backtesting.show_backtest_results')
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['UNITTEST/BTC']))
    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        'backtesting',
        '--config', 'config.json',
        '--strategy', 'StrategyTestV2',
        '--datadir', str(testdatadir),
        '--timeframe', '1m',
        '--timerange', '1510694220-1510700340',
        '--enable-position-stacking',
        '--disable-max-market-positions'
    ]
    args = get_args(args)
    start_backtesting(args)
    # check the logs, that will contain the backtest result
    exists = [
        'Parameter -i/--timeframe detected ... Using timeframe: 1m ...',
        'Ignoring max_open_trades (--disable-max-market-positions was used) ...',
        'Parameter --timerange detected: 1510694220-1510700340 ...',
        f'Using data directory: {testdatadir} ...',
        'Loading data from 2017-11-14 20:57:00 '
        'up to 2017-11-14 22:58:00 (0 days).',
        'Backtesting with data from 2017-11-14 21:17:00 '
        'up to 2017-11-14 22:58:00 (0 days).',
        'Parameter --enable-position-stacking detected ...'
    ]

    for line in exists:
        assert log_has(line, caplog)


@pytest.mark.filterwarnings("ignore:deprecated")
def test_backtest_start_multi_strat(default_conf, mocker, caplog, testdatadir):

    default_conf.update({
        "use_sell_signal": True,
        "sell_profit_only": False,
        "sell_profit_offset": 0.0,
        "ignore_roi_if_buy_signal": False,
    })
    patch_exchange(mocker)
    backtestmock = MagicMock(return_value={
        'results': pd.DataFrame(columns=BT_DATA_COLUMNS),
        'config': default_conf,
        'locks': [],
        'rejected_signals': 20,
        'final_balance': 1000,
    })
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['UNITTEST/BTC']))
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest', backtestmock)
    text_table_mock = MagicMock()
    sell_reason_mock = MagicMock()
    strattable_mock = MagicMock()
    strat_summary = MagicMock()

    mocker.patch.multiple('freqtrade.optimize.optimize_reports',
                          text_table_bt_results=text_table_mock,
                          text_table_strategy=strattable_mock,
                          generate_pair_metrics=MagicMock(),
                          generate_sell_reason_stats=sell_reason_mock,
                          generate_strategy_comparison=strat_summary,
                          generate_daily_stats=MagicMock(),
                          )
    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        'backtesting',
        '--config', 'config.json',
        '--datadir', str(testdatadir),
        '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'),
        '--timeframe', '1m',
        '--timerange', '1510694220-1510700340',
        '--enable-position-stacking',
        '--disable-max-market-positions',
        '--strategy-list',
        'StrategyTestV2',
        'TestStrategyLegacyV1',
    ]
    args = get_args(args)
    start_backtesting(args)
    # 2 backtests, 4 tables
    assert backtestmock.call_count == 2
    assert text_table_mock.call_count == 4
    assert strattable_mock.call_count == 1
    assert sell_reason_mock.call_count == 2
    assert strat_summary.call_count == 1

    # check the logs, that will contain the backtest result
    exists = [
        'Parameter -i/--timeframe detected ... Using timeframe: 1m ...',
        'Ignoring max_open_trades (--disable-max-market-positions was used) ...',
        'Parameter --timerange detected: 1510694220-1510700340 ...',
        f'Using data directory: {testdatadir} ...',
        'Loading data from 2017-11-14 20:57:00 '
        'up to 2017-11-14 22:58:00 (0 days).',
        'Backtesting with data from 2017-11-14 21:17:00 '
        'up to 2017-11-14 22:58:00 (0 days).',
        'Parameter --enable-position-stacking detected ...',
        'Running backtesting for Strategy StrategyTestV2',
        'Running backtesting for Strategy TestStrategyLegacyV1',
    ]

    for line in exists:
        assert log_has(line, caplog)


@pytest.mark.filterwarnings("ignore:deprecated")
def test_backtest_start_multi_strat_nomock(default_conf, mocker, caplog, testdatadir, capsys):
    default_conf.update({
        "use_sell_signal": True,
        "sell_profit_only": False,
        "sell_profit_offset": 0.0,
        "ignore_roi_if_buy_signal": False,
    })
    patch_exchange(mocker)
    result1 = pd.DataFrame({'pair': ['XRP/BTC', 'LTC/BTC'],
                            'profit_ratio': [0.0, 0.0],
                            'profit_abs': [0.0, 0.0],
                            'open_date': pd.to_datetime(['2018-01-29 18:40:00',
                                                         '2018-01-30 03:30:00', ], utc=True
                                                        ),
                            'close_date': pd.to_datetime(['2018-01-29 20:45:00',
                                                          '2018-01-30 05:35:00', ], utc=True),
                            'trade_duration': [235, 40],
                            'is_open': [False, False],
                            'stake_amount': [0.01, 0.01],
                            'open_rate': [0.104445, 0.10302485],
                            'close_rate': [0.104969, 0.103541],
                            'sell_reason': [SellType.ROI, SellType.ROI]
                            })
    result2 = pd.DataFrame({'pair': ['XRP/BTC', 'LTC/BTC', 'ETH/BTC'],
                            'profit_ratio': [0.03, 0.01, 0.1],
                            'profit_abs': [0.01, 0.02, 0.2],
                            'open_date': pd.to_datetime(['2018-01-29 18:40:00',
                                                         '2018-01-30 03:30:00',
                                                         '2018-01-30 05:30:00'], utc=True
                                                        ),
                            'close_date': pd.to_datetime(['2018-01-29 20:45:00',
                                                          '2018-01-30 05:35:00',
                                                          '2018-01-30 08:30:00'], utc=True),
                            'trade_duration': [47, 40, 20],
                            'is_open': [False, False, False],
                            'stake_amount': [0.01, 0.01, 0.01],
                            'open_rate': [0.104445, 0.10302485, 0.122541],
                            'close_rate': [0.104969, 0.103541, 0.123541],
                            'sell_reason': [SellType.ROI, SellType.ROI, SellType.STOP_LOSS]
                            })
    backtestmock = MagicMock(side_effect=[
        {
            'results': result1,
            'config': default_conf,
            'locks': [],
            'rejected_signals': 20,
            'final_balance': 1000,
        },
        {
            'results': result2,
            'config': default_conf,
            'locks': [],
            'rejected_signals': 20,
            'final_balance': 1000,
        }
    ])
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['UNITTEST/BTC']))
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest', backtestmock)

    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        'backtesting',
        '--config', 'config.json',
        '--datadir', str(testdatadir),
        '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'),
        '--timeframe', '1m',
        '--timerange', '1510694220-1510700340',
        '--enable-position-stacking',
        '--disable-max-market-positions',
        '--breakdown', 'day',
        '--strategy-list',
        'StrategyTestV2',
        'TestStrategyLegacyV1',
    ]
    args = get_args(args)
    start_backtesting(args)

    # check the logs, that will contain the backtest result
    exists = [
        'Parameter -i/--timeframe detected ... Using timeframe: 1m ...',
        'Ignoring max_open_trades (--disable-max-market-positions was used) ...',
        'Parameter --timerange detected: 1510694220-1510700340 ...',
        f'Using data directory: {testdatadir} ...',
        'Loading data from 2017-11-14 20:57:00 '
        'up to 2017-11-14 22:58:00 (0 days).',
        'Backtesting with data from 2017-11-14 21:17:00 '
        'up to 2017-11-14 22:58:00 (0 days).',
        'Parameter --enable-position-stacking detected ...',
        'Running backtesting for Strategy StrategyTestV2',
        'Running backtesting for Strategy TestStrategyLegacyV1',
    ]

    for line in exists:
        assert log_has(line, caplog)

    captured = capsys.readouterr()
    assert 'BACKTESTING REPORT' in captured.out
    assert 'SELL REASON STATS' in captured.out
    assert 'DAY BREAKDOWN' in captured.out
    assert 'LEFT OPEN TRADES REPORT' in captured.out
    assert '2017-11-14 21:17:00 -> 2017-11-14 22:58:00 | Max open trades : 1' in captured.out
    assert 'STRATEGY SUMMARY' in captured.out


@pytest.mark.filterwarnings("ignore:deprecated")
def test_backtest_start_multi_strat_nomock_detail(default_conf, mocker,
                                                  caplog, testdatadir, capsys):
    # Tests detail-data loading
    default_conf.update({
        "use_sell_signal": True,
        "sell_profit_only": False,
        "sell_profit_offset": 0.0,
        "ignore_roi_if_buy_signal": False,
    })
    patch_exchange(mocker)
    result1 = pd.DataFrame({'pair': ['XRP/BTC', 'LTC/BTC'],
                            'profit_ratio': [0.0, 0.0],
                            'profit_abs': [0.0, 0.0],
                            'open_date': pd.to_datetime(['2018-01-29 18:40:00',
                                                         '2018-01-30 03:30:00', ], utc=True
                                                        ),
                            'close_date': pd.to_datetime(['2018-01-29 20:45:00',
                                                          '2018-01-30 05:35:00', ], utc=True),
                            'trade_duration': [235, 40],
                            'is_open': [False, False],
                            'stake_amount': [0.01, 0.01],
                            'open_rate': [0.104445, 0.10302485],
                            'close_rate': [0.104969, 0.103541],
                            'sell_reason': [SellType.ROI, SellType.ROI]
                            })
    result2 = pd.DataFrame({'pair': ['XRP/BTC', 'LTC/BTC', 'ETH/BTC'],
                            'profit_ratio': [0.03, 0.01, 0.1],
                            'profit_abs': [0.01, 0.02, 0.2],
                            'open_date': pd.to_datetime(['2018-01-29 18:40:00',
                                                         '2018-01-30 03:30:00',
                                                         '2018-01-30 05:30:00'], utc=True
                                                        ),
                            'close_date': pd.to_datetime(['2018-01-29 20:45:00',
                                                          '2018-01-30 05:35:00',
                                                          '2018-01-30 08:30:00'], utc=True),
                            'trade_duration': [47, 40, 20],
                            'is_open': [False, False, False],
                            'stake_amount': [0.01, 0.01, 0.01],
                            'open_rate': [0.104445, 0.10302485, 0.122541],
                            'close_rate': [0.104969, 0.103541, 0.123541],
                            'sell_reason': [SellType.ROI, SellType.ROI, SellType.STOP_LOSS]
                            })
    backtestmock = MagicMock(side_effect=[
        {
            'results': result1,
            'config': default_conf,
            'locks': [],
            'rejected_signals': 20,
            'final_balance': 1000,
        },
        {
            'results': result2,
            'config': default_conf,
            'locks': [],
            'rejected_signals': 20,
            'final_balance': 1000,
        }
    ])
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['XRP/ETH']))
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest', backtestmock)

    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        'backtesting',
        '--config', 'config.json',
        '--datadir', str(testdatadir),
        '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'),
        '--timeframe', '5m',
        '--timeframe-detail', '1m',
        '--strategy-list',
        'StrategyTestV2'
    ]
    args = get_args(args)
    start_backtesting(args)

    # check the logs, that will contain the backtest result
    exists = [
        'Parameter -i/--timeframe detected ... Using timeframe: 5m ...',
        'Parameter --timeframe-detail detected, using 1m for intra-candle backtesting ...',
        f'Using data directory: {testdatadir} ...',
        'Loading data from 2019-10-11 00:00:00 '
        'up to 2019-10-13 11:10:00 (2 days).',
        'Backtesting with data from 2019-10-11 01:40:00 '
        'up to 2019-10-13 11:10:00 (2 days).',
        'Running backtesting for Strategy StrategyTestV2',
    ]

    for line in exists:
        assert log_has(line, caplog)

    captured = capsys.readouterr()
    assert 'BACKTESTING REPORT' in captured.out
    assert 'SELL REASON STATS' in captured.out
    assert 'LEFT OPEN TRADES REPORT' in captured.out
