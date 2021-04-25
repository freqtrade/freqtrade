# pragma pylint: disable=missing-docstring,W0212,C0103
import locale
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import pandas as pd
import pytest
from arrow import Arrow
from filelock import Timeout

from freqtrade.commands.optimize_commands import setup_optimize_configuration, start_hyperopt
from freqtrade.data.history import load_data
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.hyperopt import Hyperopt
from freqtrade.optimize.hyperopt_auto import HyperOptAuto
from freqtrade.optimize.hyperopt_tools import HyperoptTools
from freqtrade.optimize.space import SKDecimal
from freqtrade.resolvers.hyperopt_resolver import HyperOptResolver
from freqtrade.state import RunMode
from freqtrade.strategy.hyper import IntParameter
from tests.conftest import (get_args, log_has, log_has_re, patch_exchange,
                            patched_configuration_load_config_file)

from .hyperopts.default_hyperopt import DefaultHyperOpt


# Functions for recurrent object patching
def create_results(mocker, hyperopt, testdatadir) -> List[Dict]:
    """
    When creating results, mock the hyperopt so that *by default*
      - we don't create any pickle'd files in the filesystem
      - we might have a pickle'd file so make sure that we return
        false when looking for it
    """
    hyperopt.results_file = testdatadir / 'optimize/ut_results.pickle'

    mocker.patch.object(Path, "is_file", MagicMock(return_value=False))
    stat_mock = MagicMock()
    stat_mock.st_size = 1
    mocker.patch.object(Path, "stat", MagicMock(return_value=stat_mock))

    mocker.patch.object(Path, "unlink", MagicMock(return_value=True))
    mocker.patch('freqtrade.optimize.hyperopt.dump', return_value=None)
    mocker.patch('freqtrade.optimize.hyperopt.file_dump_json')

    return [{'loss': 1, 'result': 'foo', 'params': {}, 'is_best': True}]


def test_setup_hyperopt_configuration_without_arguments(mocker, default_conf, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        'hyperopt',
        '--config', 'config.json',
        '--hyperopt', 'DefaultHyperOpt',
    ]

    config = setup_optimize_configuration(get_args(args), RunMode.HYPEROPT)
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
    assert 'runmode' in config
    assert config['runmode'] == RunMode.HYPEROPT


def test_setup_hyperopt_configuration_with_arguments(mocker, default_conf, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch(
        'freqtrade.configuration.configuration.create_datadir',
        lambda c, x: x
    )

    args = [
        'hyperopt',
        '--config', 'config.json',
        '--hyperopt', 'DefaultHyperOpt',
        '--datadir', '/foo/bar',
        '--timeframe', '1m',
        '--timerange', ':100',
        '--enable-position-stacking',
        '--disable-max-market-positions',
        '--epochs', '1000',
        '--spaces', 'default',
        '--print-all'
    ]

    config = setup_optimize_configuration(get_args(args), RunMode.HYPEROPT)
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert config['runmode'] == RunMode.HYPEROPT

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

    assert 'epochs' in config
    assert log_has('Parameter --epochs detected ... Will run Hyperopt with for 1000 epochs ...',
                   caplog)

    assert 'spaces' in config
    assert log_has('Parameter -s/--spaces detected: {}'.format(config['spaces']), caplog)
    assert 'print_all' in config
    assert log_has('Parameter --print-all detected ...', caplog)


def test_setup_hyperopt_configuration_stake_amount(mocker, default_conf) -> None:

    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        'hyperopt',
        '--config', 'config.json',
        '--hyperopt', 'DefaultHyperOpt',
        '--stake-amount', '1',
        '--starting-balance', '2'
    ]
    conf = setup_optimize_configuration(get_args(args), RunMode.HYPEROPT)
    assert isinstance(conf, dict)

    args = [
        'hyperopt',
        '--config', 'config.json',
        '--strategy', 'DefaultStrategy',
        '--stake-amount', '1',
        '--starting-balance', '0.5'
    ]
    with pytest.raises(OperationalException, match=r"Starting balance .* smaller .*"):
        setup_optimize_configuration(get_args(args), RunMode.HYPEROPT)


def test_hyperoptresolver(mocker, default_conf, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)

    hyperopt = DefaultHyperOpt
    delattr(hyperopt, 'populate_indicators')
    delattr(hyperopt, 'populate_buy_trend')
    delattr(hyperopt, 'populate_sell_trend')
    mocker.patch(
        'freqtrade.resolvers.hyperopt_resolver.HyperOptResolver.load_object',
        MagicMock(return_value=hyperopt(default_conf))
    )
    default_conf.update({'hyperopt': 'DefaultHyperOpt'})
    x = HyperOptResolver.load_hyperopt(default_conf)
    assert not hasattr(x, 'populate_indicators')
    assert not hasattr(x, 'populate_buy_trend')
    assert not hasattr(x, 'populate_sell_trend')
    assert log_has("Hyperopt class does not provide populate_indicators() method. "
                   "Using populate_indicators from the strategy.", caplog)
    assert log_has("Hyperopt class does not provide populate_sell_trend() method. "
                   "Using populate_sell_trend from the strategy.", caplog)
    assert log_has("Hyperopt class does not provide populate_buy_trend() method. "
                   "Using populate_buy_trend from the strategy.", caplog)
    assert hasattr(x, "ticker_interval")  # DEPRECATED
    assert hasattr(x, "timeframe")


def test_hyperoptresolver_wrongname(default_conf) -> None:
    default_conf.update({'hyperopt': "NonExistingHyperoptClass"})

    with pytest.raises(OperationalException, match=r'Impossible to load Hyperopt.*'):
        HyperOptResolver.load_hyperopt(default_conf)


def test_hyperoptresolver_noname(default_conf):
    default_conf['hyperopt'] = ''
    with pytest.raises(OperationalException,
                       match="No Hyperopt set. Please use `--hyperopt` to specify "
                             "the Hyperopt class to use."):
        HyperOptResolver.load_hyperopt(default_conf)


def test_start_not_installed(mocker, default_conf, import_fails) -> None:
    start_mock = MagicMock()
    patched_configuration_load_config_file(mocker, default_conf)

    mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.start', start_mock)
    patch_exchange(mocker)

    args = [
        'hyperopt',
        '--config', 'config.json',
        '--hyperopt', 'DefaultHyperOpt',
        '--hyperopt-path',
        str(Path(__file__).parent / "hyperopts"),
        '--epochs', '5',
        '--hyperopt-loss', 'SharpeHyperOptLossDaily',
    ]
    pargs = get_args(args)

    with pytest.raises(OperationalException, match=r"Please ensure that the hyperopt dependencies"):
        start_hyperopt(pargs)


def test_start(mocker, hyperopt_conf, caplog) -> None:
    start_mock = MagicMock()
    patched_configuration_load_config_file(mocker, hyperopt_conf)
    mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.start', start_mock)
    patch_exchange(mocker)

    args = [
        'hyperopt',
        '--config', 'config.json',
        '--hyperopt', 'DefaultHyperOpt',
        '--hyperopt-loss', 'SharpeHyperOptLossDaily',
        '--epochs', '5'
    ]
    pargs = get_args(args)
    start_hyperopt(pargs)

    assert log_has('Starting freqtrade in Hyperopt mode', caplog)
    assert start_mock.call_count == 1


def test_start_no_data(mocker, hyperopt_conf) -> None:
    patched_configuration_load_config_file(mocker, hyperopt_conf)
    mocker.patch('freqtrade.data.history.load_pair_history', MagicMock(return_value=pd.DataFrame))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timerange',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    patch_exchange(mocker)

    args = [
        'hyperopt',
        '--config', 'config.json',
        '--hyperopt', 'DefaultHyperOpt',
        '--hyperopt-loss', 'SharpeHyperOptLossDaily',
        '--epochs', '5'
    ]
    pargs = get_args(args)
    with pytest.raises(OperationalException, match='No data found. Terminating.'):
        start_hyperopt(pargs)


def test_start_filelock(mocker, hyperopt_conf, caplog) -> None:
    hyperopt_mock = MagicMock(side_effect=Timeout(Hyperopt.get_lock_filename(hyperopt_conf)))
    patched_configuration_load_config_file(mocker, hyperopt_conf)
    mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.__init__', hyperopt_mock)
    patch_exchange(mocker)

    args = [
        'hyperopt',
        '--config', 'config.json',
        '--hyperopt', 'DefaultHyperOpt',
        '--hyperopt-loss', 'SharpeHyperOptLossDaily',
        '--epochs', '5'
    ]
    pargs = get_args(args)
    start_hyperopt(pargs)
    assert log_has("Another running instance of freqtrade Hyperopt detected.", caplog)


def test_log_results_if_loss_improves(hyperopt, capsys) -> None:
    hyperopt.current_best_loss = 2
    hyperopt.total_epochs = 2

    hyperopt.print_results(
        {
            'loss': 1,
            'results_metrics':
                {
                    'trade_count': 1,
                    'avg_profit': 0.1,
                    'total_profit': 0.001,
                    'profit': 1.0,
                    'duration': 20.0
                },
            'total_profit': 0,
            'current_epoch': 2,  # This starts from 1 (in a human-friendly manner)
            'is_initial_point': False,
            'is_best': True
        }
    )
    out, err = capsys.readouterr()
    assert all(x in out
               for x in ["Best", "2/2", " 1", "0.10%", "0.00100000 BTC    (1.00%)", "20.0 m"])


def test_no_log_if_loss_does_not_improve(hyperopt, caplog) -> None:
    hyperopt.current_best_loss = 2
    hyperopt.print_results(
        {
            'is_best': False,
            'loss': 3,
            'current_epoch': 1,
        }
    )
    assert caplog.record_tuples == []


def test_save_results_saves_epochs(mocker, hyperopt, testdatadir, caplog) -> None:
    epochs = create_results(mocker, hyperopt, testdatadir)
    mock_dump = mocker.patch('freqtrade.optimize.hyperopt.dump', return_value=None)
    mock_dump_json = mocker.patch('freqtrade.optimize.hyperopt.file_dump_json', return_value=None)
    results_file = testdatadir / 'optimize' / 'ut_results.pickle'

    caplog.set_level(logging.DEBUG)

    hyperopt.epochs = epochs
    hyperopt._save_results()
    assert log_has(f"1 epoch saved to '{results_file}'.", caplog)
    mock_dump.assert_called_once()
    mock_dump_json.assert_called_once()

    hyperopt.epochs = epochs + epochs
    hyperopt._save_results()
    assert log_has(f"2 epochs saved to '{results_file}'.", caplog)


def test_read_results_returns_epochs(mocker, hyperopt, testdatadir, caplog) -> None:
    epochs = create_results(mocker, hyperopt, testdatadir)
    mock_load = mocker.patch('freqtrade.optimize.hyperopt_tools.load', return_value=epochs)
    results_file = testdatadir / 'optimize' / 'ut_results.pickle'
    hyperopt_epochs = HyperoptTools._read_results(results_file)
    assert log_has(f"Reading epochs from '{results_file}'", caplog)
    assert hyperopt_epochs == epochs
    mock_load.assert_called_once()


def test_load_previous_results(mocker, hyperopt, testdatadir, caplog) -> None:
    epochs = create_results(mocker, hyperopt, testdatadir)
    mock_load = mocker.patch('freqtrade.optimize.hyperopt_tools.load', return_value=epochs)
    mocker.patch.object(Path, 'is_file', MagicMock(return_value=True))
    statmock = MagicMock()
    statmock.st_size = 5
    # mocker.patch.object(Path, 'stat', MagicMock(return_value=statmock))

    results_file = testdatadir / 'optimize' / 'ut_results.pickle'

    hyperopt_epochs = HyperoptTools.load_previous_results(results_file)

    assert hyperopt_epochs == epochs
    mock_load.assert_called_once()

    del epochs[0]['is_best']
    mock_load = mocker.patch('freqtrade.optimize.hyperopt_tools.load', return_value=epochs)

    with pytest.raises(OperationalException):
        HyperoptTools.load_previous_results(results_file)


def test_roi_table_generation(hyperopt) -> None:
    params = {
        'roi_t1': 5,
        'roi_t2': 10,
        'roi_t3': 15,
        'roi_p1': 1,
        'roi_p2': 2,
        'roi_p3': 3,
    }

    assert hyperopt.custom_hyperopt.generate_roi_table(params) == {0: 6, 15: 3, 25: 1, 30: 0}


def test_start_calls_optimizer(mocker, hyperopt_conf, capsys) -> None:
    dumper = mocker.patch('freqtrade.optimize.hyperopt.dump', MagicMock())
    mocker.patch('freqtrade.optimize.hyperopt.file_dump_json')

    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data',
                 MagicMock(return_value=(MagicMock(), None)))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timerange',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    parallel = mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel',
        MagicMock(return_value=[{
            'loss': 1, 'results_explanation': 'foo result',
            'params': {'buy': {}, 'sell': {}, 'roi': {}, 'stoploss': 0.0},
            'results_metrics':
            {
                'trade_count': 1,
                'avg_profit': 0.1,
                'total_profit': 0.001,
                'profit': 1.0,
                'duration': 20.0
            },
        }])
    )
    patch_exchange(mocker)
    # Co-test loading timeframe from strategy
    del hyperopt_conf['timeframe']

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.backtesting.strategy.ohlcvdata_to_dataframe = MagicMock()
    hyperopt.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    hyperopt.start()

    parallel.assert_called_once()

    out, err = capsys.readouterr()
    assert 'Best result:\n\n*    1/1: foo result Objective: 1.00000\n' in out
    assert dumper.called
    # Should be called twice, once for historical candle data, once to save evaluations
    assert dumper.call_count == 2
    assert hasattr(hyperopt.backtesting.strategy, "advise_sell")
    assert hasattr(hyperopt.backtesting.strategy, "advise_buy")
    assert hasattr(hyperopt, "max_open_trades")
    assert hyperopt.max_open_trades == hyperopt_conf['max_open_trades']
    assert hasattr(hyperopt, "position_stacking")


def test_format_results(hyperopt):
    # Test with BTC as stake_currency
    trades = [
        ('ETH/BTC', 2, 2, 123),
        ('LTC/BTC', 1, 1, 123),
        ('XPR/BTC', -1, -2, -246)
    ]
    labels = ['currency', 'profit_ratio', 'profit_abs', 'trade_duration']
    df = pd.DataFrame.from_records(trades, columns=labels)
    results_metrics = hyperopt._calculate_results_metrics(df)
    results_explanation = hyperopt._format_results_explanation_string(results_metrics)
    total_profit = results_metrics['total_profit']

    results = {
        'loss': 0.0,
        'params_dict': None,
        'params_details': None,
        'results_metrics': results_metrics,
        'results_explanation': results_explanation,
        'total_profit': total_profit,
        'current_epoch': 1,
        'is_initial_point': True,
    }

    result = HyperoptTools._format_explanation_string(results, 1)
    assert result.find(' 66.67%')
    assert result.find('Total profit 1.00000000 BTC')
    assert result.find('2.0000Σ %')

    # Test with EUR as stake_currency
    trades = [
        ('ETH/EUR', 2, 2, 123),
        ('LTC/EUR', 1, 1, 123),
        ('XPR/EUR', -1, -2, -246)
    ]
    df = pd.DataFrame.from_records(trades, columns=labels)
    results_metrics = hyperopt._calculate_results_metrics(df)
    results['total_profit'] = results_metrics['total_profit']
    result = HyperoptTools._format_explanation_string(results, 1)
    assert result.find('Total profit 1.00000000 EUR')


@pytest.mark.parametrize("spaces, expected_results", [
    (['buy'],
     {'buy': True, 'sell': False, 'roi': False, 'stoploss': False, 'trailing': False}),
    (['sell'],
     {'buy': False, 'sell': True, 'roi': False, 'stoploss': False, 'trailing': False}),
    (['roi'],
     {'buy': False, 'sell': False, 'roi': True, 'stoploss': False, 'trailing': False}),
    (['stoploss'],
     {'buy': False, 'sell': False, 'roi': False, 'stoploss': True, 'trailing': False}),
    (['trailing'],
     {'buy': False, 'sell': False, 'roi': False, 'stoploss': False, 'trailing': True}),
    (['buy', 'sell', 'roi', 'stoploss'],
     {'buy': True, 'sell': True, 'roi': True, 'stoploss': True, 'trailing': False}),
    (['buy', 'sell', 'roi', 'stoploss', 'trailing'],
     {'buy': True, 'sell': True, 'roi': True, 'stoploss': True, 'trailing': True}),
    (['buy', 'roi'],
     {'buy': True, 'sell': False, 'roi': True, 'stoploss': False, 'trailing': False}),
    (['all'],
     {'buy': True, 'sell': True, 'roi': True, 'stoploss': True, 'trailing': True}),
    (['default'],
     {'buy': True, 'sell': True, 'roi': True, 'stoploss': True, 'trailing': False}),
    (['default', 'trailing'],
     {'buy': True, 'sell': True, 'roi': True, 'stoploss': True, 'trailing': True}),
    (['all', 'buy'],
     {'buy': True, 'sell': True, 'roi': True, 'stoploss': True, 'trailing': True}),
    (['default', 'buy'],
     {'buy': True, 'sell': True, 'roi': True, 'stoploss': True, 'trailing': False}),
])
def test_has_space(hyperopt, spaces, expected_results):
    for s in ['buy', 'sell', 'roi', 'stoploss', 'trailing']:
        hyperopt.config.update({'spaces': spaces})
        assert hyperopt.has_space(s) == expected_results[s]


def test_populate_indicators(hyperopt, testdatadir) -> None:
    data = load_data(testdatadir, '1m', ['UNITTEST/BTC'], fill_up_missing=True)
    dataframes = hyperopt.backtesting.strategy.ohlcvdata_to_dataframe(data)
    dataframe = hyperopt.custom_hyperopt.populate_indicators(dataframes['UNITTEST/BTC'],
                                                             {'pair': 'UNITTEST/BTC'})

    # Check if some indicators are generated. We will not test all of them
    assert 'adx' in dataframe
    assert 'mfi' in dataframe
    assert 'rsi' in dataframe


def test_buy_strategy_generator(hyperopt, testdatadir) -> None:
    data = load_data(testdatadir, '1m', ['UNITTEST/BTC'], fill_up_missing=True)
    dataframes = hyperopt.backtesting.strategy.ohlcvdata_to_dataframe(data)
    dataframe = hyperopt.custom_hyperopt.populate_indicators(dataframes['UNITTEST/BTC'],
                                                             {'pair': 'UNITTEST/BTC'})

    populate_buy_trend = hyperopt.custom_hyperopt.buy_strategy_generator(
        {
            'adx-value': 20,
            'fastd-value': 20,
            'mfi-value': 20,
            'rsi-value': 20,
            'adx-enabled': True,
            'fastd-enabled': True,
            'mfi-enabled': True,
            'rsi-enabled': True,
            'trigger': 'bb_lower'
        }
    )
    result = populate_buy_trend(dataframe, {'pair': 'UNITTEST/BTC'})
    # Check if some indicators are generated. We will not test all of them
    assert 'buy' in result
    assert 1 in result['buy']


def test_sell_strategy_generator(hyperopt, testdatadir) -> None:
    data = load_data(testdatadir, '1m', ['UNITTEST/BTC'], fill_up_missing=True)
    dataframes = hyperopt.backtesting.strategy.ohlcvdata_to_dataframe(data)
    dataframe = hyperopt.custom_hyperopt.populate_indicators(dataframes['UNITTEST/BTC'],
                                                             {'pair': 'UNITTEST/BTC'})

    populate_sell_trend = hyperopt.custom_hyperopt.sell_strategy_generator(
        {
            'sell-adx-value': 20,
            'sell-fastd-value': 75,
            'sell-mfi-value': 80,
            'sell-rsi-value': 20,
            'sell-adx-enabled': True,
            'sell-fastd-enabled': True,
            'sell-mfi-enabled': True,
            'sell-rsi-enabled': True,
            'sell-trigger': 'sell-bb_upper'
        }
    )
    result = populate_sell_trend(dataframe, {'pair': 'UNITTEST/BTC'})
    # Check if some indicators are generated. We will not test all of them
    print(result)
    assert 'sell' in result
    assert 1 in result['sell']


def test_generate_optimizer(mocker, hyperopt_conf) -> None:
    hyperopt_conf.update({'spaces': 'all',
                          'hyperopt_min_trades': 1,
                          })

    trades = [
        ('TRX/BTC', 0.023117, 0.000233, 100)
    ]
    labels = ['currency', 'profit_ratio', 'profit_abs', 'trade_duration']
    backtest_result = pd.DataFrame.from_records(trades, columns=labels)

    mocker.patch(
        'freqtrade.optimize.hyperopt.Backtesting.backtest',
        MagicMock(return_value=backtest_result)
    )
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timerange',
        MagicMock(return_value=(Arrow(2017, 12, 10), Arrow(2017, 12, 13)))
    )
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.hyperopt.load', MagicMock())

    optimizer_param = {
        'adx-value': 0,
        'fastd-value': 35,
        'mfi-value': 0,
        'rsi-value': 0,
        'adx-enabled': False,
        'fastd-enabled': True,
        'mfi-enabled': False,
        'rsi-enabled': False,
        'trigger': 'macd_cross_signal',
        'sell-adx-value': 0,
        'sell-fastd-value': 75,
        'sell-mfi-value': 0,
        'sell-rsi-value': 0,
        'sell-adx-enabled': False,
        'sell-fastd-enabled': True,
        'sell-mfi-enabled': False,
        'sell-rsi-enabled': False,
        'sell-trigger': 'macd_cross_signal',
        'roi_t1': 60.0,
        'roi_t2': 30.0,
        'roi_t3': 20.0,
        'roi_p1': 0.01,
        'roi_p2': 0.01,
        'roi_p3': 0.1,
        'stoploss': -0.4,
        'trailing_stop': True,
        'trailing_stop_positive': 0.02,
        'trailing_stop_positive_offset_p1': 0.05,
        'trailing_only_offset_is_reached': False,
    }
    response_expected = {
        'loss': 1.9840569076926293,
        'results_explanation': ('     1 trades. 1/0/0 Wins/Draws/Losses. '
                                'Avg profit   2.31%. Median profit   2.31%. Total profit  '
                                '0.00023300 BTC (   2.31\N{GREEK CAPITAL LETTER SIGMA}%). '
                                'Avg duration 100.0 min.'
                                ).encode(locale.getpreferredencoding(), 'replace').decode('utf-8'),
        'params_details': {'buy': {'adx-enabled': False,
                                   'adx-value': 0,
                                   'fastd-enabled': True,
                                   'fastd-value': 35,
                                   'mfi-enabled': False,
                                   'mfi-value': 0,
                                   'rsi-enabled': False,
                                   'rsi-value': 0,
                                   'trigger': 'macd_cross_signal'},
                           'roi': {0: 0.12000000000000001,
                                   20.0: 0.02,
                                   50.0: 0.01,
                                   110.0: 0},
                           'sell': {'sell-adx-enabled': False,
                                    'sell-adx-value': 0,
                                    'sell-fastd-enabled': True,
                                    'sell-fastd-value': 75,
                                    'sell-mfi-enabled': False,
                                    'sell-mfi-value': 0,
                                    'sell-rsi-enabled': False,
                                    'sell-rsi-value': 0,
                                    'sell-trigger': 'macd_cross_signal'},
                           'stoploss': {'stoploss': -0.4},
                           'trailing': {'trailing_only_offset_is_reached': False,
                                        'trailing_stop': True,
                                        'trailing_stop_positive': 0.02,
                                        'trailing_stop_positive_offset': 0.07}},
        'params_dict': optimizer_param,
        'results_metrics': {'avg_profit': 2.3117,
                            'draws': 0,
                            'duration': 100.0,
                            'losses': 0,
                            'winsdrawslosses': '   1    0    0',
                            'median_profit': 2.3117,
                            'profit': 2.3117,
                            'total_profit': 0.000233,
                            'trade_count': 1,
                            'wins': 1},
        'total_profit': 0.00023300
    }

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.dimensions = hyperopt.hyperopt_space()
    generate_optimizer_value = hyperopt.generate_optimizer(list(optimizer_param.values()))
    assert generate_optimizer_value == response_expected


def test_clean_hyperopt(mocker, hyperopt_conf, caplog):
    patch_exchange(mocker)

    mocker.patch("freqtrade.optimize.hyperopt.Path.is_file", MagicMock(return_value=True))
    unlinkmock = mocker.patch("freqtrade.optimize.hyperopt.Path.unlink", MagicMock())
    h = Hyperopt(hyperopt_conf)

    assert unlinkmock.call_count == 2
    assert log_has(f"Removing `{h.data_pickle_file}`.", caplog)


def test_print_json_spaces_all(mocker, hyperopt_conf, capsys) -> None:
    dumper = mocker.patch('freqtrade.optimize.hyperopt.dump', MagicMock())
    mocker.patch('freqtrade.optimize.hyperopt.file_dump_json')

    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data',
                 MagicMock(return_value=(MagicMock(), None)))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timerange',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    parallel = mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel',
        MagicMock(return_value=[{
            'loss': 1, 'results_explanation': 'foo result', 'params': {},
            'params_details': {
                'buy': {'mfi-value': None},
                'sell': {'sell-mfi-value': None},
                'roi': {}, 'stoploss': {'stoploss': None},
                'trailing': {'trailing_stop': None}
            },
            'results_metrics':
            {
                'trade_count': 1,
                'avg_profit': 0.1,
                'total_profit': 0.001,
                'profit': 1.0,
                'duration': 20.0
            }
        }])
    )
    patch_exchange(mocker)

    hyperopt_conf.update({'spaces': 'all',
                          'hyperopt_jobs': 1,
                          'print_json': True,
                          })

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.backtesting.strategy.ohlcvdata_to_dataframe = MagicMock()
    hyperopt.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    hyperopt.start()

    parallel.assert_called_once()

    out, err = capsys.readouterr()
    result_str = (
        '{"params":{"mfi-value":null,"sell-mfi-value":null},"minimal_roi"'
        ':{},"stoploss":null,"trailing_stop":null}'
    )
    assert result_str in out  # noqa: E501
    assert dumper.called
    # Should be called twice, once for historical candle data, once to save evaluations
    assert dumper.call_count == 2


def test_print_json_spaces_default(mocker, hyperopt_conf, capsys) -> None:
    dumper = mocker.patch('freqtrade.optimize.hyperopt.dump', MagicMock())
    mocker.patch('freqtrade.optimize.hyperopt.file_dump_json')
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data',
                 MagicMock(return_value=(MagicMock(), None)))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timerange',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    parallel = mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel',
        MagicMock(return_value=[{
            'loss': 1, 'results_explanation': 'foo result', 'params': {},
            'params_details': {
                'buy': {'mfi-value': None},
                'sell': {'sell-mfi-value': None},
                'roi': {}, 'stoploss': {'stoploss': None}
            },
            'results_metrics':
            {
                'trade_count': 1,
                'avg_profit': 0.1,
                'total_profit': 0.001,
                'profit': 1.0,
                'duration': 20.0
            }
        }])
    )
    patch_exchange(mocker)

    hyperopt_conf.update({'print_json': True})

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.backtesting.strategy.ohlcvdata_to_dataframe = MagicMock()
    hyperopt.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    hyperopt.start()

    parallel.assert_called_once()

    out, err = capsys.readouterr()
    assert '{"params":{"mfi-value":null,"sell-mfi-value":null},"minimal_roi":{},"stoploss":null}' in out  # noqa: E501
    assert dumper.called
    # Should be called twice, once for historical candle data, once to save evaluations
    assert dumper.call_count == 2


def test_print_json_spaces_roi_stoploss(mocker, hyperopt_conf, capsys) -> None:
    dumper = mocker.patch('freqtrade.optimize.hyperopt.dump', MagicMock())
    mocker.patch('freqtrade.optimize.hyperopt.file_dump_json')
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data',
                 MagicMock(return_value=(MagicMock(), None)))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timerange',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    parallel = mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel',
        MagicMock(return_value=[{
            'loss': 1, 'results_explanation': 'foo result', 'params': {},
            'params_details': {'roi': {}, 'stoploss': {'stoploss': None}},
            'results_metrics':
            {
                'trade_count': 1,
                'avg_profit': 0.1,
                'total_profit': 0.001,
                'profit': 1.0,
                'duration': 20.0
            }
        }])
    )
    patch_exchange(mocker)

    hyperopt_conf.update({'spaces': 'roi stoploss',
                          'hyperopt_jobs': 1,
                          'print_json': True,
                          })

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.backtesting.strategy.ohlcvdata_to_dataframe = MagicMock()
    hyperopt.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    hyperopt.start()

    parallel.assert_called_once()

    out, err = capsys.readouterr()
    assert '{"minimal_roi":{},"stoploss":null}' in out
    assert dumper.called
    # Should be called twice, once for historical candle data, once to save evaluations
    assert dumper.call_count == 2


def test_simplified_interface_roi_stoploss(mocker, hyperopt_conf, capsys) -> None:
    dumper = mocker.patch('freqtrade.optimize.hyperopt.dump', MagicMock())
    mocker.patch('freqtrade.optimize.hyperopt.file_dump_json')
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data',
                 MagicMock(return_value=(MagicMock(), None)))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timerange',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    parallel = mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel',
        MagicMock(return_value=[{
            'loss': 1, 'results_explanation': 'foo result', 'params': {'stoploss': 0.0},
            'results_metrics':
            {
                'trade_count': 1,
                'avg_profit': 0.1,
                'total_profit': 0.001,
                'profit': 1.0,
                'duration': 20.0
            }
        }])
    )
    patch_exchange(mocker)

    hyperopt_conf.update({'spaces': 'roi stoploss'})

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.backtesting.strategy.ohlcvdata_to_dataframe = MagicMock()
    hyperopt.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    del hyperopt.custom_hyperopt.__class__.buy_strategy_generator
    del hyperopt.custom_hyperopt.__class__.sell_strategy_generator
    del hyperopt.custom_hyperopt.__class__.indicator_space
    del hyperopt.custom_hyperopt.__class__.sell_indicator_space

    hyperopt.start()

    parallel.assert_called_once()

    out, err = capsys.readouterr()
    assert 'Best result:\n\n*    1/1: foo result Objective: 1.00000\n' in out
    assert dumper.called
    # Should be called twice, once for historical candle data, once to save evaluations
    assert dumper.call_count == 2
    assert hasattr(hyperopt.backtesting.strategy, "advise_sell")
    assert hasattr(hyperopt.backtesting.strategy, "advise_buy")
    assert hasattr(hyperopt, "max_open_trades")
    assert hyperopt.max_open_trades == hyperopt_conf['max_open_trades']
    assert hasattr(hyperopt, "position_stacking")


def test_simplified_interface_all_failed(mocker, hyperopt_conf) -> None:
    mocker.patch('freqtrade.optimize.hyperopt.dump', MagicMock())
    mocker.patch('freqtrade.optimize.hyperopt.file_dump_json')
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data',
                 MagicMock(return_value=(MagicMock(), None)))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timerange',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    patch_exchange(mocker)

    hyperopt_conf.update({'spaces': 'all', })

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.backtesting.strategy.ohlcvdata_to_dataframe = MagicMock()
    hyperopt.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    del hyperopt.custom_hyperopt.__class__.buy_strategy_generator
    del hyperopt.custom_hyperopt.__class__.sell_strategy_generator
    del hyperopt.custom_hyperopt.__class__.indicator_space
    del hyperopt.custom_hyperopt.__class__.sell_indicator_space

    with pytest.raises(OperationalException, match=r"The 'buy' space is included into *"):
        hyperopt.start()


def test_simplified_interface_buy(mocker, hyperopt_conf, capsys) -> None:
    dumper = mocker.patch('freqtrade.optimize.hyperopt.dump', MagicMock())
    mocker.patch('freqtrade.optimize.hyperopt.file_dump_json')
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data',
                 MagicMock(return_value=(MagicMock(), None)))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timerange',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    parallel = mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel',
        MagicMock(return_value=[{
            'loss': 1, 'results_explanation': 'foo result', 'params': {},
            'results_metrics':
            {
                'trade_count': 1,
                'avg_profit': 0.1,
                'total_profit': 0.001,
                'profit': 1.0,
                'duration': 20.0
            }
        }])
    )
    patch_exchange(mocker)

    hyperopt_conf.update({'spaces': 'buy'})

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.backtesting.strategy.ohlcvdata_to_dataframe = MagicMock()
    hyperopt.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    # TODO: sell_strategy_generator() is actually not called because
    # run_optimizer_parallel() is mocked
    del hyperopt.custom_hyperopt.__class__.sell_strategy_generator
    del hyperopt.custom_hyperopt.__class__.sell_indicator_space

    hyperopt.start()

    parallel.assert_called_once()

    out, err = capsys.readouterr()
    assert 'Best result:\n\n*    1/1: foo result Objective: 1.00000\n' in out
    assert dumper.called
    # Should be called twice, once for historical candle data, once to save evaluations
    assert dumper.call_count == 2
    assert hasattr(hyperopt.backtesting.strategy, "advise_sell")
    assert hasattr(hyperopt.backtesting.strategy, "advise_buy")
    assert hasattr(hyperopt, "max_open_trades")
    assert hyperopt.max_open_trades == hyperopt_conf['max_open_trades']
    assert hasattr(hyperopt, "position_stacking")


def test_simplified_interface_sell(mocker, hyperopt_conf, capsys) -> None:
    dumper = mocker.patch('freqtrade.optimize.hyperopt.dump', MagicMock())
    mocker.patch('freqtrade.optimize.hyperopt.file_dump_json')
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data',
                 MagicMock(return_value=(MagicMock(), None)))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timerange',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    parallel = mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel',
        MagicMock(return_value=[{
            'loss': 1, 'results_explanation': 'foo result', 'params': {},
            'results_metrics':
            {
                'trade_count': 1,
                'avg_profit': 0.1,
                'total_profit': 0.001,
                'profit': 1.0,
                'duration': 20.0
            }
        }])
    )
    patch_exchange(mocker)

    hyperopt_conf.update({'spaces': 'sell', })

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.backtesting.strategy.ohlcvdata_to_dataframe = MagicMock()
    hyperopt.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    # TODO: buy_strategy_generator() is actually not called because
    # run_optimizer_parallel() is mocked
    del hyperopt.custom_hyperopt.__class__.buy_strategy_generator
    del hyperopt.custom_hyperopt.__class__.indicator_space

    hyperopt.start()

    parallel.assert_called_once()

    out, err = capsys.readouterr()
    assert 'Best result:\n\n*    1/1: foo result Objective: 1.00000\n' in out
    assert dumper.called
    # Should be called twice, once for historical candle data, once to save evaluations
    assert dumper.call_count == 2
    assert hasattr(hyperopt.backtesting.strategy, "advise_sell")
    assert hasattr(hyperopt.backtesting.strategy, "advise_buy")
    assert hasattr(hyperopt, "max_open_trades")
    assert hyperopt.max_open_trades == hyperopt_conf['max_open_trades']
    assert hasattr(hyperopt, "position_stacking")


@pytest.mark.parametrize("method,space", [
    ('buy_strategy_generator', 'buy'),
    ('indicator_space', 'buy'),
    ('sell_strategy_generator', 'sell'),
    ('sell_indicator_space', 'sell'),
])
def test_simplified_interface_failed(mocker, hyperopt_conf, method, space) -> None:
    mocker.patch('freqtrade.optimize.hyperopt.dump', MagicMock())
    mocker.patch('freqtrade.optimize.hyperopt.file_dump_json')
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data',
                 MagicMock(return_value=(MagicMock(), None)))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timerange',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    patch_exchange(mocker)

    hyperopt_conf.update({'spaces': space})

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.backtesting.strategy.ohlcvdata_to_dataframe = MagicMock()
    hyperopt.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    delattr(hyperopt.custom_hyperopt.__class__, method)

    with pytest.raises(OperationalException, match=f"The '{space}' space is included into *"):
        hyperopt.start()


def test_print_epoch_details(capsys):
    test_result = {
        'params_details': {
            'trailing': {
                'trailing_stop': True,
                'trailing_stop_positive': 0.02,
                'trailing_stop_positive_offset': 0.04,
                'trailing_only_offset_is_reached': True
            },
            'roi': {
                0: 0.18,
                90: 0.14,
                225: 0.05,
                430: 0},
        },
        'results_explanation': 'foo result',
        'is_initial_point': False,
        'total_profit': 0,
        'current_epoch': 2,  # This starts from 1 (in a human-friendly manner)
        'is_best': True
    }

    HyperoptTools.print_epoch_details(test_result, 5, False, no_header=True)
    captured = capsys.readouterr()
    assert '# Trailing stop:' in captured.out
    # re.match(r"Pairs for .*", captured.out)
    assert re.search(r'^\s+trailing_stop = True$', captured.out, re.MULTILINE)
    assert re.search(r'^\s+trailing_stop_positive = 0.02$', captured.out, re.MULTILINE)
    assert re.search(r'^\s+trailing_stop_positive_offset = 0.04$', captured.out, re.MULTILINE)
    assert re.search(r'^\s+trailing_only_offset_is_reached = True$', captured.out, re.MULTILINE)

    assert '# ROI table:' in captured.out
    assert re.search(r'^\s+minimal_roi = \{$', captured.out, re.MULTILINE)
    assert re.search(r'^\s+\"90\"\:\s0.14,\s*$', captured.out, re.MULTILINE)


def test_in_strategy_auto_hyperopt(mocker, hyperopt_conf, tmpdir, fee) -> None:
    patch_exchange(mocker)
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    (Path(tmpdir) / 'hyperopt_results').mkdir(parents=True)
    # No hyperopt needed
    del hyperopt_conf['hyperopt']
    hyperopt_conf.update({
        'strategy': 'HyperoptableStrategy',
        'user_data_dir': Path(tmpdir),
    })
    hyperopt = Hyperopt(hyperopt_conf)
    assert isinstance(hyperopt.custom_hyperopt, HyperOptAuto)
    assert isinstance(hyperopt.backtesting.strategy.buy_rsi, IntParameter)

    assert hyperopt.backtesting.strategy.buy_rsi.hyperopt is True
    assert hyperopt.backtesting.strategy.buy_rsi.value == 35
    buy_rsi_range = hyperopt.backtesting.strategy.buy_rsi.range
    assert isinstance(buy_rsi_range, range)
    # Range from 0 - 50 (inclusive)
    assert len(list(buy_rsi_range)) == 51

    hyperopt.start()


def test_SKDecimal():
    space = SKDecimal(1, 2, decimals=2)
    assert 1.5 in space
    assert 2.5 not in space
    assert space.low == 100
    assert space.high == 200

    assert space.inverse_transform([200]) == [2.0]
    assert space.inverse_transform([100]) == [1.0]
    assert space.inverse_transform([150, 160]) == [1.5, 1.6]

    assert space.transform([1.5]) == [150]
    assert space.transform([2.0]) == [200]
    assert space.transform([1.0]) == [100]
    assert space.transform([1.5, 1.6]) == [150, 160]
