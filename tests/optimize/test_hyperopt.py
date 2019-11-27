# pragma pylint: disable=missing-docstring,W0212,C0103
import locale
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pandas as pd
import pytest
from arrow import Arrow
from filelock import Timeout

from freqtrade import OperationalException
from freqtrade.data.converter import parse_ticker_dataframe
from freqtrade.data.history import load_tickerdata_file
from freqtrade.optimize import setup_configuration, start_hyperopt
from freqtrade.optimize.default_hyperopt import DefaultHyperOpt
from freqtrade.optimize.default_hyperopt_loss import DefaultHyperOptLoss
from freqtrade.optimize.hyperopt import Hyperopt
from freqtrade.resolvers.hyperopt_resolver import (HyperOptLossResolver,
                                                   HyperOptResolver)
from freqtrade.state import RunMode
from freqtrade.strategy.interface import SellType
from tests.conftest import (get_args, log_has, log_has_re, patch_exchange,
                            patched_configuration_load_config_file)


@pytest.fixture(scope='function')
def hyperopt(default_conf, mocker):
    default_conf.update({
        'spaces': ['all'],
        'hyperopt': 'DefaultHyperOpt',
    })
    patch_exchange(mocker)
    return Hyperopt(default_conf)


@pytest.fixture(scope='function')
def hyperopt_results():
    return pd.DataFrame(
        {
            'pair': ['ETH/BTC', 'ETH/BTC', 'ETH/BTC'],
            'profit_percent': [-0.1, 0.2, 0.3],
            'profit_abs': [-0.2, 0.4, 0.6],
            'trade_duration': [10, 30, 10],
            'sell_reason': [SellType.STOP_LOSS, SellType.ROI, SellType.ROI]
        }
    )


# Functions for recurrent object patching
def create_trials(mocker, hyperopt, testdatadir) -> None:
    """
    When creating trials, mock the hyperopt Trials so that *by default*
      - we don't create any pickle'd files in the filesystem
      - we might have a pickle'd file so make sure that we return
        false when looking for it
    """
    hyperopt.trials_file = testdatadir / 'optimize/ut_trials.pickle'

    mocker.patch.object(Path, "is_file", MagicMock(return_value=False))
    stat_mock = MagicMock()
    stat_mock.st_size = PropertyMock(return_value=1)
    mocker.patch.object(Path, "stat", MagicMock(return_value=False))

    mocker.patch.object(Path, "unlink", MagicMock(return_value=True))
    mocker.patch('freqtrade.optimize.hyperopt.dump', return_value=None)

    return [{'loss': 1, 'result': 'foo', 'params': {}}]


def test_setup_hyperopt_configuration_without_arguments(mocker, default_conf, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        'hyperopt',
        '--config', 'config.json',
        '--hyperopt', 'DefaultHyperOpt',
    ]

    config = setup_configuration(get_args(args), RunMode.HYPEROPT)
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert log_has('Using data directory: {} ...'.format(config['datadir']), caplog)
    assert 'ticker_interval' in config
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
        '--ticker-interval', '1m',
        '--timerange', ':100',
        '--enable-position-stacking',
        '--disable-max-market-positions',
        '--epochs', '1000',
        '--spaces', 'all',
        '--print-all'
    ]

    config = setup_configuration(get_args(args), RunMode.HYPEROPT)
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert config['runmode'] == RunMode.HYPEROPT

    assert log_has('Using data directory: {} ...'.format(config['datadir']), caplog)
    assert 'ticker_interval' in config
    assert log_has('Parameter -i/--ticker-interval detected ... Using ticker_interval: 1m ...',
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


def test_hyperoptresolver(mocker, default_conf, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)

    hyperopt = DefaultHyperOpt
    delattr(hyperopt, 'populate_indicators')
    delattr(hyperopt, 'populate_buy_trend')
    delattr(hyperopt, 'populate_sell_trend')
    mocker.patch(
        'freqtrade.resolvers.hyperopt_resolver.HyperOptResolver._load_hyperopt',
        MagicMock(return_value=hyperopt(default_conf))
    )
    default_conf.update({'hyperopt': 'DefaultHyperOpt'})
    x = HyperOptResolver(default_conf).hyperopt
    assert not hasattr(x, 'populate_indicators')
    assert not hasattr(x, 'populate_buy_trend')
    assert not hasattr(x, 'populate_sell_trend')
    assert log_has("Hyperopt class does not provide populate_indicators() method. "
                   "Using populate_indicators from the strategy.", caplog)
    assert log_has("Hyperopt class does not provide populate_sell_trend() method. "
                   "Using populate_sell_trend from the strategy.", caplog)
    assert log_has("Hyperopt class does not provide populate_buy_trend() method. "
                   "Using populate_buy_trend from the strategy.", caplog)
    assert hasattr(x, "ticker_interval")


def test_hyperoptresolver_wrongname(mocker, default_conf, caplog) -> None:
    default_conf.update({'hyperopt': "NonExistingHyperoptClass"})

    with pytest.raises(OperationalException, match=r'Impossible to load Hyperopt.*'):
        HyperOptResolver(default_conf).hyperopt


def test_hyperoptresolver_noname(default_conf):
    default_conf['hyperopt'] = ''
    with pytest.raises(OperationalException,
                       match="No Hyperopt set. Please use `--hyperopt` to specify "
                             "the Hyperopt class to use."):
        HyperOptResolver(default_conf)


def test_hyperoptlossresolver(mocker, default_conf, caplog) -> None:

    hl = DefaultHyperOptLoss
    mocker.patch(
        'freqtrade.resolvers.hyperopt_resolver.HyperOptLossResolver._load_hyperoptloss',
        MagicMock(return_value=hl)
    )
    x = HyperOptLossResolver(default_conf).hyperoptloss
    assert hasattr(x, "hyperopt_loss_function")


def test_hyperoptlossresolver_wrongname(mocker, default_conf, caplog) -> None:
    default_conf.update({'hyperopt_loss': "NonExistingLossClass"})

    with pytest.raises(OperationalException, match=r'Impossible to load HyperoptLoss.*'):
        HyperOptLossResolver(default_conf).hyperopt


def test_start_not_installed(mocker, default_conf, caplog, import_fails) -> None:
    start_mock = MagicMock()
    patched_configuration_load_config_file(mocker, default_conf)

    mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.start', start_mock)
    patch_exchange(mocker)

    args = [
        'hyperopt',
        '--config', 'config.json',
        '--hyperopt', 'DefaultHyperOpt',
        '--epochs', '5'
    ]
    args = get_args(args)

    with pytest.raises(OperationalException, match=r"Please ensure that the hyperopt dependencies"):
        start_hyperopt(args)


def test_start(mocker, default_conf, caplog) -> None:
    start_mock = MagicMock()
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.start', start_mock)
    patch_exchange(mocker)

    args = [
        'hyperopt',
        '--config', 'config.json',
        '--hyperopt', 'DefaultHyperOpt',
        '--epochs', '5'
    ]
    args = get_args(args)
    start_hyperopt(args)

    assert log_has('Starting freqtrade in Hyperopt mode', caplog)
    assert start_mock.call_count == 1


def test_start_no_data(mocker, default_conf, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('freqtrade.data.history.load_pair_history', MagicMock(return_value=None))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timeframe',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    patch_exchange(mocker)

    args = [
        'hyperopt',
        '--config', 'config.json',
        '--hyperopt', 'DefaultHyperOpt',
        '--epochs', '5'
    ]
    args = get_args(args)
    with pytest.raises(OperationalException, match='No data found. Terminating.'):
        start_hyperopt(args)


def test_start_filelock(mocker, default_conf, caplog) -> None:
    start_mock = MagicMock(side_effect=Timeout(Hyperopt.get_lock_filename(default_conf)))
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.start', start_mock)
    patch_exchange(mocker)

    args = [
        'hyperopt',
        '--config', 'config.json',
        '--hyperopt', 'DefaultHyperOpt',
        '--epochs', '5'
    ]
    args = get_args(args)
    start_hyperopt(args)
    assert log_has("Another running instance of freqtrade Hyperopt detected.", caplog)


def test_loss_calculation_prefer_correct_trade_count(default_conf, hyperopt_results) -> None:
    hl = HyperOptLossResolver(default_conf).hyperoptloss
    correct = hl.hyperopt_loss_function(hyperopt_results, 600)
    over = hl.hyperopt_loss_function(hyperopt_results, 600 + 100)
    under = hl.hyperopt_loss_function(hyperopt_results, 600 - 100)
    assert over > correct
    assert under > correct


def test_loss_calculation_prefer_shorter_trades(default_conf, hyperopt_results) -> None:
    resultsb = hyperopt_results.copy()
    resultsb.loc[1, 'trade_duration'] = 20

    hl = HyperOptLossResolver(default_conf).hyperoptloss
    longer = hl.hyperopt_loss_function(hyperopt_results, 100)
    shorter = hl.hyperopt_loss_function(resultsb, 100)
    assert shorter < longer


def test_loss_calculation_has_limited_profit(default_conf, hyperopt_results) -> None:
    results_over = hyperopt_results.copy()
    results_over['profit_percent'] = hyperopt_results['profit_percent'] * 2
    results_under = hyperopt_results.copy()
    results_under['profit_percent'] = hyperopt_results['profit_percent'] / 2

    hl = HyperOptLossResolver(default_conf).hyperoptloss
    correct = hl.hyperopt_loss_function(hyperopt_results, 600)
    over = hl.hyperopt_loss_function(results_over, 600)
    under = hl.hyperopt_loss_function(results_under, 600)
    assert over < correct
    assert under > correct


def test_sharpe_loss_prefers_higher_profits(default_conf, hyperopt_results) -> None:
    results_over = hyperopt_results.copy()
    results_over['profit_percent'] = hyperopt_results['profit_percent'] * 2
    results_under = hyperopt_results.copy()
    results_under['profit_percent'] = hyperopt_results['profit_percent'] / 2

    default_conf.update({'hyperopt_loss': 'SharpeHyperOptLoss'})
    hl = HyperOptLossResolver(default_conf).hyperoptloss
    correct = hl.hyperopt_loss_function(hyperopt_results, len(hyperopt_results),
                                        datetime(2019, 1, 1), datetime(2019, 5, 1))
    over = hl.hyperopt_loss_function(results_over, len(hyperopt_results),
                                     datetime(2019, 1, 1), datetime(2019, 5, 1))
    under = hl.hyperopt_loss_function(results_under, len(hyperopt_results),
                                      datetime(2019, 1, 1), datetime(2019, 5, 1))
    assert over < correct
    assert under > correct


def test_onlyprofit_loss_prefers_higher_profits(default_conf, hyperopt_results) -> None:
    results_over = hyperopt_results.copy()
    results_over['profit_percent'] = hyperopt_results['profit_percent'] * 2
    results_under = hyperopt_results.copy()
    results_under['profit_percent'] = hyperopt_results['profit_percent'] / 2

    default_conf.update({'hyperopt_loss': 'OnlyProfitHyperOptLoss'})
    hl = HyperOptLossResolver(default_conf).hyperoptloss
    correct = hl.hyperopt_loss_function(hyperopt_results, len(hyperopt_results),
                                        datetime(2019, 1, 1), datetime(2019, 5, 1))
    over = hl.hyperopt_loss_function(results_over, len(hyperopt_results),
                                     datetime(2019, 1, 1), datetime(2019, 5, 1))
    under = hl.hyperopt_loss_function(results_under, len(hyperopt_results),
                                      datetime(2019, 1, 1), datetime(2019, 5, 1))
    assert over < correct
    assert under > correct


def test_log_results_if_loss_improves(hyperopt, capsys) -> None:
    hyperopt.current_best_loss = 2
    hyperopt.total_epochs = 2
    hyperopt.print_results(
        {
            'is_best': False,
            'loss': 1,
            'current_epoch': 2,  # This starts from 1 (in a human-friendly manner)
            'results_explanation': 'foo.',
            'is_initial_point': False
        }
    )
    out, err = capsys.readouterr()
    assert '    2/2: foo. Objective: 1.00000' in out


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


def test_save_trials_saves_trials(mocker, hyperopt, testdatadir, caplog) -> None:
    trials = create_trials(mocker, hyperopt, testdatadir)
    mock_dump = mocker.patch('freqtrade.optimize.hyperopt.dump', return_value=None)
    trials_file = testdatadir / 'optimize' / 'ut_trials.pickle'

    hyperopt.trials = trials
    hyperopt.save_trials(final=True)
    assert log_has("Saving 1 epoch.", caplog)
    assert log_has(f"1 epoch saved to '{trials_file}'.", caplog)
    mock_dump.assert_called_once()

    hyperopt.trials = trials + trials
    hyperopt.save_trials(final=True)
    assert log_has("Saving 2 epochs.", caplog)
    assert log_has(f"2 epochs saved to '{trials_file}'.", caplog)


def test_read_trials_returns_trials_file(mocker, hyperopt, testdatadir, caplog) -> None:
    trials = create_trials(mocker, hyperopt, testdatadir)
    mock_load = mocker.patch('freqtrade.optimize.hyperopt.load', return_value=trials)
    trials_file = testdatadir / 'optimize' / 'ut_trials.pickle'
    hyperopt_trial = hyperopt._read_trials(trials_file)
    assert log_has(f"Reading Trials from '{trials_file}'", caplog)
    assert hyperopt_trial == trials
    mock_load.assert_called_once()


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


def test_start_calls_optimizer(mocker, default_conf, caplog, capsys) -> None:
    dumper = mocker.patch('freqtrade.optimize.hyperopt.dump', MagicMock())
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data',
                 MagicMock(return_value=(MagicMock(), None)))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timeframe',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    parallel = mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel',
        MagicMock(return_value=[{'loss': 1, 'results_explanation': 'foo result',
                                 'params': {'buy': {}, 'sell': {}, 'roi': {}, 'stoploss': 0.0}}])
    )
    patch_exchange(mocker)

    default_conf.update({'config': 'config.json.example',
                         'hyperopt': 'DefaultHyperOpt',
                         'epochs': 1,
                         'timerange': None,
                         'spaces': 'all',
                         'hyperopt_jobs': 1, })

    hyperopt = Hyperopt(default_conf)
    hyperopt.backtesting.strategy.tickerdata_to_dataframe = MagicMock()
    hyperopt.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    hyperopt.start()

    parallel.assert_called_once()

    out, err = capsys.readouterr()
    assert 'Best result:\n\n*    1/1: foo result Objective: 1.00000\n' in out
    assert dumper.called
    # Should be called twice, once for tickerdata, once to save evaluations
    assert dumper.call_count == 2
    assert hasattr(hyperopt.backtesting.strategy, "advise_sell")
    assert hasattr(hyperopt.backtesting.strategy, "advise_buy")
    assert hasattr(hyperopt, "max_open_trades")
    assert hyperopt.max_open_trades == default_conf['max_open_trades']
    assert hasattr(hyperopt, "position_stacking")


def test_format_results(hyperopt):
    # Test with BTC as stake_currency
    trades = [
        ('ETH/BTC', 2, 2, 123),
        ('LTC/BTC', 1, 1, 123),
        ('XPR/BTC', -1, -2, -246)
    ]
    labels = ['currency', 'profit_percent', 'profit_abs', 'trade_duration']
    df = pd.DataFrame.from_records(trades, columns=labels)
    results_metrics = hyperopt._calculate_results_metrics(df)
    result = hyperopt._format_explanation_string(results_metrics, 1)
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
    result = hyperopt._format_explanation_string(results_metrics, 1)
    assert result.find('Total profit 1.00000000 EUR')


def test_has_space(hyperopt):
    hyperopt.config.update({'spaces': ['buy', 'roi']})
    assert hyperopt.has_space('roi')
    assert hyperopt.has_space('buy')
    assert not hyperopt.has_space('stoploss')

    hyperopt.config.update({'spaces': ['all']})
    assert hyperopt.has_space('buy')


def test_populate_indicators(hyperopt, testdatadir) -> None:
    tick = load_tickerdata_file(testdatadir, 'UNITTEST/BTC', '1m')
    tickerlist = {'UNITTEST/BTC': parse_ticker_dataframe(tick, '1m', pair="UNITTEST/BTC",
                                                         fill_missing=True)}
    dataframes = hyperopt.backtesting.strategy.tickerdata_to_dataframe(tickerlist)
    dataframe = hyperopt.custom_hyperopt.populate_indicators(dataframes['UNITTEST/BTC'],
                                                             {'pair': 'UNITTEST/BTC'})

    # Check if some indicators are generated. We will not test all of them
    assert 'adx' in dataframe
    assert 'mfi' in dataframe
    assert 'rsi' in dataframe


def test_buy_strategy_generator(hyperopt, testdatadir) -> None:
    tick = load_tickerdata_file(testdatadir, 'UNITTEST/BTC', '1m')
    tickerlist = {'UNITTEST/BTC': parse_ticker_dataframe(tick, '1m', pair="UNITTEST/BTC",
                                                         fill_missing=True)}
    dataframes = hyperopt.backtesting.strategy.tickerdata_to_dataframe(tickerlist)
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


def test_generate_optimizer(mocker, default_conf) -> None:
    default_conf.update({'config': 'config.json.example',
                         'hyperopt': 'DefaultHyperOpt',
                         'timerange': None,
                         'spaces': 'all',
                         'hyperopt_min_trades': 1,
                         })

    trades = [
        ('TRX/BTC', 0.023117, 0.000233, 100)
    ]
    labels = ['currency', 'profit_percent', 'profit_abs', 'trade_duration']
    backtest_result = pd.DataFrame.from_records(trades, columns=labels)

    mocker.patch(
        'freqtrade.optimize.hyperopt.Backtesting.backtest',
        MagicMock(return_value=backtest_result)
    )
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timeframe',
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
    }
    response_expected = {
        'loss': 1.9840569076926293,
        'results_explanation': ('     1 trades. Avg profit   2.31%. Total profit  0.00023300 BTC '
                                '(   2.31\N{GREEK CAPITAL LETTER SIGMA}%). Avg duration 100.0 mins.'
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
                           'stoploss': -0.4},
        'params_dict': optimizer_param,
        'results_metrics': {'avg_profit': 2.3117,
                            'duration': 100.0,
                            'profit': 2.3117,
                            'total_profit': 0.000233,
                            'trade_count': 1},
        'total_profit': 0.00023300
    }

    hyperopt = Hyperopt(default_conf)
    hyperopt.dimensions = hyperopt.hyperopt_space()
    generate_optimizer_value = hyperopt.generate_optimizer(list(optimizer_param.values()))
    assert generate_optimizer_value == response_expected


def test_clean_hyperopt(mocker, default_conf, caplog):
    patch_exchange(mocker)
    default_conf.update({'config': 'config.json.example',
                         'hyperopt': 'DefaultHyperOpt',
                         'epochs': 1,
                         'timerange': None,
                         'spaces': 'all',
                         'hyperopt_jobs': 1,
                         })
    mocker.patch("freqtrade.optimize.hyperopt.Path.is_file", MagicMock(return_value=True))
    unlinkmock = mocker.patch("freqtrade.optimize.hyperopt.Path.unlink", MagicMock())
    h = Hyperopt(default_conf)

    assert unlinkmock.call_count == 2
    assert log_has(f"Removing `{h.tickerdata_pickle}`.", caplog)


def test_continue_hyperopt(mocker, default_conf, caplog):
    patch_exchange(mocker)
    default_conf.update({'config': 'config.json.example',
                         'hyperopt': 'DefaultHyperOpt',
                         'epochs': 1,
                         'timerange': None,
                         'spaces': 'all',
                         'hyperopt_jobs': 1,
                         'hyperopt_continue': True
                         })
    mocker.patch("freqtrade.optimize.hyperopt.Path.is_file", MagicMock(return_value=True))
    unlinkmock = mocker.patch("freqtrade.optimize.hyperopt.Path.unlink", MagicMock())
    Hyperopt(default_conf)

    assert unlinkmock.call_count == 0
    assert log_has(f"Continuing on previous hyperopt results.", caplog)


def test_print_json_spaces_all(mocker, default_conf, caplog, capsys) -> None:
    dumper = mocker.patch('freqtrade.optimize.hyperopt.dump', MagicMock())
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data',
                 MagicMock(return_value=(MagicMock(), None)))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timeframe',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    parallel = mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel',
        MagicMock(return_value=[{'loss': 1, 'results_explanation': 'foo result', 'params': {}}])
    )
    patch_exchange(mocker)

    default_conf.update({'config': 'config.json.example',
                         'hyperopt': 'DefaultHyperOpt',
                         'epochs': 1,
                         'timerange': None,
                         'spaces': 'all',
                         'hyperopt_jobs': 1,
                         'print_json': True,
                         })

    hyperopt = Hyperopt(default_conf)
    hyperopt.backtesting.strategy.tickerdata_to_dataframe = MagicMock()
    hyperopt.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    hyperopt.start()

    parallel.assert_called_once()

    out, err = capsys.readouterr()
    assert '{"params":{"mfi-value":null,"fastd-value":null,"adx-value":null,"rsi-value":null,"mfi-enabled":null,"fastd-enabled":null,"adx-enabled":null,"rsi-enabled":null,"trigger":null,"sell-mfi-value":null,"sell-fastd-value":null,"sell-adx-value":null,"sell-rsi-value":null,"sell-mfi-enabled":null,"sell-fastd-enabled":null,"sell-adx-enabled":null,"sell-rsi-enabled":null,"sell-trigger":null},"minimal_roi":{},"stoploss":null}' in out  # noqa: E501
    assert dumper.called
    # Should be called twice, once for tickerdata, once to save evaluations
    assert dumper.call_count == 2


def test_print_json_spaces_roi_stoploss(mocker, default_conf, caplog, capsys) -> None:
    dumper = mocker.patch('freqtrade.optimize.hyperopt.dump', MagicMock())
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data',
                 MagicMock(return_value=(MagicMock(), None)))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timeframe',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    parallel = mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel',
        MagicMock(return_value=[{'loss': 1, 'results_explanation': 'foo result', 'params': {},
                                'params_details': {'roi': {}, 'stoploss': None}}])
    )
    patch_exchange(mocker)

    default_conf.update({'config': 'config.json.example',
                         'hyperopt': 'DefaultHyperOpt',
                         'epochs': 1,
                         'timerange': None,
                         'spaces': 'roi stoploss',
                         'hyperopt_jobs': 1,
                         'print_json': True,
                         })

    hyperopt = Hyperopt(default_conf)
    hyperopt.backtesting.strategy.tickerdata_to_dataframe = MagicMock()
    hyperopt.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    hyperopt.start()

    parallel.assert_called_once()

    out, err = capsys.readouterr()
    assert '{"minimal_roi":{},"stoploss":null}' in out
    assert dumper.called
    # Should be called twice, once for tickerdata, once to save evaluations
    assert dumper.call_count == 2


def test_simplified_interface_roi_stoploss(mocker, default_conf, caplog, capsys) -> None:
    dumper = mocker.patch('freqtrade.optimize.hyperopt.dump', MagicMock())
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data',
                 MagicMock(return_value=(MagicMock(), None)))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timeframe',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    parallel = mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel',
        MagicMock(return_value=[{
            'loss': 1, 'results_explanation': 'foo result', 'params': {'stoploss': 0.0}}])
    )
    patch_exchange(mocker)

    default_conf.update({'config': 'config.json.example',
                         'hyperopt': 'DefaultHyperOpt',
                         'epochs': 1,
                         'timerange': None,
                         'spaces': 'roi stoploss',
                         'hyperopt_jobs': 1, })

    hyperopt = Hyperopt(default_conf)
    hyperopt.backtesting.strategy.tickerdata_to_dataframe = MagicMock()
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
    # Should be called twice, once for tickerdata, once to save evaluations
    assert dumper.call_count == 2
    assert hasattr(hyperopt.backtesting.strategy, "advise_sell")
    assert hasattr(hyperopt.backtesting.strategy, "advise_buy")
    assert hasattr(hyperopt, "max_open_trades")
    assert hyperopt.max_open_trades == default_conf['max_open_trades']
    assert hasattr(hyperopt, "position_stacking")


def test_simplified_interface_all_failed(mocker, default_conf, caplog, capsys) -> None:
    mocker.patch('freqtrade.optimize.hyperopt.dump', MagicMock())
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data',
                 MagicMock(return_value=(MagicMock(), None)))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timeframe',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    patch_exchange(mocker)

    default_conf.update({'config': 'config.json.example',
                         'hyperopt': 'DefaultHyperOpt',
                         'epochs': 1,
                         'timerange': None,
                         'spaces': 'all',
                         'hyperopt_jobs': 1, })

    hyperopt = Hyperopt(default_conf)
    hyperopt.backtesting.strategy.tickerdata_to_dataframe = MagicMock()
    hyperopt.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    del hyperopt.custom_hyperopt.__class__.buy_strategy_generator
    del hyperopt.custom_hyperopt.__class__.sell_strategy_generator
    del hyperopt.custom_hyperopt.__class__.indicator_space
    del hyperopt.custom_hyperopt.__class__.sell_indicator_space

    with pytest.raises(OperationalException, match=r"The 'buy' space is included into *"):
        hyperopt.start()


def test_simplified_interface_buy(mocker, default_conf, caplog, capsys) -> None:
    dumper = mocker.patch('freqtrade.optimize.hyperopt.dump', MagicMock())
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data',
                 MagicMock(return_value=(MagicMock(), None)))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timeframe',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    parallel = mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel',
        MagicMock(return_value=[{'loss': 1, 'results_explanation': 'foo result', 'params': {}}])
    )
    patch_exchange(mocker)

    default_conf.update({'config': 'config.json.example',
                         'hyperopt': 'DefaultHyperOpt',
                         'epochs': 1,
                         'timerange': None,
                         'spaces': 'buy',
                         'hyperopt_jobs': 1, })

    hyperopt = Hyperopt(default_conf)
    hyperopt.backtesting.strategy.tickerdata_to_dataframe = MagicMock()
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
    # Should be called twice, once for tickerdata, once to save evaluations
    assert dumper.call_count == 2
    assert hasattr(hyperopt.backtesting.strategy, "advise_sell")
    assert hasattr(hyperopt.backtesting.strategy, "advise_buy")
    assert hasattr(hyperopt, "max_open_trades")
    assert hyperopt.max_open_trades == default_conf['max_open_trades']
    assert hasattr(hyperopt, "position_stacking")


def test_simplified_interface_sell(mocker, default_conf, caplog, capsys) -> None:
    dumper = mocker.patch('freqtrade.optimize.hyperopt.dump', MagicMock())
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data',
                 MagicMock(return_value=(MagicMock(), None)))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timeframe',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    parallel = mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel',
        MagicMock(return_value=[{'loss': 1, 'results_explanation': 'foo result', 'params': {}}])
    )
    patch_exchange(mocker)

    default_conf.update({'config': 'config.json.example',
                         'hyperopt': 'DefaultHyperOpt',
                         'epochs': 1,
                         'timerange': None,
                         'spaces': 'sell',
                         'hyperopt_jobs': 1, })

    hyperopt = Hyperopt(default_conf)
    hyperopt.backtesting.strategy.tickerdata_to_dataframe = MagicMock()
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
    # Should be called twice, once for tickerdata, once to save evaluations
    assert dumper.call_count == 2
    assert hasattr(hyperopt.backtesting.strategy, "advise_sell")
    assert hasattr(hyperopt.backtesting.strategy, "advise_buy")
    assert hasattr(hyperopt, "max_open_trades")
    assert hyperopt.max_open_trades == default_conf['max_open_trades']
    assert hasattr(hyperopt, "position_stacking")


@pytest.mark.parametrize("method,space", [
    ('buy_strategy_generator', 'buy'),
    ('indicator_space', 'buy'),
    ('sell_strategy_generator', 'sell'),
    ('sell_indicator_space', 'sell'),
])
def test_simplified_interface_failed(mocker, default_conf, caplog, capsys, method, space) -> None:
    mocker.patch('freqtrade.optimize.hyperopt.dump', MagicMock())
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data',
                 MagicMock(return_value=(MagicMock(), None)))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timeframe',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    patch_exchange(mocker)

    default_conf.update({'config': 'config.json.example',
                         'hyperopt': 'DefaultHyperOpt',
                         'epochs': 1,
                         'timerange': None,
                         'spaces': space,
                         'hyperopt_jobs': 1, })

    hyperopt = Hyperopt(default_conf)
    hyperopt.backtesting.strategy.tickerdata_to_dataframe = MagicMock()
    hyperopt.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    delattr(hyperopt.custom_hyperopt.__class__, method)

    with pytest.raises(OperationalException, match=f"The '{space}' space is included into *"):
        hyperopt.start()
