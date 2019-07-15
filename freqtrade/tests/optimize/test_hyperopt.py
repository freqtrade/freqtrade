# pragma pylint: disable=missing-docstring,W0212,C0103
import os
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
from arrow import Arrow
from filelock import Timeout

from freqtrade import DependencyException
from freqtrade.data.converter import parse_ticker_dataframe
from freqtrade.data.history import load_tickerdata_file
from freqtrade.optimize import setup_configuration, start_hyperopt
from freqtrade.optimize.default_hyperopt import DefaultHyperOpts
from freqtrade.optimize.hyperopt import (HYPEROPT_LOCKFILE, TICKERDATA_PICKLE,
                                         Hyperopt)
from freqtrade.optimize.hyperopt_loss import hyperopt_loss_legacy, hyperopt_loss_sharpe
from freqtrade.resolvers.hyperopt_resolver import HyperOptResolver
from freqtrade.state import RunMode
from freqtrade.strategy.interface import SellType
from freqtrade.tests.conftest import (get_args, log_has, log_has_re,
                                      patch_exchange,
                                      patched_configuration_load_config_file)


@pytest.fixture(scope='function')
def hyperopt(default_conf, mocker):
    patch_exchange(mocker)
    return Hyperopt(default_conf)


@pytest.fixture(scope='function')
def hyperopt_results():
    return pd.DataFrame(
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


# Functions for recurrent object patching
def create_trials(mocker, hyperopt) -> None:
    """
    When creating trials, mock the hyperopt Trials so that *by default*
      - we don't create any pickle'd files in the filesystem
      - we might have a pickle'd file so make sure that we return
        false when looking for it
    """
    hyperopt.trials_file = os.path.join('freqtrade', 'tests', 'optimize', 'ut_trials.pickle')

    mocker.patch('freqtrade.optimize.hyperopt.os.path.exists', return_value=False)
    mocker.patch('freqtrade.optimize.hyperopt.os.path.getsize', return_value=1)
    mocker.patch('freqtrade.optimize.hyperopt.os.remove', return_value=True)
    mocker.patch('freqtrade.optimize.hyperopt.dump', return_value=None)

    return [{'loss': 1, 'result': 'foo', 'params': {}}]


def test_setup_hyperopt_configuration_without_arguments(mocker, default_conf, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        '--config', 'config.json',
        'hyperopt'
    ]

    config = setup_configuration(get_args(args), RunMode.HYPEROPT)
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert log_has(
        'Using data directory: {} ...'.format(config['datadir']),
        caplog.record_tuples
    )
    assert 'ticker_interval' in config
    assert not log_has_re('Parameter -i/--ticker-interval detected .*', caplog.record_tuples)

    assert 'live' not in config
    assert not log_has('Parameter -l/--live detected ...', caplog.record_tuples)

    assert 'position_stacking' not in config
    assert not log_has('Parameter --enable-position-stacking detected ...', caplog.record_tuples)

    assert 'refresh_pairs' not in config
    assert not log_has('Parameter -r/--refresh-pairs-cached detected ...', caplog.record_tuples)

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
        '--config', 'config.json',
        '--datadir', '/foo/bar',
        'hyperopt',
        '--ticker-interval', '1m',
        '--timerange', ':100',
        '--refresh-pairs-cached',
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

    assert log_has(
        'Using data directory: {} ...'.format(config['datadir']),
        caplog.record_tuples
    )
    assert 'ticker_interval' in config
    assert log_has('Parameter -i/--ticker-interval detected ... Using ticker_interval: 1m ...',
                   caplog.record_tuples)

    assert 'position_stacking' in config
    assert log_has('Parameter --enable-position-stacking detected ...', caplog.record_tuples)

    assert 'use_max_market_positions' in config
    assert log_has('Parameter --disable-max-market-positions detected ...', caplog.record_tuples)
    assert log_has('max_open_trades set to unlimited ...', caplog.record_tuples)

    assert 'refresh_pairs' in config
    assert log_has('Parameter -r/--refresh-pairs-cached detected ...', caplog.record_tuples)

    assert 'timerange' in config
    assert log_has(
        'Parameter --timerange detected: {} ...'.format(config['timerange']),
        caplog.record_tuples
    )

    assert 'epochs' in config
    assert log_has('Parameter --epochs detected ... Will run Hyperopt with for 1000 epochs ...',
                   caplog.record_tuples)

    assert 'spaces' in config
    assert log_has(
        'Parameter -s/--spaces detected: {}'.format(config['spaces']),
        caplog.record_tuples
    )
    assert 'print_all' in config
    assert log_has('Parameter --print-all detected ...', caplog.record_tuples)


def test_hyperoptresolver(mocker, default_conf, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)

    hyperopts = DefaultHyperOpts
    delattr(hyperopts, 'populate_buy_trend')
    delattr(hyperopts, 'populate_sell_trend')
    mocker.patch(
        'freqtrade.resolvers.hyperopt_resolver.HyperOptResolver._load_hyperopt',
        MagicMock(return_value=hyperopts)
    )
    x = HyperOptResolver(default_conf, ).hyperopt
    assert not hasattr(x, 'populate_buy_trend')
    assert not hasattr(x, 'populate_sell_trend')
    assert log_has("Custom Hyperopt does not provide populate_sell_trend. "
                   "Using populate_sell_trend from DefaultStrategy.", caplog.record_tuples)
    assert log_has("Custom Hyperopt does not provide populate_buy_trend. "
                   "Using populate_buy_trend from DefaultStrategy.", caplog.record_tuples)
    assert hasattr(x, "ticker_interval")


def test_start(mocker, default_conf, caplog) -> None:
    start_mock = MagicMock()
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.start', start_mock)
    patch_exchange(mocker)

    args = [
        '--config', 'config.json',
        'hyperopt',
        '--epochs', '5'
    ]
    args = get_args(args)
    start_hyperopt(args)

    import pprint
    pprint.pprint(caplog.record_tuples)

    assert log_has(
        'Starting freqtrade in Hyperopt mode',
        caplog.record_tuples
    )
    assert start_mock.call_count == 1


def test_start_no_data(mocker, default_conf, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('freqtrade.optimize.hyperopt.load_data', MagicMock(return_value={}))
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timeframe',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    patch_exchange(mocker)

    args = [
        '--config', 'config.json',
        'hyperopt',
        '--epochs', '5'
    ]
    args = get_args(args)
    start_hyperopt(args)

    import pprint
    pprint.pprint(caplog.record_tuples)

    assert log_has('No data found. Terminating.', caplog.record_tuples)


def test_start_failure(mocker, default_conf, caplog) -> None:
    start_mock = MagicMock()
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.start', start_mock)
    patch_exchange(mocker)

    args = [
        '--config', 'config.json',
        '--strategy', 'TestStrategy',
        'hyperopt',
        '--epochs', '5'
    ]
    args = get_args(args)
    with pytest.raises(DependencyException):
        start_hyperopt(args)
    assert log_has(
        "Please don't use --strategy for hyperopt.",
        caplog.record_tuples
    )


def test_start_filelock(mocker, default_conf, caplog) -> None:
    start_mock = MagicMock(side_effect=Timeout(HYPEROPT_LOCKFILE))
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.start', start_mock)
    patch_exchange(mocker)

    args = [
        '--config', 'config.json',
        'hyperopt',
        '--epochs', '5'
    ]
    args = get_args(args)
    start_hyperopt(args)
    assert log_has(
        "Another running instance of freqtrade Hyperopt detected.",
        caplog.record_tuples
    )


def test_loss_calculation_prefer_correct_trade_count(hyperopt_results) -> None:
    correct = hyperopt_loss_legacy(hyperopt_results, 600)
    over = hyperopt_loss_legacy(hyperopt_results, 600 + 100)
    under = hyperopt_loss_legacy(hyperopt_results, 600 - 100)
    assert over > correct
    assert under > correct


def test_loss_calculation_prefer_shorter_trades(hyperopt_results) -> None:
    resultsb = hyperopt_results.copy()
    resultsb['trade_duration'][1] = 20

    longer = hyperopt_loss_legacy(hyperopt_results, 100)
    shorter = hyperopt_loss_legacy(resultsb, 100)
    assert shorter < longer


def test_loss_calculation_has_limited_profit(hyperopt_results) -> None:
    results_over = hyperopt_results.copy()
    results_over['profit_percent'] = hyperopt_results['profit_percent'] * 2
    results_under = hyperopt_results.copy()
    results_under['profit_percent'] = hyperopt_results['profit_percent'] / 2

    correct = hyperopt_loss_legacy(hyperopt_results, 600)
    over = hyperopt_loss_legacy(results_over, 600)
    under = hyperopt_loss_legacy(results_under, 600)
    assert over < correct
    assert under > correct
    

def test_sharpe_loss_prefers_higher_profits(hyperopt_results) -> None:
    results_over = hyperopt_results.copy()
    results_over['profit_percent'] = hyperopt_results['profit_percent'] * 2
    results_under = hyperopt_results.copy()
    results_under['profit_percent'] = hyperopt_results['profit_percent'] / 2

    correct = hyperopt_loss_sharpe(hyperopt_results, len(
        hyperopt_results), datetime(2019, 1, 1), datetime(2019, 5, 1))
    over = hyperopt_loss_sharpe(results_over, len(hyperopt_results),
                                datetime(2019, 1, 1), datetime(2019, 5, 1))
    under = hyperopt_loss_sharpe(results_under, len(hyperopt_results),
                                 datetime(2019, 1, 1), datetime(2019, 5, 1))
    assert over < correct
    assert under > correct


def test_log_results_if_loss_improves(hyperopt, capsys) -> None:
    hyperopt.current_best_loss = 2
    hyperopt.log_results(
        {
            'loss': 1,
            'current_tries': 1,
            'total_tries': 2,
            'result': 'foo.',
            'initial_point': False
        }
    )
    out, err = capsys.readouterr()
    assert '    2/2: foo. Objective: 1.00000' in out


def test_no_log_if_loss_does_not_improve(hyperopt, caplog) -> None:
    hyperopt.current_best_loss = 2
    hyperopt.log_results(
        {
            'loss': 3,
        }
    )
    assert caplog.record_tuples == []


def test_save_trials_saves_trials(mocker, hyperopt, caplog) -> None:
    trials = create_trials(mocker, hyperopt)
    mock_dump = mocker.patch('freqtrade.optimize.hyperopt.dump', return_value=None)
    hyperopt.trials = trials
    hyperopt.save_trials()

    trials_file = os.path.join('freqtrade', 'tests', 'optimize', 'ut_trials.pickle')
    assert log_has(
        'Saving 1 evaluations to \'{}\''.format(trials_file),
        caplog.record_tuples
    )
    mock_dump.assert_called_once()


def test_read_trials_returns_trials_file(mocker, hyperopt, caplog) -> None:
    trials = create_trials(mocker, hyperopt)
    mock_load = mocker.patch('freqtrade.optimize.hyperopt.load', return_value=trials)
    hyperopt_trial = hyperopt.read_trials()
    trials_file = os.path.join('freqtrade', 'tests', 'optimize', 'ut_trials.pickle')
    assert log_has(
        'Reading Trials from \'{}\''.format(trials_file),
        caplog.record_tuples
    )
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


def test_start_calls_optimizer(mocker, default_conf, caplog) -> None:
    dumper = mocker.patch('freqtrade.optimize.hyperopt.dump', MagicMock())
    mocker.patch('freqtrade.optimize.hyperopt.load_data', MagicMock())
    mocker.patch(
        'freqtrade.optimize.hyperopt.get_timeframe',
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13)))
    )

    parallel = mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel',
        MagicMock(return_value=[{'loss': 1, 'result': 'foo result', 'params': {}}])
    )
    patch_exchange(mocker)

    default_conf.update({'config': 'config.json.example',
                         'epochs': 1,
                         'timerange': None,
                         'spaces': 'all',
                         'hyperopt_jobs': 1, })

    hyperopt = Hyperopt(default_conf)
    hyperopt.strategy.tickerdata_to_dataframe = MagicMock()

    hyperopt.start()
    parallel.assert_called_once()
    assert log_has('Best result:\nfoo result\nwith values:\n', caplog.record_tuples)
    assert dumper.called
    # Should be called twice, once for tickerdata, once to save evaluations
    assert dumper.call_count == 2
    assert hasattr(hyperopt, "advise_sell")
    assert hasattr(hyperopt, "advise_buy")
    assert hasattr(hyperopt, "max_open_trades")
    assert hyperopt.max_open_trades == default_conf['max_open_trades']


def test_format_results(hyperopt):
    # Test with BTC as stake_currency
    trades = [
        ('ETH/BTC', 2, 2, 123),
        ('LTC/BTC', 1, 1, 123),
        ('XPR/BTC', -1, -2, -246)
    ]
    labels = ['currency', 'profit_percent', 'profit_abs', 'trade_duration']
    df = pd.DataFrame.from_records(trades, columns=labels)

    result = hyperopt.format_results(df)
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
    result = hyperopt.format_results(df)
    assert result.find('Total profit 1.00000000 EUR')


def test_has_space(hyperopt):
    hyperopt.config.update({'spaces': ['buy', 'roi']})
    assert hyperopt.has_space('roi')
    assert hyperopt.has_space('buy')
    assert not hyperopt.has_space('stoploss')

    hyperopt.config.update({'spaces': ['all']})
    assert hyperopt.has_space('buy')


def test_populate_indicators(hyperopt) -> None:
    tick = load_tickerdata_file(None, 'UNITTEST/BTC', '1m')
    tickerlist = {'UNITTEST/BTC': parse_ticker_dataframe(tick, '1m', pair="UNITTEST/BTC",
                                                         fill_missing=True)}
    dataframes = hyperopt.strategy.tickerdata_to_dataframe(tickerlist)
    dataframe = hyperopt.custom_hyperopt.populate_indicators(dataframes['UNITTEST/BTC'],
                                                             {'pair': 'UNITTEST/BTC'})

    # Check if some indicators are generated. We will not test all of them
    assert 'adx' in dataframe
    assert 'mfi' in dataframe
    assert 'rsi' in dataframe


def test_buy_strategy_generator(hyperopt) -> None:
    tick = load_tickerdata_file(None, 'UNITTEST/BTC', '1m')
    tickerlist = {'UNITTEST/BTC': parse_ticker_dataframe(tick, '1m', pair="UNITTEST/BTC",
                                                         fill_missing=True)}
    dataframes = hyperopt.strategy.tickerdata_to_dataframe(tickerlist)
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
    default_conf.update({'config': 'config.json.example'})
    default_conf.update({'timerange': None})
    default_conf.update({'spaces': 'all'})
    default_conf.update({'hyperopt_min_trades': 1})

    trades = [
        ('POWR/BTC', 0.023117, 0.000233, 100)
    ]
    labels = ['currency', 'profit_percent', 'profit_abs', 'trade_duration']
    backtest_result = pd.DataFrame.from_records(trades, columns=labels)

    mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.backtest',
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
        'result': '     1 trades. Avg profit  2.31%. Total profit  0.00023300 BTC '
                  '(   2.31Σ%). Avg duration 100.0 mins.',
        'params': optimizer_param
    }

    hyperopt = Hyperopt(default_conf)
    generate_optimizer_value = hyperopt.generate_optimizer(list(optimizer_param.values()))
    assert generate_optimizer_value == response_expected


def test_clean_hyperopt(mocker, default_conf, caplog):
    patch_exchange(mocker)
    default_conf.update({'config': 'config.json.example',
                         'epochs': 1,
                         'timerange': None,
                         'spaces': 'all',
                         'hyperopt_jobs': 1,
                         })
    mocker.patch("freqtrade.optimize.hyperopt.Path.is_file", MagicMock(return_value=True))
    unlinkmock = mocker.patch("freqtrade.optimize.hyperopt.Path.unlink", MagicMock())
    hyp = Hyperopt(default_conf)

    hyp.clean_hyperopt()
    assert unlinkmock.call_count == 2
    assert log_has(f"Removing `{TICKERDATA_PICKLE}`.", caplog.record_tuples)
