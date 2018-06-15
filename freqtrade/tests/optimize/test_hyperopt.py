# pragma pylint: disable=missing-docstring,W0212,C0103
import os
import signal
from copy import deepcopy
from unittest.mock import MagicMock

import pandas as pd
import pytest

from freqtrade.optimize.__init__ import load_tickerdata_file
from freqtrade.optimize.hyperopt import Hyperopt, start
from freqtrade.strategy.resolver import StrategyResolver
from freqtrade.tests.conftest import log_has
from freqtrade.tests.optimize.test_backtesting import get_args

# Avoid to reinit the same object again and again
_HYPEROPT_INITIALIZED = False
_HYPEROPT = None


@pytest.fixture(scope='function')
def init_hyperopt(default_conf, mocker):
    global _HYPEROPT_INITIALIZED, _HYPEROPT
    if not _HYPEROPT_INITIALIZED:
        mocker.patch('freqtrade.exchange.validate_pairs', MagicMock(return_value=True))
        mocker.patch('freqtrade.exchange.validate_pairs', MagicMock())
        _HYPEROPT = Hyperopt(default_conf)
        _HYPEROPT_INITIALIZED = True


# Functions for recurrent object patching
def create_trials(mocker) -> None:
    """
    When creating trials, mock the hyperopt Trials so that *by default*
      - we don't create any pickle'd files in the filesystem
      - we might have a pickle'd file so make sure that we return
        false when looking for it
    """
    _HYPEROPT.trials_file = os.path.join('freqtrade', 'tests', 'optimize', 'ut_trials.pickle')

    mocker.patch('freqtrade.optimize.hyperopt.os.path.exists', return_value=False)
    mocker.patch('freqtrade.optimize.hyperopt.os.path.getsize', return_value=1)
    mocker.patch('freqtrade.optimize.hyperopt.os.remove', return_value=True)
    mocker.patch('freqtrade.optimize.hyperopt.pickle.dump', return_value=None)

    return mocker.Mock(
        results=[
            {
                'loss': 1,
                'result': 'foo',
                'status': 'ok'
            }
        ],
        best_trial={'misc': {'vals': {'adx': 999}}}
    )


# Unit tests
def test_start(mocker, default_conf, caplog) -> None:
    """
    Test start() function
    """
    start_mock = MagicMock()
    mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.start', start_mock)
    mocker.patch('freqtrade.freqtradebot.exchange.validate_pairs', MagicMock())

    args = [
        '--config', 'config.json',
        '--strategy', 'DefaultStrategy',
        'hyperopt',
        '--epochs', '5'
    ]
    args = get_args(args)
    StrategyResolver({'strategy': 'DefaultStrategy'})
    start(args)

    import pprint
    pprint.pprint(caplog.record_tuples)

    assert log_has(
        'Starting freqtrade in Hyperopt mode',
        caplog.record_tuples
    )
    assert start_mock.call_count == 1


def test_loss_calculation_prefer_correct_trade_count(init_hyperopt) -> None:
    """
    Test Hyperopt.calculate_loss()
    """
    hyperopt = _HYPEROPT
    StrategyResolver({'strategy': 'DefaultStrategy'})

    correct = hyperopt.calculate_loss(1, hyperopt.target_trades, 20)
    over = hyperopt.calculate_loss(1, hyperopt.target_trades + 100, 20)
    under = hyperopt.calculate_loss(1, hyperopt.target_trades - 100, 20)
    assert over > correct
    assert under > correct


def test_loss_calculation_prefer_shorter_trades(init_hyperopt) -> None:
    """
    Test Hyperopt.calculate_loss()
    """
    hyperopt = _HYPEROPT

    shorter = hyperopt.calculate_loss(1, 100, 20)
    longer = hyperopt.calculate_loss(1, 100, 30)
    assert shorter < longer


def test_loss_calculation_has_limited_profit(init_hyperopt) -> None:
    hyperopt = _HYPEROPT

    correct = hyperopt.calculate_loss(hyperopt.expected_max_profit, hyperopt.target_trades, 20)
    over = hyperopt.calculate_loss(hyperopt.expected_max_profit * 2, hyperopt.target_trades, 20)
    under = hyperopt.calculate_loss(hyperopt.expected_max_profit / 2, hyperopt.target_trades, 20)
    assert over == correct
    assert under > correct


def test_log_results_if_loss_improves(init_hyperopt, capsys) -> None:
    hyperopt = _HYPEROPT
    hyperopt.current_best_loss = 2
    hyperopt.log_results(
        {
            'loss': 1,
            'current_tries': 1,
            'total_tries': 2,
            'result': 'foo'
        }
    )
    out, err = capsys.readouterr()
    assert '    1/2: foo. Loss 1.00000'in out


def test_no_log_if_loss_does_not_improve(init_hyperopt, caplog) -> None:
    hyperopt = _HYPEROPT
    hyperopt.current_best_loss = 2
    hyperopt.log_results(
        {
            'loss': 3,
        }
    )
    assert caplog.record_tuples == []


def test_fmin_best_results(mocker, init_hyperopt, default_conf, caplog) -> None:
    fmin_result = {
        "macd_below_zero": 0,
        "adx": 1,
        "adx-value": 15.0,
        "fastd": 1,
        "fastd-value": 40.0,
        "green_candle": 1,
        "mfi": 0,
        "over_sar": 0,
        "rsi": 1,
        "rsi-value": 37.0,
        "trigger": 2,
        "uptrend_long_ema": 1,
        "uptrend_short_ema": 0,
        "uptrend_sma": 0,
        "stoploss": -0.1,
        "roi_t1": 1,
        "roi_t2": 2,
        "roi_t3": 3,
        "roi_p1": 1,
        "roi_p2": 2,
        "roi_p3": 3,
    }

    conf = deepcopy(default_conf)
    conf.update({'config': 'config.json.example'})
    conf.update({'epochs': 1})
    conf.update({'timerange': None})
    conf.update({'spaces': 'all'})

    mocker.patch('freqtrade.optimize.hyperopt.load_data', MagicMock())
    mocker.patch('freqtrade.optimize.hyperopt.fmin', return_value=fmin_result)
    mocker.patch('freqtrade.freqtradebot.exchange.validate_pairs', MagicMock())

    StrategyResolver({'strategy': 'DefaultStrategy'})
    hyperopt = Hyperopt(conf)
    hyperopt.trials = create_trials(mocker)
    hyperopt.tickerdata_to_dataframe = MagicMock()
    hyperopt.start()

    exists = [
        'Best parameters:',
        '"adx": {\n        "enabled": true,\n        "value": 15.0\n    },',
        '"fastd": {\n        "enabled": true,\n        "value": 40.0\n    },',
        '"green_candle": {\n        "enabled": true\n    },',
        '"macd_below_zero": {\n        "enabled": false\n    },',
        '"mfi": {\n        "enabled": false\n    },',
        '"over_sar": {\n        "enabled": false\n    },',
        '"roi_p1": 1.0,',
        '"roi_p2": 2.0,',
        '"roi_p3": 3.0,',
        '"roi_t1": 1.0,',
        '"roi_t2": 2.0,',
        '"roi_t3": 3.0,',
        '"rsi": {\n        "enabled": true,\n        "value": 37.0\n    },',
        '"stoploss": -0.1,',
        '"trigger": {\n        "type": "faststoch10"\n    },',
        '"uptrend_long_ema": {\n        "enabled": true\n    },',
        '"uptrend_short_ema": {\n        "enabled": false\n    },',
        '"uptrend_sma": {\n        "enabled": false\n    }',
        'ROI table:\n{0: 6.0, 3.0: 3.0, 5.0: 1.0, 6.0: 0}',
        'Best Result:\nfoo'
    ]
    for line in exists:
        assert line in caplog.text


def test_fmin_throw_value_error(mocker, init_hyperopt, default_conf, caplog) -> None:
    mocker.patch('freqtrade.optimize.hyperopt.load_data', MagicMock())
    mocker.patch('freqtrade.optimize.hyperopt.fmin', side_effect=ValueError())

    conf = deepcopy(default_conf)
    conf.update({'config': 'config.json.example'})
    conf.update({'epochs': 1})
    conf.update({'timerange': None})
    conf.update({'spaces': 'all'})
    mocker.patch('freqtrade.freqtradebot.exchange.validate_pairs', MagicMock())

    StrategyResolver({'strategy': 'DefaultStrategy'})
    hyperopt = Hyperopt(conf)
    hyperopt.trials = create_trials(mocker)
    hyperopt.tickerdata_to_dataframe = MagicMock()

    hyperopt.start()

    exists = [
        'Best Result:',
        'Sorry, Hyperopt was not able to find good parameters. Please try with more epochs '
        '(param: -e).',
    ]

    for line in exists:
        assert line in caplog.text


def test_resuming_previous_hyperopt_results_succeeds(mocker, init_hyperopt, default_conf) -> None:
    trials = create_trials(mocker)

    conf = deepcopy(default_conf)
    conf.update({'config': 'config.json.example'})
    conf.update({'epochs': 1})
    conf.update({'timerange': None})
    conf.update({'spaces': 'all'})

    mocker.patch('freqtrade.optimize.hyperopt.os.path.exists', return_value=True)
    mocker.patch('freqtrade.optimize.hyperopt.len', return_value=len(trials.results))
    mock_read = mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.read_trials',
        return_value=trials
    )
    mock_save = mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.save_trials',
        return_value=None
    )
    mocker.patch('freqtrade.optimize.hyperopt.sorted', return_value=trials.results)
    mocker.patch('freqtrade.optimize.hyperopt.load_data', MagicMock())
    mocker.patch('freqtrade.optimize.hyperopt.fmin', return_value={})
    mocker.patch('freqtrade.exchange.validate_pairs', MagicMock())

    StrategyResolver({'strategy': 'DefaultStrategy'})
    hyperopt = Hyperopt(conf)
    hyperopt.trials = trials
    hyperopt.tickerdata_to_dataframe = MagicMock()

    hyperopt.start()

    mock_read.assert_called_once()
    mock_save.assert_called_once()

    current_tries = hyperopt.current_tries
    total_tries = hyperopt.total_tries

    assert current_tries == len(trials.results)
    assert total_tries == (current_tries + len(trials.results))


def test_save_trials_saves_trials(mocker, init_hyperopt, caplog) -> None:
    create_trials(mocker)
    mock_dump = mocker.patch('freqtrade.optimize.hyperopt.pickle.dump', return_value=None)

    hyperopt = _HYPEROPT
    mocker.patch('freqtrade.optimize.hyperopt.open', return_value=hyperopt.trials_file)

    hyperopt.save_trials()

    trials_file = os.path.join('freqtrade', 'tests', 'optimize', 'ut_trials.pickle')
    assert log_has(
        'Saving Trials to \'{}\''.format(trials_file),
        caplog.record_tuples
    )
    mock_dump.assert_called_once()


def test_read_trials_returns_trials_file(mocker, init_hyperopt, caplog) -> None:
    trials = create_trials(mocker)
    mock_load = mocker.patch('freqtrade.optimize.hyperopt.pickle.load', return_value=trials)
    mock_open = mocker.patch('freqtrade.optimize.hyperopt.open', return_value=mock_load)

    hyperopt = _HYPEROPT
    hyperopt_trial = hyperopt.read_trials()
    trials_file = os.path.join('freqtrade', 'tests', 'optimize', 'ut_trials.pickle')
    assert log_has(
        'Reading Trials from \'{}\''.format(trials_file),
        caplog.record_tuples
    )
    assert hyperopt_trial == trials
    mock_open.assert_called_once()
    mock_load.assert_called_once()


def test_roi_table_generation(init_hyperopt) -> None:
    params = {
        'roi_t1': 5,
        'roi_t2': 10,
        'roi_t3': 15,
        'roi_p1': 1,
        'roi_p2': 2,
        'roi_p3': 3,
    }

    hyperopt = _HYPEROPT
    assert hyperopt.generate_roi_table(params) == {0: 6, 15: 3, 25: 1, 30: 0}


def test_start_calls_fmin(mocker, init_hyperopt, default_conf) -> None:
    trials = create_trials(mocker)
    mocker.patch('freqtrade.optimize.hyperopt.sorted', return_value=trials.results)
    mocker.patch('freqtrade.optimize.hyperopt.load_data', MagicMock())
    mocker.patch('freqtrade.exchange.validate_pairs', MagicMock())
    mock_fmin = mocker.patch('freqtrade.optimize.hyperopt.fmin', return_value={})

    conf = deepcopy(default_conf)
    conf.update({'config': 'config.json.example'})
    conf.update({'epochs': 1})
    conf.update({'timerange': None})
    conf.update({'spaces': 'all'})

    hyperopt = Hyperopt(conf)
    hyperopt.trials = trials
    hyperopt.tickerdata_to_dataframe = MagicMock()

    hyperopt.start()
    mock_fmin.assert_called_once()


def test_format_results(init_hyperopt):
    """
    Test Hyperopt.format_results()
    """

    # Test with BTC as stake_currency
    trades = [
        ('ETH/BTC', 2, 2, 123),
        ('LTC/BTC', 1, 1, 123),
        ('XPR/BTC', -1, -2, -246)
    ]
    labels = ['currency', 'profit_percent', 'profit_BTC', 'duration']
    df = pd.DataFrame.from_records(trades, columns=labels)

    result = _HYPEROPT.format_results(df)
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
    result = _HYPEROPT.format_results(df)
    assert result.find('Total profit 1.00000000 EUR')


def test_signal_handler(mocker, init_hyperopt):
    """
    Test Hyperopt.signal_handler()
    """
    m = MagicMock()
    mocker.patch('sys.exit', m)
    mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.save_trials', m)
    mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.log_trials_result', m)

    hyperopt = _HYPEROPT
    hyperopt.signal_handler(signal.SIGTERM, None)
    assert m.call_count == 3


def test_has_space(init_hyperopt):
    """
    Test Hyperopt.has_space() method
    """
    _HYPEROPT.config.update({'spaces': ['buy', 'roi']})
    assert _HYPEROPT.has_space('roi')
    assert _HYPEROPT.has_space('buy')
    assert not _HYPEROPT.has_space('stoploss')

    _HYPEROPT.config.update({'spaces': ['all']})
    assert _HYPEROPT.has_space('buy')


def test_populate_indicators(init_hyperopt) -> None:
    """
    Test Hyperopt.populate_indicators()
    """
    tick = load_tickerdata_file(None, 'UNITTEST/BTC', '1m')
    tickerlist = {'UNITTEST/BTC': tick}
    dataframes = _HYPEROPT.tickerdata_to_dataframe(tickerlist)
    dataframe = _HYPEROPT.populate_indicators(dataframes['UNITTEST/BTC'])

    # Check if some indicators are generated. We will not test all of them
    assert 'adx' in dataframe
    assert 'ao' in dataframe
    assert 'cci' in dataframe


def test_buy_strategy_generator(init_hyperopt) -> None:
    """
    Test Hyperopt.buy_strategy_generator()
    """
    tick = load_tickerdata_file(None, 'UNITTEST/BTC', '1m')
    tickerlist = {'UNITTEST/BTC': tick}
    dataframes = _HYPEROPT.tickerdata_to_dataframe(tickerlist)
    dataframe = _HYPEROPT.populate_indicators(dataframes['UNITTEST/BTC'])

    populate_buy_trend = _HYPEROPT.buy_strategy_generator(
        {
            'uptrend_long_ema': {
                'enabled': True
            },
            'macd_below_zero': {
                'enabled': True
            },
            'uptrend_short_ema': {
                'enabled': True
            },
            'mfi': {
                'enabled': True,
                'value': 20
            },
            'fastd': {
                'enabled': True,
                'value': 20
            },
            'adx': {
                'enabled': True,
                'value': 20
            },
            'rsi': {
                'enabled': True,
                'value': 20
            },
            'over_sar': {
                'enabled': True,
            },
            'green_candle': {
                'enabled': True,
            },
            'uptrend_sma': {
                'enabled': True,
            },

            'trigger': {
                'type': 'lower_bb'
            }
        }
    )
    result = populate_buy_trend(dataframe)
    # Check if some indicators are generated. We will not test all of them
    assert 'buy' in result
    assert 1 in result['buy']


def test_generate_optimizer(mocker, init_hyperopt, default_conf) -> None:
    """
    Test Hyperopt.generate_optimizer() function
    """
    conf = deepcopy(default_conf)
    conf.update({'config': 'config.json.example'})
    conf.update({'timerange': None})
    conf.update({'spaces': 'all'})

    trades = [
        ('POWR/BTC', 0.023117, 0.000233, 100)
    ]
    labels = ['currency', 'profit_percent', 'profit_BTC', 'duration']
    backtest_result = pd.DataFrame.from_records(trades, columns=labels)

    mocker.patch(
        'freqtrade.optimize.hyperopt.Hyperopt.backtest',
        MagicMock(return_value=backtest_result)
    )
    mocker.patch('freqtrade.exchange.validate_pairs', MagicMock())

    optimizer_param = {
        'adx': {'enabled': False},
        'fastd': {'enabled': True, 'value': 35.0},
        'green_candle': {'enabled': True},
        'macd_below_zero': {'enabled': True},
        'mfi': {'enabled': False},
        'over_sar': {'enabled': False},
        'roi_p1': 0.01,
        'roi_p2': 0.01,
        'roi_p3': 0.1,
        'roi_t1': 60.0,
        'roi_t2': 30.0,
        'roi_t3': 20.0,
        'rsi': {'enabled': False},
        'stoploss': -0.4,
        'trigger': {'type': 'macd_cross_signal'},
        'uptrend_long_ema': {'enabled': False},
        'uptrend_short_ema': {'enabled': True},
        'uptrend_sma': {'enabled': True}
    }

    response_expected = {
        'loss': 1.9840569076926293,
        'result': '     1 trades. Avg profit  2.31%. Total profit  0.00023300 BTC '
                  '(0.0231Σ%). Avg duration 100.0 mins.',
        'status': 'ok'
    }

    hyperopt = Hyperopt(conf)
    generate_optimizer_value = hyperopt.generate_optimizer(optimizer_param)
    assert generate_optimizer_value == response_expected
