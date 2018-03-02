# pragma pylint: disable=missing-docstring,W0212,C0103
import logging
import os
import pytest
from copy import deepcopy

#from freqtrade.optimize.hyperopt import EXPECTED_MAX_PROFIT, start, \
#    log_results, save_trials, read_trials, generate_roi_table
from unittest.mock import MagicMock

from freqtrade.optimize.hyperopt import Hyperopt, start
import freqtrade.tests.conftest as tt  # test tools


# Avoid to reinit the same object again and again
_HYPEROPT = Hyperopt(tt.default_conf())


# Functions for recurrent object patching
def create_trials(mocker) -> None:
    """
    When creating trials, mock the hyperopt Trials so that *by default*
      - we don't create any pickle'd files in the filesystem
      - we might have a pickle'd file so make sure that we return
        false when looking for it
    """
    _HYPEROPT.trials_file = os.path.join('freqtrade', 'tests', 'optimize','ut_trials.pickle')

    mocker.patch('freqtrade.optimize.hyperopt.os.path.exists', return_value=False)
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
def test_loss_calculation_prefer_correct_trade_count() -> None:
    """
    Test Hyperopt.calculate_loss()
    """
    hyperopt = _HYPEROPT

    correct = hyperopt.calculate_loss(1, hyperopt.target_trades, 20)
    over = hyperopt.calculate_loss(1, hyperopt.target_trades + 100, 20)
    under = hyperopt.calculate_loss(1, hyperopt.target_trades - 100, 20)
    assert over > correct
    assert under > correct


def test_loss_calculation_prefer_shorter_trades() -> None:
    """
    Test Hyperopt.calculate_loss()
    """
    hyperopt = _HYPEROPT

    shorter = hyperopt.calculate_loss(1, 100, 20)
    longer = hyperopt.calculate_loss(1, 100, 30)
    assert shorter < longer


def test_loss_calculation_has_limited_profit() -> None:
    hyperopt = _HYPEROPT

    correct = hyperopt.calculate_loss(hyperopt.expected_max_profit, hyperopt.target_trades, 20)
    over = hyperopt.calculate_loss(hyperopt.expected_max_profit * 2, hyperopt.target_trades, 20)
    under = hyperopt.calculate_loss(hyperopt.expected_max_profit / 2, hyperopt.target_trades, 20)
    assert over == correct
    assert under > correct


def test_log_results_if_loss_improves(caplog) -> None:
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
    assert tt.log_has('    1/2: foo. Loss 1.00000', caplog.record_tuples)


def test_no_log_if_loss_does_not_improve(caplog) -> None:
    hyperopt = _HYPEROPT
    hyperopt.current_best_loss = 2
    hyperopt.log_results(
        {
            'loss': 3,
        }
    )
    assert caplog.record_tuples == []


def test_fmin_best_results(mocker, default_conf, caplog) -> None:
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

    mocker.patch('freqtrade.optimize.hyperopt.load_data', MagicMock())
    mocker.patch('freqtrade.optimize.hyperopt.fmin', return_value=fmin_result)
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimize_conf', return_value=conf)
    mocker.patch('freqtrade.logger.Logger.set_format', MagicMock())

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
        'ROI table:\n{\'0\': 6.0, \'3.0\': 3.0, \'5.0\': 1.0, \'6.0\': 0}',
        'Best Result:\nfoo'
    ]
    for line in exists:
        assert line in caplog.text


def test_fmin_throw_value_error(mocker, default_conf, caplog) -> None:
    mocker.patch('freqtrade.optimize.hyperopt.load_data', MagicMock())
    mocker.patch('freqtrade.optimize.hyperopt.fmin', side_effect=ValueError())

    conf = deepcopy(default_conf)
    conf.update({'config': 'config.json.example'})
    conf.update({'epochs': 1})
    conf.update({'timerange': None})
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimize_conf', return_value=conf)
    mocker.patch('freqtrade.logger.Logger.set_format', MagicMock())

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


def test_resuming_previous_hyperopt_results_succeeds(mocker, default_conf) -> None:
    trials = create_trials(mocker)

    conf = deepcopy(default_conf)
    conf.update({'config': 'config.json.example'})
    conf.update({'epochs': 1})
    conf.update({'mongodb': False})
    conf.update({'timerange': None})

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
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimize_conf', return_value=conf)
    mocker.patch('freqtrade.logger.Logger.set_format', MagicMock())

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


def test_save_trials_saves_trials(mocker, caplog) -> None:
    create_trials(mocker)
    mock_dump = mocker.patch('freqtrade.optimize.hyperopt.pickle.dump', return_value=None)

    hyperopt = _HYPEROPT
    mocker.patch('freqtrade.optimize.hyperopt.open', return_value=hyperopt.trials_file)

    hyperopt.save_trials()

    assert tt.log_has(
        'Saving Trials to \'freqtrade/tests/optimize/ut_trials.pickle\'',
        caplog.record_tuples
    )
    mock_dump.assert_called_once()


def test_read_trials_returns_trials_file(mocker, default_conf, caplog) -> None:
    trials = create_trials(mocker)
    mock_load = mocker.patch('freqtrade.optimize.hyperopt.pickle.load', return_value=trials)
    mock_open = mocker.patch('freqtrade.optimize.hyperopt.open', return_value=mock_load)

    hyperopt = _HYPEROPT
    hyperopt_trial = hyperopt.read_trials()
    assert tt.log_has(
        'Reading Trials from \'freqtrade/tests/optimize/ut_trials.pickle\'',
        caplog.record_tuples
    )
    assert hyperopt_trial == trials
    mock_open.assert_called_once()
    mock_load.assert_called_once()


def test_roi_table_generation() -> None:
    params = {
        'roi_t1': 5,
        'roi_t2': 10,
        'roi_t3': 15,
        'roi_p1': 1,
        'roi_p2': 2,
        'roi_p3': 3,
    }

    hyperopt = _HYPEROPT
    assert hyperopt.generate_roi_table(params) == {'0': 6, '15': 3, '25': 1, '30': 0}


def test_start_calls_fmin(mocker, default_conf) -> None:
    trials = create_trials(mocker)
    mocker.patch('freqtrade.optimize.hyperopt.sorted', return_value=trials.results)
    mocker.patch('freqtrade.optimize.hyperopt.load_data', MagicMock())
    mock_fmin = mocker.patch('freqtrade.optimize.hyperopt.fmin', return_value={})

    conf = deepcopy(default_conf)
    conf.update({'config': 'config.json.example'})
    conf.update({'epochs': 1})
    conf.update({'mongodb': False})
    conf.update({'timerange': None})

    hyperopt = Hyperopt(conf)
    hyperopt.trials = trials
    hyperopt.tickerdata_to_dataframe = MagicMock()

    hyperopt.start()
    mock_fmin.assert_called_once()


def test_start_uses_mongotrials(mocker, default_conf) -> None:
    mocker.patch('freqtrade.optimize.hyperopt.load_data', MagicMock())
    mock_fmin = mocker.patch('freqtrade.optimize.hyperopt.fmin', return_value={})
    mock_mongotrials = mocker.patch(
        'freqtrade.optimize.hyperopt.MongoTrials',
        return_value=create_trials(mocker)
    )

    conf = deepcopy(default_conf)
    conf.update({'config': 'config.json.example'})
    conf.update({'epochs': 1})
    conf.update({'mongodb': True})
    conf.update({'timerange': None})
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimize_conf', return_value=conf)

    hyperopt = Hyperopt(conf)
    hyperopt.tickerdata_to_dataframe = MagicMock()

    hyperopt.start()
    mock_mongotrials.assert_called_once()
    mock_fmin.assert_called_once()
