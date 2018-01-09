# pragma pylint: disable=missing-docstring,W0212,C0103
import pickle
import os
import pytest
import freqtrade.optimize.hyperopt
from freqtrade.optimize.hyperopt import calculate_loss, TARGET_TRADES, EXPECTED_MAX_PROFIT, start, \
    log_results


def test_loss_calculation_prefer_correct_trade_count():
    correct = calculate_loss(1, TARGET_TRADES, 20)
    over = calculate_loss(1, TARGET_TRADES + 100, 20)
    under = calculate_loss(1, TARGET_TRADES - 100, 20)
    assert over > correct
    assert under > correct


def test_loss_calculation_prefer_shorter_trades():
    shorter = calculate_loss(1, 100, 20)
    longer = calculate_loss(1, 100, 30)
    assert shorter < longer


def test_loss_calculation_has_limited_profit():
    correct = calculate_loss(EXPECTED_MAX_PROFIT, TARGET_TRADES, 20)
    over = calculate_loss(EXPECTED_MAX_PROFIT * 2, TARGET_TRADES, 20)
    under = calculate_loss(EXPECTED_MAX_PROFIT / 2, TARGET_TRADES, 20)
    assert over == correct
    assert under > correct


def create_trials(mocker):
    """
    When creating trials, mock the hyperopt Trials so that *by default*
      - we don't create any pickle'd files in the filesystem
      - we might have a pickle'd file so make sure that we return
        false when looking for it
    """
    mocker.patch('freqtrade.optimize.hyperopt.TRIALS_FILE',
                 return_value='freqtrade/tests/optimize/ut_trials.pickle')
    mocker.patch('freqtrade.optimize.hyperopt.os.path.exists',
                 return_value=False)
    mocker.patch('freqtrade.optimize.hyperopt.save_trials',
                 return_value=None)
    mocker.patch('freqtrade.optimize.hyperopt.read_trials',
                 return_value=None)
    return mocker.Mock(
        results=[{
            'loss': 1,
            'result': 'foo',
            'status': 'ok'
        }],
        best_trial={'misc': {'vals': {'adx': 999}}}
    )


def test_start_calls_fmin(mocker):
    trials = create_trials(mocker)
    mocker.patch('freqtrade.optimize.hyperopt.TRIALS', return_value=trials)
    mocker.patch('freqtrade.optimize.hyperopt.sorted',
                 return_value=trials.results)
    mocker.patch('freqtrade.optimize.preprocess')
    mocker.patch('freqtrade.optimize.load_data')
    mock_fmin = mocker.patch('freqtrade.optimize.hyperopt.fmin', return_value={})

    args = mocker.Mock(epochs=1, config='config.json.example', mongodb=False)
    start(args)

    mock_fmin.assert_called_once()


def test_start_uses_mongotrials(mocker):
    mock_mongotrials = mocker.patch('freqtrade.optimize.hyperopt.MongoTrials',
                                    return_value=create_trials(mocker))
    mocker.patch('freqtrade.optimize.preprocess')
    mocker.patch('freqtrade.optimize.load_data')
    mocker.patch('freqtrade.optimize.hyperopt.fmin', return_value={})

    args = mocker.Mock(epochs=1, config='config.json.example', mongodb=True)
    start(args)

    mock_mongotrials.assert_called_once()


def test_log_results_if_loss_improves(mocker):
    logger = mocker.patch('freqtrade.optimize.hyperopt.logger.info')
    global CURRENT_BEST_LOSS
    CURRENT_BEST_LOSS = 2
    log_results({
        'loss': 1,
        'current_tries': 1,
        'total_tries': 2,
        'result': 'foo'
    })

    logger.assert_called_once()


def test_no_log_if_loss_does_not_improve(mocker):
    logger = mocker.patch('freqtrade.optimize.hyperopt.logger.info')
    global CURRENT_BEST_LOSS
    CURRENT_BEST_LOSS = 2
    log_results({
        'loss': 3,
    })

    assert not logger.called


def test_fmin_best_results(mocker, caplog):
    fmin_result = {
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
    }

    mocker.patch('freqtrade.optimize.hyperopt.MongoTrials', return_value=create_trials(mocker))
    mocker.patch('freqtrade.optimize.preprocess')
    mocker.patch('freqtrade.optimize.load_data')
    mocker.patch('freqtrade.optimize.hyperopt.fmin', return_value=fmin_result)

    args = mocker.Mock(epochs=1, config='config.json.example')
    start(args)

    exists = [
        'Best parameters',
        '"adx": {\n        "enabled": true,\n        "value": 15.0\n    },',
        '"green_candle": {\n        "enabled": true\n    },',
        '"mfi": {\n        "enabled": false\n    },',
        '"trigger": {\n        "type": "ao_cross_zero"\n    },',
        '"stoploss": -0.1',
    ]

    for line in exists:
        assert line in caplog.text


def test_fmin_throw_value_error(mocker, caplog):
    mocker.patch('freqtrade.optimize.hyperopt.MongoTrials', return_value=create_trials(mocker))
    mocker.patch('freqtrade.optimize.preprocess')
    mocker.patch('freqtrade.optimize.load_data')
    mocker.patch('freqtrade.optimize.hyperopt.fmin', side_effect=ValueError())

    args = mocker.Mock(epochs=1, config='config.json.example')
    start(args)

    exists = [
        'Best Result:',
        'Sorry, Hyperopt was not able to find good parameters. Please try with more epochs '
        '(param: -e).',
    ]

    for line in exists:
        assert line in caplog.text


def test_resuming_previous_hyperopt_results_succeeds(mocker):
    import freqtrade.optimize.hyperopt as hyperopt
    trials = create_trials(mocker)
    mocker.patch('freqtrade.optimize.hyperopt.TRIALS',
                 return_value=trials)
    mocker.patch('freqtrade.optimize.hyperopt.os.path.exists',
                 return_value=True)
    mocker.patch('freqtrade.optimize.hyperopt.len',
                 return_value=len(trials.results))
    mock_read = mocker.patch('freqtrade.optimize.hyperopt.read_trials',
                             return_value=trials)
    mock_save = mocker.patch('freqtrade.optimize.hyperopt.save_trials',
                             return_value=None)
    mocker.patch('freqtrade.optimize.hyperopt.sorted',
                 return_value=trials.results)
    mocker.patch('freqtrade.optimize.preprocess')
    mocker.patch('freqtrade.optimize.load_data')
    mocker.patch('freqtrade.optimize.hyperopt.fmin',
                 return_value={})
    args = mocker.Mock(epochs=1,
                       config='config.json.example',
                       mongodb=False)

    start(args)

    mock_read.assert_called_once()
    mock_save.assert_called_once()

    current_tries = hyperopt._CURRENT_TRIES
    total_tries = hyperopt.TOTAL_TRIES

    assert current_tries == len(trials.results)
    assert total_tries == (current_tries + len(trials.results))

