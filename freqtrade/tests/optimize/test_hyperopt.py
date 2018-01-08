# pragma pylint: disable=missing-docstring,W0212,C0103

from freqtrade.optimize.hyperopt import calculate_loss, TARGET_TRADES, EXPECTED_MAX_PROFIT, start, \
    log_results


def test_loss_calculation_prefer_correct_trade_count():
    correct = calculate_loss(1, TARGET_TRADES)
    over = calculate_loss(1, TARGET_TRADES + 100)
    under = calculate_loss(1, TARGET_TRADES - 100)
    assert over > correct
    assert under > correct


def test_loss_calculation_has_limited_profit():
    correct = calculate_loss(EXPECTED_MAX_PROFIT, TARGET_TRADES)
    over = calculate_loss(EXPECTED_MAX_PROFIT * 2, TARGET_TRADES)
    under = calculate_loss(EXPECTED_MAX_PROFIT / 2, TARGET_TRADES)
    assert over == correct
    assert under > correct


def create_trials(mocker):
    return mocker.Mock(
        results=[{
            'loss': 1,
            'result': 'foo'
        }]
    )


def test_start_calls_fmin(mocker):
    mocker.patch('freqtrade.optimize.hyperopt.Trials', return_value=create_trials(mocker))
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
      "uptrend_sma": 0
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
        '"trigger": {\n        "type": "ao_cross_zero"\n    },'
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
