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
