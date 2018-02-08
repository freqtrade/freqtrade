# pragma pylint: disable=missing-docstring,W0212,C0103
import logging

from unittest.mock import MagicMock

import pandas as pd

from freqtrade.optimize.hyperopt import calculate_loss, TARGET_TRADES, EXPECTED_MAX_PROFIT, start, \
    log_results, save_trials, read_trials, generate_roi_table

import freqtrade.optimize.hyperopt as hyperopt


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
    mocker.patch('freqtrade.optimize.hyperopt.os.remove',
                 return_value=True)
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
    mocker.patch('freqtrade.optimize.tickerdata_to_dataframe')
    mocker.patch('freqtrade.optimize.hyperopt.TRIALS', return_value=trials)
    mocker.patch('freqtrade.optimize.hyperopt.sorted',
                 return_value=trials.results)
    mocker.patch('freqtrade.optimize.preprocess')
    mocker.patch('freqtrade.optimize.load_data')
    mock_fmin = mocker.patch('freqtrade.optimize.hyperopt.fmin', return_value={})

    args = mocker.Mock(epochs=1, config='config.json.example', mongodb=False,
                       timerange=None)
    start(args)

    mock_fmin.assert_called_once()


def test_start_uses_mongotrials(mocker):
    mock_mongotrials = mocker.patch('freqtrade.optimize.hyperopt.MongoTrials',
                                    return_value=create_trials(mocker))
    mocker.patch('freqtrade.optimize.tickerdata_to_dataframe')
    mocker.patch('freqtrade.optimize.load_data')
    mocker.patch('freqtrade.optimize.hyperopt.fmin', return_value={})

    args = mocker.Mock(epochs=1, config='config.json.example', mongodb=True,
                       timerange=None)
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
    caplog.set_level(logging.INFO)
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

    mocker.patch('freqtrade.optimize.hyperopt.MongoTrials', return_value=create_trials(mocker))
    mocker.patch('freqtrade.optimize.tickerdata_to_dataframe')
    mocker.patch('freqtrade.optimize.load_data')
    mocker.patch('freqtrade.optimize.hyperopt.fmin', return_value=fmin_result)

    args = mocker.Mock(epochs=1, config='config.json.example',
                       timerange=None)
    start(args)

    exists = [
        'Best parameters',
        '"adx": {\n        "enabled": true,\n        "value": 15.0\n    },',
        '"green_candle": {\n        "enabled": true\n    },',
        '"mfi": {\n        "enabled": false\n    },',
        '"trigger": {\n        "type": "faststoch10"\n    },',
        '"stoploss": -0.1',
    ]

    for line in exists:
        assert line in caplog.text


def test_fmin_throw_value_error(mocker, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch('freqtrade.optimize.hyperopt.MongoTrials', return_value=create_trials(mocker))
    mocker.patch('freqtrade.optimize.tickerdata_to_dataframe')
    mocker.patch('freqtrade.optimize.load_data')
    mocker.patch('freqtrade.optimize.hyperopt.fmin', side_effect=ValueError())

    args = mocker.Mock(epochs=1, config='config.json.example',
                       timerange=None)
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
                       mongodb=False,
                       timerange=None)

    start(args)

    mock_read.assert_called_once()
    mock_save.assert_called_once()

    current_tries = hyperopt._CURRENT_TRIES
    total_tries = hyperopt.TOTAL_TRIES

    assert current_tries == len(trials.results)
    assert total_tries == (current_tries + len(trials.results))


def test_save_trials_saves_trials(mocker):
    trials = create_trials(mocker)
    mock_dump = mocker.patch('freqtrade.optimize.hyperopt.pickle.dump',
                             return_value=None)
    trials_path = mocker.patch('freqtrade.optimize.hyperopt.TRIALS_FILE',
                               return_value='ut_trials.pickle')
    mocker.patch('freqtrade.optimize.hyperopt.open',
                 return_value=trials_path)
    save_trials(trials, trials_path)

    mock_dump.assert_called_once_with(trials, trials_path)


def test_read_trials_returns_trials_file(mocker):
    trials = create_trials(mocker)
    mock_load = mocker.patch('freqtrade.optimize.hyperopt.pickle.load',
                             return_value=trials)
    mock_open = mocker.patch('freqtrade.optimize.hyperopt.open',
                             return_value=mock_load)

    assert read_trials() == trials
    mock_open.assert_called_once()
    mock_load.assert_called_once()


def test_roi_table_generation():
    params = {
        'roi_t1': 5,
        'roi_t2': 10,
        'roi_t3': 15,
        'roi_p1': 1,
        'roi_p2': 2,
        'roi_p3': 3,
    }
    assert generate_roi_table(params) == {'0': 6, '15': 3, '25': 1, '30': 0}


# test log_trials_result
# test buy_strategy_generator def populate_buy_trend
# test optimizer if 'ro_t1' in params

def test_format_results():
    trades = [('BTC_ETH', 2, 2, 123),
              ('BTC_LTC', 1, 1, 123),
              ('BTC_XRP', -1, -2, -246)]
    labels = ['currency', 'profit_percent', 'profit_BTC', 'duration']
    df = pd.DataFrame.from_records(trades, columns=labels)
    x = hyperopt.format_results(df)
    assert x.find(' 66.67%')


def test_signal_handler(mocker):
    m = MagicMock()
    mocker.patch('sys.exit', m)
    mocker.patch('freqtrade.optimize.hyperopt.save_trials', m)
    mocker.patch('freqtrade.optimize.hyperopt.log_trials_result', m)
    hyperopt.signal_handler(9, None)
    assert m.call_count == 3
