"""
Unit test file for main.py
"""

import logging
from unittest.mock import MagicMock

import pytest

from freqtrade.main import main, set_loggers
from freqtrade.tests.conftest import log_has


def test_parse_args_backtesting(mocker) -> None:
    """
    Test that main() can start backtesting and also ensure we can pass some specific arguments
    further argument parsing is done in test_arguments.py
    """
    backtesting_mock = mocker.patch('freqtrade.optimize.backtesting.start', MagicMock())
    main(['backtesting'])
    assert backtesting_mock.call_count == 1
    call_args = backtesting_mock.call_args[0][0]
    assert call_args.config == 'config.json'
    assert call_args.live is False
    assert call_args.loglevel == 20
    assert call_args.subparser == 'backtesting'
    assert call_args.func is not None
    assert call_args.ticker_interval is None


def test_main_start_hyperopt(mocker) -> None:
    """
    Test that main() can start hyperopt
    """
    hyperopt_mock = mocker.patch('freqtrade.optimize.hyperopt.start', MagicMock())
    main(['hyperopt'])
    assert hyperopt_mock.call_count == 1
    call_args = hyperopt_mock.call_args[0][0]
    assert call_args.config == 'config.json'
    assert call_args.loglevel == 20
    assert call_args.subparser == 'hyperopt'
    assert call_args.func is not None


def test_set_loggers() -> None:
    """
    Test set_loggers() update the logger level for third-party libraries
    """
    previous_value1 = logging.getLogger('requests.packages.urllib3').level
    previous_value2 = logging.getLogger('telegram').level

    set_loggers()

    value1 = logging.getLogger('requests.packages.urllib3').level
    assert previous_value1 is not value1
    assert value1 is logging.INFO

    value2 = logging.getLogger('telegram').level
    assert previous_value2 is not value2
    assert value2 is logging.INFO


def test_main(mocker, caplog) -> None:
    """
    Test main() function
    In this test we are skipping the while True loop by throwing an exception.
    """
    mocker.patch.multiple(
        'freqtrade.freqtradebot.FreqtradeBot',
        _init_modules=MagicMock(),
        worker=MagicMock(
            side_effect=KeyboardInterrupt
        ),
        clean=MagicMock(),
    )
    args = ['-c', 'config.json.example']

    # Test Main + the KeyboardInterrupt exception
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        main(args)
        log_has('Starting freqtrade', caplog.record_tuples)
        log_has('Got SIGINT, aborting ...', caplog.record_tuples)
        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == 42

    # Test the BaseException case
    mocker.patch(
        'freqtrade.freqtradebot.FreqtradeBot.worker',
        MagicMock(side_effect=BaseException)
    )
    with pytest.raises(SystemExit):
        main(args)
        log_has('Got fatal exception!', caplog.record_tuples)
