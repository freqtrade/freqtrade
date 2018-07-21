"""
Unit test file for main.py
"""

from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from freqtrade import OperationalException
from freqtrade.arguments import Arguments
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.main import main, reconfigure
from freqtrade.state import State
from freqtrade.tests.conftest import log_has, patch_exchange


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
    assert call_args.loglevel == 0
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
    assert call_args.loglevel == 0
    assert call_args.subparser == 'hyperopt'
    assert call_args.func is not None


def test_main_fatal_exception(mocker, default_conf, caplog) -> None:
    """
    Test main() function
    In this test we are skipping the while True loop by throwing an exception.
    """
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.FreqtradeBot',
        _init_modules=MagicMock(),
        worker=MagicMock(side_effect=Exception),
        cleanup=MagicMock(),
    )
    mocker.patch(
        'freqtrade.configuration.Configuration._load_config_file',
        lambda *args, **kwargs: default_conf
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    args = ['-c', 'config.json.example']

    # Test Main + the KeyboardInterrupt exception
    with pytest.raises(SystemExit):
        main(args)
    assert log_has('Using config: config.json.example ...', caplog.record_tuples)
    assert log_has('Fatal exception!', caplog.record_tuples)


def test_main_keyboard_interrupt(mocker, default_conf, caplog) -> None:
    """
    Test main() function
    In this test we are skipping the while True loop by throwing an exception.
    """
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.FreqtradeBot',
        _init_modules=MagicMock(),
        worker=MagicMock(side_effect=KeyboardInterrupt),
        cleanup=MagicMock(),
    )
    mocker.patch(
        'freqtrade.configuration.Configuration._load_config_file',
        lambda *args, **kwargs: default_conf
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    args = ['-c', 'config.json.example']

    # Test Main + the KeyboardInterrupt exception
    with pytest.raises(SystemExit):
        main(args)
    assert log_has('Using config: config.json.example ...', caplog.record_tuples)
    assert log_has('SIGINT received, aborting ...', caplog.record_tuples)


def test_main_operational_exception(mocker, default_conf, caplog) -> None:
    """
    Test main() function
    In this test we are skipping the while True loop by throwing an exception.
    """
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.FreqtradeBot',
        _init_modules=MagicMock(),
        worker=MagicMock(side_effect=OperationalException('Oh snap!')),
        cleanup=MagicMock(),
    )
    mocker.patch(
        'freqtrade.configuration.Configuration._load_config_file',
        lambda *args, **kwargs: default_conf
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    args = ['-c', 'config.json.example']

    # Test Main + the KeyboardInterrupt exception
    with pytest.raises(SystemExit):
        main(args)
    assert log_has('Using config: config.json.example ...', caplog.record_tuples)
    assert log_has('Oh snap!', caplog.record_tuples)


def test_main_reload_conf(mocker, default_conf, caplog) -> None:
    """
    Test main() function
    In this test we are skipping the while True loop by throwing an exception.
    """
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.FreqtradeBot',
        _init_modules=MagicMock(),
        worker=MagicMock(return_value=State.RELOAD_CONF),
        cleanup=MagicMock(),
    )
    mocker.patch(
        'freqtrade.configuration.Configuration._load_config_file',
        lambda *args, **kwargs: default_conf
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    # Raise exception as side effect to avoid endless loop
    reconfigure_mock = mocker.patch(
        'freqtrade.main.reconfigure', MagicMock(side_effect=Exception)
    )

    with pytest.raises(SystemExit):
        main(['-c', 'config.json.example'])

    assert reconfigure_mock.call_count == 1
    assert log_has('Using config: config.json.example ...', caplog.record_tuples)


def test_reconfigure(mocker, default_conf) -> None:
    """ Test recreate() function """
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.freqtradebot.FreqtradeBot',
        _init_modules=MagicMock(),
        worker=MagicMock(side_effect=OperationalException('Oh snap!')),
        cleanup=MagicMock(),
    )
    mocker.patch(
        'freqtrade.configuration.Configuration._load_config_file',
        lambda *args, **kwargs: default_conf
    )
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())

    freqtrade = FreqtradeBot(default_conf)

    # Renew mock to return modified data
    conf = deepcopy(default_conf)
    conf['stake_amount'] += 1
    mocker.patch(
        'freqtrade.configuration.Configuration._load_config_file',
        lambda *args, **kwargs: conf
    )

    # reconfigure should return a new instance
    freqtrade2 = reconfigure(
        freqtrade,
        Arguments(['-c', 'config.json.example'], '').get_parsed_arg()
    )

    # Verify we have a new instance with the new config
    assert freqtrade is not freqtrade2
    assert freqtrade.config['stake_amount'] + 1 == freqtrade2.config['stake_amount']
