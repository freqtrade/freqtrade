"""
Unit test file for rpc/client_manager.py
"""

import logging
from copy import deepcopy
from unittest.mock import MagicMock

from freqtrade.clients.client_manager import ClientManager
from freqtrade.clients.rpc.telegram import Telegram
from freqtrade.tests.conftest import log_has, get_patched_freqtradebot


def test_client_manager_object() -> None:
    """
    Test the Arguments object has the mandatory methods
    :return: None
    """
    assert hasattr(ClientManager, '_init')
    assert hasattr(ClientManager, 'send_msg')
    assert hasattr(ClientManager, 'cleanup')


def test__init__(mocker, default_conf) -> None:
    """
    Test __init__() method
    """
    init_mock = mocker.patch('freqtrade.clients.client_manager.ClientManager._init', MagicMock())
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)

    client_manager = ClientManager(freqtradebot)
    assert client_manager.freqtrade == freqtradebot
    assert client_manager.registered_modules == []
    assert client_manager.telegram is None
    assert init_mock.call_count == 1


def test_init_telegram_disabled(mocker, default_conf, caplog) -> None:
    """
    Test _init() method with Telegram disabled
    """
    caplog.set_level(logging.DEBUG)

    conf = deepcopy(default_conf)
    conf['telegram']['enabled'] = False

    freqtradebot = get_patched_freqtradebot(mocker, conf)
    client_manager = ClientManager(freqtradebot)

    assert not log_has('Enabling rpc.telegram ...', caplog.record_tuples)
    assert client_manager.registered_modules == []
    assert client_manager.telegram is None


def test_init_telegram_enabled(mocker, default_conf, caplog) -> None:
    """
    Test _init() method with Telegram enabled
    """
    caplog.set_level(logging.DEBUG)
    mocker.patch('freqtrade.clients.rpc.telegram.Telegram._init', MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    client_manager = ClientManager(freqtradebot)

    assert log_has('Enabling rpc.telegram ...', caplog.record_tuples)
    len_modules = len(client_manager.registered_modules)
    assert len_modules == 1
    assert 'telegram' in client_manager.registered_modules
    assert isinstance(client_manager.telegram, Telegram)


def test_cleanup_telegram_disabled(mocker, default_conf, caplog) -> None:
    """
    Test cleanup() method with Telegram disabled
    """
    caplog.set_level(logging.DEBUG)
    telegram_mock = mocker.patch('freqtrade.clients.rpc.telegram.Telegram.cleanup', MagicMock())

    conf = deepcopy(default_conf)
    conf['telegram']['enabled'] = False

    freqtradebot = get_patched_freqtradebot(mocker, conf)
    client_manager = ClientManager(freqtradebot)
    client_manager.cleanup()

    assert not log_has('Cleaning up rpc.telegram ...', caplog.record_tuples)
    assert telegram_mock.call_count == 0


def test_cleanup_telegram_enabled(mocker, default_conf, caplog) -> None:
    """
    Test cleanup() method with Telegram enabled
    """
    caplog.set_level(logging.DEBUG)
    mocker.patch('freqtrade.clients.rpc.telegram.Telegram._init', MagicMock())
    telegram_mock = mocker.patch('freqtrade.clients.rpc.telegram.Telegram.cleanup', MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    client_manager = ClientManager(freqtradebot)

    # Check we have Telegram as a registered modules
    assert 'telegram' in client_manager.registered_modules

    client_manager.cleanup()
    assert log_has('Cleaning up rpc.telegram ...', caplog.record_tuples)
    assert 'telegram' not in client_manager.registered_modules
    assert telegram_mock.call_count == 1


def test_send_msg_telegram_disabled(mocker, default_conf, caplog) -> None:
    """
    Test send_msg() method with Telegram disabled
    """
    telegram_mock = mocker.patch('freqtrade.clients.rpc.telegram.Telegram.send_msg', MagicMock())

    conf = deepcopy(default_conf)
    conf['telegram']['enabled'] = False

    freqtradebot = get_patched_freqtradebot(mocker, conf)
    client_manager = ClientManager(freqtradebot)
    client_manager.send_msg('test')

    assert log_has('test', caplog.record_tuples)
    assert telegram_mock.call_count == 0


def test_send_msg_telegram_enabled(mocker, default_conf, caplog) -> None:
    """
    Test send_msg() method with Telegram disabled
    """
    telegram_mock = mocker.patch('freqtrade.clients.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.clients.rpc.telegram.Telegram._init', MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    client_manager = ClientManager(freqtradebot)
    client_manager.send_msg('test')

    assert log_has('test', caplog.record_tuples)
    assert telegram_mock.call_count == 1
