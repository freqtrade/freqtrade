# pragma pylint: disable=missing-docstring, too-many-arguments, too-many-ancestors, C0103
from unittest.mock import MagicMock

from copy import deepcopy

from freqtrade.rpc import init, cleanup, send_msg


def test_init_telegram_enabled(default_conf, mocker):
    module_list = []
    mocker.patch('freqtrade.rpc.REGISTERED_MODULES', module_list)
    telegram_mock = mocker.patch('freqtrade.rpc.telegram.init', MagicMock())

    init(default_conf)

    assert telegram_mock.call_count == 1
    assert 'telegram' in module_list


def test_init_telegram_disabled(default_conf, mocker):
    module_list = []
    mocker.patch('freqtrade.rpc.REGISTERED_MODULES', module_list)
    telegram_mock = mocker.patch('freqtrade.rpc.telegram.init', MagicMock())

    conf = deepcopy(default_conf)
    conf['telegram']['enabled'] = False
    init(conf)

    assert telegram_mock.call_count == 0
    assert 'telegram' not in module_list


def test_cleanup_telegram_enabled(mocker):
    mocker.patch('freqtrade.rpc.REGISTERED_MODULES', ['telegram'])
    telegram_mock = mocker.patch('freqtrade.rpc.telegram.cleanup', MagicMock())
    cleanup()
    assert telegram_mock.call_count == 1


def test_cleanup_telegram_disabled(mocker):
    mocker.patch('freqtrade.rpc.REGISTERED_MODULES', [])
    telegram_mock = mocker.patch('freqtrade.rpc.telegram.cleanup', MagicMock())
    cleanup()
    assert telegram_mock.call_count == 0


def test_send_msg_telegram_enabled(mocker):
    mocker.patch('freqtrade.rpc.REGISTERED_MODULES', ['telegram'])
    telegram_mock = mocker.patch('freqtrade.rpc.telegram.send_msg', MagicMock())
    send_msg('test')
    assert telegram_mock.call_count == 1


def test_send_msg_telegram_disabled(mocker):
    mocker.patch('freqtrade.rpc.REGISTERED_MODULES', [])
    telegram_mock = mocker.patch('freqtrade.rpc.telegram.send_msg', MagicMock())
    send_msg('test')
    assert telegram_mock.call_count == 0
