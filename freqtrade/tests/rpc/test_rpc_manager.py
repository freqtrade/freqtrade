# pragma pylint: disable=missing-docstring, C0103

import logging
from unittest.mock import MagicMock

from freqtrade.rpc import RPCMessageType, RPCManager
from freqtrade.tests.conftest import log_has, get_patched_freqtradebot


def test__init__(mocker, default_conf) -> None:
    default_conf['telegram']['enabled'] = False

    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))
    assert rpc_manager.registered_modules == []


def test_init_telegram_disabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf['telegram']['enabled'] = False
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    assert not log_has('Enabling rpc.telegram ...', caplog.record_tuples)
    assert rpc_manager.registered_modules == []


def test_init_telegram_enabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    assert log_has('Enabling rpc.telegram ...', caplog.record_tuples)
    len_modules = len(rpc_manager.registered_modules)
    assert len_modules == 1
    assert 'telegram' in [mod.name for mod in rpc_manager.registered_modules]


def test_cleanup_telegram_disabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    telegram_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.cleanup', MagicMock())
    default_conf['telegram']['enabled'] = False

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc_manager = RPCManager(freqtradebot)
    rpc_manager.cleanup()

    assert not log_has('Cleaning up rpc.telegram ...', caplog.record_tuples)
    assert telegram_mock.call_count == 0


def test_cleanup_telegram_enabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    telegram_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.cleanup', MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc_manager = RPCManager(freqtradebot)

    # Check we have Telegram as a registered modules
    assert 'telegram' in [mod.name for mod in rpc_manager.registered_modules]

    rpc_manager.cleanup()
    assert log_has('Cleaning up rpc.telegram ...', caplog.record_tuples)
    assert 'telegram' not in [mod.name for mod in rpc_manager.registered_modules]
    assert telegram_mock.call_count == 1


def test_send_msg_telegram_disabled(mocker, default_conf, caplog) -> None:
    telegram_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    default_conf['telegram']['enabled'] = False

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc_manager = RPCManager(freqtradebot)
    rpc_manager.send_msg({
        'type': RPCMessageType.STATUS_NOTIFICATION,
        'status': 'test'
    })

    assert log_has("Sending rpc message: {'type': status, 'status': 'test'}", caplog.record_tuples)
    assert telegram_mock.call_count == 0


def test_send_msg_telegram_enabled(mocker, default_conf, caplog) -> None:
    telegram_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc_manager = RPCManager(freqtradebot)
    rpc_manager.send_msg({
        'type': RPCMessageType.STATUS_NOTIFICATION,
        'status': 'test'
    })

    assert log_has("Sending rpc message: {'type': status, 'status': 'test'}", caplog.record_tuples)
    assert telegram_mock.call_count == 1


def test_init_webhook_disabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf['telegram']['enabled'] = False
    default_conf['webhook'] = {'enabled': False}
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    assert not log_has('Enabling rpc.webhook ...', caplog.record_tuples)
    assert rpc_manager.registered_modules == []


def test_init_webhook_enabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf['telegram']['enabled'] = False
    default_conf['webhook'] = {'enabled': True, 'url': "https://DEADBEEF.com"}
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    assert log_has('Enabling rpc.webhook ...', caplog.record_tuples)
    assert len(rpc_manager.registered_modules) == 1
    assert 'webhook' in [mod.name for mod in rpc_manager.registered_modules]


def test_startupmessages_telegram_enabled(mocker, default_conf, caplog) -> None:
    telegram_mock = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc_manager = RPCManager(freqtradebot)
    rpc_manager.startup_messages(default_conf, freqtradebot.pairlists)

    assert telegram_mock.call_count == 3
    assert "*Exchange:* `bittrex`" in telegram_mock.call_args_list[1][0][0]['status']

    telegram_mock.reset_mock()
    default_conf['dry_run'] = True
    default_conf['whitelist'] = {'method': 'VolumePairList',
                                 'config': {'number_assets': 20}
                                 }

    rpc_manager.startup_messages(default_conf,  freqtradebot.pairlists)
    assert telegram_mock.call_count == 3
    assert "Dry run is enabled." in telegram_mock.call_args_list[0][0][0]['status']
