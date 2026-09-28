# pragma pylint: disable=missing-docstring, C0103
import logging
import threading
import time
from collections import deque
from unittest.mock import MagicMock

from freqtrade.enums import RPCMessageType
from freqtrade.rpc import RPCManager
from freqtrade.rpc.api_server.webserver import ApiServer
from tests.conftest import get_patched_freqtradebot, log_has


def test__init__(mocker, default_conf) -> None:
    default_conf["telegram"]["enabled"] = False

    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))
    assert rpc_manager.registered_modules == []


def test_init_telegram_disabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf["telegram"]["enabled"] = False
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    assert not log_has("Enabling rpc.telegram ...", caplog)
    assert rpc_manager.registered_modules == []


def test_init_telegram_enabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf["telegram"]["enabled"] = True
    mocker.patch("freqtrade.rpc.telegram.Telegram._init", MagicMock())
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    assert log_has("Enabling rpc.telegram ...", caplog)
    len_modules = len(rpc_manager.registered_modules)
    assert len_modules == 1
    assert "telegram" in [mod.name for mod in rpc_manager.registered_modules]


def test_cleanup_telegram_disabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    telegram_mock = mocker.patch("freqtrade.rpc.telegram.Telegram.cleanup", MagicMock())
    default_conf["telegram"]["enabled"] = False

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc_manager = RPCManager(freqtradebot)
    rpc_manager.cleanup()

    assert not log_has("Cleaning up rpc.telegram ...", caplog)
    assert telegram_mock.call_count == 0


def test_cleanup_telegram_enabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf["telegram"]["enabled"] = True
    mocker.patch("freqtrade.rpc.telegram.Telegram._init", MagicMock())
    telegram_mock = mocker.patch("freqtrade.rpc.telegram.Telegram.cleanup", MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc_manager = RPCManager(freqtradebot)

    # Check we have Telegram as a registered modules
    assert "telegram" in [mod.name for mod in rpc_manager.registered_modules]

    rpc_manager.cleanup()
    assert log_has("Cleaning up rpc.telegram ...", caplog)
    assert "telegram" not in [mod.name for mod in rpc_manager.registered_modules]
    assert telegram_mock.call_count == 1


def test_send_msg_telegram_disabled(mocker, default_conf, caplog) -> None:
    telegram_mock = mocker.patch("freqtrade.rpc.telegram.Telegram.send_msg", MagicMock())
    default_conf["telegram"]["enabled"] = False

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc_manager = RPCManager(freqtradebot)
    rpc_manager.send_msg({"type": RPCMessageType.STATUS, "status": "test"})

    assert log_has("Sending rpc message: {'type': status, 'status': 'test'}", caplog)
    assert telegram_mock.call_count == 0


def test_send_msg_telegram_error(mocker, default_conf, caplog) -> None:
    mocker.patch("freqtrade.rpc.telegram.Telegram._init", MagicMock())
    mocker.patch("freqtrade.rpc.telegram.Telegram.send_msg", side_effect=ValueError())
    default_conf["telegram"]["enabled"] = True
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc_manager = RPCManager(freqtradebot)
    rpc_manager.send_msg({"type": RPCMessageType.STATUS, "status": "test"})
    rpc_manager.flush()

    assert log_has("Sending rpc message: {'type': status, 'status': 'test'}", caplog)
    assert log_has("Exception occurred within RPC module telegram", caplog)


def test_process_msg_queue(mocker, default_conf, caplog) -> None:
    telegram_mock = mocker.patch("freqtrade.rpc.telegram.Telegram.send_msg")
    default_conf["telegram"]["enabled"] = True
    default_conf["telegram"]["allow_custom_messages"] = True
    mocker.patch("freqtrade.rpc.telegram.Telegram._init")

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc_manager = RPCManager(freqtradebot)
    queue = deque()
    queue.append("Test message")
    queue.append("Test message 2")
    rpc_manager.process_msg_queue(queue)
    rpc_manager.flush()

    assert log_has("Sending rpc strategy_msg: Test message", caplog)
    assert log_has("Sending rpc strategy_msg: Test message 2", caplog)
    assert telegram_mock.call_count == 2


def test_send_msg_telegram_enabled(mocker, default_conf, caplog) -> None:
    default_conf["telegram"]["enabled"] = True
    telegram_mock = mocker.patch("freqtrade.rpc.telegram.Telegram.send_msg")
    mocker.patch("freqtrade.rpc.telegram.Telegram._init")
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc_manager = RPCManager(freqtradebot)
    rpc_manager.send_msg({"type": RPCMessageType.STATUS, "status": "test"})
    rpc_manager.flush()

    assert log_has("Sending rpc message: {'type': status, 'status': 'test'}", caplog)
    assert telegram_mock.call_count == 1


def test_init_webhook_disabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf["telegram"]["enabled"] = False
    default_conf["webhook"] = {"enabled": False}
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    assert not log_has("Enabling rpc.webhook ...", caplog)
    assert rpc_manager.registered_modules == []


def test_init_webhook_enabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf["telegram"]["enabled"] = False
    default_conf["webhook"] = {"enabled": True, "url": "https://DEADBEEF.com"}
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    assert log_has("Enabling rpc.webhook ...", caplog)
    assert len(rpc_manager.registered_modules) == 1
    assert "webhook" in [mod.name for mod in rpc_manager.registered_modules]


def test_send_msg_webhook_CustomMessagetype(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf["telegram"]["enabled"] = False
    default_conf["webhook"] = {"enabled": True, "url": "https://DEADBEEF.com"}
    mocker.patch(
        "freqtrade.rpc.webhook.Webhook.send_msg", MagicMock(side_effect=NotImplementedError)
    )
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    assert "webhook" in [mod.name for mod in rpc_manager.registered_modules]
    rpc_manager.send_msg({"type": RPCMessageType.STARTUP, "status": "TestMessage"})
    rpc_manager.flush()
    assert log_has("Message type 'startup' not implemented by handler webhook.", caplog)


def test_startupmessages_telegram_enabled(mocker, default_conf) -> None:
    default_conf["telegram"]["enabled"] = True
    telegram_mock = mocker.patch("freqtrade.rpc.telegram.Telegram.send_msg", MagicMock())
    mocker.patch("freqtrade.rpc.telegram.Telegram._init", MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc_manager = RPCManager(freqtradebot)
    rpc_manager.startup_messages(default_conf, freqtradebot.pairlists, freqtradebot.protections)
    rpc_manager.flush()

    assert telegram_mock.call_count == 3
    assert "*Exchange:* `binance`" in telegram_mock.call_args_list[1][0][0]["status"]

    telegram_mock.reset_mock()
    default_conf["dry_run"] = True
    default_conf["whitelist"] = {"method": "VolumePairList", "config": {"number_assets": 20}}
    default_conf["_strategy_protections"] = [
        {"method": "StoplossGuard", "lookback_period": 60, "trade_limit": 2, "stop_duration": 60}
    ]
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)

    rpc_manager.startup_messages(default_conf, freqtradebot.pairlists, freqtradebot.protections)
    rpc_manager.flush()
    assert telegram_mock.call_count == 4
    assert "Dry run is enabled." in telegram_mock.call_args_list[0][0][0]["status"]
    assert "StoplossGuard" in telegram_mock.call_args_list[-1][0][0]["status"]


def test_init_apiserver_disabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    run_mock = MagicMock()
    mocker.patch("freqtrade.rpc.api_server.ApiServer.start_api", run_mock)
    default_conf["telegram"]["enabled"] = False
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    assert not log_has("Enabling rpc.api_server", caplog)
    assert rpc_manager.registered_modules == []
    assert run_mock.call_count == 0


def test_init_apiserver_enabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    run_mock = MagicMock()
    mocker.patch("freqtrade.rpc.api_server.ApiServer.start_api", run_mock)

    default_conf["telegram"]["enabled"] = False
    default_conf["api_server"] = {
        "enabled": True,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
        "username": "TestUser",
        "password": "TestPass",
    }
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    # Sleep to allow the thread to start
    time.sleep(0.5)
    assert log_has("Enabling rpc.api_server", caplog)
    assert len(rpc_manager.registered_modules) == 1
    assert "apiserver" in [mod.name for mod in rpc_manager.registered_modules]
    assert run_mock.call_count == 1
    ApiServer.shutdown()


def test_send_msg_slow_handler_does_not_block(mocker, default_conf) -> None:
    default_conf["telegram"]["enabled"] = True
    default_conf["webhook"] = {"enabled": True, "url": "https://DEADBEEF.com"}
    mocker.patch("freqtrade.rpc.telegram.Telegram._init")
    mocker.patch("freqtrade.rpc.telegram.Telegram.cleanup")
    release = threading.Event()
    webhook_mock = mocker.patch(
        "freqtrade.rpc.webhook.Webhook.send_msg", side_effect=lambda msg: release.wait(5)
    )
    telegram_mock = mocker.patch("freqtrade.rpc.telegram.Telegram.send_msg")
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    start = time.monotonic()
    rpc_manager.send_msg({"type": RPCMessageType.STATUS, "status": "test"})
    rpc_manager.send_msg({"type": RPCMessageType.STATUS, "status": "test2"})
    assert time.monotonic() - start < 1

    # Telegram is not held up by the blocked webhook
    rpc_manager._queues["telegram"].join()
    assert telegram_mock.call_count == 2
    assert webhook_mock.call_count <= 1

    release.set()
    rpc_manager.flush()
    assert webhook_mock.call_count == 2
    # Order is preserved per handler
    assert [c[0][0]["status"] for c in webhook_mock.call_args_list] == ["test", "test2"]
    rpc_manager.cleanup()


def test_send_msg_worker_survives_exception(mocker, default_conf, caplog) -> None:
    default_conf["telegram"]["enabled"] = True
    mocker.patch("freqtrade.rpc.telegram.Telegram._init")
    mocker.patch("freqtrade.rpc.telegram.Telegram.cleanup")
    telegram_mock = mocker.patch(
        "freqtrade.rpc.telegram.Telegram.send_msg", side_effect=[ValueError(), None]
    )
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    rpc_manager.send_msg({"type": RPCMessageType.STATUS, "status": "test"})
    rpc_manager.send_msg({"type": RPCMessageType.STATUS, "status": "test2"})
    rpc_manager.flush()
    assert log_has("Exception occurred within RPC module telegram", caplog)
    assert telegram_mock.call_count == 2
    rpc_manager.cleanup()


def test_cleanup_delivers_pending_messages(mocker, default_conf, caplog) -> None:
    default_conf["telegram"]["enabled"] = True
    mocker.patch("freqtrade.rpc.telegram.Telegram._init")
    calls = []
    mocker.patch(
        "freqtrade.rpc.telegram.Telegram.send_msg",
        side_effect=lambda msg: (time.sleep(0.05), calls.append(msg["status"])),
    )
    mocker.patch(
        "freqtrade.rpc.telegram.Telegram.cleanup", side_effect=lambda: calls.append("cleanup")
    )
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))
    worker = rpc_manager._workers["telegram"]
    assert worker.name == "FTRPC-telegram"

    rpc_manager.send_msg({"type": RPCMessageType.STATUS, "status": "test"})
    rpc_manager.send_msg({"type": RPCMessageType.STATUS, "status": "test2"})
    rpc_manager.cleanup()

    assert calls == ["test", "test2", "cleanup"]
    assert not worker.is_alive()
    assert rpc_manager._queues == {}


def test_cleanup_timeout(mocker, default_conf, caplog) -> None:
    default_conf["telegram"]["enabled"] = True
    mocker.patch("freqtrade.rpc.telegram.Telegram._init")
    cleanup_mock = mocker.patch("freqtrade.rpc.telegram.Telegram.cleanup")
    sending = threading.Event()
    release = threading.Event()
    send_mock = mocker.patch(
        "freqtrade.rpc.telegram.Telegram.send_msg",
        side_effect=lambda m: (sending.set(), release.wait(5)),
    )
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    rpc_manager.send_msg({"type": RPCMessageType.STATUS, "status": "test"})
    rpc_manager.send_msg({"type": RPCMessageType.STATUS, "status": "test2"})
    assert sending.wait(5)
    # Simulate a worker which doesn't finish within either timeout
    worker = rpc_manager._workers["telegram"]
    worker_mock = MagicMock(is_alive=worker.is_alive)
    rpc_manager._workers["telegram"] = worker_mock
    rpc_manager.cleanup()

    assert worker_mock.join.call_count == 2
    assert 9 < worker_mock.join.call_args_list[0][1]["timeout"] <= 10
    assert worker_mock.join.call_args_list[1][1]["timeout"] == 10
    assert log_has(
        "RPC module telegram did not finish sending pending messages - discarding 1 messages.",
        caplog,
    )
    assert log_has("RPC module telegram is still sending - cleaning up anyway.", caplog)
    assert cleanup_mock.call_count == 1
    release.set()
    worker.join(5)
    assert not worker.is_alive()
    # "test2" was discarded
    assert send_mock.call_count == 1


def test_send_msg_queue_warning(mocker, default_conf, caplog) -> None:
    default_conf["telegram"]["enabled"] = True
    mocker.patch("freqtrade.rpc.telegram.Telegram._init")
    mocker.patch("freqtrade.rpc.telegram.Telegram.cleanup")
    release = threading.Event()
    mocker.patch("freqtrade.rpc.telegram.Telegram.send_msg", side_effect=lambda m: release.wait(5))
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    for _ in range(101):
        rpc_manager.send_msg({"type": RPCMessageType.STATUS, "status": "test"})
    assert log_has("RPC module telegram is slow - 100 messages pending.", caplog)
    release.set()
    rpc_manager.cleanup()


def test_send_msg_apiserver_inline(mocker, default_conf) -> None:
    mocker.patch("freqtrade.rpc.api_server.ApiServer.start_api")
    default_conf["telegram"]["enabled"] = False
    default_conf["api_server"] = {
        "enabled": True,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
        "username": "TestUser",
        "password": "TestPass",
    }
    send_mock = mocker.patch("freqtrade.rpc.api_server.ApiServer.send_msg")
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))
    assert rpc_manager._queues == {}

    rpc_manager.send_msg({"type": RPCMessageType.STATUS, "status": "test"})
    # Delivered synchronously - no flush needed
    assert send_mock.call_count == 1
    ApiServer.shutdown()
