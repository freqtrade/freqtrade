"""
Unit test file for rpc/external_message_consumer.py
"""
import asyncio
import functools
import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import websockets

from freqtrade.data.dataprovider import DataProvider
from freqtrade.rpc.external_message_consumer import ExternalMessageConsumer
from tests.conftest import log_has, log_has_re, log_has_when


_TEST_WS_TOKEN = "secret_Ws_t0ken"
_TEST_WS_HOST = "127.0.0.1"
_TEST_WS_PORT = 9989


@pytest.fixture
def patched_emc(default_conf, mocker):
    default_conf.update({
        "external_message_consumer": {
            "enabled": True,
            "producers": [
                {
                    "name": "default",
                    "host": "null",
                    "port": 9891,
                    "ws_token": _TEST_WS_TOKEN
                }
            ]
        }
    })
    dataprovider = DataProvider(default_conf, None, None, None)
    emc = ExternalMessageConsumer(default_conf, dataprovider)

    try:
        yield emc
    finally:
        emc.shutdown()


def test_emc_start(patched_emc, caplog):
    # Test if the message was printed
    assert log_has_when("Starting ExternalMessageConsumer", caplog, "setup")
    # Test if the thread and loop objects were created
    assert patched_emc._thread and patched_emc._loop

    # Test we call start again nothing happens
    prev_thread = patched_emc._thread
    patched_emc.start()
    assert prev_thread == patched_emc._thread


def test_emc_shutdown(patched_emc, caplog):
    patched_emc.shutdown()

    assert log_has("Stopping ExternalMessageConsumer", caplog)
    # Test the loop has stopped
    assert patched_emc._loop is None
    # Test if the thread has stopped
    assert patched_emc._thread is None

    caplog.clear()
    patched_emc.shutdown()

    # Test func didn't run again as it was called once already
    assert not log_has("Stopping ExternalMessageConsumer", caplog)


def test_emc_init(patched_emc):
    # Test the settings were set correctly
    assert patched_emc.initial_candle_limit <= 1500
    assert patched_emc.wait_timeout > 0
    assert patched_emc.sleep_time > 0


# Parametrize this?
def test_emc_handle_producer_message(patched_emc, caplog, ohlcv_history):
    test_producer = {"name": "test", "url": "ws://test", "ws_token": "test"}
    producer_name = test_producer['name']
    invalid_msg = r"Invalid message .+"

    caplog.set_level(logging.DEBUG)

    # Test handle whitelist message
    whitelist_message = {"type": "whitelist", "data": ["BTC/USDT"]}
    patched_emc.handle_producer_message(test_producer, whitelist_message)

    assert log_has(f"Received message of type `whitelist` from `{producer_name}`", caplog)
    assert log_has(
        f"Consumed message from `{producer_name}` of type `RPCMessageType.WHITELIST`", caplog)

    # Test handle analyzed_df single candle message
    df_message = {
        "type": "analyzed_df",
        "data": {
            "key": ("BTC/USDT", "5m", "spot"),
            "df": ohlcv_history,
            "la": datetime.now(timezone.utc)
        }
    }
    patched_emc.handle_producer_message(test_producer, df_message)

    assert log_has(f"Received message of type `analyzed_df` from `{producer_name}`", caplog)
    assert log_has_re(r"Holes in data or no existing df,.+", caplog)

    # Test unhandled message
    unhandled_message = {"type": "status", "data": "RUNNING"}
    patched_emc.handle_producer_message(test_producer, unhandled_message)

    assert log_has_re(r"Received unhandled message\: .*", caplog)

    # Test malformed messages
    caplog.clear()
    malformed_message = {"type": "whitelist", "data": {"pair": "BTC/USDT"}}
    patched_emc.handle_producer_message(test_producer, malformed_message)

    assert log_has_re(invalid_msg, caplog)
    caplog.clear()

    malformed_message = {
        "type": "analyzed_df",
        "data": {
            "key": "BTC/USDT",
            "df": ohlcv_history,
            "la": datetime.now(timezone.utc)
        }
    }
    patched_emc.handle_producer_message(test_producer, malformed_message)

    assert log_has(f"Received message of type `analyzed_df` from `{producer_name}`", caplog)
    assert log_has_re(invalid_msg, caplog)
    caplog.clear()

    # Empty dataframe
    malformed_message = {
            "type": "analyzed_df",
            "data": {
                "key": ("BTC/USDT", "5m", "spot"),
                "df": ohlcv_history.loc[ohlcv_history['open'] < 0],
                "la": datetime.now(timezone.utc)
                }
            }
    patched_emc.handle_producer_message(test_producer, malformed_message)

    assert log_has(f"Received message of type `analyzed_df` from `{producer_name}`", caplog)
    assert not log_has_re(invalid_msg, caplog)
    assert log_has_re(r"Received Empty Dataframe for.+", caplog)

    caplog.clear()
    malformed_message = {"some": "stuff"}
    patched_emc.handle_producer_message(test_producer, malformed_message)

    assert log_has_re(invalid_msg, caplog)
    caplog.clear()

    caplog.clear()
    malformed_message = {"type": "whitelist", "data": None}
    patched_emc.handle_producer_message(test_producer, malformed_message)

    assert log_has_re(r"Empty message .+", caplog)


async def test_emc_create_connection_success(default_conf, caplog, mocker):
    default_conf.update({
        "external_message_consumer": {
            "enabled": True,
            "producers": [
                {
                    "name": "default",
                    "host": _TEST_WS_HOST,
                    "port": _TEST_WS_PORT,
                    "ws_token": _TEST_WS_TOKEN
                }
            ],
            "wait_timeout": 60,
            "ping_timeout": 60,
            "sleep_timeout": 60
        }
    })

    mocker.patch('freqtrade.rpc.external_message_consumer.ExternalMessageConsumer.start',
                 MagicMock())
    dp = DataProvider(default_conf, None, None, None)
    emc = ExternalMessageConsumer(default_conf, dp)

    test_producer = default_conf['external_message_consumer']['producers'][0]
    lock = asyncio.Lock()

    emc._running = True

    async def eat(websocket):
        emc._running = False

    try:
        async with websockets.serve(eat, _TEST_WS_HOST, _TEST_WS_PORT):
            await emc._create_connection(test_producer, lock)

        assert log_has_re(r"Connected to channel.+", caplog)
    finally:
        emc.shutdown()


@pytest.mark.parametrize('host,port', [
    (_TEST_WS_HOST, -1),
    ("10000.1241..2121/", _TEST_WS_PORT),
])
async def test_emc_create_connection_invalid_url(default_conf, caplog, mocker, host, port):
    default_conf.update({
        "external_message_consumer": {
            "enabled": True,
            "producers": [
                {
                    "name": "default",
                    "host": host,
                    "port": port,
                    "ws_token": _TEST_WS_TOKEN
                }
            ],
            "wait_timeout": 60,
            "ping_timeout": 60,
            "sleep_timeout": 60
        }
    })

    dp = DataProvider(default_conf, None, None, None)
    # Handle start explicitly to avoid messing with threading in tests
    mocker.patch("freqtrade.rpc.external_message_consumer.ExternalMessageConsumer.start")
    mocker.patch("freqtrade.rpc.api_server.ws.channel.create_channel")
    emc = ExternalMessageConsumer(default_conf, dp)

    try:
        emc._running = True
        await emc._create_connection(emc.producers[0], asyncio.Lock())
        assert log_has_re(r".+ is an invalid WebSocket URL .+", caplog)
    finally:
        emc.shutdown()


async def test_emc_create_connection_error(default_conf, caplog, mocker):
    default_conf.update({
        "external_message_consumer": {
            "enabled": True,
            "producers": [
                {
                    "name": "default",
                    "host": _TEST_WS_HOST,
                    "port": _TEST_WS_PORT,
                    "ws_token": _TEST_WS_TOKEN
                }
            ],
            "wait_timeout": 60,
            "ping_timeout": 60,
            "sleep_timeout": 60
        }
    })

    # Test unexpected error
    mocker.patch('websockets.connect', side_effect=RuntimeError)

    dp = DataProvider(default_conf, None, None, None)
    emc = ExternalMessageConsumer(default_conf, dp)

    try:
        await asyncio.sleep(0.05)
        assert log_has("Unexpected error has occurred:", caplog)
    finally:
        emc.shutdown()


async def test_emc_receive_messages_valid(default_conf, caplog, mocker):
    caplog.set_level(logging.DEBUG)

    default_conf.update({
        "external_message_consumer": {
            "enabled": True,
            "producers": [
                {
                    "name": "default",
                    "host": _TEST_WS_HOST,
                    "port": _TEST_WS_PORT,
                    "ws_token": _TEST_WS_TOKEN
                }
            ],
            "wait_timeout": 1,
            "ping_timeout": 60,
            "sleep_time": 60
        }
    })

    mocker.patch('freqtrade.rpc.external_message_consumer.ExternalMessageConsumer.start',
                 MagicMock())

    lock = asyncio.Lock()
    test_producer = default_conf['external_message_consumer']['producers'][0]

    dp = DataProvider(default_conf, None, None, None)
    emc = ExternalMessageConsumer(default_conf, dp)

    loop = asyncio.get_event_loop()
    def change_running(emc): emc._running = not emc._running

    class TestChannel:
        async def recv(self, *args, **kwargs):
            return {"type": "whitelist", "data": ["BTC/USDT"]}

        async def ping(self, *args, **kwargs):
            return asyncio.Future()

    try:
        change_running(emc)
        loop.call_soon(functools.partial(change_running, emc=emc))
        await emc._receive_messages(TestChannel(), test_producer, lock)

        assert log_has_re(r"Received message of type `whitelist`.+", caplog)
    finally:
        emc.shutdown()


async def test_emc_receive_messages_invalid(default_conf, caplog, mocker):
    default_conf.update({
        "external_message_consumer": {
            "enabled": True,
            "producers": [
                {
                    "name": "default",
                    "host": _TEST_WS_HOST,
                    "port": _TEST_WS_PORT,
                    "ws_token": _TEST_WS_TOKEN
                }
            ],
            "wait_timeout": 1,
            "ping_timeout": 60,
            "sleep_time": 60
        }
    })

    mocker.patch('freqtrade.rpc.external_message_consumer.ExternalMessageConsumer.start',
                 MagicMock())

    lock = asyncio.Lock()
    test_producer = default_conf['external_message_consumer']['producers'][0]

    dp = DataProvider(default_conf, None, None, None)
    emc = ExternalMessageConsumer(default_conf, dp)

    loop = asyncio.get_event_loop()
    def change_running(emc): emc._running = not emc._running

    class TestChannel:
        async def recv(self, *args, **kwargs):
            return {"type": ["BTC/USDT"]}

        async def ping(self, *args, **kwargs):
            return asyncio.Future()

    try:
        change_running(emc)
        loop.call_soon(functools.partial(change_running, emc=emc))
        await emc._receive_messages(TestChannel(), test_producer, lock)

        assert log_has_re(r"Invalid message from.+", caplog)
    finally:
        emc.shutdown()


async def test_emc_receive_messages_timeout(default_conf, caplog, mocker):
    default_conf.update({
        "external_message_consumer": {
            "enabled": True,
            "producers": [
                {
                    "name": "default",
                    "host": _TEST_WS_HOST,
                    "port": _TEST_WS_PORT,
                    "ws_token": _TEST_WS_TOKEN
                }
            ],
            "wait_timeout": 0.1,
            "ping_timeout": 1,
            "sleep_time": 1
        }
    })

    mocker.patch('freqtrade.rpc.external_message_consumer.ExternalMessageConsumer.start',
                 MagicMock())

    lock = asyncio.Lock()
    test_producer = default_conf['external_message_consumer']['producers'][0]

    dp = DataProvider(default_conf, None, None, None)
    emc = ExternalMessageConsumer(default_conf, dp)

    loop = asyncio.get_event_loop()
    def change_running(emc): emc._running = not emc._running

    class TestChannel:
        async def recv(self, *args, **kwargs):
            await asyncio.sleep(0.2)

        async def ping(self, *args, **kwargs):
            return asyncio.Future()

    try:
        change_running(emc)
        loop.call_soon(functools.partial(change_running, emc=emc))

        with pytest.raises(asyncio.TimeoutError):
            await emc._receive_messages(TestChannel(), test_producer, lock)

        assert log_has_re(r"Ping error.+", caplog)
    finally:
        emc.shutdown()


async def test_emc_receive_messages_handle_error(default_conf, caplog, mocker):
    default_conf.update({
        "external_message_consumer": {
            "enabled": True,
            "producers": [
                {
                    "name": "default",
                    "host": _TEST_WS_HOST,
                    "port": _TEST_WS_PORT,
                    "ws_token": _TEST_WS_TOKEN
                }
            ],
            "wait_timeout": 1,
            "ping_timeout": 1,
            "sleep_time": 1
        }
    })

    mocker.patch('freqtrade.rpc.external_message_consumer.ExternalMessageConsumer.start',
                 MagicMock())

    lock = asyncio.Lock()
    test_producer = default_conf['external_message_consumer']['producers'][0]

    dp = DataProvider(default_conf, None, None, None)
    emc = ExternalMessageConsumer(default_conf, dp)

    emc.handle_producer_message = MagicMock(side_effect=Exception)

    loop = asyncio.get_event_loop()
    def change_running(emc): emc._running = not emc._running

    class TestChannel:
        async def recv(self, *args, **kwargs):
            return {"type": "whitelist", "data": ["BTC/USDT"]}

        async def ping(self, *args, **kwargs):
            return asyncio.Future()

    try:
        change_running(emc)
        loop.call_soon(functools.partial(change_running, emc=emc))
        await emc._receive_messages(TestChannel(), test_producer, lock)

        assert log_has_re(r"Error handling producer message.+", caplog)
    finally:
        emc.shutdown()
