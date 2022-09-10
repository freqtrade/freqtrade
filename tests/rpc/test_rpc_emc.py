"""
Unit test file for rpc/external_message_consumer.py
"""
import logging
from datetime import datetime, timezone

import pytest

from freqtrade.data.dataprovider import DataProvider
from freqtrade.rpc.external_message_consumer import ExternalMessageConsumer
from tests.conftest import log_has, log_has_re, log_has_when


@pytest.fixture
def patched_emc(default_conf, mocker):
    default_conf.update({
        "external_message_consumer": {
            "enabled": True,
            "producers": [
                {
                    "name": "default",
                    "url": "ws://127.0.0.1:8080/api/v1/message/ws",
                    "ws_token": "secret_Ws_t0ken"
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


def test_emc_init(patched_emc, default_conf, mocker, caplog):
    # Test the settings were set correctly
    assert patched_emc.initial_candle_limit <= 1500
    assert patched_emc.wait_timeout > 0
    assert patched_emc.sleep_time > 0

    default_conf.update({
        "external_message_consumer": {
            "enabled": True,
            "producers": []
        }
    })
    dataprovider = DataProvider(default_conf, None, None, None)
    with pytest.raises(ValueError) as exc:
        ExternalMessageConsumer(default_conf, dataprovider)

    # Make sure we failed because of no producers
    assert str(exc.value) == "You must specify at least 1 Producer to connect to."


# Parametrize this?
def test_emc_handle_producer_message(patched_emc, caplog, ohlcv_history):
    test_producer = {"name": "test", "url": "ws://test", "ws_token": "test"}
    producer_name = test_producer['name']

    caplog.set_level(logging.DEBUG)

    # Test handle whitelist message
    whitelist_message = {"type": "whitelist", "data": ["BTC/USDT"]}
    patched_emc.handle_producer_message(test_producer, whitelist_message)

    assert log_has(f"Received message of type `whitelist` from `{producer_name}`", caplog)
    assert log_has(
        f"Consumed message from `{producer_name}` of type `RPCMessageType.WHITELIST`", caplog)

    # Test handle analyzed_df message
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
    assert log_has(
        f"Consumed message from `{producer_name}` of type `RPCMessageType.ANALYZED_DF`", caplog)

    # Test unhandled message
    unhandled_message = {"type": "status", "data": "RUNNING"}
    patched_emc.handle_producer_message(test_producer, unhandled_message)

    assert log_has_re(r"Received unhandled message\: .*", caplog)

    # Test malformed messages
    caplog.clear()
    malformed_message = {"type": "whitelist", "data": {"pair": "BTC/USDT"}}
    patched_emc.handle_producer_message(test_producer, malformed_message)

    assert log_has_re(r"Invalid message .+", caplog)

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
    assert log_has_re(r"Invalid message .+", caplog)

    caplog.clear()
    malformed_message = {"some": "stuff"}
    patched_emc.handle_producer_message(test_producer, malformed_message)

    assert log_has_re(r"Invalid message .+", caplog)

    caplog.clear()
    malformed_message = {"type": "whitelist", "data": None}
    patched_emc.handle_producer_message(test_producer, malformed_message)

    assert log_has_re(r"Invalid message .+", caplog)
