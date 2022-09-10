"""
Unit test file for rpc/external_message_consumer.py
"""
import pytest

from freqtrade.data.dataprovider import DataProvider
from freqtrade.rpc.external_message_consumer import ExternalMessageConsumer
from tests.conftest import log_has, log_has_when


@pytest.fixture(autouse=True)
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
