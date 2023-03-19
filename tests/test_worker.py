import logging
import time
from datetime import timedelta
from unittest.mock import MagicMock, PropertyMock

import time_machine

from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import State
from freqtrade.worker import Worker
from tests.conftest import EXMS, get_patched_worker, log_has, log_has_re


def test_worker_state(mocker, default_conf, markets) -> None:
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    worker = get_patched_worker(mocker, default_conf)
    assert worker.freqtrade.state is State.RUNNING

    default_conf.pop('initial_state')
    worker = Worker(args=None, config=default_conf)
    assert worker.freqtrade.state is State.STOPPED


def test_worker_running(mocker, default_conf, caplog) -> None:
    mock_throttle = MagicMock()
    mocker.patch('freqtrade.worker.Worker._throttle', mock_throttle)
    mocker.patch('freqtrade.persistence.Trade.stoploss_reinitialization', MagicMock())

    worker = get_patched_worker(mocker, default_conf)

    state = worker._worker(old_state=None)
    assert state is State.RUNNING
    assert log_has('Changing state to: RUNNING', caplog)
    assert mock_throttle.call_count == 1
    # Check strategy is loaded, and received a dataprovider object
    assert worker.freqtrade.strategy
    assert worker.freqtrade.strategy.dp
    assert isinstance(worker.freqtrade.strategy.dp, DataProvider)


def test_worker_stopped(mocker, default_conf, caplog) -> None:
    mock_throttle = MagicMock()
    mocker.patch('freqtrade.worker.Worker._throttle', mock_throttle)

    worker = get_patched_worker(mocker, default_conf)
    worker.freqtrade.state = State.STOPPED
    state = worker._worker(old_state=State.RUNNING)
    assert state is State.STOPPED
    assert log_has('Changing state from RUNNING to: STOPPED', caplog)
    assert mock_throttle.call_count == 1


def test_throttle(mocker, default_conf, caplog) -> None:
    def throttled_func():
        return 42

    caplog.set_level(logging.DEBUG)
    worker = get_patched_worker(mocker, default_conf)

    start = time.time()
    result = worker._throttle(throttled_func, throttle_secs=0.1)
    end = time.time()

    assert result == 42
    assert 0.3 > end - start > 0.1
    assert log_has_re(r"Throttling with 'throttled_func\(\)': sleep for \d\.\d{2} s.*", caplog)

    result = worker._throttle(throttled_func, throttle_secs=-1)
    assert result == 42


def test_throttle_sleep_time(mocker, default_conf, caplog) -> None:

    caplog.set_level(logging.DEBUG)
    worker = get_patched_worker(mocker, default_conf)
    sleep_mock = mocker.patch("freqtrade.worker.Worker._sleep")
    with time_machine.travel("2022-09-01 05:00:00 +00:00") as t:
        def throttled_func(x=1):
            t.shift(timedelta(seconds=x))
            return 42

        assert worker._throttle(throttled_func, throttle_secs=5) == 42
        # This moves the clock by 1 second
        assert sleep_mock.call_count == 1
        assert 3.8 < sleep_mock.call_args[0][0] < 4.1

        sleep_mock.reset_mock()
        # This moves the clock by 1 second
        assert worker._throttle(throttled_func, throttle_secs=10) == 42
        assert sleep_mock.call_count == 1
        assert 8.8 < sleep_mock.call_args[0][0] < 9.1

        sleep_mock.reset_mock()
        # This moves the clock by 5 second, so we only throttle by 5s
        assert worker._throttle(throttled_func, throttle_secs=10, x=5) == 42
        assert sleep_mock.call_count == 1
        assert 4.8 < sleep_mock.call_args[0][0] < 5.1

        t.move_to("2022-09-01 05:01:00 +00:00")
        sleep_mock.reset_mock()
        # Throttle for more than 5m (1 timeframe)
        assert worker._throttle(throttled_func, throttle_secs=400, x=5) == 42
        assert sleep_mock.call_count == 1
        assert 394.8 < sleep_mock.call_args[0][0] < 395.1

        t.move_to("2022-09-01 05:01:00 +00:00")

        sleep_mock.reset_mock()
        # Throttle for more than 5m (1 timeframe)
        assert worker._throttle(throttled_func, throttle_secs=400, timeframe='5m',
                                timeframe_offset=0.4, x=5) == 42
        assert sleep_mock.call_count == 1
        # 300 (5m) - 60 (1m - see set time above) - 5 (duration of throttled_func) = 235
        assert 235.2 < sleep_mock.call_args[0][0] < 235.6

        t.move_to("2022-09-01 05:04:51 +00:00")
        sleep_mock.reset_mock()
        # Offset of 5s, so we hit the sweet-spot between "candle" and "candle offset"
        # Which should not get a throttle iteration to avoid late candle fetching
        assert worker._throttle(throttled_func, throttle_secs=10, timeframe='5m',
                                timeframe_offset=5, x=1.2) == 42
        assert sleep_mock.call_count == 1
        # Time is slightly bigger than throttle secs due to the high timeframe offset.
        assert 11.1 < sleep_mock.call_args[0][0] < 13.2


def test_throttle_with_assets(mocker, default_conf) -> None:
    def throttled_func(nb_assets=-1):
        return nb_assets

    worker = get_patched_worker(mocker, default_conf)

    result = worker._throttle(throttled_func, throttle_secs=0.1, nb_assets=666)
    assert result == 666

    result = worker._throttle(throttled_func, throttle_secs=0.1)
    assert result == -1


def test_worker_heartbeat_running(default_conf, mocker, caplog):
    message = r"Bot heartbeat\. PID=.*state='RUNNING'"

    mock_throttle = MagicMock()
    mocker.patch('freqtrade.worker.Worker._throttle', mock_throttle)
    worker = get_patched_worker(mocker, default_conf)

    worker.freqtrade.state = State.RUNNING
    worker._worker(old_state=State.STOPPED)
    assert log_has_re(message, caplog)

    caplog.clear()
    # Message is not shown before interval is up
    worker._worker(old_state=State.RUNNING)
    assert not log_has_re(message, caplog)

    caplog.clear()
    # Set clock - 70 seconds
    worker._heartbeat_msg -= 70
    worker._worker(old_state=State.RUNNING)
    assert log_has_re(message, caplog)


def test_worker_heartbeat_stopped(default_conf, mocker, caplog):
    message = r"Bot heartbeat\. PID=.*state='STOPPED'"

    mock_throttle = MagicMock()
    mocker.patch('freqtrade.worker.Worker._throttle', mock_throttle)
    worker = get_patched_worker(mocker, default_conf)

    worker.freqtrade.state = State.STOPPED
    worker._worker(old_state=State.RUNNING)
    assert log_has_re(message, caplog)

    caplog.clear()
    # Message is not shown before interval is up
    worker._worker(old_state=State.STOPPED)
    assert not log_has_re(message, caplog)

    caplog.clear()
    # Set clock - 70 seconds
    worker._heartbeat_msg -= 70
    worker._worker(old_state=State.STOPPED)
    assert log_has_re(message, caplog)
