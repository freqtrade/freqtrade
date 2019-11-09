import logging
import time
from unittest.mock import MagicMock, PropertyMock

from freqtrade.data.dataprovider import DataProvider
from freqtrade.state import State
from freqtrade.worker import Worker
from tests.conftest import get_patched_worker, log_has


def test_worker_state(mocker, default_conf, markets) -> None:
    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets))
    worker = get_patched_worker(mocker, default_conf)
    assert worker.state is State.RUNNING

    default_conf.pop('initial_state')
    worker = Worker(args=None, config=default_conf)
    assert worker.state is State.STOPPED


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
    mock_sleep = mocker.patch('time.sleep', return_value=None)

    worker = get_patched_worker(mocker, default_conf)
    worker.state = State.STOPPED
    state = worker._worker(old_state=State.RUNNING)
    assert state is State.STOPPED
    assert log_has('Changing state to: STOPPED', caplog)
    assert mock_throttle.call_count == 0
    assert mock_sleep.call_count == 1


def test_throttle(mocker, default_conf, caplog) -> None:
    def throttled_func():
        return 42

    caplog.set_level(logging.DEBUG)
    worker = get_patched_worker(mocker, default_conf)

    start = time.time()
    result = worker._throttle(throttled_func, min_secs=0.1)
    end = time.time()

    assert result == 42
    assert end - start > 0.1
    assert log_has('Throttling throttled_func for 0.10 seconds', caplog)

    result = worker._throttle(throttled_func, min_secs=-1)
    assert result == 42


def test_throttle_with_assets(mocker, default_conf) -> None:
    def throttled_func(nb_assets=-1):
        return nb_assets

    worker = get_patched_worker(mocker, default_conf)

    result = worker._throttle(throttled_func, min_secs=0.1, nb_assets=666)
    assert result == 666

    result = worker._throttle(throttled_func, min_secs=0.1)
    assert result == -1
