import logging
from logging.handlers import QueueHandler
from queue import Queue
from types import SimpleNamespace

import pytest

from freqtrade.optimize.hyperopt.hyperopt_logger import logging_mp_handle, logging_mp_setup


@pytest.fixture(autouse=True)
def reset_root_logging():
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    ft_level = logging.getLogger("freqtrade").level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    logging.getLogger("freqtrade").setLevel(ft_level)


def _queue_handlers():
    """Minor helper to get all QueueHandlers from the logger. Used only in tests in this file."""
    return [h for h in logging.getLogger().handlers if isinstance(h, QueueHandler)]


def test_logging_mp_setup_main_process():
    logging_mp_setup(Queue(), logging.INFO)
    assert _queue_handlers() == []


def test_logging_mp_setup_child_process(mocker):
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_logger.current_process",
        return_value=SimpleNamespace(name="LokyProcess-1"),
    )
    q1 = Queue()
    logging_mp_setup(q1, logging.INFO)
    assert len(_queue_handlers()) == 1
    assert _queue_handlers()[0].queue is q1

    # Repeated calls (one per epoch) must not stack up handlers.
    for _ in range(5):
        logging_mp_setup(q1, logging.INFO)
    assert len(_queue_handlers()) == 1

    # A new queue is picked up by the existing handler.
    q2 = Queue()
    logging_mp_setup(q2, logging.INFO)
    assert len(_queue_handlers()) == 1
    assert _queue_handlers()[0].queue is q2

    logging.getLogger("someStrategy").warning("test message")
    assert q1.empty()
    assert q2.qsize() == 1


def test_logging_mp_handle(caplog):
    q: Queue = Queue()
    q.put(logging.LogRecord("ft_test", logging.WARNING, "path", 1, "test message", None, None))
    q.put(None)
    q.put(logging.LogRecord("ft_test", logging.WARNING, "path", 1, "not handled", None, None))

    with caplog.at_level(logging.WARNING):
        logging_mp_handle(q)

    assert "test message" in caplog.text
    assert "not handled" not in caplog.text
