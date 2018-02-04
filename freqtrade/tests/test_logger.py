"""
Unit test file for logger.py
"""

import logging
from freqtrade.logger import Logger


def test_logger_object() -> None:
    """
    Test the Constants object has the mandatory Constants
    :return: None
    """
    logger = Logger()
    assert logger.name == ''
    assert logger.level == 20
    assert logger.level is logging.INFO
    assert hasattr(logger, 'get_logger')

    logger = Logger(name='Foo', level=logging.WARNING)
    assert logger.name == 'Foo'
    assert logger.name != ''
    assert logger.level == 30
    assert logger.level is logging.WARNING


def test_get_logger() -> None:
    """
    Test logger.get_logger()
    :return: None
    """
    logger = Logger(name='Foo', level=logging.WARNING)
    get_logger = logger.get_logger()
    assert get_logger is not None
    assert hasattr(get_logger, 'debug')
    assert hasattr(get_logger, 'info')
    assert hasattr(get_logger, 'warning')
    assert hasattr(get_logger, 'critical')
    assert hasattr(get_logger, 'exception')


def test_sending_msg(caplog) -> None:
    """
    Test send a logging message
    :return: None
    """
    logger = Logger(name='FooBar', level=logging.WARNING).get_logger()

    logger.info('I am an INFO message')
    assert('FooBar', logging.INFO, 'I am an INFO message') not in caplog.record_tuples

    logger.warning('I am an WARNING message')
    assert ('FooBar', logging.WARNING, 'I am an WARNING message') in caplog.record_tuples
