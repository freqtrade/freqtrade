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
    Test Logger.get_logger() and Logger._init_logger()
    :return: None
    """
    logger = Logger(name='get_logger', level=logging.WARNING)
    get_logger = logger.get_logger()
    assert logger.logger is get_logger
    assert get_logger is not None
    assert hasattr(get_logger, 'debug')
    assert hasattr(get_logger, 'info')
    assert hasattr(get_logger, 'warning')
    assert hasattr(get_logger, 'critical')
    assert hasattr(get_logger, 'exception')


def test_set_name() -> None:
    """
    Test Logger.set_name()
    :return: None
    """
    logger = Logger(name='set_name')
    assert logger.name == 'set_name'

    logger.set_name('set_name_new')
    assert logger.name == 'set_name_new'


def test_set_level() -> None:
    """
    Test Logger.set_name()
    :return: None
    """
    logger = Logger(name='Foo', level=logging.WARNING)
    assert logger.level == logging.WARNING
    assert logger.get_logger().level == logging.WARNING

    logger.set_level(logging.INFO)
    assert logger.level == logging.INFO
    assert logger.get_logger().level == logging.INFO


def test_sending_msg(caplog) -> None:
    """
    Test send a logging message
    :return: None
    """
    logger = Logger(name='sending_msg', level=logging.WARNING).get_logger()

    logger.info('I am an INFO message')
    assert('sending_msg', logging.INFO, 'I am an INFO message') not in caplog.record_tuples

    logger.warning('I am an WARNING message')
    assert ('sending_msg', logging.WARNING, 'I am an WARNING message') in caplog.record_tuples


def test_set_format(caplog) -> None:
    """
    Test Logger.set_format()
    :return: None
    """
    log = Logger(name='set_format')
    logger = log.get_logger()

    logger.info('I am the first message')
    assert ('set_format', logging.INFO, 'I am the first message') in caplog.record_tuples

    log.set_format(log_format='%(message)s', propagate=True)
    logger.info('I am the second message')
    assert ('set_format', logging.INFO, 'I am the second message') in caplog.record_tuples
