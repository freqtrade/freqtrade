"""
Unit test file for constants.py
"""

from freqtrade.constants import Constants


def test_constant_object() -> None:
    """
    Test the Constants object has the mandatory Constants
    :return: None
    """
    constant = Constants()
    assert hasattr(constant, 'CONF_SCHEMA')
    assert hasattr(constant, 'DYNAMIC_WHITELIST')
    assert hasattr(constant, 'PROCESS_THROTTLE_SECS')
    assert hasattr(constant, 'TICKER_INTERVAL')
    assert hasattr(constant, 'HYPEROPT_EPOCH')
    assert hasattr(constant, 'RETRY_TIMEOUT')


def test_conf_schema() -> None:
    """
    Test the CONF_SCHEMA is from the right type
    :return:
    """
    constant = Constants()
    assert isinstance(constant.CONF_SCHEMA, dict)
