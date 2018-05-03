"""
Unit test file for constants.py
"""

from freqtrade import constants


def test_constant_object() -> None:
    """
    Test the Constants object has the mandatory Constants
    """
    assert hasattr(constants, 'CONF_SCHEMA')
    assert hasattr(constants, 'DYNAMIC_WHITELIST')
    assert hasattr(constants, 'PROCESS_THROTTLE_SECS')
    assert hasattr(constants, 'TICKER_INTERVAL')
    assert hasattr(constants, 'HYPEROPT_EPOCH')
    assert hasattr(constants, 'RETRY_TIMEOUT')
    assert hasattr(constants, 'DEFAULT_STRATEGY')


def test_conf_schema() -> None:
    """
    Test the CONF_SCHEMA is from the right type
    """
    assert isinstance(constants.CONF_SCHEMA, dict)
