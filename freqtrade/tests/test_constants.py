"""
Unit test file for constants.py
"""

from freqtrade.constants import Constants


def test_constant_object() -> None:
    """
    Test the Constants object has the mandatory Constants
    :return: None
    """
    assert hasattr(Constants, 'CONF_SCHEMA')
    assert hasattr(Constants, 'DYNAMIC_WHITELIST')
    assert hasattr(Constants, 'PROCESS_THROTTLE_SECS')
    assert hasattr(Constants, 'TICKER_INTERVAL')
    assert hasattr(Constants, 'HYPEROPT_EPOCH')
    assert hasattr(Constants, 'RETRY_TIMEOUT')
    assert hasattr(Constants, 'DEFAULT_STRATEGY')



def test_conf_schema() -> None:
    """
    Test the CONF_SCHEMA is from the right type
    :return:
    """
    constant = Constants()
    assert isinstance(constant.CONF_SCHEMA, dict)
