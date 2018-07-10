"""
Unit test file for constants.py
"""

from freqtrade.state import State


def test_state_object() -> None:
    """
    Test the State object has the mandatory states
    :return: None
    """
    assert hasattr(State, 'RUNNING')
    assert hasattr(State, 'STOPPED')
