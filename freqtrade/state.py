# pragma pylint: disable=too-few-public-methods

"""
Bot state constant
"""
import enum


class State(enum.Enum):
    """
    Bot running states
    """
    RUNNING = 0
    STOPPED = 1
