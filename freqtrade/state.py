# pragma pylint: disable=too-few-public-methods

"""
Bot state constant
"""
import enum


class State(enum.Enum):
    """
    Bot application states
    """
    RUNNING = 0
    STOPPED = 1
    RELOAD_CONF = 2
