# pragma pylint: disable=too-few-public-methods

"""
Bot state constant
"""
from enum import Enum


class State(Enum):
    """
    Bot application states
    """
    RUNNING = 1
    STOPPED = 2
    RELOAD_CONFIG = 3

    def __str__(self):
        return f"{self.name.lower()}"
