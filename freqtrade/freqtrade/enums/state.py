from enum import Enum


class State(Enum):
    """
    Bot application states
    """

    RUNNING = 1
    PAUSED = 2
    STOPPED = 3
    RELOAD_CONFIG = 4

    def __str__(self):
        return f"{self.name.lower()}"
