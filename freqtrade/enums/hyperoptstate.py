from enum import Enum


class HyperoptState(Enum):
    """ Hyperopt states """
    STARTUP = 1
    DATALOAD = 2
    INDICATORS = 3
    OPTIMIZE = 0

    def __str__(self):
        return f"{self.name.lower()}"
