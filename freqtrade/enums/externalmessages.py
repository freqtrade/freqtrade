from enum import Enum


class WaitDataPolicy(str, Enum):
    none = "none"
    one = "one"
    all = "all"
