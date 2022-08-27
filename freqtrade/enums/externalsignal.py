from enum import Enum


class ExternalSignalModeType(str, Enum):
    leader = "leader"
    follower = "follower"


class LeaderMessageType(str, Enum):
    default = "default"
    pairlist = "pairlist"
    analyzed_df = "analyzed_df"


class WaitDataPolicy(str, Enum):
    none = "none"
    one = "one"
    all = "all"
