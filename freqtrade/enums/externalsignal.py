from enum import Enum


class ExternalSignalModeType(str, Enum):
    leader = "leader"
    follower = "follower"


class LeaderMessageType(str, Enum):
    pairlist = "pairlist"
    analyzed_df = "analyzed_df"
