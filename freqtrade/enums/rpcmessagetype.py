from enum import Enum


class RPCMessageType(str, Enum):
    STATUS = "status"
    WARNING = "warning"
    EXCEPTION = "exception"
    STARTUP = "startup"

    ENTRY = "entry"
    ENTRY_FILL = "entry_fill"
    ENTRY_CANCEL = "entry_cancel"

    EXIT = "exit"
    EXIT_FILL = "exit_fill"
    EXIT_CANCEL = "exit_cancel"

    PROTECTION_TRIGGER = "protection_trigger"
    PROTECTION_TRIGGER_GLOBAL = "protection_trigger_global"

    STRATEGY_MSG = "strategy_msg"

    WHITELIST = "whitelist"
    ANALYZED_DF = "analyzed_df"
    NEW_CANDLE = "new_candle"

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


# Enum for parsing requests from ws consumers
class RPCRequestType(str, Enum):
    SUBSCRIBE = "subscribe"

    WHITELIST = "whitelist"
    ANALYZED_DF = "analyzed_df"

    def __str__(self):
        return self.value


NO_ECHO_MESSAGES = (RPCMessageType.ANALYZED_DF, RPCMessageType.WHITELIST, RPCMessageType.NEW_CANDLE)
