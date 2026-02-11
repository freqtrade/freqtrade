from enum import StrEnum


class RPCMessageType(StrEnum):
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
        # TODO: do we still need to overwrite __repr__? Impact needs to be looked at in detail
        return self.value


# Enum for parsing requests from ws consumers
class RPCRequestType(StrEnum):
    SUBSCRIBE = "subscribe"

    WHITELIST = "whitelist"
    ANALYZED_DF = "analyzed_df"


NO_ECHO_MESSAGES = (RPCMessageType.ANALYZED_DF, RPCMessageType.WHITELIST, RPCMessageType.NEW_CANDLE)
