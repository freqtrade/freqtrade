from enum import Enum


class RPCMessageType(Enum):
    STATUS = 'status'
    WARNING = 'warning'
    STARTUP = 'startup'
    BUY = 'buy'
    BUY_FILL = 'buy_fill'
    BUY_CANCEL = 'buy_cancel'
    SELL = 'sell'
    SELL_FILL = 'sell_fill'
    SELL_CANCEL = 'sell_cancel'
    PROTECTION_TRIGGER = 'protection_trigger'
    PROTECTION_TRIGGER_GLOBAL = 'protection_trigger_global'

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value
