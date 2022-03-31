from enum import Enum


class RPCMessageType(Enum):
    STATUS = 'status'
    WARNING = 'warning'
    STARTUP = 'startup'

    BUY = 'buy'
    BUY_FILL = 'buy_fill'
    BUY_CANCEL = 'buy_cancel'

    SHORT = 'short'
    SHORT_FILL = 'short_fill'
    SHORT_CANCEL = 'short_cancel'

    # TODO: The below messagetypes should be renamed to "exit"!
    # Careful - has an impact on webhooks, therefore needs proper communication
    SELL = 'sell'
    SELL_FILL = 'sell_fill'
    SELL_CANCEL = 'sell_cancel'

    PROTECTION_TRIGGER = 'protection_trigger'
    PROTECTION_TRIGGER_GLOBAL = 'protection_trigger_global'

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value
