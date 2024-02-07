from enum import Enum


class ExitType(Enum):
    """
    Enum to distinguish between exit reasons
    """
    ROI = "roi"
    STOP_LOSS = "stop_loss"
    STOPLOSS_ON_EXCHANGE = "stoploss_on_exchange"
    TRAILING_STOP_LOSS = "trailing_stop_loss"
    LIQUIDATION = "liquidation"
    EXIT_SIGNAL = "exit_signal"
    FORCE_EXIT = "force_exit"
    EMERGENCY_EXIT = "emergency_exit"
    CUSTOM_EXIT = "custom_exit"
    PARTIAL_EXIT = "partial_exit"
    SOLD_ON_EXCHANGE = "sold_on_exchange"
    NONE = ""

    def __str__(self):
        # explicitly convert to String to help with exporting data.
        return self.value
