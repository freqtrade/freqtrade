from enum import Enum


class ExitType(Enum):
    """
    Enum to distinguish between sell reasons
    """
    ROI = "roi"
    STOP_LOSS = "stop_loss"
    STOPLOSS_ON_EXCHANGE = "stoploss_on_exchange"
    TRAILING_STOP_LOSS = "trailing_stop_loss"
    EXIT_SIGNAL = "exit_signal"
    FORCE_SELL = "force_exit"
    EMERGENCY_SELL = "emergency_sell"
    CUSTOM_SELL = "custom_sell"
    NONE = ""

    def __str__(self):
        # explicitly convert to String to help with exporting data.
        return self.value
