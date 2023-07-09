from strenum import StrEnum


class ExitType(StrEnum):
    """
    Enum to distinguish between exit reasons
    """
    ROI = "roi"
    STOP_LOSS = "stop_loss"
    STOPLOSS_ON_EXCHANGE = "stoploss_on_exchange"
    TAKE_PROFIT_ON_EXCHANGE = "takeprofit_on_exchange"
    TRAILING_STOP_LOSS = "trailing_stop_loss"
    LIQUIDATION = "liquidation"
    EXIT_SIGNAL = "exit_signal"
    FORCE_EXIT = "force_exit"
    EMERGENCY_EXIT = "emergency_exit"
    CUSTOM_EXIT = "custom_exit"
    PARTIAL_EXIT = "partial_exit"
    NONE = ""