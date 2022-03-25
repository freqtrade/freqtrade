from freqtrade.enums.selltype import SellType


class ExitCheckTuple:
    """
    NamedTuple for Exit type + reason
    """
    exit_type: SellType
    exit_reason: str = ''

    def __init__(self, exit_type: SellType, exit_reason: str = ''):
        self.exit_type = exit_type
        self.exit_reason = exit_reason or exit_type.value

    @property
    def exit_flag(self):
        return self.exit_type != SellType.NONE
