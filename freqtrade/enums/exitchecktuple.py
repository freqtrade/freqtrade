from freqtrade.enums.exittype import ExitType


class ExitCheckTuple:
    """
    NamedTuple for Exit type + reason
    """
    exit_type: ExitType
    exit_reason: str = ''

    def __init__(self, exit_type: ExitType, exit_reason: str = ''):
        self.exit_type = exit_type
        self.exit_reason = exit_reason or exit_type.value

    @property
    def exit_flag(self):
        return self.exit_type != ExitType.NONE

    def __eq__(self, other):
        return self.exit_type == other.exit_type and self.exit_reason == other.exit_reason

    def __repr__(self):
        return f"ExitCheckTuple({self.exit_type}, {self.exit_reason})"
