from enum import StrEnum


class ConditionalExitKind(StrEnum):
    stoploss = "stoploss"
    trailing = "trailing"

    @classmethod
    def from_value(cls, value: "ConditionalExitKind | str | None") -> "ConditionalExitKind | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            return None
