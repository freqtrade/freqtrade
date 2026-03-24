from enum import StrEnum


class ConditionalExitKind(StrEnum):
    stoploss = "stoploss"

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

    @classmethod
    def is_valid(cls, value: "ConditionalExitKind | str | None") -> bool:
        return cls.from_value(value) is not None
