from enum import StrEnum


class OrderRole(StrEnum):
    entry = "entry"
    exit = "exit"
    conditional_exit = "conditional_exit"

    @classmethod
    def from_value(cls, value: "OrderRole | str | None") -> "OrderRole | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            return None
