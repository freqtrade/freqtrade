from enum import StrEnum


class OrderRole(StrEnum):
    entry = "entry"
    exit = "exit"
    conditional_exit = "conditional_exit"
