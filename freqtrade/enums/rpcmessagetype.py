from enum import Enum


class RPCMessageType(Enum):
    STATUS = 'status'
    WARNING = 'warning'
    STARTUP = 'startup'

    ENTRY = 'entry'
    ENTRY_FILL = 'entry_fill'
    ENTRY_CANCEL = 'entry_cancel'

    EXIT = 'exit'
    EXIT_FILL = 'exit_fill'
    EXIT_CANCEL = 'exit_cancel'

    PROTECTION_TRIGGER = 'protection_trigger'
    PROTECTION_TRIGGER_GLOBAL = 'protection_trigger_global'

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value
