from datetime import datetime
from logging import Handler

from rich._null_file import NullFile
from rich.console import Console
from rich.text import Text


class FtRichHandler(Handler):
    """
    Basic colorized logging handler using Rich.
    Does not support all features of the standard logging handler, and uses a hard-coded log format
    """

    def __init__(self, console: Console, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._console = console

    def emit(self, record):
        try:
            msg = self.format(record)
            # Format log message
            log_time = Text(
                datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
                if record.created
                else "N/A",
            )
            name = Text(record.name, style="violet")
            log_level = Text(record.levelname, style=f"logging.level.{record.levelname.lower()}")
            gray_sep = Text(" - ", style="gray46")

            if isinstance(self._console.file, NullFile):
                # Handles pythonw, where stdout/stderr are null, and we return NullFile
                # instance from Console.file. In this case, we still want to make a log record
                # even though we won't be writing anything to a file.
                self.handleError(record)
                return

            self._console.print(
                Text() + log_time + gray_sep + name + gray_sep + log_level + gray_sep + msg
            )

        except RecursionError:
            raise
        except ImportError:
            # Error when shutting down the console...
            pass
        except Exception:
            self.handleError(record)
