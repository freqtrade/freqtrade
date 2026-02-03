import logging
import logging.config
import os
from copy import deepcopy
from logging import Formatter
from pathlib import Path
from typing import Any

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.loggers.buffering_handler import FTBufferingHandler
from freqtrade.loggers.ft_rich_handler import FtRichHandler
from freqtrade.loggers.rich_console import get_rich_console


# from freqtrade.loggers.std_err_stream_handler import FTStdErrStreamHandler


logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Initialize bufferhandler - will be used for /log endpoints
bufferHandler = FTBufferingHandler(1000)
bufferHandler.setFormatter(Formatter(LOGFORMAT))


error_console = get_rich_console(stderr=True, color_system=None)


def get_existing_handlers(handlertype):
    """
    Returns Existing handler or None (if the handler has not yet been added to the root handlers).
    """
    return next((h for h in logging.root.handlers if isinstance(h, handlertype)), None)


def setup_logging_pre() -> None:
    """
    Early setup for logging.
    Uses INFO loglevel and only the Streamhandler.
    Early messages (before proper logging setup) will therefore only be sent to additional
    logging handlers after the real initialization, because we don't know which
    ones the user desires beforehand.
    """
    rh = FtRichHandler(console=error_console)
    rh.setFormatter(Formatter("%(message)s"))
    logging.basicConfig(
        level=logging.INFO,
        format=LOGFORMAT,
        handlers=[
            # FTStdErrStreamHandler(),
            rh,
            bufferHandler,
        ],
    )


FT_LOGGING_CONFIG = {
    "version": 1,
    # "incremental": True,
    "disable_existing_loggers": False,
    "formatters": {
        "basic": {"format": "%(message)s"},
        "standard": {
            "format": LOGFORMAT,
        },
    },
    "handlers": {
        "console": {
            "class": "freqtrade.loggers.ft_rich_handler.FtRichHandler",
            "formatter": "basic",
        },
    },
    "root": {
        "handlers": [
            "console",
            # "file",
        ],
        "level": "INFO",
    },
}


def _set_log_levels(
    log_config: dict[str, Any], verbosity: int = 0, api_verbosity: str = "info"
) -> None:
    """
    Set the logging level for the different loggers
    """
    if "loggers" not in log_config:
        log_config["loggers"] = {}

    # Set default levels for third party libraries
    third_party_loggers = {
        "freqtrade": logging.INFO if verbosity < 1 else logging.DEBUG,
        "freqtrade.exchange.exchange_ws": logging.INFO if verbosity <= 1 else logging.DEBUG,
        "requests": logging.INFO if verbosity <= 1 else logging.DEBUG,
        "urllib3": logging.INFO if verbosity <= 1 else logging.DEBUG,
        "asyncio": logging.INFO if verbosity <= 1 else logging.DEBUG,
        "httpcore": logging.INFO if verbosity <= 1 else logging.DEBUG,
        "ccxt.base.exchange": logging.INFO if verbosity <= 2 else logging.DEBUG,
        "telegram": logging.INFO,
        "httpx": logging.WARNING,
        "werkzeug": logging.ERROR if api_verbosity == "error" else logging.INFO,
    }

    # Add third party loggers to the configuration
    for logger_name, level in third_party_loggers.items():
        if logger_name not in log_config["loggers"]:
            log_config["loggers"][logger_name] = {
                "level": logging.getLevelName(level),
                "propagate": True,
            }


def _add_root_handler(log_config: dict[str, Any], handler_name: str):
    if handler_name not in log_config["root"]["handlers"]:
        log_config["root"]["handlers"].append(handler_name)


def _add_formatter(log_config: dict[str, Any], format_name: str, format_: str):
    if format_name not in log_config["formatters"]:
        log_config["formatters"][format_name] = {"format": format_}


def _create_log_config(config: Config) -> dict[str, Any]:
    # Get log_config from user config or use default
    log_config = deepcopy(config.get("log_config", FT_LOGGING_CONFIG))

    if logfile := config.get("logfile"):
        s = logfile.split(":")
        if s[0] == "syslog":
            logger.warning(
                "DEPRECATED: Configuring syslog logging via command line is deprecated."
                "Please use the log_config option in the configuration file instead."
            )
            # Add syslog handler to the config
            log_config["handlers"]["syslog"] = {
                "class": "logging.handlers.SysLogHandler",
                "formatter": "syslog_format",
                "address": (s[1], int(s[2])) if len(s) > 2 else s[1] if len(s) > 1 else "/dev/log",
            }

            _add_formatter(log_config, "syslog_format", "%(name)s - %(levelname)s - %(message)s")
            _add_root_handler(log_config, "syslog")

        elif s[0] == "journald":  # pragma: no cover
            # Check if we have the module available
            logger.warning(
                "DEPRECATED: Configuring Journald logging via command line is deprecated."
                "Please use the log_config option in the configuration file instead."
            )
            try:
                from cysystemd.journal import JournaldLogHandler  # noqa: F401
            except ImportError:
                raise OperationalException(
                    "You need the cysystemd python package be installed in "
                    "order to use logging to journald."
                )

            # Add journald handler to the config
            log_config["handlers"]["journald"] = {
                "class": "cysystemd.journal.JournaldLogHandler",
                "formatter": "journald_format",
            }

            _add_formatter(log_config, "journald_format", "%(name)s - %(levelname)s - %(message)s")
            _add_root_handler(log_config, "journald")

        else:
            # Regular file logging
            # Update existing file handler configuration
            if "file" in log_config["handlers"]:
                log_config["handlers"]["file"]["filename"] = logfile
            else:
                log_config["handlers"]["file"] = {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "standard",
                    "filename": logfile,
                    "maxBytes": 1024 * 1024 * 10,  # 10Mb
                    "backupCount": 10,
                }
            _add_root_handler(log_config, "file")

    # Dynamically update some handlers
    for handler_config in log_config.get("handlers", {}).values():
        if handler_config.get("class") == "freqtrade.loggers.ft_rich_handler.FtRichHandler":
            handler_config["console"] = error_console
        elif handler_config.get("class") == "logging.handlers.RotatingFileHandler":
            logfile_path = Path(handler_config["filename"])
            try:
                # Create parent for filehandler
                logfile_path.parent.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise OperationalException(
                    f'Failed to create or access log file "{logfile_path.absolute()}". '
                    "Please make sure you have the write permission to the log file or its parent "
                    "directories. If you're running freqtrade using docker, you see this error "
                    "message probably because you've logged in as the root user, please switch to "
                    "non-root user, delete and recreate the directories you need, and then try "
                    "again."
                )
    return log_config


def setup_logging(config: Config) -> None:
    """
    Process -v/--verbose, --logfile options
    """
    verbosity = config["verbosity"]
    if os.environ.get("PYTEST_VERSION") is None or config.get("ft_tests_force_logging"):
        log_config = _create_log_config(config)
        _set_log_levels(
            log_config, verbosity, config.get("api_server", {}).get("verbosity", "info")
        )

        logging.config.dictConfig(log_config)

    # Add buffer handler to root logger
    if bufferHandler not in logging.root.handlers:
        logging.root.addHandler(bufferHandler)

    # Set color system for console output
    if config.get("print_colorized", True):
        logger.info("Enabling colorized output.")
        error_console._color_system = error_console._detect_color_system()

    logger.info("Logfile configured")

    # Set verbosity levels
    logging.root.setLevel(logging.INFO if verbosity < 1 else logging.DEBUG)

    logger.info("Verbosity set to %s", verbosity)
