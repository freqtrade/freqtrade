import logging
import re
import sys

import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.loggers import (
    FTBufferingHandler,
    FtRichHandler,
    setup_logging,
    setup_logging_pre,
)
from freqtrade.loggers.set_log_levels import (
    reduce_verbosity_for_bias_tester,
    restore_verbosity_for_bias_tester,
)


@pytest.mark.usefixtures("keep_log_config_loggers")
def test_set_loggers() -> None:
    # Reset Logging to Debug, otherwise this fails randomly as it's set globally
    logging.getLogger("requests").setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.DEBUG)
    logging.getLogger("ccxt.base.exchange").setLevel(logging.DEBUG)
    logging.getLogger("telegram").setLevel(logging.DEBUG)

    previous_value1 = logging.getLogger("requests").level
    previous_value2 = logging.getLogger("ccxt.base.exchange").level
    previous_value3 = logging.getLogger("telegram").level
    config = {
        "verbosity": 1,
        "ft_tests_force_logging": True,
    }
    setup_logging(config)

    value1 = logging.getLogger("requests").level
    assert previous_value1 is not value1
    assert value1 is logging.INFO

    value2 = logging.getLogger("ccxt.base.exchange").level
    assert previous_value2 is not value2
    assert value2 is logging.INFO

    value3 = logging.getLogger("telegram").level
    assert previous_value3 is not value3
    assert value3 is logging.INFO
    config["verbosity"] = 2
    setup_logging(config)

    assert logging.getLogger("requests").level is logging.DEBUG
    assert logging.getLogger("ccxt.base.exchange").level is logging.INFO
    assert logging.getLogger("telegram").level is logging.INFO
    assert logging.getLogger("werkzeug").level is logging.INFO

    config["verbosity"] = 3
    config["api_server"] = {"verbosity": "error"}
    setup_logging(config)

    assert logging.getLogger("requests").level is logging.DEBUG
    assert logging.getLogger("ccxt.base.exchange").level is logging.DEBUG
    assert logging.getLogger("telegram").level is logging.INFO
    assert logging.getLogger("werkzeug").level is logging.ERROR


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@pytest.mark.usefixtures("keep_log_config_loggers")
def test_set_loggers_syslog():
    logger = logging.getLogger()
    orig_handlers = logger.handlers
    logger.handlers = []

    config = {
        "ft_tests_force_logging": True,
        "verbosity": 2,
        "logfile": "syslog:/dev/log",
    }

    setup_logging_pre()
    setup_logging(config)
    assert len(logger.handlers) == 3
    assert [x for x in logger.handlers if isinstance(x, logging.handlers.SysLogHandler)]
    assert [x for x in logger.handlers if isinstance(x, FtRichHandler)]
    assert [x for x in logger.handlers if isinstance(x, FTBufferingHandler)]
    # setting up logging again should NOT cause the loggers to be added a second time.
    setup_logging(config)
    assert len(logger.handlers) == 3
    # reset handlers to not break pytest
    logger.handlers = orig_handlers


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@pytest.mark.usefixtures("keep_log_config_loggers")
def test_set_loggers_Filehandler(tmp_path):
    logger = logging.getLogger()
    orig_handlers = logger.handlers
    logger.handlers = []
    logfile = tmp_path / "logs/ft_logfile.log"
    config = {
        "ft_tests_force_logging": True,
        "verbosity": 2,
        "logfile": str(logfile),
    }

    setup_logging_pre()
    setup_logging(config)
    assert len(logger.handlers) == 3
    assert [x for x in logger.handlers if isinstance(x, logging.handlers.RotatingFileHandler)]
    assert [x for x in logger.handlers if isinstance(x, FtRichHandler)]
    assert [x for x in logger.handlers if isinstance(x, FTBufferingHandler)]
    # setting up logging again should NOT cause the loggers to be added a second time.
    setup_logging(config)
    assert len(logger.handlers) == 3
    # reset handlers to not break pytest
    if logfile.exists:
        logfile.unlink()
    logger.handlers = orig_handlers


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@pytest.mark.usefixtures("keep_log_config_loggers")
def test_set_loggers_Filehandler_without_permission(tmp_path):
    logger = logging.getLogger()
    orig_handlers = logger.handlers
    logger.handlers = []

    try:
        tmp_path.chmod(0o400)
        logfile = tmp_path / "logs/ft_logfile.log"
        config = {
            "ft_tests_force_logging": True,
            "verbosity": 2,
            "logfile": str(logfile),
        }

        setup_logging_pre()
        with pytest.raises(OperationalException):
            setup_logging(config)

        logger.handlers = orig_handlers
    finally:
        tmp_path.chmod(0o700)


@pytest.mark.skip(reason="systemd is not installed on every system, so we're not testing this.")
@pytest.mark.usefixtures("keep_log_config_loggers")
def test_set_loggers_journald():
    logger = logging.getLogger()
    orig_handlers = logger.handlers
    logger.handlers = []

    config = {
        "ft_tests_force_logging": True,
        "verbosity": 2,
        "logfile": "journald",
    }

    setup_logging_pre()
    setup_logging(config)
    assert len(logger.handlers) == 3
    assert [x for x in logger.handlers if type(x).__name__ == "JournaldLogHandler"]
    assert [x for x in logger.handlers if isinstance(x, FtRichHandler)]
    # reset handlers to not break pytest
    logger.handlers = orig_handlers


@pytest.mark.usefixtures("keep_log_config_loggers")
def test_set_loggers_journald_importerror(import_fails):
    logger = logging.getLogger()
    orig_handlers = logger.handlers
    logger.handlers = []

    config = {
        "ft_tests_force_logging": True,
        "verbosity": 2,
        "logfile": "journald",
    }
    with pytest.raises(OperationalException, match=r"You need the cysystemd python package.*"):
        setup_logging(config)
    logger.handlers = orig_handlers


@pytest.mark.usefixtures("keep_log_config_loggers")
def test_set_loggers_json_format(capsys):
    logger = logging.getLogger()
    orig_handlers = logger.handlers
    logger.handlers = []

    config = {
        "ft_tests_force_logging": True,
        "verbosity": 2,
        "log_config": {
            "version": 1,
            "formatters": {
                "json": {
                    "()": "freqtrade.loggers.json_formatter.JsonFormatter",
                    "fmt_dict": {
                        "timestamp": "asctime",
                        "level": "levelname",
                        "logger": "name",
                        "message": "message",
                    },
                }
            },
            "handlers": {
                "json": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                }
            },
            "root": {
                "handlers": ["json"],
                "level": "DEBUG",
            },
        },
    }

    setup_logging_pre()
    setup_logging(config)
    assert len(logger.handlers) == 2
    assert [x for x in logger.handlers if type(x).__name__ == "StreamHandler"]
    assert [x for x in logger.handlers if isinstance(x, FTBufferingHandler)]

    logger.info("Test message")

    captured = capsys.readouterr()
    assert re.search(r'{"timestamp": ".*"Test message".*', captured.err)

    # reset handlers to not break pytest
    logger.handlers = orig_handlers


def test_reduce_verbosity():
    setup_logging_pre()
    reduce_verbosity_for_bias_tester()
    prior_level = logging.getLogger("freqtrade").getEffectiveLevel()

    assert logging.getLogger("freqtrade.resolvers").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("freqtrade.strategy.hyper").getEffectiveLevel() == logging.WARNING
    # base level wasn't changed
    assert logging.getLogger("freqtrade").getEffectiveLevel() == prior_level

    restore_verbosity_for_bias_tester()

    assert logging.getLogger("freqtrade.resolvers").getEffectiveLevel() == prior_level
    assert logging.getLogger("freqtrade.strategy.hyper").getEffectiveLevel() == prior_level
    assert logging.getLogger("freqtrade").getEffectiveLevel() == prior_level
    # base level wasn't changed
