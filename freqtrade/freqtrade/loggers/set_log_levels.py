import logging


logger = logging.getLogger(__name__)


__BIAS_TESTER_LOGGERS = [
    "freqtrade.resolvers",
    "freqtrade.strategy.hyper",
    "freqtrade.configuration.config_validation",
]


def reduce_verbosity_for_bias_tester() -> None:
    """
    Reduce verbosity for bias tester.
    It loads the same strategy several times, which would spam the log.
    """
    logger.info("Reducing verbosity for bias tester.")
    for logger_name in __BIAS_TESTER_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def restore_verbosity_for_bias_tester() -> None:
    """
    Restore verbosity after bias tester.
    """
    logger.info("Restoring log verbosity.")
    log_level = logging.NOTSET
    for logger_name in __BIAS_TESTER_LOGGERS:
        logging.getLogger(logger_name).setLevel(log_level)
