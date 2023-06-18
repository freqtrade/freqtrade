
import logging


logger = logging.getLogger(__name__)


def set_loggers(verbosity: int = 0, api_verbosity: str = 'info') -> None:
    """
    Set the logging level for third party libraries
    :return: None
    """

    logging.getLogger('requests').setLevel(
        logging.INFO if verbosity <= 1 else logging.DEBUG
    )
    logging.getLogger("urllib3").setLevel(
        logging.INFO if verbosity <= 1 else logging.DEBUG
    )
    logging.getLogger('ccxt.base.exchange').setLevel(
        logging.INFO if verbosity <= 2 else logging.DEBUG
    )
    logging.getLogger('telegram').setLevel(logging.INFO)
    logging.getLogger('httpx').setLevel(logging.WARNING)

    logging.getLogger('werkzeug').setLevel(
        logging.ERROR if api_verbosity == 'error' else logging.INFO
    )


__BIAS_TESTER_LOGGERS = [
    'freqtrade.resolvers',
    'freqtrade.strategy.hyper',
    'freqtrade.configuration.config_validation',
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
