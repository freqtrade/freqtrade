import logging
import sys

from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List


logger = logging.getLogger(__name__)


def _set_loggers(verbosity: int = 0) -> None:
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


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Process -v/--verbose, --logfile options
    """
    # Log level
    verbosity = config['verbosity']

    # Log to stdout, not stderr
    log_handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if config.get('logfile'):
        log_handlers.append(RotatingFileHandler(config['logfile'],
                                                maxBytes=1024 * 1024,  # 1Mb
                                                backupCount=10))

    logging.basicConfig(
        level=logging.INFO if verbosity < 1 else logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=log_handlers
    )
    _set_loggers(verbosity)
    logger.info('Verbosity set to %s', verbosity)
