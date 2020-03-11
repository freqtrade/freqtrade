import logging
import sys

from logging import Formatter
from logging.handlers import RotatingFileHandler, SysLogHandler
from typing import Any, Dict, List

from freqtrade.exceptions import OperationalException


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

    # Log to stderr
    log_handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    logfile = config.get('logfile')
    if logfile:
        s = logfile.split(':')
        if s[0] == 'syslog':
            # Address can be either a string (socket filename) for Unix domain socket or
            # a tuple (hostname, port) for UDP socket.
            # Address can be omitted (i.e. simple 'syslog' used as the value of
            # config['logfilename']), which defaults to '/dev/log', applicable for most
            # of the systems.
            address = (s[1], int(s[2])) if len(s) > 2 else s[1] if len(s) > 1 else '/dev/log'
            handler = SysLogHandler(address=address)
            # No datetime field for logging into syslog, to allow syslog
            # to perform reduction of repeating messages if this is set in the
            # syslog config. The messages should be equal for this.
            handler.setFormatter(Formatter('%(name)s - %(levelname)s - %(message)s'))
            log_handlers.append(handler)
        elif s[0] == 'journald':
            try:
                from systemd.journal import JournaldLogHandler
            except ImportError:
                raise OperationalException("You need the systemd python package be installed in "
                                           "order to use logging to journald.")
            handler = JournaldLogHandler()
            # No datetime field for logging into journald, to allow syslog
            # to perform reduction of repeating messages if this is set in the
            # syslog config. The messages should be equal for this.
            handler.setFormatter(Formatter('%(name)s - %(levelname)s - %(message)s'))
            log_handlers.append(handler)
        else:
            log_handlers.append(RotatingFileHandler(logfile,
                                                    maxBytes=1024 * 1024,  # 1Mb
                                                    backupCount=10))

    logging.basicConfig(
        level=logging.INFO if verbosity < 1 else logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=log_handlers
    )
    _set_loggers(verbosity)
    logger.info('Verbosity set to %s', verbosity)
