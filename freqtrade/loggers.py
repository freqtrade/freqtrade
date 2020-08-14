import logging
import queue
import sys
from logging import Formatter
from logging.handlers import RotatingFileHandler, SysLogHandler, QueueHandler, QueueListener
from typing import Any, Dict, List

from freqtrade.exceptions import OperationalException

logger = logging.getLogger(__name__)
log_queue = queue.Queue(-1)
LOGFORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def _set_loggers(verbosity: int = 0, api_verbosity: str = 'info') -> None:
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

    logging.getLogger('werkzeug').setLevel(
        logging.ERROR if api_verbosity == 'error' else logging.INFO
    )


def setup_logging_pre() -> None:
    """
    Setup early logging.
    This uses a queuehandler, which delays logging.
    # TODO: How does QueueHandler work if no listenerhandler is attached??
    """
    logging.root.setLevel(logging.INFO)
    fmt = logging.Formatter(LOGFORMAT)

    queue_handler = QueueHandler(log_queue)
    queue_handler.setFormatter(fmt)
    logger.root.addHandler(queue_handler)

    # Add streamhandler here to capture Errors before QueueListener is started
    sth = logging.StreamHandler(sys.stderr)
    sth.setFormatter(fmt)
    logger.root.addHandler(sth)


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Process -v/--verbose, --logfile options
    """
    # Log level
    verbosity = config['verbosity']

    # Log to stderr
    log_handlers: List[logging.Handler] = []

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

    listener = QueueListener(log_queue, *log_handlers)

    # logging.root.setFormatter(logging.Formatter(LOGFORMAT))
    logging.root.setLevel(logging.INFO if verbosity < 1 else logging.DEBUG)
    listener.start()
    _set_loggers(verbosity, config.get('api_server', {}).get('verbosity', 'info'))
    logger.info('Verbosity set to %s', verbosity)
