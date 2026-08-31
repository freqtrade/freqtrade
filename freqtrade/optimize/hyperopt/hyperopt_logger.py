import logging
from logging.handlers import QueueHandler
from multiprocessing import Queue, current_process
from queue import Empty

from freqtrade.loggers import get_existing_handlers


logger = logging.getLogger(__name__)


def logging_mp_setup(log_queue: Queue, verbosity: int):
    """
    Setup logging in a child process.
    Must be called in the child process before logging.
    log_queue MUST be passed to the child process via inheritance
        Which essentially means that the log_queue must be a global, created in the same
        file as Parallel is initialized.
    This is called once per epoch - but the worker process itself is reused for the whole run.
    """
    current_proc = current_process().name
    if current_proc != "MainProcess":
        root = logging.getLogger()
        root.setLevel(verbosity)
        if (h := get_existing_handlers(QueueHandler)) is not None:
            # Re-point the existing handler
            h.queue = log_queue
        else:
            root.addHandler(QueueHandler(log_queue))
        # Disable freqtrade logging outside of the main process
        # This only leaves logging from the strategy (unless it's prefixed with "freqtrade.")
        # and eventually from other libraries.
        if verbosity > logging.DEBUG:
            logging.getLogger("freqtrade").setLevel(logging.WARNING)


def logging_mp_handle(q: Queue):
    """
    Handle logging from a child process.
    Must be called in the parent process to handle log messages from the child process.
    """

    try:
        while True:
            record = q.get(block=False)
            if record is None:
                break
            logger.handle(record)

    except Empty:
        pass
