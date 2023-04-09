import logging
import sys
from contextlib import contextmanager

from rich.progress import Progress


@contextmanager
def FtProgress(*args, **kwargs):
    """
    Wrapper around rich.progress.Progress to fix issues with logging.
    """
    try:
        __logger = kwargs.pop('logger', None)
        streamhandlers = [x for x in __logger.root.handlers if type(x) == logging.StreamHandler]
        __prior_stderr = []

        with Progress(*args, **kwargs) as progress:
            for handler in streamhandlers:
                __prior_stderr.append(handler.stream)
                handler.setStream(sys.stderr)

            yield progress

    finally:
        for idx, handler in  enumerate(streamhandlers):
            handler.setStream(__prior_stderr[idx])
