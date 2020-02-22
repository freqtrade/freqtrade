import logging

from freqtrade.exceptions import StrategyError

logger = logging.getLogger(__name__)


def strategy_safe_wrapper(f, message: str = "", default_retval=None):
    """
    Wrapper around user-provided methods and functions.
    Caches all exceptions and returns either the default_retval (if it's not None) or raises
    a StrategyError exception, which then needs to be handled by the calling method.
    """
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as error:
            logger.warning(
                f"{message}"
                f"Strategy caused the following exception: {error}"
                f"{f}"
            )
            if default_retval is None:
                raise StrategyError(str(error)) from error
            return default_retval
        except Exception as error:
            logger.exception(
                f"{message}"
                f"Unexpected error {error} calling {f}"
            )
            if default_retval is None:
                raise StrategyError(str(error)) from error
            return default_retval

    return wrapper
