import logging

from freqtrade.exceptions import StrategyError

logger = logging.getLogger(__name__)


def strategy_safe_wrapper(f, message: str = "", default_retval=None):
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as error:
            logger.warning(
                f"{message}"
                f"Strategy caused the following exception: {error}"
                f"{f}"
            )
            if not default_retval:
                raise StrategyError(str(error)) from error
            return default_retval
        except Exception as error:
            logger.exception(
                f"Unexpected error {error} calling {f}"
            )
            if not default_retval:
                raise StrategyError(str(error)) from error
            return default_retval

    return wrapper
