import logging
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from freqtrade.exceptions import StrategyError
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


F = TypeVar('F', bound=Callable[..., Any])


def strategy_safe_wrapper(f: F, message: str = "", default_retval=None, supress_error=False) -> F:
    """
    Wrapper around user-provided methods and functions.
    Caches all exceptions and returns either the default_retval (if it's not None) or raises
    a StrategyError exception, which then needs to be handled by the calling method.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            if 'trade' in kwargs:
                # Protect accidental modifications from within the strategy
                trade = kwargs['trade']
                if isinstance(trade, Trade):
                    if trade in Trade.query.session.dirty:
                        Trade.commit()
                    return_vals = f(*args, **kwargs)
                    if trade in Trade.query.session.dirty:
                        logger.warning(f"The `trade` parameter have changed "
                                       f"in the function `{f.__name__}`, this may "
                                       f"lead to unexpected behavior.")
                    return return_vals
            return f(*args, **kwargs)
        except ValueError as error:
            logger.warning(
                f"{message}"
                f"Strategy caused the following exception: {error}"
                f"{f}"
            )
            if default_retval is None and not supress_error:
                raise StrategyError(str(error)) from error
            return default_retval
        except Exception as error:
            logger.exception(
                f"{message}"
                f"Unexpected error {error} calling {f}"
            )
            if default_retval is None and not supress_error:
                raise StrategyError(str(error)) from error
            return default_retval

    return cast(F, wrapper)
