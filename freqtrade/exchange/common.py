import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, cast, overload

from freqtrade.constants import ExchangeConfig
from freqtrade.exceptions import DDosProtection, RetryableOrderError, TemporaryError
from freqtrade.mixins import LoggingMixin


logger = logging.getLogger(__name__)
__logging_mixin = None


def _reset_logging_mixin():
    """
    Reset global logging mixin - used in tests only.
    """
    global __logging_mixin
    __logging_mixin = LoggingMixin(logger)


def _get_logging_mixin():
    # Logging-mixin to cache kucoin responses
    # Only to be used in retrier
    global __logging_mixin
    if not __logging_mixin:
        __logging_mixin = LoggingMixin(logger)
    return __logging_mixin


# Maximum default retry count.
# Functions are always called RETRY_COUNT + 1 times (for the original call)
API_RETRY_COUNT = 4
API_FETCH_ORDER_RETRY_COUNT = 5

BAD_EXCHANGES = {
    "bitmex": "Various reasons.",
    "phemex": "Does not provide history.",
    "probit": "Requires additional, regular calls to `signIn()`.",
    "poloniex": "Does not provide fetch_order endpoint to fetch both open and closed orders.",
}

MAP_EXCHANGE_CHILDCLASS = {
    'binanceus': 'binance',
    'binanceje': 'binance',
    'binanceusdm': 'binance',
    'okex': 'okx',
    'gateio': 'gate',
}

SUPPORTED_EXCHANGES = [
    'binance',
    'bittrex',
    'gate',
    'huobi',
    'kraken',
    'okx',
]

EXCHANGE_HAS_REQUIRED = [
    # Required / private
    'fetchOrder',
    'cancelOrder',
    'createOrder',
    'fetchBalance',

    # Public endpoints
    'fetchOHLCV',
]

EXCHANGE_HAS_OPTIONAL = [
    # Private
    'fetchMyTrades',  # Trades for order - fee detection
    'createLimitOrder', 'createMarketOrder',  # Either OR for orders
    # 'setLeverage',  # Margin/Futures trading
    # 'setMarginMode',  # Margin/Futures trading
    # 'fetchFundingHistory', # Futures trading
    # Public
    'fetchOrderBook', 'fetchL2OrderBook', 'fetchTicker',  # OR for pricing
    'fetchTickers',  # For volumepairlist?
    'fetchTrades',  # Downloading trades data
    # 'fetchFundingRateHistory',  # Futures trading
    # 'fetchPositions',  # Futures trading
    # 'fetchLeverageTiers',  # Futures initialization
    # 'fetchMarketLeverageTiers',  # Futures initialization
    # 'fetchOpenOrders', 'fetchClosedOrders',  # 'fetchOrders',  # Refinding balance...
]


def remove_exchange_credentials(exchange_config: ExchangeConfig, dry_run: bool) -> None:
    """
    Removes exchange keys from the configuration and specifies dry-run
    Used for backtesting / hyperopt / edge and utils.
    Modifies the input dict!
    """
    if dry_run:
        exchange_config['key'] = ''
        exchange_config['apiKey'] = ''
        exchange_config['secret'] = ''
        exchange_config['password'] = ''
        exchange_config['uid'] = ''


def calculate_backoff(retrycount, max_retries):
    """
    Calculate backoff
    """
    return (max_retries - retrycount) ** 2 + 1


def retrier_async(f):
    async def wrapper(*args, **kwargs):
        count = kwargs.pop('count', API_RETRY_COUNT)
        kucoin = args[0].name == "KuCoin"  # Check if the exchange is KuCoin.
        try:
            return await f(*args, **kwargs)
        except TemporaryError as ex:
            msg = f'{f.__name__}() returned exception: "{ex}". '
            if count > 0:
                msg += f'Retrying still for {count} times.'
                count -= 1
                kwargs['count'] = count
                if isinstance(ex, DDosProtection):
                    if kucoin and "429000" in str(ex):
                        # Temporary fix for 429000 error on kucoin
                        # see https://github.com/freqtrade/freqtrade/issues/5700 for details.
                        _get_logging_mixin().log_once(
                            f"Kucoin 429 error, avoid triggering DDosProtection backoff delay. "
                            f"{count} tries left before giving up", logmethod=logger.warning)
                        # Reset msg to avoid logging too many times.
                        msg = ''
                    else:
                        backoff_delay = calculate_backoff(count + 1, API_RETRY_COUNT)
                        logger.info(f"Applying DDosProtection backoff delay: {backoff_delay}")
                        await asyncio.sleep(backoff_delay)
                if msg:
                    logger.warning(msg)
                return await wrapper(*args, **kwargs)
            else:
                logger.warning(msg + 'Giving up.')
                raise ex
    return wrapper


F = TypeVar('F', bound=Callable[..., Any])


# Type shenanigans
@overload
def retrier(_func: F) -> F:
    ...


@overload
def retrier(*, retries=API_RETRY_COUNT) -> Callable[[F], F]:
    ...


def retrier(_func: Optional[F] = None, *, retries=API_RETRY_COUNT):
    def decorator(f: F) -> F:
        @wraps(f)
        def wrapper(*args, **kwargs):
            count = kwargs.pop('count', retries)
            try:
                return f(*args, **kwargs)
            except (TemporaryError, RetryableOrderError) as ex:
                msg = f'{f.__name__}() returned exception: "{ex}". '
                if count > 0:
                    logger.warning(msg + f'Retrying still for {count} times.')
                    count -= 1
                    kwargs.update({'count': count})
                    if isinstance(ex, (DDosProtection, RetryableOrderError)):
                        # increasing backoff
                        backoff_delay = calculate_backoff(count + 1, retries)
                        logger.info(f"Applying DDosProtection backoff delay: {backoff_delay}")
                        time.sleep(backoff_delay)
                    return wrapper(*args, **kwargs)
                else:
                    logger.warning(msg + 'Giving up.')
                    raise ex
        return cast(F, wrapper)
    # Support both @retrier and @retrier(retries=2) syntax
    if _func is None:
        return decorator
    else:
        return decorator(_func)
