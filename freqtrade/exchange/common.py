import asyncio
import logging
import time
from functools import wraps

from freqtrade.exceptions import DDosProtection, RetryableOrderError, TemporaryError


logger = logging.getLogger(__name__)


# Maximum default retry count.
# Functions are always called RETRY_COUNT + 1 times (for the original call)
API_RETRY_COUNT = 4
API_FETCH_ORDER_RETRY_COUNT = 5

BAD_EXCHANGES = {
    "bitmex": "Various reasons.",
    "phemex": "Does not provide history. ",
    "poloniex": "Does not provide fetch_order endpoint to fetch both open and closed orders.",
}

MAP_EXCHANGE_CHILDCLASS = {
    'binanceus': 'binance',
    'binanceje': 'binance',
}


EXCHANGE_HAS_REQUIRED = [
    # Required / private
    'fetchOrder',
    'cancelOrder',
    'createOrder',
    # 'createLimitOrder', 'createMarketOrder',
    'fetchBalance',

    # Public endpoints
    'loadMarkets',
    'fetchOHLCV',
]

EXCHANGE_HAS_OPTIONAL = [
    # Private
    'fetchMyTrades',  # Trades for order - fee detection
    # Public
    'fetchOrderBook', 'fetchL2OrderBook', 'fetchTicker',  # OR for pricing
    'fetchTickers',  # For volumepairlist?
    'fetchTrades',  # Downloading trades data
]


def remove_credentials(config) -> None:
    """
    Removes exchange keys from the configuration and specifies dry-run
    Used for backtesting / hyperopt / edge and utils.
    Modifies the input dict!
    """
    if config.get('dry_run', False):
        config['exchange']['key'] = ''
        config['exchange']['secret'] = ''
        config['exchange']['password'] = ''
        config['exchange']['uid'] = ''


def calculate_backoff(retrycount, max_retries):
    """
    Calculate backoff
    """
    return (max_retries - retrycount) ** 2 + 1


def retrier_async(f):
    async def wrapper(*args, **kwargs):
        count = kwargs.pop('count', API_RETRY_COUNT)
        try:
            return await f(*args, **kwargs)
        except TemporaryError as ex:
            logger.warning('%s() returned exception: "%s"', f.__name__, ex)
            if count > 0:
                logger.warning('retrying %s() still for %s times', f.__name__, count)
                count -= 1
                kwargs.update({'count': count})
                if isinstance(ex, DDosProtection):
                    backoff_delay = calculate_backoff(count + 1, API_RETRY_COUNT)
                    logger.info(f"Applying DDosProtection backoff delay: {backoff_delay}")
                    await asyncio.sleep(backoff_delay)
                return await wrapper(*args, **kwargs)
            else:
                logger.warning('Giving up retrying: %s()', f.__name__)
                raise ex
    return wrapper


def retrier(_func=None, retries=API_RETRY_COUNT):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            count = kwargs.pop('count', retries)
            try:
                return f(*args, **kwargs)
            except (TemporaryError, RetryableOrderError) as ex:
                logger.warning('%s() returned exception: "%s"', f.__name__, ex)
                if count > 0:
                    logger.warning('retrying %s() still for %s times', f.__name__, count)
                    count -= 1
                    kwargs.update({'count': count})
                    if isinstance(ex, (DDosProtection, RetryableOrderError)):
                        # increasing backoff
                        backoff_delay = calculate_backoff(count + 1, retries)
                        logger.info(f"Applying DDosProtection backoff delay: {backoff_delay}")
                        time.sleep(backoff_delay)
                    return wrapper(*args, **kwargs)
                else:
                    logger.warning('Giving up retrying: %s()', f.__name__)
                    raise ex
        return wrapper
    # Support both @retrier and @retrier(retries=2) syntax
    if _func is None:
        return decorator
    else:
        return decorator(_func)
