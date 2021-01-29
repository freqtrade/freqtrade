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
    "bitstamp": "Does not provide history. "
                "Details in https://github.com/freqtrade/freqtrade/issues/1983",
    "hitbtc": "This API cannot be used with Freqtrade. "
              "Use `hitbtc2` exchange id to access this exchange.",
    "phemex": "Does not provide history. ",
    "poloniex": "Does not provide fetch_order endpoint to fetch both open and closed orders.",
    **dict.fromkeys([
        'adara',
        'anxpro',
        'bigone',
        'coinbase',
        'coinexchange',
        'coinmarketcap',
        'lykke',
        'xbtce',
    ], "Does not provide timeframes. ccxt fetchOHLCV: False"),
    **dict.fromkeys([
        'bcex',
        'bit2c',
        'bitbay',
        'bitflyer',
        'bitforex',
        'bithumb',
        'bitso',
        'bitstamp1',
        'bl3p',
        'braziliex',
        'btcbox',
        'btcchina',
        'btctradeim',
        'btctradeua',
        'bxinth',
        'chilebit',
        'coincheck',
        'coinegg',
        'coinfalcon',
        'coinfloor',
        'coingi',
        'coinmate',
        'coinone',
        'coinspot',
        'coolcoin',
        'crypton',
        'deribit',
        'exmo',
        'exx',
        'flowbtc',
        'foxbit',
        'fybse',
        # 'hitbtc',
        'ice3x',
        'independentreserve',
        'indodax',
        'itbit',
        'lakebtc',
        'latoken',
        'liquid',
        'livecoin',
        'luno',
        'mixcoins',
        'negociecoins',
        'nova',
        'paymium',
        'southxchange',
        'stronghold',
        'surbitcoin',
        'therock',
        'tidex',
        'vaultoro',
        'vbtc',
        'virwox',
        'yobit',
        'zaif',
    ], "Does not provide timeframes. ccxt fetchOHLCV: emulated"),
}

MAP_EXCHANGE_CHILDCLASS = {
    'binanceus': 'binance',
    'binanceje': 'binance',
}


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
                    if isinstance(ex, DDosProtection) or isinstance(ex, RetryableOrderError):
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
