import logging

from freqtrade.exceptions import TemporaryError

logger = logging.getLogger(__name__)


API_RETRY_COUNT = 4
BAD_EXCHANGES = {
    "bitmex": "Various reasons.",
    "bitstamp": "Does not provide history. "
                "Details in https://github.com/freqtrade/freqtrade/issues/1983",
    "hitbtc": "This API cannot be used with Freqtrade. "
              "Use `hitbtc2` exchange id to access this exchange.",
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


def retrier_async(f):
    async def wrapper(*args, **kwargs):
        count = kwargs.pop('count', API_RETRY_COUNT)
        try:
            return await f(*args, **kwargs)
        except TemporaryError as ex:
            logger.warning('%s() returned exception: "%s"', f.__name__, ex)
            if count > 0:
                count -= 1
                kwargs.update({'count': count})
                logger.warning('retrying %s() still for %s times', f.__name__, count)
                return await wrapper(*args, **kwargs)
            else:
                logger.warning('Giving up retrying: %s()', f.__name__)
                raise ex
    return wrapper


def retrier(f):
    def wrapper(*args, **kwargs):
        count = kwargs.pop('count', API_RETRY_COUNT)
        try:
            return f(*args, **kwargs)
        except TemporaryError as ex:
            logger.warning('%s() returned exception: "%s"', f.__name__, ex)
            if count > 0:
                count -= 1
                kwargs.update({'count': count})
                logger.warning('retrying %s() still for %s times', f.__name__, count)
                return wrapper(*args, **kwargs)
            else:
                logger.warning('Giving up retrying: %s()', f.__name__)
                raise ex
    return wrapper
