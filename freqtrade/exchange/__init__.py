# pragma pylint: disable=W0603
""" Cryptocurrency Exchanges support """
import enum
import logging
import ccxt
from random import randint
from typing import List, Dict, Any, Optional
from cachetools import cached, TTLCache

import arrow
import requests

from freqtrade import OperationalException

logger = logging.getLogger(__name__)

# Current selected exchange
_API = None
_CONF: dict = {}
API_RETRY_COUNT = 4

# Holds all open sell orders for dry_run
_DRY_RUN_OPEN_ORDERS: Dict[str, Any] = {}

def retrier(f):
    def wrapper(*args, **kwargs):
        count = kwargs.pop('count', API_RETRY_COUNT)
        try:
            return f(*args, **kwargs)
        # TODO dont be a gotta-catch-them-all pokemon collector
        except Exception as ex:
            logger.warn('%s returned exception: "%s"', f, ex)
            if count > 0:
                count -= 1
                kwargs.update({'count': count})
                logger.warn('retrying %s still for %s times', f, count)
                return wrapper(*args, **kwargs)
            else:
                raise OperationalException('Giving up retrying: %s', f)
    return wrapper


def _get_market_url(exchange):
    "get market url for exchange"
    # TODO: PR to ccxt
    base = exchange.urls.get('www')
    market = ""
    if 'bittrex' in get_name():
        market = base + '/Market/Index?MarketName={}'
    if 'binance' in get_name():
        market = base + '/trade.html?symbol={}'

    return market


def init(config: dict) -> None:
    """
    Initializes this module with the given config,
    it does basic validation whether the specified
    exchange and pairs are valid.
    :param config: config to use
    :return: None
    """
    global _CONF, _API

    _CONF.update(config)

    if config['dry_run']:
        logger.info('Instance is running with dry_run enabled')

    exchange_config = config['exchange']

    # Find matching class for the given exchange name
    name = exchange_config['name']

    # TODO add check for a list of supported exchanges

    try:
        # exchange_class = Exchanges[name.upper()].value
        _API = getattr(ccxt, name.lower())({
            'apiKey': exchange_config.get('key'),
            'secret': exchange_config.get('secret'),
        })
        logger.info('Using Exchange %s', name.capitalize())
    except KeyError:
        raise OperationalException('Exchange {} is not supported'.format(name))

    # we need load api markets
    _API.load_markets()

    # Check if all pairs are available
    validate_pairs(config['exchange']['pair_whitelist'])


def validate_pairs(pairs: List[str]) -> None:
    """
    Checks if all given pairs are tradable on the current exchange.
    Raises OperationalException if one pair is not available.
    :param pairs: list of pairs
    :return: None
    """

    if not _API.markets:
        _API.load_markets()

    try:
        markets = _API.markets
    except requests.exceptions.RequestException as e:
        logger.warning('Unable to validate pairs (assuming they are correct). Reason: %s', e)
        return

    stake_cur = _CONF['stake_currency']
    for pair in pairs:
        # Note: ccxt has BaseCurrency/QuoteCurrency format for pairs
        pair = pair.replace('_', '/')
        # TODO: add a support for having coins in BTC/USDT format
        if not pair.endswith(stake_cur):
            raise OperationalException(
                'Pair {} not compatible with stake_currency: {}'.format(pair, stake_cur)
            )
        if pair not in markets:
            raise OperationalException(
                'Pair {} is not available at {}'.format(pair, _API.name.lower()))


def buy(pair: str, rate: float, amount: float) -> str:
    if _CONF['dry_run']:
        global _DRY_RUN_OPEN_ORDERS
        order_id = 'dry_run_buy_{}'.format(randint(0, 10**6))
        _DRY_RUN_OPEN_ORDERS[order_id] = {
            'pair': pair,
            'rate': rate,
            'amount': amount,
            'type': 'LIMIT_BUY',
            'remaining': 0.0,
            'opened': arrow.utcnow().datetime,
            'closed': arrow.utcnow().datetime,
        }
        return order_id

    return _API.buy(pair, rate, amount)


def sell(pair: str, rate: float, amount: float) -> str:
    if _CONF['dry_run']:
        global _DRY_RUN_OPEN_ORDERS
        order_id = 'dry_run_sell_{}'.format(randint(0, 10**6))
        _DRY_RUN_OPEN_ORDERS[order_id] = {
            'pair': pair,
            'rate': rate,
            'amount': amount,
            'type': 'LIMIT_SELL',
            'remaining': 0.0,
            'opened': arrow.utcnow().datetime,
            'closed': arrow.utcnow().datetime,
        }
        return order_id

    return _API.sell(pair, rate, amount)


def get_balance(currency: str) -> float:
    if _CONF['dry_run']:
        return 999.9

    return _API.fetch_balance()[currency]


def get_balances():
    if _CONF['dry_run']:
        return []

    return _API.fetch_balance()

# @cached(TTLCache(maxsize=100, ttl=30))
@retrier
def get_ticker(pair: str, refresh: Optional[bool] = True) -> dict:
    return _API.fetch_ticker(pair)


# @cached(TTLCache(maxsize=100, ttl=30))
@retrier
def get_ticker_history(pair: str, tick_interval) -> List[Dict]:
    # TODO: tickers need to be in format 1m,5m
    # fetch_ohlcv returns an [[datetime,o,h,l,c,v]]
    if not _API.markets:
        _API.load_markets()
    ohlcv = _API.fetch_ohlcv(pair, str(tick_interval)+'m')

    return ohlcv


def cancel_order(order_id: str) -> None:
    if _CONF['dry_run']:
        return

    return _API.cancel_order(order_id)


def get_order(order_id: str) -> Dict:
    if _CONF['dry_run']:
        order = _DRY_RUN_OPEN_ORDERS[order_id]
        order.update({
            'id': order_id
        })
        return order

    return _API.get_order(order_id)


def get_pair_detail_url(pair: str) -> str:
    return _get_market_url(_API).format(
        _API.markets[pair]['id']
    )


def get_markets() -> List[str]:
    return _API.get_markets()


def get_market_summaries() -> List[Dict]:
    return _API.fetch_tickers()


def get_name() -> str:
    return _API.__class__.__name__


def get_fee_maker() -> float:
    return _API.fees['trading']['maker']


def get_fee_taker() -> float:
    return _API.fees['trading']['taker']


def get_fee() -> float:
    return _API.fees['trading']


def get_wallet_health() -> List[Dict]:
    if not _API.markets:
        _API.load_markets()

    return _API.markets
