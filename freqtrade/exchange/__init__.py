# pragma pylint: disable=W0603
""" Cryptocurrency Exchanges support """
import logging
import ccxt
from random import randint
from typing import List, Dict, Any, Optional
from datetime import datetime

import arrow
from cachetools import cached, TTLCache

from freqtrade import OperationalException, DependencyException, NetworkException

logger = logging.getLogger(__name__)

# Current selected exchange
_API: ccxt.Exchange = None
_CONF: dict = {}

# Cache for ticker data
_TICKER_CACHE: dict = {}

# Holds all open sell orders for dry_run
_DRY_RUN_OPEN_ORDERS: Dict[str, Any] = {}


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

    if name not in ccxt.exchanges:
        raise OperationalException('Exchange {} is not supported'.format(name))

    try:
        _API = getattr(ccxt, name.lower())({
            'apiKey': exchange_config.get('key'),
            'secret': exchange_config.get('secret'),
            'password': exchange_config.get('password'),
            'uid': exchange_config.get('uid'),
        })
    except KeyError:
        raise OperationalException('Exchange {} is not supported'.format(name))

    # Check if all pairs are available
    validate_pairs(config['exchange']['pair_whitelist'])


def validate_pairs(pairs: List[str]) -> None:
    """
    Checks if all given pairs are tradable on the current exchange.
    Raises OperationalException if one pair is not available.
    :param pairs: list of pairs
    :return: None
    """

    try:
        markets = _API.load_markets()
    except ccxt.BaseError as e:
        logger.warning('Unable to validate pairs (assuming they are correct). Reason: %s', e)
        return

    stake_cur = _CONF['stake_currency']
    for pair in pairs:
        # Note: ccxt has BaseCurrency/QuoteCurrency format for pairs
        # pair = pair.replace('_', '/')

        # TODO: add a support for having coins in BTC/USDT format
        if not pair.endswith(stake_cur):
            raise OperationalException(
                'Pair {} not compatible with stake_currency: {}'.format(pair, stake_cur)
            )
        if pair not in markets:
            raise OperationalException(
                'Pair {} is not available at {}'.format(pair, _API.id.lower()))


def buy(pair: str, rate: float, amount: float) -> Dict:
    if _CONF['dry_run']:
        global _DRY_RUN_OPEN_ORDERS
        order_id = 'dry_run_buy_{}'.format(randint(0, 10**6))
        _DRY_RUN_OPEN_ORDERS[order_id] = {
            'pair': pair,
            'price': rate,
            'amount': amount,
            'type': 'limit',
            'side': 'buy',
            'remaining': 0.0,
            'datetime': arrow.utcnow().isoformat(),
            'status': 'closed'
        }
        return {'id': order_id}

    try:
        return _API.create_limit_buy_order(pair, amount, rate)
    except ccxt.InsufficientFunds as e:
        raise DependencyException(
            'Insufficient funds to create limit buy order on market {}.'
            'Tried to buy amount {} at rate {} (total {}).'
            'Message: {}'.format(pair, amount, rate, rate*amount, e)
        )
    except ccxt.InvalidOrder as e:
        raise DependencyException(
            'Could not create limit buy order on market {}.'
            'Tried to buy amount {} at rate {} (total {}).'
            'Message: {}'.format(pair, amount, rate, rate*amount, e)
        )
    except ccxt.NetworkError as e:
        raise NetworkException(
            'Could not place buy order due to networking error. Message: {}'.format(e)
        )
    except ccxt.BaseError as e:
        raise OperationalException(e)


def sell(pair: str, rate: float, amount: float) -> Dict:
    if _CONF['dry_run']:
        global _DRY_RUN_OPEN_ORDERS
        order_id = 'dry_run_sell_{}'.format(randint(0, 10**6))
        _DRY_RUN_OPEN_ORDERS[order_id] = {
            'pair': pair,
            'price': rate,
            'amount': amount,
            'type': 'limit',
            'side': 'sell',
            'remaining': 0.0,
            'datetime': arrow.utcnow().isoformat(),
            'status': 'closed'
        }
        return {'id': order_id}

    try:
        return _API.create_limit_sell_order(pair, amount, rate)
    except ccxt.InsufficientFunds as e:
        raise DependencyException(
            'Insufficient funds to create limit sell order on market {}.'
            'Tried to sell amount {} at rate {} (total {}).'
            'Message: {}'.format(pair, amount, rate, rate*amount, e)
        )
    except ccxt.InvalidOrder as e:
        raise DependencyException(
            'Could not create limit sell order on market {}.'
            'Tried to sell amount {} at rate {} (total {}).'
            'Message: {}'.format(pair, amount, rate, rate*amount, e)
        )
    except ccxt.NetworkError as e:
        raise NetworkException(
            'Could not place sell order due to networking error. Message: {}'.format(e)
        )
    except ccxt.BaseError as e:
        raise OperationalException(e)


def get_balance(currency: str) -> float:
    if _CONF['dry_run']:
        return 999.9

    # ccxt exception is already handled by get_balances
    balances = get_balances()
    return balances[currency]['free']


def get_balances() -> dict:
    if _CONF['dry_run']:
        return {}

    try:
        balances = _API.fetch_balance()
        # Remove additional info from ccxt results
        balances.pop("info", None)
        balances.pop("free", None)
        balances.pop("total", None)
        balances.pop("used", None)

        return balances
    except ccxt.NetworkError as e:
        raise NetworkException(
            'Could not get balance due to networking error. Message: {}'.format(e)
        )
    except ccxt.BaseError as e:
        raise OperationalException(e)


def get_ticker(pair: str, refresh: Optional[bool] = True) -> dict:
    global _TICKER_CACHE
    try:
        if not refresh:
            if _TICKER_CACHE:
                return _TICKER_CACHE
        _TICKER_CACHE = _API.fetch_ticker(pair)
        return _TICKER_CACHE
    except ccxt.NetworkError as e:
        raise NetworkException(
            'Could not load tickers due to networking error. Message: {}'.format(e)
        )
    except ccxt.BaseError as e:
        raise OperationalException(e)


@cached(TTLCache(maxsize=100, ttl=30))
def get_ticker_history(pair: str, tick_interval: str) -> List[Dict]:
    if 'fetchOHLCV' not in _API.has or not _API.has['fetchOHLCV']:
        raise OperationalException(
            'Exhange {} does not support fetching historical candlestick data.'.format(_API.name)
        )

    try:
        history = _API.fetch_ohlcv(pair, timeframe=tick_interval)
        history_json = []
        for candlestick in history:
            history_json.append({
                'T': datetime.fromtimestamp(candlestick[0]/1000.0).strftime('%Y-%m-%dT%H:%M:%S.%f'),
                'O': candlestick[1],
                'H': candlestick[2],
                'L': candlestick[3],
                'C': candlestick[4],
                'V': candlestick[5],
            })
        return history_json
    except IndexError as e:
        logger.warning('Empty ticker history. Msg %s', str(e))
        return []
    except ccxt.NetworkError as e:
        raise NetworkException(
            'Could not load ticker history due to networking error. Message: {}'.format(e)
        )
    except ccxt.BaseError as e:
        raise OperationalException('Could not fetch ticker data. Msg: {}'.format(e))


def cancel_order(order_id: str, pair: str) -> None:
    if _CONF['dry_run']:
        return

    try:
        return _API.cancel_order(order_id, pair)
    except ccxt.NetworkError as e:
        raise NetworkException(
            'Could not get order due to networking error. Message: {}'.format(e)
        )
    except ccxt.InvalidOrder as e:
        raise DependencyException(
            'Could not cancel order. Message: {}'.format(e)
        )
    except ccxt.BaseError as e:
        raise OperationalException(e)


def get_order(order_id: str, pair: str) -> Dict:
    if _CONF['dry_run']:
        order = _DRY_RUN_OPEN_ORDERS[order_id]
        order.update({
            'id': order_id
        })
        return order
    try:
        return _API.fetch_order(order_id, pair)
    except ccxt.NetworkError as e:
        raise NetworkException(
            'Could not get order due to networking error. Message: {}'.format(e)
        )
    except ccxt.InvalidOrder as e:
        raise DependencyException(
            'Could not get order. Message: {}'.format(e)
        )
    except ccxt.BaseError as e:
        raise OperationalException(e)


# TODO: reimplement, not part of ccxt
def get_pair_detail_url(pair: str) -> str:
    return ""


def get_markets() -> List[dict]:
    try:
        return _API.fetch_markets()
    except ccxt.NetworkError as e:
        raise NetworkException(
            'Could not load markets due to networking error. Message: {}'.format(e)
        )
    except ccxt.BaseError as e:
        raise OperationalException(e)


def get_name() -> str:
    return _API.name


def get_fee() -> float:
    # validate that markets are loaded before trying to get fee
    if _API.markets is None or len(_API.markets) == 0:
        _API.load_markets()

    return _API.calculate_fee('ETH/BTC', '', '', 1, 1)['rate']
