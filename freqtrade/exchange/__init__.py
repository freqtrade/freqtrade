# pragma pylint: disable=W0603
""" Cryptocurrency Exchanges support """
import logging
from random import randint
from typing import List, Dict, Any, Optional

import ccxt
import arrow

from freqtrade import OperationalException, DependencyException, TemporaryError


logger = logging.getLogger(__name__)

# Current selected exchange
_API: ccxt.Exchange = None

_CONF: Dict = {}
API_RETRY_COUNT = 4

# Holds all open sell orders for dry_run
_DRY_RUN_OPEN_ORDERS: Dict[str, Any] = {}

# Urls to exchange markets, insert quote and base with .format()
_EXCHANGE_URLS = {
    ccxt.bittrex.__name__: '/Market/Index?MarketName={quote}-{base}',
    ccxt.binance.__name__: '/tradeDetail.html?symbol={base}_{quote}'
}


def retrier(f):
    def wrapper(*args, **kwargs):
        count = kwargs.pop('count', API_RETRY_COUNT)
        try:
            return f(*args, **kwargs)
        except (TemporaryError, DependencyException) as ex:
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
            'enableRateLimit': True,
        })
    except (KeyError, AttributeError):
        raise OperationalException('Exchange {} is not supported'.format(name))

    logger.info('Using Exchange "%s"', get_name())

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
        # TODO: add a support for having coins in BTC/USDT format
        if not pair.endswith(stake_cur):
            raise OperationalException(
                'Pair {} not compatible with stake_currency: {}'.format(pair, stake_cur)
            )
        if pair not in markets:
            raise OperationalException(
                'Pair {} is not available at {}'.format(pair, get_name()))


def exchange_has(endpoint: str) -> bool:
    """
    Checks if exchange implements a specific API endpoint.
    Wrapper around ccxt 'has' attribute
    :param endpoint: Name of endpoint (e.g. 'fetchOHLCV', 'fetchTickers')
    :return: bool
    """
    return endpoint in _API.has and _API.has[endpoint]


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
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        raise TemporaryError(
            'Could not place buy order due to {}. Message: {}'.format(
                e.__class__.__name__, e))
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
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        raise TemporaryError(
            'Could not place sell order due to {}. Message: {}'.format(
                e.__class__.__name__, e))
    except ccxt.BaseError as e:
        raise OperationalException(e)


def get_balance(currency: str) -> float:
    if _CONF['dry_run']:
        return 999.9

    # ccxt exception is already handled by get_balances
    balances = get_balances()
    return balances[currency]['free']


@retrier
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
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        raise TemporaryError(
            'Could not get balance due to {}. Message: {}'.format(
                e.__class__.__name__, e))
    except ccxt.BaseError as e:
        raise OperationalException(e)


@retrier
def get_tickers() -> Dict:
    try:
        return _API.fetch_tickers()
    except ccxt.NotSupported as e:
        raise OperationalException(
            'Exchange {} does not support fetching tickers in batch.'
            'Message: {}'.format(_API.name, e)
        )
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        raise TemporaryError(
            'Could not load tickers due to {}. Message: {}'.format(
                e.__class__.__name__, e))
    except ccxt.BaseError as e:
        raise OperationalException(e)


# TODO: remove refresh argument, keeping it to keep track of where it was intended to be used
@retrier
def get_ticker(pair: str, refresh: Optional[bool] = True) -> dict:
    try:
        return _API.fetch_ticker(pair)
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        raise TemporaryError(
            'Could not load ticker history due to {}. Message: {}'.format(
                e.__class__.__name__, e))
    except ccxt.BaseError as e:
        raise OperationalException(e)


@retrier
def get_ticker_history(pair: str, tick_interval: str) -> List[Dict]:
    try:
        return _API.fetch_ohlcv(pair, timeframe=tick_interval)
    except ccxt.NotSupported as e:
        raise OperationalException(
            'Exchange {} does not support fetching historical candlestick data.'
            'Message: {}'.format(_API.name, e)
        )
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        raise TemporaryError(
            'Could not load ticker history due to {}. Message: {}'.format(
                e.__class__.__name__, e))
    except ccxt.BaseError as e:
        raise OperationalException('Could not fetch ticker data. Msg: {}'.format(e))


@retrier
def cancel_order(order_id: str, pair: str) -> None:
    if _CONF['dry_run']:
        return

    try:
        return _API.cancel_order(order_id, pair)
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        raise TemporaryError(
            'Could not cancel order due to {}. Message: {}'.format(
                e.__class__.__name__, e))
    except ccxt.BaseError as e:
        raise OperationalException(e)


@retrier
def get_order(order_id: str, pair: str) -> Dict:
    if _CONF['dry_run']:
        order = _DRY_RUN_OPEN_ORDERS[order_id]
        order.update({
            'id': order_id
        })
        return order
    try:
        return _API.fetch_order(order_id, pair)
    except ccxt.InvalidOrder as e:
        raise DependencyException(
            'Could not get order. Message: {}'.format(e)
        )
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        raise TemporaryError(
            'Could not get order due to {}. Message: {}'.format(
                e.__class__.__name__, e))
    except ccxt.BaseError as e:
        raise OperationalException(e)


@retrier
def get_pair_detail_url(pair: str) -> str:
    try:
        url_base = _API.urls.get('www')
        base, quote = pair.split('/')

        return url_base + _EXCHANGE_URLS[_API.id].format(base=base, quote=quote)
    except KeyError:
        logger.warning('Could not get exchange url for %s', get_name())
        return ""


@retrier
def get_markets() -> List[dict]:
    try:
        return _API.fetch_markets()
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        raise TemporaryError(
            'Could not load markets due to {}. Message: {}'.format(
                e.__class__.__name__, e))
    except ccxt.BaseError as e:
        raise OperationalException(e)


def get_name() -> str:
    return _API.name


def get_id() -> str:
    return _API.id


@retrier
def get_fee(symbol='ETH/BTC', type='', side='', amount=1,
            price=1, taker_or_maker='maker') -> float:
    try:
        # validate that markets are loaded before trying to get fee
        if _API.markets is None or len(_API.markets) == 0:
            _API.load_markets()

        return _API.calculate_fee(symbol=symbol, type=type, side=side, amount=amount,
                                  price=price, takerOrMaker=taker_or_maker)['rate']
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        raise TemporaryError(
            'Could not get fee info due to {}. Message: {}'.format(
                e.__class__.__name__, e))
    except ccxt.BaseError as e:
        raise OperationalException(e)
