import enum
import logging
from typing import List

from bittrex.bittrex import Bittrex
from poloniex import Poloniex

logger = logging.getLogger(__name__)

# Current selected exchange
EXCHANGE = None
_API = None
_CONF = {}


class Exchange(enum.Enum):
    POLONIEX = 0
    BITTREX = 1


def init(config: dict) -> None:
    """
    Initializes this module with the given config,
    it does basic validation whether the specified
    exchange and pairs are valid.
    :param config: config to use
    :return: None
    """
    global _API, EXCHANGE

    _CONF.update(config)

    if config['dry_run']:
        logger.info('Instance is running with dry_run enabled')

    use_poloniex = config.get('poloniex', {}).get('enabled', False)
    use_bittrex = config.get('bittrex', {}).get('enabled', False)

    if use_poloniex:
        EXCHANGE = Exchange.POLONIEX
        _API = Poloniex(key=config['poloniex']['key'], secret=config['poloniex']['secret'])
    elif use_bittrex:
        EXCHANGE = Exchange.BITTREX
        _API = Bittrex(api_key=config['bittrex']['key'], api_secret=config['bittrex']['secret'])
    else:
        raise RuntimeError('No exchange specified. Aborting!')

    # Check if all pairs are available
    markets = get_markets()
    for pair in config[EXCHANGE.name.lower()]['pair_whitelist']:
        if pair not in markets:
            raise RuntimeError('Pair {} is not available at Poloniex'.format(pair))


def buy(pair: str, rate: float, amount: float) -> str:
    """
    Places a limit buy order.
    :param pair: Pair as str, format: BTC_ETH
    :param rate: Rate limit for order
    :param amount: The amount to purchase
    :return: order_id of the placed buy order
    """
    if _CONF['dry_run']:
        return 'dry_run'
    elif EXCHANGE == Exchange.POLONIEX:
        _API.buy(pair, rate, amount)
        # TODO: return order id
    elif EXCHANGE == Exchange.BITTREX:
        data = _API.buy_limit(pair.replace('_', '-'), amount, rate)
        if not data['success']:
            raise RuntimeError('BITTREX: {}'.format(data['message']))
        return data['result']['uuid']


def sell(pair: str, rate: float, amount: float) -> str:
    """
    Places a limit sell order.
    :param pair: Pair as str, format: BTC_ETH
    :param rate: Rate limit for order
    :param amount: The amount to sell
    :return: None
    """
    if _CONF['dry_run']:
        return 'dry_run'
    elif EXCHANGE == Exchange.POLONIEX:
        _API.sell(pair, rate, amount)
        # TODO: return order id
    elif EXCHANGE == Exchange.BITTREX:
        data = _API.sell_limit(pair.replace('_', '-'), amount, rate)
        if not data['success']:
            raise RuntimeError('BITTREX: {}'.format(data['message']))
        return data['result']['uuid']


def get_balance(currency: str) -> float:
    """
    Get account balance.
    :param currency: currency as str, format: BTC
    :return: float
    """
    if _CONF['dry_run']:
        return 999.9
    elif EXCHANGE == Exchange.POLONIEX:
        data = _API.returnBalances()
        return float(data[currency])
    elif EXCHANGE == Exchange.BITTREX:
        data = _API.get_balance(currency)
        if not data['success']:
            raise RuntimeError('BITTREX: {}'.format(data['message']))
        return float(data['result']['Balance'] or 0.0)


def get_ticker(pair: str) -> dict:
    """
    Get Ticker for given pair.
    :param pair: Pair as str, format: BTC_ETC
    :return: dict
    """
    if EXCHANGE == Exchange.POLONIEX:
        data = _API.returnTicker()
        return {
            'bid': float(data[pair]['highestBid']),
            'ask': float(data[pair]['lowestAsk']),
            'last': float(data[pair]['last'])
        }
    elif EXCHANGE == Exchange.BITTREX:
        data = _API.get_ticker(pair.replace('_', '-'))
        if not data['success']:
            raise RuntimeError('BITTREX: {}'.format(data['message']))
        return {
            'bid': float(data['result']['Bid']),
            'ask': float(data['result']['Ask']),
            'last': float(data['result']['Last']),
        }


def cancel_order(order_id: str) -> None:
    """
    Cancel order for given order_id
    :param order_id: id as str
    :return: None
    """
    if _CONF['dry_run']:
        pass
    elif EXCHANGE == Exchange.POLONIEX:
        raise NotImplemented('Not implemented')
    elif EXCHANGE == Exchange.BITTREX:
        data = _API.cancel(order_id)
        if not data['success']:
            raise RuntimeError('BITTREX: {}'.format(data['message']))


def get_open_orders(pair: str) -> List[dict]:
    """
    Get all open orders for given pair.
    :param pair: Pair as str, format: BTC_ETC
    :return: list of dicts
    """
    if _CONF['dry_run']:
        return []
    elif EXCHANGE == Exchange.POLONIEX:
        raise NotImplemented('Not implemented')
    elif EXCHANGE == Exchange.BITTREX:
        data = _API.get_open_orders(pair.replace('_', '-'))
        if not data['success']:
            raise RuntimeError('BITTREX: {}'.format(data['message']))
        return [{
            'id': entry['OrderUuid'],
            'type': entry['OrderType'],
            'opened': entry['Opened'],
            'rate': entry['PricePerUnit'],
            'amount': entry['Quantity'],
            'remaining': entry['QuantityRemaining'],
        } for entry in data['result']]


def get_pair_detail_url(pair: str) -> str:
    """
    Returns the market detail url for the given pair
    :param pair: pair as str, format: BTC_ANT
    :return: url as str
    """
    if EXCHANGE == Exchange.POLONIEX:
        raise NotImplemented('Not implemented')
    elif EXCHANGE == Exchange.BITTREX:
        return 'https://bittrex.com/Market/Index?MarketName={}'.format(pair.replace('_', '-'))


def get_markets() -> List[str]:
    """
    Returns all available markets
    :return: list of all available pairs
    """
    if EXCHANGE == Exchange.POLONIEX:
        # TODO: implement
        raise NotImplemented('Not implemented')
    elif EXCHANGE == Exchange. BITTREX:
        data = _API.get_markets()
        if not data['success']:
            raise RuntimeError('BITTREX: {}'.format(data['message']))
        return [m['MarketName'].replace('-', '_') for m in data['result']]
