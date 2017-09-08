import enum
import logging
from typing import List

from bittrex.bittrex import Bittrex
from poloniex import Poloniex

logger = logging.getLogger(__name__)


cur_exchange = None
_api = None
_conf = {}


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
    global _api, cur_exchange

    _conf.update(config)

    if config['dry_run']:
        logger.info('Instance is running with dry_run enabled')

    use_poloniex = config.get('poloniex', {}).get('enabled', False)
    use_bittrex = config.get('bittrex', {}).get('enabled', False)

    if use_poloniex:
        cur_exchange = Exchange.POLONIEX
        _api = Poloniex(key=config['poloniex']['key'], secret=config['poloniex']['secret'])
    elif use_bittrex:
        cur_exchange = Exchange.BITTREX
        _api = Bittrex(api_key=config['bittrex']['key'], api_secret=config['bittrex']['secret'])
    else:
        raise RuntimeError('No exchange specified. Aborting!')

    # Check if all pairs are available
    markets = get_markets()
    for pair in config[cur_exchange.name.lower()]['pair_whitelist']:
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
    if _conf['dry_run']:
        return 'dry_run'
    elif cur_exchange == Exchange.POLONIEX:
        _api.buy(pair, rate, amount)
        # TODO: return order id
    elif cur_exchange == Exchange.BITTREX:
        data = _api.buy_limit(pair.replace('_', '-'), amount, rate)
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
    if _conf['dry_run']:
        return 'dry_run'
    elif cur_exchange == Exchange.POLONIEX:
        _api.sell(pair, rate, amount)
        # TODO: return order id
    elif cur_exchange == Exchange.BITTREX:
        data = _api.sell_limit(pair.replace('_', '-'), amount, rate)
        if not data['success']:
            raise RuntimeError('BITTREX: {}'.format(data['message']))
        return data['result']['uuid']


def get_balance(currency: str) -> float:
    """
    Get account balance.
    :param currency: currency as str, format: BTC
    :return: float
    """
    if _conf['dry_run']:
        return 999.9
    elif cur_exchange == Exchange.POLONIEX:
        data = _api.returnBalances()
        return float(data[currency])
    elif cur_exchange == Exchange.BITTREX:
        data = _api.get_balance(currency)
        if not data['success']:
            raise RuntimeError('BITTREX: {}'.format(data['message']))
        return float(data['result']['Balance'] or 0.0)


def get_ticker(pair: str) -> dict:
    """
    Get Ticker for given pair.
    :param pair: Pair as str, format: BTC_ETC
    :return: dict
    """
    if cur_exchange == Exchange.POLONIEX:
        data = _api.returnTicker()
        return {
            'bid': float(data[pair]['highestBid']),
            'ask': float(data[pair]['lowestAsk']),
            'last': float(data[pair]['last'])
        }
    elif cur_exchange == Exchange.BITTREX:
        data = _api.get_ticker(pair.replace('_', '-'))
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
    if _conf['dry_run']:
        pass
    elif cur_exchange == Exchange.POLONIEX:
        raise NotImplemented('Not implemented')
    elif cur_exchange == Exchange.BITTREX:
        data = _api.cancel(order_id)
        if not data['success']:
            raise RuntimeError('BITTREX: {}'.format(data['message']))


def get_open_orders(pair: str) -> List[dict]:
    """
    Get all open orders for given pair.
    :param pair: Pair as str, format: BTC_ETC
    :return: list of dicts
    """
    if _conf['dry_run']:
        return []
    elif cur_exchange == Exchange.POLONIEX:
        raise NotImplemented('Not implemented')
    elif cur_exchange == Exchange.BITTREX:
        data = _api.get_open_orders(pair.replace('_', '-'))
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
    if cur_exchange == Exchange.POLONIEX:
        raise NotImplemented('Not implemented')
    elif cur_exchange == Exchange.BITTREX:
        return 'https://bittrex.com/Market/Index?MarketName={}'.format(pair.replace('_', '-'))


def get_markets() -> List[str]:
    """
    Returns all available markets
    :return: list of all available pairs
    """
    if cur_exchange == Exchange.POLONIEX:
        # TODO: implement
        raise NotImplemented('Not implemented')
    elif cur_exchange == Exchange. BITTREX:
        data = _api.get_markets()
        if not data['success']:
            raise RuntimeError('BITTREX: {}'.format(data['message']))
        return [m['MarketName'].replace('-', '_') for m in data['result']]
