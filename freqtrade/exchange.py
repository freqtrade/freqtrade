import enum
import logging
from typing import List

import arrow

from freqtrade.exchanges import Exchange
from freqtrade.exchanges.bittrex import Bittrex

logger = logging.getLogger(__name__)

# Current selected exchange
EXCHANGE: Exchange = None
_CONF: dict = {}


class Exchanges(enum.Enum):
    """
    Maps supported exchange names to correspondent classes.
    """
    BITTREX = Bittrex


def init(config: dict) -> None:
    """
    Initializes this module with the given config,
    it does basic validation whether the specified
    exchange and pairs are valid.
    :param config: config to use
    :return: None
    """
    global _CONF, EXCHANGE

    _CONF.update(config)

    if config['dry_run']:
        logger.info('Instance is running with dry_run enabled')

    exchange_config = config['exchange']
    name = exchange_config['name']

    # Find matching class for the given exchange name
    exchange_class = None
    for exchange in Exchanges:
        if name.upper() == exchange.name:
            exchange_class = exchange.value
            break
    if not exchange_class:
        raise RuntimeError('Exchange {} is not supported'.format(name))

    if not exchange_config.get('enabled', False):
        raise RuntimeError('Exchange {} is disabled'.format(name))

    EXCHANGE = exchange_class(exchange_config)

    # Check if all pairs are available
    validate_pairs(config['exchange']['pair_whitelist'])


def validate_pairs(pairs: List[str]) -> None:
    """
    Checks if all given pairs are tradable on the current exchange.
    Raises RuntimeError if one pair is not available.
    :param pairs: list of pairs
    :return: None
    """
    markets = EXCHANGE.get_markets()
    for pair in pairs:
        if pair not in markets:
            raise RuntimeError('Pair {} is not available at {}'.format(pair, EXCHANGE.name.lower()))


def buy(pair: str, rate: float, amount: float) -> str:
    if _CONF['dry_run']:
        return 'dry_run'

    return EXCHANGE.buy(pair, rate, amount)


def sell(pair: str, rate: float, amount: float) -> str:
    if _CONF['dry_run']:
        return 'dry_run'

    return EXCHANGE.sell(pair, rate, amount)


def get_balance(currency: str) -> float:
    if _CONF['dry_run']:
        return 999.9

    return EXCHANGE.get_balance(currency)


def get_ticker(pair: str) -> dict:
    return EXCHANGE.get_ticker(pair)


def get_ticker_history(pair: str, minimum_date: arrow.Arrow):
    return EXCHANGE.get_ticker_history(pair, minimum_date)


def cancel_order(order_id: str) -> None:
    if _CONF['dry_run']:
        return

    return EXCHANGE.cancel_order(order_id)


def get_open_orders(pair: str) -> List[dict]:
    if _CONF['dry_run']:
        return []

    return EXCHANGE.get_open_orders(pair)


def get_pair_detail_url(pair: str) -> str:
    return EXCHANGE.get_pair_detail_url(pair)


def get_markets() -> List[str]:
    return EXCHANGE.get_markets()
