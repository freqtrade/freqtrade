import enum
import logging
from typing import List, Dict

import arrow

from freqtrade.exchange.bittrex import Bittrex
from freqtrade.exchange.interface import Exchange

logger = logging.getLogger(__name__)

# Current selected exchange
_API: Exchange = None
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
    global _CONF, _API

    _CONF.update(config)

    if config['dry_run']:
        logger.info('Instance is running with dry_run enabled')

    exchange_config = config['exchange']

    # Find matching class for the given exchange name
    name = exchange_config['name']
    try:
        exchange_class = Exchanges[name.upper()].value
    except KeyError:
        raise RuntimeError('Exchange {} is not supported'.format(name))

    _API = exchange_class(exchange_config)

    # Check if all pairs are available
    validate_pairs(config['exchange']['pair_whitelist'])


def validate_pairs(pairs: List[str]) -> None:
    """
    Checks if all given pairs are tradable on the current exchange.
    Raises RuntimeError if one pair is not available.
    :param pairs: list of pairs
    :return: None
    """
    markets = _API.get_markets()
    for pair in pairs:
        if pair not in markets:
            raise RuntimeError('Pair {} is not available at {}'.format(pair, _API.name.lower()))


def buy(pair: str, rate: float, amount: float) -> str:
    if _CONF['dry_run']:
        return 'dry_run'

    return _API.buy(pair, rate, amount)


def sell(pair: str, rate: float, amount: float) -> str:
    if _CONF['dry_run']:
        return 'dry_run'

    return _API.sell(pair, rate, amount)


def get_balance(currency: str) -> float:
    if _CONF['dry_run']:
        return 999.9

    return _API.get_balance(currency)


def get_balances():
    return _API.get_balances()


def get_ticker(pair: str) -> dict:
    return _API.get_ticker(pair)


def get_ticker_history(pair: str, minimum_date: arrow.Arrow):
    return _API.get_ticker_history(pair, minimum_date)


def cancel_order(order_id: str) -> None:
    if _CONF['dry_run']:
        return

    return _API.cancel_order(order_id)


def get_order(order_id: str) -> Dict:
    if _CONF['dry_run']:
        return {}

    return _API.get_order(order_id)


def get_pair_detail_url(pair: str) -> str:
    return _API.get_pair_detail_url(pair)


def get_markets() -> List[str]:
    return _API.get_markets()


def get_name() -> str:
    return _API.name


def get_sleep_time() -> float:
    return _API.sleep_time


def get_fee() -> float:
    return _API.fee
