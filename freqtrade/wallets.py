# pragma pylint: disable=W0603
""" Wallet """
import logging
from typing import Dict, Any, NamedTuple, Optional
from collections import namedtuple
from freqtrade.exchange import Exchange

logger = logging.getLogger(__name__)


class Wallet(NamedTuple):
    exchange: Optional[str] = None
    currency: Optional[str] = None
    free: float = 0
    used: float = 0
    total: float = 0


class Wallets(object):

    # wallet data structure
    wallet = namedtuple(
        'wallet',
        ['exchange', 'currency', 'free', 'used', 'total']
    )

    def __init__(self, exchange: Exchange) -> None:
        self.exchange = exchange
        self.wallets: Dict[str, Any] = {}
        self.update()

    def update(self) -> None:
        balances = self.exchange.get_balances()

        for currency in balances:
            self.wallets[currency] = Wallet(
                self.exchange.id,
                currency,
                balances[currency].get('free', None),
                balances[currency].get('used', None),
                balances[currency].get('total', None)
            )

        logger.info('Wallets synced ...')
