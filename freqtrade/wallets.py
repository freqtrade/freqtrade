# pragma pylint: disable=W0603
""" Wallet """
import logging
from typing import Dict
from collections import namedtuple
from freqtrade.exchange import Exchange

logger = logging.getLogger(__name__)


class Wallets(object):

    # wallet data structure
    wallet = namedtuple(
        'wallet',
        ['exchange', 'currency', 'free', 'used', 'total']
    )

    def __init__(self, exchange: Exchange) -> None:
        self.exchange = exchange
        self.wallets: Dict[str, self.wallet] = {}
        self._update_wallets()

    def _update_wallets(self) -> None:
        balances = self.exchange.get_balances()

        for currency in balances:
            info = {
                'exchange': self.exchange.id,
                'currency': currency,
                'free': balances[currency]['free'],
                'used': balances[currency]['used'],
                'total': balances[currency]['total']
            }

            self.wallets[currency] = self.wallet(**info)

        logger.info('Wallets synced ...')

    def update(self) -> None:
        self._update_wallets()
