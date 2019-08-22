# pragma pylint: disable=W0603
""" Wallet """

import logging
from typing import Dict, NamedTuple, Optional
from freqtrade.exchange import Exchange, get_exchange
from freqtrade import constants

logger = logging.getLogger(__name__)


# wallet data structure
class Wallet(NamedTuple):
    currency: str
    free: float = 0
    used: float = 0
    total: float = 0


class Wallets(object):

    def __init__(self, config: dict, exchange_name: Optional[str] = None) -> None:
        self._config = config
        self._exchange_name = exchange_name
        self._wallets: Dict[str, Wallet] = {}

        self.update()

    def get_free(self, currency) -> float:

        if self._config['dry_run']:
            return self._config.get('dry_run_wallet', constants.DRY_RUN_WALLET)

        balance = self._wallets.get(currency)
        if balance and balance.free:
            return balance.free
        else:
            return 0

    def get_used(self, currency) -> float:

        if self._config['dry_run']:
            return self._config.get('dry_run_wallet', constants.DRY_RUN_WALLET)

        balance = self._wallets.get(currency)
        if balance and balance.used:
            return balance.used
        else:
            return 0

    def get_total(self, currency) -> float:

        if self._config['dry_run']:
            return self._config.get('dry_run_wallet', constants.DRY_RUN_WALLET)

        balance = self._wallets.get(currency)
        if balance and balance.total:
            return balance.total
        else:
            return 0

    def update(self) -> None:

        exchange: Exchange = get_exchange(self._config, self._exchange_name)
        balances = exchange.get_balances()

        for currency in balances:
            self._wallets[currency] = Wallet(
                currency,
                balances[currency].get('free', None),
                balances[currency].get('used', None),
                balances[currency].get('total', None)
            )

        logger.info('Wallets synced.')
