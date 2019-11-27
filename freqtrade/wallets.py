# pragma pylint: disable=W0603
""" Wallet """

import logging
from typing import Dict, NamedTuple, Any
from freqtrade.exchange import Exchange
from freqtrade import constants

logger = logging.getLogger(__name__)


# wallet data structure
class Wallet(NamedTuple):
    currency: str
    free: float = 0
    used: float = 0
    total: float = 0


class Wallets:

    def __init__(self, config: dict, exchange: Exchange) -> None:
        self._config = config
        self._exchange = exchange
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

        balances = self._exchange.get_balances()

        for currency in balances:
            self._wallets[currency] = Wallet(
                currency,
                balances[currency].get('free', None),
                balances[currency].get('used', None),
                balances[currency].get('total', None)
            )

        logger.info('Wallets synced.')

    def get_all_balances(self) -> Dict[str, Any]:
        return self._wallets
