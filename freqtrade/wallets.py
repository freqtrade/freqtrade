# pragma pylint: disable=W0603
""" Wallet """

import logging
from typing import Any, Dict, NamedTuple

import arrow

from freqtrade.exchange import Exchange
from freqtrade.persistence import Trade

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
        self.start_cap = config['dry_run_wallet']
        self._last_wallet_refresh = 0
        self.update()

    def get_free(self, currency: str) -> float:
        balance = self._wallets.get(currency)
        if balance and balance.free:
            return balance.free
        else:
            return 0

    def get_used(self, currency: str) -> float:
        balance = self._wallets.get(currency)
        if balance and balance.used:
            return balance.used
        else:
            return 0

    def get_total(self, currency: str) -> float:
        balance = self._wallets.get(currency)
        if balance and balance.total:
            return balance.total
        else:
            return 0

    def _update_dry(self) -> None:
        """
        Update from database in dry-run mode
        - Apply apply profits of closed trades on top of stake amount
        - Subtract currently tied up stake_amount in open trades
        - update balances for currencies currently in trades
        """
        # Recreate _wallets to reset closed trade balances
        _wallets = {}
        closed_trades = Trade.get_trades(Trade.is_open.is_(False)).all()
        open_trades = Trade.get_trades(Trade.is_open.is_(True)).all()
        tot_profit = sum([trade.calc_profit() for trade in closed_trades])
        tot_in_trades = sum([trade.stake_amount for trade in open_trades])

        current_stake = self.start_cap + tot_profit - tot_in_trades
        _wallets[self._config['stake_currency']] = Wallet(
            self._config['stake_currency'],
            current_stake,
            0,
            current_stake
        )

        for trade in open_trades:
            curr = self._exchange.get_pair_base_currency(trade.pair)
            _wallets[curr] = Wallet(
                curr,
                trade.amount,
                0,
                trade.amount
            )
        self._wallets = _wallets

    def _update_live(self) -> None:
        balances = self._exchange.get_balances()

        for currency in balances:
            self._wallets[currency] = Wallet(
                currency,
                balances[currency].get('free', None),
                balances[currency].get('used', None),
                balances[currency].get('total', None)
            )

    def update(self, require_update: bool = True) -> None:
        """
        Updates wallets from the configured version.
        By default, updates from the exchange.
        Update-skipping should only be used for user-invoked /balance calls, since
        for trading operations, the latest balance is needed.
        :param require_update: Allow skipping an update if balances were recently refreshed
        """
        if (require_update or (self._last_wallet_refresh + 3600 < arrow.utcnow().timestamp)):
            if self._config['dry_run']:
                self._update_dry()
            else:
                self._update_live()
            logger.info('Wallets synced.')
            self._last_wallet_refresh = arrow.utcnow().timestamp

    def get_all_balances(self) -> Dict[str, Any]:
        return self._wallets
