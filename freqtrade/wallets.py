# pragma pylint: disable=W0603
""" Wallet """

import logging
from typing import Dict, NamedTuple, Any
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

        self.update()

    def get_free(self, currency) -> float:

        balance = self._wallets.get(currency)
        if balance and balance.free:
            return balance.free
        else:
            return 0

    def get_used(self, currency) -> float:

        balance = self._wallets.get(currency)
        if balance and balance.used:
            return balance.used
        else:
            return 0

    def get_total(self, currency) -> float:

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
            curr = trade.pair.split('/')[0]
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

    def update(self) -> None:
        if self._config['dry_run']:
            self._update_dry()
        else:
            self._update_live()
        logger.info('Wallets synced.')

    def get_all_balances(self) -> Dict[str, Any]:
        return self._wallets
