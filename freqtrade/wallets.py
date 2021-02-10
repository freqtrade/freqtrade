# pragma pylint: disable=W0603
""" Wallet """

import logging
from copy import deepcopy
from typing import Any, Dict, NamedTuple

import arrow

from freqtrade.constants import UNLIMITED_STAKE_AMOUNT
from freqtrade.exceptions import DependencyException
from freqtrade.exchange import Exchange
from freqtrade.persistence import Trade
from freqtrade.state import RunMode


logger = logging.getLogger(__name__)


# wallet data structure
class Wallet(NamedTuple):
    currency: str
    free: float = 0
    used: float = 0
    total: float = 0


class Wallets:

    def __init__(self, config: dict, exchange: Exchange, skip_update: bool = False) -> None:
        self._config = config
        self._log = True
        self._exchange = exchange
        self._wallets: Dict[str, Wallet] = {}
        self.start_cap = config['dry_run_wallet']
        self._last_wallet_refresh = 0
        if not skip_update:
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
        closed_trades = Trade.get_trades_proxy(is_open=False)
        open_trades = Trade.get_trades_proxy(is_open=True)
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
        # Remove currencies no longer in get_balances output
        for currency in deepcopy(self._wallets):
            if currency not in balances:
                del self._wallets[currency]

    def update(self, require_update: bool = True) -> None:
        """
        Updates wallets from the configured version.
        By default, updates from the exchange.
        Update-skipping should only be used for user-invoked /balance calls, since
        for trading operations, the latest balance is needed.
        :param require_update: Allow skipping an update if balances were recently refreshed
        """
        if (require_update or (self._last_wallet_refresh + 3600 < arrow.utcnow().int_timestamp)):
            if (not self._config['dry_run'] or self._config.get('runmode') == RunMode.LIVE):
                self._update_live()
            else:
                self._update_dry()
            if self._log:
                logger.info('Wallets synced.')
            self._last_wallet_refresh = arrow.utcnow().int_timestamp

    def get_all_balances(self) -> Dict[str, Any]:
        return self._wallets

    def _get_available_stake_amount(self) -> float:
        """
        Return the total currently available balance in stake currency,
        respecting tradable_balance_ratio.
        Calculated as
        (<open_trade stakes> + free amount ) * tradable_balance_ratio - <open_trade stakes>
        """
        val_tied_up = Trade.total_open_trades_stakes()

        # Ensure <tradable_balance_ratio>% is used from the overall balance
        # Otherwise we'd risk lowering stakes with each open trade.
        # (tied up + current free) * ratio) - tied up
        available_amount = ((val_tied_up + self.get_free(self._config['stake_currency'])) *
                            self._config['tradable_balance_ratio']) - val_tied_up
        return available_amount

    def _calculate_unlimited_stake_amount(self, free_open_trades: int) -> float:
        """
        Calculate stake amount for "unlimited" stake amount
        :return: 0 if max number of trades reached, else stake_amount to use.
        """
        if not free_open_trades:
            return 0

        available_amount = self._get_available_stake_amount()

        return available_amount / free_open_trades

    def _check_available_stake_amount(self, stake_amount: float) -> float:
        """
        Check if stake amount can be fulfilled with the available balance
        for the stake currency
        :return: float: Stake amount
        :raise: DependencyException if balance is lower than stake-amount
        """
        available_amount = self._get_available_stake_amount()

        if self._config['amend_last_stake_amount']:
            # Remaining amount needs to be at least stake_amount * last_stake_amount_min_ratio
            # Otherwise the remaining amount is too low to trade.
            if available_amount > (stake_amount * self._config['last_stake_amount_min_ratio']):
                stake_amount = min(stake_amount, available_amount)
            else:
                stake_amount = 0

        if available_amount < stake_amount:
            raise DependencyException(
                f"Available balance ({available_amount} {self._config['stake_currency']}) is "
                f"lower than stake amount ({stake_amount} {self._config['stake_currency']})"
            )

        return stake_amount

    def get_trade_stake_amount(self, pair: str, free_open_trades: int, edge=None) -> float:
        """
        Calculate stake amount for the trade
        :return: float: Stake amount
        :raise: DependencyException if the available stake amount is too low
        """
        stake_amount: float
        # Ensure wallets are uptodate.
        self.update()

        if edge:
            stake_amount = edge.stake_amount(
                pair,
                self.get_free(self._config['stake_currency']),
                self.get_total(self._config['stake_currency']),
                Trade.total_open_trades_stakes()
            )
        else:
            stake_amount = self._config['stake_amount']
            if stake_amount == UNLIMITED_STAKE_AMOUNT:
                stake_amount = self._calculate_unlimited_stake_amount(free_open_trades)

        return self._check_available_stake_amount(stake_amount)
