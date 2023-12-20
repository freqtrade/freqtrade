# pragma pylint: disable=W0603
""" Wallet """

import logging
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, NamedTuple, Optional

from freqtrade.constants import UNLIMITED_STAKE_AMOUNT, Config, IntOrInf
from freqtrade.enums import RunMode, TradingMode
from freqtrade.exceptions import DependencyException
from freqtrade.exchange import Exchange
from freqtrade.misc import safe_value_fallback
from freqtrade.persistence import LocalTrade, Trade
from freqtrade.util.datetime_helpers import dt_now


logger = logging.getLogger(__name__)


# wallet data structure
class Wallet(NamedTuple):
    currency: str
    free: float = 0
    used: float = 0
    total: float = 0


class PositionWallet(NamedTuple):
    symbol: str
    position: float = 0
    leverage: float = 0
    collateral: float = 0
    side: str = 'long'


class Wallets:

    def __init__(self, config: Config, exchange: Exchange, log: bool = True) -> None:
        self._config = config
        self._log = log
        self._exchange = exchange
        self._wallets: Dict[str, Wallet] = {}
        self._positions: Dict[str, PositionWallet] = {}
        self.start_cap = config['dry_run_wallet']
        self._last_wallet_refresh: Optional[datetime] = None
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
        _positions = {}
        open_trades = Trade.get_trades_proxy(is_open=True)
        # If not backtesting...
        # TODO: potentially remove the ._log workaround to determine backtest mode.
        if self._log:
            tot_profit = Trade.get_total_closed_profit()
        else:
            tot_profit = LocalTrade.total_profit
        tot_profit += sum(trade.realized_profit for trade in open_trades)
        tot_in_trades = sum(trade.stake_amount for trade in open_trades)
        used_stake = 0.0

        if self._config.get('trading_mode', 'spot') != TradingMode.FUTURES:
            current_stake = self.start_cap + tot_profit - tot_in_trades
            total_stake = current_stake
            for trade in open_trades:
                curr = self._exchange.get_pair_base_currency(trade.pair)
                _wallets[curr] = Wallet(
                    curr,
                    trade.amount,
                    0,
                    trade.amount
                )
        else:
            tot_in_trades = 0
            for position in open_trades:
                # size = self._exchange._contracts_to_amount(position.pair, position['contracts'])
                size = position.amount
                collateral = position.stake_amount
                leverage = position.leverage
                tot_in_trades += collateral
                _positions[position.pair] = PositionWallet(
                    position.pair, position=size,
                    leverage=leverage,
                    collateral=collateral,
                    side=position.trade_direction
                )
            current_stake = self.start_cap + tot_profit - tot_in_trades
            used_stake = tot_in_trades
            total_stake = current_stake + tot_in_trades

        _wallets[self._config['stake_currency']] = Wallet(
            currency=self._config['stake_currency'],
            free=current_stake,
            used=used_stake,
            total=total_stake
        )
        self._wallets = _wallets
        self._positions = _positions

    def _update_live(self) -> None:
        balances = self._exchange.get_balances()

        for currency in balances:
            if isinstance(balances[currency], dict):
                self._wallets[currency] = Wallet(
                    currency,
                    balances[currency].get('free'),
                    balances[currency].get('used'),
                    balances[currency].get('total')
                )
        # Remove currencies no longer in get_balances output
        for currency in deepcopy(self._wallets):
            if currency not in balances:
                del self._wallets[currency]

        positions = self._exchange.fetch_positions()
        self._positions = {}
        for position in positions:
            symbol = position['symbol']
            if position['side'] is None or position['collateral'] == 0.0:
                # Position is not open ...
                continue
            size = self._exchange._contracts_to_amount(symbol, position['contracts'])
            collateral = safe_value_fallback(position, 'collateral', 'initialMargin', 0.0)
            leverage = position['leverage']
            self._positions[symbol] = PositionWallet(
                symbol, position=size,
                leverage=leverage,
                collateral=collateral,
                side=position['side']
            )

    def update(self, require_update: bool = True) -> None:
        """
        Updates wallets from the configured version.
        By default, updates from the exchange.
        Update-skipping should only be used for user-invoked /balance calls, since
        for trading operations, the latest balance is needed.
        :param require_update: Allow skipping an update if balances were recently refreshed
        """
        now = dt_now()
        if (
            require_update
            or self._last_wallet_refresh is None
            or (self._last_wallet_refresh + timedelta(seconds=3600) < now)
        ):
            if (not self._config['dry_run'] or self._config.get('runmode') == RunMode.LIVE):
                self._update_live()
            else:
                self._update_dry()
            if self._log:
                logger.info('Wallets synced.')
            self._last_wallet_refresh = dt_now()

    def get_all_balances(self) -> Dict[str, Wallet]:
        return self._wallets

    def get_all_positions(self) -> Dict[str, PositionWallet]:
        return self._positions

    def _check_exit_amount(self, trade: Trade) -> bool:
        if trade.trading_mode != TradingMode.FUTURES:
            # Slightly higher offset than in safe_exit_amount.
            wallet_amount: float = self.get_total(trade.safe_base_currency) * (2 - 0.981)
        else:
            # wallet_amount: float = self.wallets.get_free(trade.safe_base_currency)
            position = self._positions.get(trade.pair)
            if position is None:
                # We don't own anything :O
                return False
            wallet_amount = position.position

        if wallet_amount >= trade.amount:
            return True
        return False

    def check_exit_amount(self, trade: Trade) -> bool:
        """
        Checks if the exit amount is available in the wallet.
        :param trade: Trade to check
        :return: True if the exit amount is available, False otherwise
        """
        if not self._check_exit_amount(trade):
            # Update wallets just to make sure
            self.update()
            return self._check_exit_amount(trade)

        return True

    def get_starting_balance(self) -> float:
        """
        Retrieves starting balance - based on either available capital,
        or by using current balance subtracting
        """
        if "available_capital" in self._config:
            return self._config['available_capital']
        else:
            tot_profit = Trade.get_total_closed_profit()
            open_stakes = Trade.total_open_trades_stakes()
            available_balance = self.get_free(self._config['stake_currency'])
            return available_balance - tot_profit + open_stakes

    def get_total_stake_amount(self):
        """
        Return the total currently available balance in stake currency, including tied up stake and
        respecting tradable_balance_ratio.
        Calculated as
        (<open_trade stakes> + free amount) * tradable_balance_ratio
        """
        val_tied_up = Trade.total_open_trades_stakes()
        if "available_capital" in self._config:
            starting_balance = self._config['available_capital']
            tot_profit = Trade.get_total_closed_profit()
            available_amount = starting_balance + tot_profit

        else:
            # Ensure <tradable_balance_ratio>% is used from the overall balance
            # Otherwise we'd risk lowering stakes with each open trade.
            # (tied up + current free) * ratio) - tied up
            available_amount = ((val_tied_up + self.get_free(self._config['stake_currency'])) *
                                self._config['tradable_balance_ratio'])
        return available_amount

    def get_available_stake_amount(self) -> float:
        """
        Return the total currently available balance in stake currency,
        respecting tradable_balance_ratio.
        Calculated as
        (<open_trade stakes> + free amount) * tradable_balance_ratio - <open_trade stakes>
        """

        free = self.get_free(self._config['stake_currency'])
        return min(self.get_total_stake_amount() - Trade.total_open_trades_stakes(), free)

    def _calculate_unlimited_stake_amount(self, available_amount: float,
                                          val_tied_up: float, max_open_trades: IntOrInf) -> float:
        """
        Calculate stake amount for "unlimited" stake amount
        :return: 0 if max number of trades reached, else stake_amount to use.
        """
        if max_open_trades == 0:
            return 0

        possible_stake = (available_amount + val_tied_up) / max_open_trades
        # Theoretical amount can be above available amount - therefore limit to available amount!
        return min(possible_stake, available_amount)

    def _check_available_stake_amount(self, stake_amount: float, available_amount: float) -> float:
        """
        Check if stake amount can be fulfilled with the available balance
        for the stake currency
        :return: float: Stake amount
        :raise: DependencyException if balance is lower than stake-amount
        """

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

    def get_trade_stake_amount(
            self, pair: str, max_open_trades: IntOrInf, edge=None, update: bool = True) -> float:
        """
        Calculate stake amount for the trade
        :return: float: Stake amount
        :raise: DependencyException if the available stake amount is too low
        """
        stake_amount: float
        # Ensure wallets are uptodate.
        if update:
            self.update()
        val_tied_up = Trade.total_open_trades_stakes()
        available_amount = self.get_available_stake_amount()

        if edge:
            stake_amount = edge.stake_amount(
                pair,
                self.get_free(self._config['stake_currency']),
                self.get_total(self._config['stake_currency']),
                val_tied_up
            )
        else:
            stake_amount = self._config['stake_amount']
            if stake_amount == UNLIMITED_STAKE_AMOUNT:
                stake_amount = self._calculate_unlimited_stake_amount(
                    available_amount, val_tied_up, max_open_trades)

        return self._check_available_stake_amount(stake_amount, available_amount)

    def validate_stake_amount(self, pair: str, stake_amount: Optional[float],
                              min_stake_amount: Optional[float], max_stake_amount: float,
                              trade_amount: Optional[float]):
        if not stake_amount:
            logger.debug(f"Stake amount is {stake_amount}, ignoring possible trade for {pair}.")
            return 0

        max_allowed_stake = min(max_stake_amount, self.get_available_stake_amount())
        if trade_amount:
            # if in a trade, then the resulting trade size cannot go beyond the max stake
            # Otherwise we could no longer exit.
            max_allowed_stake = min(max_allowed_stake, max_stake_amount - trade_amount)

        if min_stake_amount is not None and min_stake_amount > max_allowed_stake:
            if self._log:
                logger.warning("Minimum stake amount > available balance. "
                               f"{min_stake_amount} > {max_allowed_stake}")
            return 0
        if min_stake_amount is not None and stake_amount < min_stake_amount:
            if self._log:
                logger.info(
                    f"Stake amount for pair {pair} is too small "
                    f"({stake_amount} < {min_stake_amount}), adjusting to {min_stake_amount}."
                )
            if stake_amount * 1.3 < min_stake_amount:
                # Top-cap stake-amount adjustments to +30%.
                if self._log:
                    logger.info(
                        f"Adjusted stake amount for pair {pair} is more than 30% bigger than "
                        f"the desired stake amount of ({stake_amount:.8f} * 1.3 = "
                        f"{stake_amount * 1.3:.8f}) < {min_stake_amount}), ignoring trade."
                    )
                return 0
            stake_amount = min_stake_amount

        if stake_amount > max_allowed_stake:
            if self._log:
                logger.info(
                    f"Stake amount for pair {pair} is too big "
                    f"({stake_amount} > {max_allowed_stake}), adjusting to {max_allowed_stake}."
                )
            stake_amount = max_allowed_stake
        return stake_amount
