"""Hyperliquid exchange subclass"""

import logging
from copy import deepcopy
from datetime import datetime
from typing import Any

from freqtrade.constants import BuySell
from freqtrade.enums import MarginMode, TradingMode
from freqtrade.enums.runmode import NON_UTIL_MODES
from freqtrade.exceptions import ConfigurationError, ExchangeError, OperationalException
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import CcxtBalances, CcxtOrder, CcxtPosition, FtHas
from freqtrade.util.datetime_helpers import dt_from_ts


logger = logging.getLogger(__name__)


class Hyperliquid(Exchange):
    """Hyperliquid exchange class.
    Contains adjustments needed for Freqtrade to work with this exchange.
    """

    _ft_has: FtHas = {
        "ohlcv_has_history": False,
        "l2_limit_range": [20],
        "trades_has_history": False,
        "tickers_have_bid_ask": False,
        "stoploss_on_exchange": False,
        "exchange_has_overrides": {"fetchTrades": False},
        "marketOrderRequiresPrice": True,
        "download_data_parallel_quick": False,
        "ws_enabled": True,
    }
    _ft_has_futures: FtHas = {
        "stoploss_on_exchange": True,
        "stoploss_order_types": {"limit": "limit"},
        "stoploss_blocks_assets": False,
        "stop_price_prop": "stopPrice",
        "funding_fee_candle_limit": 500,
        "uses_leverage_tiers": False,
        "mark_ohlcv_price": "futures",
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.SPOT, MarginMode.NONE),
        (TradingMode.FUTURES, MarginMode.ISOLATED),
        (TradingMode.FUTURES, MarginMode.CROSS),
    ]

    @property
    def _ccxt_config(self) -> dict:
        # ccxt Hyperliquid defaults to swap
        config = {}
        if self.trading_mode == TradingMode.SPOT:
            config.update({"options": {"defaultType": "spot"}})
        config.update(super()._ccxt_config)
        return config

    def _get_configured_hip3_dexes(self) -> list[str]:
        """Get list of configured HIP-3 DEXes."""
        return self._config.get("exchange", {}).get("hip3_dexes", [])

    def validate_config(self, config: dict) -> None:
        """Validate HIP-3 configuration at bot startup."""
        super().validate_config(config)
        configured = self._get_configured_hip3_dexes()
        if not configured or not self.markets:
            return
        if self.trading_mode != TradingMode.FUTURES:
            if configured:
                raise ConfigurationError(
                    "HIP-3 DEXes are only supported in FUTURES trading mode. "
                    "Please update your configuration!"
                )
            return
        if configured and self.margin_mode != MarginMode.ISOLATED:
            raise ConfigurationError(
                "HIP-3 DEXes require 'isolated' margin mode. "
                f"Current margin mode: '{self.margin_mode.value}'. "
                "Please update your configuration!"
            )

        available = {
            m.get("info", {}).get("dex")
            for m in self.get_markets(
                quote_currencies=[self._config["stake_currency"]],
                tradable_only=True,
                active_only=True,
            ).values()
            if m.get("info", {}).get("hip3")
        }
        available.discard(None)

        invalid = set(configured) - available
        if invalid:
            raise ConfigurationError(
                f"Invalid HIP-3 DEXes configured: {sorted(invalid)}. "
                f"Available DEXes matching your stake currency ({self._config['stake_currency']}): "
                f"{sorted(available)}. "
                f"Check your 'hip3_dexes' configuration!"
            )

    def market_is_tradable(self, market: dict[str, Any]) -> bool:
        """Check if market is tradable, including HIP-3 markets."""
        parent_check = super().market_is_tradable(market)

        market_info = market.get("info", {})
        if market_info.get("hip3") and self._config["runmode"] in NON_UTIL_MODES:
            configured = self._get_configured_hip3_dexes()
            if not configured:
                return False

            market_dex = market_info.get("dex")
            return parent_check and market_dex in configured

        return parent_check

    def get_balances(self, params: dict | None = None) -> CcxtBalances:
        """Fetch balances from default DEX and HIP-3 DEXes needed by tradable pairs.
        This override is not absolutely necessary and is only there for correct used / total values
        which are however not used by Freqtrade in futures mode at the moment.
        """
        balances = super().get_balances()
        dexes = self._get_configured_hip3_dexes()
        for dex in dexes:
            try:
                dex_balance = super().get_balances(params={"dex": dex})

                for currency, amount_info in dex_balance.items():
                    if currency in ["info", "free", "used", "total", "datetime", "timestamp"]:
                        continue

                    if currency not in balances:
                        balances[currency] = amount_info
                    else:
                        balances[currency]["free"] += amount_info["free"]
                        balances[currency]["used"] += amount_info["used"]
                        balances[currency]["total"] += amount_info["total"]

            except Exception as e:
                logger.error(f"Could not fetch balance for HIP-3 DEX '{dex}': {e}")

        if dexes:
            self._log_exchange_response("fetch_balance", balances, add_info="combined")
        return balances

    def fetch_positions(
        self, pair: str | None = None, params: dict | None = None
    ) -> list[CcxtPosition]:
        """Fetch positions from default DEX and HIP-3 DEXes needed by tradable pairs."""
        positions = super().fetch_positions(pair)
        dexes = self._get_configured_hip3_dexes()
        for dex in dexes:
            try:
                positions.extend(super().fetch_positions(pair, params={"dex": dex}))
            except Exception as e:
                logger.error(f"Could not fetch positions from HIP-3 DEX '{dex}': {e}")
        if dexes:
            self._log_exchange_response("fetch_positions", positions, add_info="combined")
        return positions

    def get_max_leverage(self, pair: str, stake_amount: float | None) -> float:
        # There are no leverage tiers
        if self.trading_mode == TradingMode.FUTURES:
            return self.markets[pair]["limits"]["leverage"]["max"]
        else:
            return 1.0

    def _lev_prep(self, pair: str, leverage: float, side: BuySell, accept_fail: bool = False):
        if self.trading_mode != TradingMode.SPOT:
            # Hyperliquid expects leverage to be an int
            leverage = int(leverage)
            # Hyperliquid needs the parameter leverage.
            # Don't use _set_leverage(), as this sets margin back to cross
            self.set_margin_mode(pair, self.margin_mode, params={"leverage": leverage})

    def dry_run_liquidation_price(
        self,
        pair: str,
        open_rate: float,  # Entry price of position
        is_short: bool,
        amount: float,
        stake_amount: float,
        leverage: float,
        wallet_balance: float,  # Or margin balance
        open_trades: list,
    ) -> float | None:
        """
        Optimized
        Docs: https://hyperliquid.gitbook.io/hyperliquid-docs/trading/liquidations
        Below can be done in fewer lines of code, but like this it matches the documentation.

        Tested with 196 unique ccxt fetch_positions() position outputs
        - Only first output per position where pnl=0.0
        - Compare against returned liquidation price
        Positions: 197 Average deviation: 0.00028980% Max deviation: 0.01309453%
        Positions info:
        {'leverage': {1.0: 23, 2.0: 155, 3.0: 8, 4.0: 7, 5.0: 4},
        'side': {'long': 133, 'short': 64},
        'symbol': {'BTC/USDC:USDC': 81,
                   'DOGE/USDC:USDC': 20,
                   'ETH/USDC:USDC': 53,
                   'SOL/USDC:USDC': 43}}
        """
        # Defining/renaming variables to match the documentation
        position_size = amount
        price = open_rate
        position_value = price * position_size
        max_leverage = self.markets[pair]["limits"]["leverage"]["max"]

        # Docs: The maintenance margin is half of the initial margin at max leverage,
        #       which varies from 3-50x. In other words, the maintenance margin is between 1%
        #       (for 50x max leverage assets) and 16.7% (for 3x max leverage assets)
        #       depending on the asset
        # The key thing here is 'Half of the initial margin at max leverage'.
        # A bit ambiguous, but this interpretation leads to accurate results:
        #       1. Start from the position value
        #       2. Assume max leverage, calculate the initial margin by dividing the position value
        #          by the max leverage
        #       3. Divide this by 2
        maintenance_margin_required = position_value / max_leverage / 2

        if self.margin_mode == MarginMode.ISOLATED:
            # Docs: margin_available (isolated) = isolated_margin - maintenance_margin_required
            margin_available = stake_amount - maintenance_margin_required
        elif self.margin_mode == MarginMode.CROSS:
            # Docs: margin_available (cross) = account_value - maintenance_margin_required
            margin_available = wallet_balance - maintenance_margin_required
        else:
            raise OperationalException("Unsupported margin mode for liquidation price calculation")

        # Docs: The maintenance margin is half of the initial margin at max leverage
        # The docs don't explicitly specify maintenance leverage, but this works.
        # Double because of the statement 'half of the initial margin at max leverage'
        maintenance_leverage = max_leverage * 2

        # Docs: l = 1 / MAINTENANCE_LEVERAGE (Using 'll' to comply with PEP8: E741)
        ll = 1 / maintenance_leverage

        # Docs: side = 1 for long and -1 for short
        side = -1 if is_short else 1

        # Docs: liq_price = price - side * margin_available / position_size / (1 - l * side)
        liq_price = price - side * margin_available / position_size / (1 - ll * side)

        if self.trading_mode == TradingMode.FUTURES:
            return liq_price
        else:
            raise OperationalException(
                "Freqtrade only supports isolated futures for leverage trading"
            )

    def get_funding_fees(
        self, pair: str, amount: float, is_short: bool, open_date: datetime
    ) -> float:
        """
        Fetch funding fees, either from the exchange (live) or calculates them
        based on funding rate/mark price history
        :param pair: The quote/base pair of the trade
        :param is_short: trade direction
        :param amount: Trade amount
        :param open_date: Open date of the trade
        :return: funding fee since open_date
        :raises: ExchangeError if something goes wrong.
        """
        # Hyperliquid does not have fetchFundingHistory
        if self.trading_mode == TradingMode.FUTURES:
            try:
                return self._fetch_and_calculate_funding_fees(pair, amount, is_short, open_date)
            except ExchangeError:
                logger.warning(f"Could not update funding fees for {pair}.")
        return 0.0

    def _adjust_hyperliquid_order(
        self,
        order: dict,
    ) -> dict:
        """
        Adjusts order response for Hyperliquid
        :param order: Order response from Hyperliquid
        :return: Adjusted order response
        """
        if (
            order["average"] is None
            and order["status"] in ("canceled", "closed")
            and order["filled"] > 0
        ):
            # Hyperliquid does not fill the average price in the order response
            # Fetch trades to calculate the average price to have the actual price
            # the order was executed at
            trades = self.get_trades_for_order(
                order["id"], order["symbol"], since=dt_from_ts(order["timestamp"])
            )

            if trades:
                total_amount = sum(t["amount"] for t in trades)
                order["average"] = (
                    sum(t["price"] * t["amount"] for t in trades) / total_amount
                    if total_amount
                    else None
                )
        return order

    def fetch_order(self, order_id: str, pair: str, params: dict | None = None) -> CcxtOrder:
        order = super().fetch_order(order_id, pair, params)

        order = self._adjust_hyperliquid_order(order)
        self._log_exchange_response("fetch_order2", order)

        return order

    def fetch_orders(
        self, pair: str, since: datetime, params: dict | None = None
    ) -> list[CcxtOrder]:
        orders = super().fetch_orders(pair, since, params)
        for idx, order in enumerate(deepcopy(orders)):
            order2 = self._adjust_hyperliquid_order(order)
            orders[idx] = order2

        self._log_exchange_response("fetch_orders2", orders)
        return orders
