import logging
from datetime import datetime, timedelta

import ccxt

from freqtrade.constants import BuySell
from freqtrade.enums import OPTIMIZE_MODES, MarginMode, PriceType, TradingMode
from freqtrade.exceptions import DDosProtection, ExchangeError, OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier
from freqtrade.exchange.exchange_types import CcxtOrder, FtHas
from freqtrade.misc import deep_merge_dicts
from freqtrade.util import dt_from_ts, dt_ts


logger = logging.getLogger(__name__)


class Bybit(Exchange):
    """Bybit exchange class.
    Contains adjustments needed for Freqtrade to work with this exchange.
    """

    unified_account = False

    _ft_has: FtHas = {
        "ohlcv_has_history": True,
        "order_time_in_force": ["GTC", "FOK", "IOC", "PO"],
        "ws_enabled": True,
        "trades_has_history": False,  # Endpoint doesn't support pagination
        "fetch_orders_limit_minutes": 7 * 1440,  # 7 days
        "exchange_has_overrides": {
            # Bybit spot does not support fetch_order
            # Unless the account is unified.
            # TODO: Can be removed once bybit fully forces all accounts to unified mode.
            "fetchOrder": False,
        },
    }
    _ft_has_futures: FtHas = {
        "ohlcv_has_history": True,
        "funding_fee_candle_limit": 200,
        "stoploss_on_exchange": True,
        "stoploss_order_types": {"limit": "limit", "market": "market"},
        "stoploss_blocks_assets": False,
        # bybit response parsing fails to populate stopLossPrice
        "stop_price_prop": "stopPrice",
        "stop_price_type_field": "triggerBy",
        "stop_price_type_value_mapping": {
            PriceType.LAST: "LastPrice",
            PriceType.MARK: "MarkPrice",
            PriceType.INDEX: "IndexPrice",
        },
        "exchange_has_overrides": {
            "fetchOrder": True,
        },
        "has_delisting": True,
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.SPOT, MarginMode.NONE),
        (TradingMode.FUTURES, MarginMode.ISOLATED),
        # (TradingMode.FUTURES, MarginMode.CROSS),
    ]

    @property
    def _ccxt_config(self) -> dict:
        # Parameters to add directly to ccxt sync/async initialization.
        # ccxt defaults to swap mode.
        config = {}
        if self.trading_mode == TradingMode.SPOT:
            config.update({"options": {"defaultType": "spot"}})
        elif self.trading_mode == TradingMode.FUTURES:
            config.update({"options": {"defaultSettle": self._config["stake_currency"]}})
        config = deep_merge_dicts(config, super()._ccxt_config)
        return config

    @retrier
    def additional_exchange_init(self) -> None:
        """
        Additional exchange initialization logic.
        .api will be available at this point.
        Must be overridden in child methods if required.
        """
        try:
            if not self._config["dry_run"]:
                if self.trading_mode == TradingMode.FUTURES:
                    position_mode = self._api.set_position_mode(False)
                    self._log_exchange_response("set_position_mode", position_mode)
                is_unified = self._api.is_unified_enabled()
                # Returns a tuple of bools, first for margin, second for Account
                if is_unified and len(is_unified) > 1 and is_unified[1]:
                    self.unified_account = True
                    logger.info(
                        "Bybit: Unified account. Assuming dedicated subaccount for this bot."
                    )
                else:
                    self.unified_account = False
                    logger.info("Bybit: Standard account.")
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"Error in additional_exchange_init due to {e.__class__.__name__}. Message: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _lev_prep(self, pair: str, leverage: float, side: BuySell, accept_fail: bool = False):
        if self.trading_mode != TradingMode.SPOT:
            params = {"leverage": leverage}
            self.set_margin_mode(pair, self.margin_mode, accept_fail=True, params=params)
            self._set_leverage(leverage, pair, accept_fail=True)

    def _get_params(
        self,
        side: BuySell,
        ordertype: str,
        leverage: float,
        reduceOnly: bool,
        time_in_force: str = "GTC",
    ) -> dict:
        params = super()._get_params(
            side=side,
            ordertype=ordertype,
            leverage=leverage,
            reduceOnly=reduceOnly,
            time_in_force=time_in_force,
        )
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode:
            params["position_idx"] = 0
        return params

    def _get_stop_params(self, side: BuySell, ordertype: str, stop_price: float) -> dict:
        params = super()._get_stop_params(
            side=side,
            ordertype=ordertype,
            stop_price=stop_price,
        )
        # work around ccxt bug introduced in https://github.com/ccxt/ccxt/pull/25887
        # Where create_order ain't returning an ID any longer.
        params.update(
            {
                "method": "privatePostV5OrderCreate",
            }
        )
        return params

    def _order_needs_price(self, side: BuySell, ordertype: str) -> bool:
        # Bybit requires price for market orders - but only for classic accounts,
        # and only in spot mode
        return (
            ordertype != "market"
            or (
                side == "buy" and not self.unified_account and self.trading_mode == TradingMode.SPOT
            )
            or self._ft_has.get("marketOrderRequiresPrice", False)
        )

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
        Important: Must be fetching data from cached values as this is used by backtesting!
        PERPETUAL:
         bybit:
          https://www.bybithelp.com/HelpCenterKnowledge/bybitHC_Article?language=en_US&id=000001067
          USDT:
            https://www.bybit.com/en/help-center/article/Liquidation-Price-Calculation-under-Isolated-Mode-Unified-Trading-Account#b
          USDC:
            https://www.bybit.com/en/help-center/article/Liquidation-Price-Calculation-under-Isolated-Mode-Unified-Trading-Account#c

        Long USDT:
            Liquidation Price = (
                Entry Price - [(Initial Margin - Maintenance Margin)/Contract Quantity]
                - (Extra Margin Added/Contract Quantity))
        Short USDT:
            Liquidation Price = (
                Entry Price + [(Initial Margin - Maintenance Margin)/Contract Quantity]
                + (Extra Margin Added/Contract Quantity))

        Long USDC:
        Liquidation Price = (
            Position Entry Price - [
                (Initial Margin + Extra Margin Added - Maintenance Margin) / Position Size
            ]
        )

        Short USDC:
        Liquidation Price = (
            Position Entry Price + [
                (Initial Margin + Extra Margin Added - Maintenance Margin) / Position Size
            ]
        )

        Implementation Note: Extra margin is currently not used.
        Due to this - the liquidation formula between USDT and USDC is the same.

        :param pair: Pair to calculate liquidation price for
        :param open_rate: Entry price of position
        :param is_short: True if the trade is a short, false otherwise
        :param amount: Absolute value of position size incl. leverage (in base currency)
        :param stake_amount: Stake amount - Collateral in settle currency.
        :param leverage: Leverage used for this position.
        :param wallet_balance: Amount of margin_mode in the wallet being used to trade
            Cross-Margin Mode: crossWalletBalance
            Isolated-Margin Mode: isolatedWalletBalance
        :param open_trades: List of other open trades in the same wallet
        """

        market = self.markets[pair]
        mm_ratio, _ = self.get_maintenance_ratio_and_amt(pair, stake_amount)

        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            if market["inverse"]:
                raise OperationalException("Freqtrade does not yet support inverse contracts")
            position_value = amount * open_rate
            initial_margin = position_value / leverage
            maintenance_margin = position_value * mm_ratio
            margin_diff_per_contract = (initial_margin - maintenance_margin) / amount

            # See docstring - ignores extra margin!
            if is_short:
                return open_rate + margin_diff_per_contract
            else:
                return open_rate - margin_diff_per_contract

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
        # Bybit does not provide "applied" funding fees per position.
        if self.trading_mode == TradingMode.FUTURES:
            try:
                return self._fetch_and_calculate_funding_fees(pair, amount, is_short, open_date)
            except ExchangeError:
                logger.warning(f"Could not update funding fees for {pair}.")
        return 0.0

    def fetch_order(self, order_id: str, pair: str, params: dict | None = None) -> CcxtOrder:
        if self.exchange_has("fetchOrder"):
            # Set acknowledged to True to avoid ccxt exception
            params = {"acknowledged": True}

        order = super().fetch_order(order_id, pair, params)
        if not order:
            order = self.fetch_order_emulated(order_id, pair, {})
        if (
            order.get("status") == "canceled"
            and order.get("filled") == 0.0
            and order.get("remaining") == 0.0
        ):
            # Canceled orders will have "remaining=0" on bybit.
            order["remaining"] = None
        return order

    @retrier
    def get_leverage_tiers(self) -> dict[str, list[dict]]:
        """
        Cache leverage tiers for 1 day, since they are not expected to change often, and
        bybit requires pagination to fetch all tiers.
        """

        # Load cached tiers
        tiers_cached = self.load_cached_leverage_tiers(
            self._config["stake_currency"], timedelta(days=1)
        )
        if tiers_cached:
            return tiers_cached

        # Fetch tiers from exchange
        tiers = super().get_leverage_tiers()

        self.cache_leverage_tiers(tiers, self._config["stake_currency"])
        return tiers

    def check_delisting_time(self, pair: str) -> datetime | None:
        """
        Check if the pair gonna be delisted.
        By default, it returns None.
        :param pair: Market symbol
        :return: Datetime if the pair gonna be delisted, None otherwise
        """
        if self._config["runmode"] in OPTIMIZE_MODES:
            return None

        if self.trading_mode == TradingMode.FUTURES:
            return self._check_delisting_futures(pair)
        return None

    def _check_delisting_futures(self, pair: str) -> datetime | None:
        delivery_time = self.markets.get(pair, {}).get("info", {}).get("deliveryTime", 0)
        if delivery_time:
            if isinstance(delivery_time, str) and (delivery_time != ""):
                delivery_time = int(delivery_time)

            if not isinstance(delivery_time, int) or delivery_time <= 0:
                return None

            max_delivery = dt_ts() + (
                14 * 24 * 60 * 60 * 1000
            )  # Assume exchange don't announce delisting more than 14 days in advance

            if delivery_time < max_delivery:
                return dt_from_ts(delivery_time)

        return None
