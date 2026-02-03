import logging
from datetime import datetime, timedelta

import ccxt

from freqtrade.constants import BuySell
from freqtrade.enums import OPTIMIZE_MODES, CandleType, MarginMode, TradingMode
from freqtrade.exceptions import (
    DDosProtection,
    OperationalException,
    RetryableOrderError,
    TemporaryError,
)
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import API_RETRY_COUNT, retrier
from freqtrade.exchange.exchange_types import CcxtOrder, FtHas
from freqtrade.util import dt_from_ts, dt_now, dt_ts


logger = logging.getLogger(__name__)


class Bitget(Exchange):
    """Bitget exchange class.
    Contains adjustments needed for Freqtrade to work with this exchange.
    """

    _ft_has: FtHas = {
        "stoploss_on_exchange": True,
        "stop_price_param": "stopPrice",
        "stop_price_prop": "stopPrice",
        "stoploss_blocks_assets": False,  # Stoploss orders do not block assets
        "stoploss_order_types": {"limit": "limit", "market": "market"},
        "stoploss_query_requires_stop_flag": True,
        "ohlcv_candle_limit": 200,  # 200 for historical candles, 1000 for recent ones.
        "order_time_in_force": ["GTC", "FOK", "IOC", "PO"],
    }
    _ft_has_futures: FtHas = {
        "funding_fee_candle_limit": 100,
        "has_delisting": True,
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.SPOT, MarginMode.NONE),
        (TradingMode.FUTURES, MarginMode.ISOLATED),
        # (TradingMode.FUTURES, MarginMode.CROSS),
    ]

    def ohlcv_candle_limit(
        self, timeframe: str, candle_type: CandleType, since_ms: int | None = None
    ) -> int:
        """
        Exchange ohlcv candle limit
        bitget has the following behaviour:
        * 1000 candles for up-to-date data
        * 200 candles for historic data (prior to a certain date)
        :param timeframe: Timeframe to check
        :param candle_type: Candle-type
        :param since_ms: Starting timestamp
        :return: Candle limit as integer
        """
        timeframe_map = self._api.options["fetchOHLCV"]["maxRecentDaysPerTimeframe"]
        days = timeframe_map.get(timeframe, 30)

        if candle_type in (CandleType.FUTURES, CandleType.SPOT, CandleType.MARK) and (
            not since_ms or dt_ts(dt_now() - timedelta(days=days)) < since_ms
        ):
            return 1000

        return super().ohlcv_candle_limit(timeframe, candle_type, since_ms)

    def _convert_stop_order(self, pair: str, order_id: str, order: CcxtOrder) -> CcxtOrder:
        if order.get("status", "open") == "closed":
            # Use orderID as cliendOrderId filter to fetch the regular followup order.
            # Could be done with "fetch_order" - but clientOid as filter doesn't seem to work
            # https://www.bitget.com/api-doc/spot/trade/Get-Order-Info

            for method in (
                self._api.fetch_canceled_and_closed_orders,
                self._api.fetch_open_orders,
            ):
                orders = method(pair)
                orders_f = [order for order in orders if order["clientOrderId"] == order_id]
                if orders_f:
                    order_reg = orders_f[0]
                    self._log_exchange_response("fetch_stoploss_order1", order_reg)
                    order_reg["id_stop"] = order_reg["id"]
                    order_reg["id"] = order_id
                    order_reg["type"] = "stoploss"
                    order_reg["status_stop"] = "triggered"
                    return order_reg
        order = self._order_contracts_to_amount(order)
        order["type"] = "stoploss"
        return order

    def _fetch_stop_order_fallback(self, order_id: str, pair: str) -> CcxtOrder:
        params2 = {
            "stop": True,
        }
        for method in (
            self._api.fetch_open_orders,
            self._api.fetch_canceled_and_closed_orders,
        ):
            try:
                orders = method(pair, params=params2)
                orders_f = [order for order in orders if order["id"] == order_id]
                if orders_f:
                    order = orders_f[0]
                    self._log_exchange_response("get_stop_order_fallback", order)
                    return self._convert_stop_order(pair, order_id, order)
            except (ccxt.OrderNotFound, ccxt.InvalidOrder):
                pass
            except ccxt.DDoSProtection as e:
                raise DDosProtection(e) from e
            except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
                raise TemporaryError(
                    f"Could not get order due to {e.__class__.__name__}. Message: {e}"
                ) from e
            except ccxt.BaseError as e:
                raise OperationalException(e) from e
        raise RetryableOrderError(f"StoplossOrder not found (pair: {pair} id: {order_id}).")

    @retrier(retries=API_RETRY_COUNT)
    def fetch_stoploss_order(
        self, order_id: str, pair: str, params: dict | None = None
    ) -> CcxtOrder:
        if self._config["dry_run"]:
            return self.fetch_dry_run_order(order_id)

        return self._fetch_stop_order_fallback(order_id, pair)

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
            # Explicitly setting margin_mode is not necessary as marginMode can be set per order.
            # self.set_margin_mode(pair, self.margin_mode, accept_fail)
            self._set_leverage(leverage, pair, accept_fail)

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
            params["marginMode"] = self.margin_mode.value.lower()
        return params

    def dry_run_liquidation_price(
        self,
        pair: str,
        open_rate: float,
        is_short: bool,
        amount: float,
        stake_amount: float,
        leverage: float,
        wallet_balance: float,
        open_trades: list,
    ) -> float | None:
        """
        Important: Must be fetching data from cached values as this is used by backtesting!


        https://www.bitget.com/support/articles/12560603808759
        MMR: Maintenance margin rate of the trading pair.

        CoinMainIndexPrice: The index price for Coin-M futures. For USDT-M futures,
                            the index price is: 1.

        TakerFeeRatio: The fee rate applied when placing taker orders.

        Position direction: The current position direction of the trading pair.
                        1 indicates a long position, and -1 indicates a short position.

        Formula:

        Estimated liquidation price = [
            position margin - position size x average entry price x position direction
        ] ÷ [position size x (MMR + TakerFeeRatio - position direction)]

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
        taker_fee_rate = market["taker"] or self._api.describe().get("fees", {}).get(
            "trading", {}
        ).get("taker", 0.001)
        mm_ratio, _ = self.get_maintenance_ratio_and_amt(pair, stake_amount)

        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            position_direction = -1 if is_short else 1

            return (wallet_balance - (amount * open_rate * position_direction)) / (
                amount * (mm_ratio + taker_fee_rate - position_direction)
            )
        else:
            raise OperationalException(
                "Freqtrade currently only supports isolated futures for bitget"
            )

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
        delivery_time = self.markets.get(pair, {}).get("info", {}).get("limitOpenTime", None)
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
