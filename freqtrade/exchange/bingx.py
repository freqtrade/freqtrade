"""Bingx exchange subclass"""

import logging

import ccxt

from freqtrade.constants import BuySell
from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas, OrderBook
from freqtrade.misc import deep_merge_dicts


logger = logging.getLogger(__name__)


class Bingx(Exchange):
    """
    Bingx exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.
    """

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 1000,
        "stoploss_on_exchange": True,
        "stoploss_order_types": {"limit": "limit", "market": "market"},
        "order_time_in_force": ["GTC", "IOC", "PO"],
        "trades_has_history": False,  # Endpoint doesn't seem to support pagination
        "exchange_has_overrides": {
            "fetchLeverageTiers": False,
            "fetchMarketLeverageTiers": False,
        },
    }

    _ft_has_futures: FtHas = {
        "funding_fee_candle_limit": 200,
        "stoploss_on_exchange": True,
        "stoploss_order_types": {"limit": "limit", "market": "market"},
        "stoploss_blocks_assets": False,
        "has_delisting": True,
        "exchange_has_overrides": {
            "fetchLeverageTiers": False,
            "fetchMarketLeverageTiers": False,
        },
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.SPOT, MarginMode.NONE),
        (TradingMode.FUTURES, MarginMode.ISOLATED),
        (TradingMode.FUTURES, MarginMode.CROSS),
    ]

    @property
    def _ccxt_config(self) -> dict:
        config: dict = {}
        if self.trading_mode == TradingMode.FUTURES:
            config.update({"options": {"defaultType": "swap"}})
        config = deep_merge_dicts(config, super()._ccxt_config)
        return config

    _bingx_hedge_mode: bool | None = None

    def additional_exchange_init(self) -> None:
        super().additional_exchange_init()
        if self.trading_mode == TradingMode.FUTURES:
            # BingX does not allow reduceOnly in Hedge mode, but Freqtrade forcefully adds it.
            # We monkey-patch CCXT's create_order to strip it out before it reaches the exchange.
            original_create_order = self._api.create_order

            def patched_create_order(*args, **kwargs):
                hedge_mode = getattr(self, "_bingx_hedge_mode", True)
                if hedge_mode:
                    if "params" in kwargs and kwargs["params"] and "reduceOnly" in kwargs["params"]:
                        kwargs["params"] = kwargs["params"].copy()
                        kwargs["params"].pop("reduceOnly")
                    elif len(args) >= 6 and args[5] and "reduceOnly" in args[5]:
                        # params is the 6th positional argument in CCXT create_order
                        args = list(args)
                        args[5] = args[5].copy()
                        args[5].pop("reduceOnly")
                        args = tuple(args)
                return original_create_order(*args, **kwargs)

            self._api.create_order = patched_create_order

            # Monkey-patch cancel_order to gracefully handle orders that BingX already cancelled.
            original_cancel_order = self._api.cancel_order

            def patched_cancel_order(order_id, symbol=None, params=None):
                if params is None:
                    params = {}
                try:
                    return original_cancel_order(order_id, symbol, params)
                except ccxt.ExchangeError as e:
                    err_str = str(e)
                    if "109400" in err_str or "109201" in err_str:
                        # 109400: order not exist (already cancelled or closed)
                        # 109201: Same order can only be submitted once per sec.
                        # Instead of raising OrderNotFound, we return a dummy 'canceled' order.
                        logger.info(
                            f"BingX order {order_id} already cancelled. Suppressing error."
                        )
                        return {"id": order_id, "status": "canceled", "info": {}}
                    raise

            self._api.cancel_order = patched_cancel_order

    def _lev_prep(
        self, pair: str, leverage: float, side: BuySell, accept_fail: bool = False
    ) -> None:
        if self.trading_mode != TradingMode.SPOT:
            # BingX setLeverage strictly requires leverage as an integer.
            # Freqtrade provides a float.
            lev_int = int(leverage)
            params = {"leverage": lev_int}
            self.set_margin_mode(pair, self.margin_mode, accept_fail=True, params=params)

            try:
                if self._bingx_hedge_mode is None or not self._bingx_hedge_mode:
                    try:
                        res = self._api.set_leverage(
                            leverage=lev_int, symbol=pair, params={"side": "BOTH"}
                        )
                        self._log_exchange_response("set_leverage", res)
                        self._bingx_hedge_mode = False
                        return
                    except Exception as e:
                        if (
                            "109400" in str(e)
                            or "Hedge" in str(e)
                            or "LONG or SHORT" in str(e)
                            or "Invalid parameters" in str(e)
                        ):
                            self._bingx_hedge_mode = True
                        else:
                            raise

                if self._bingx_hedge_mode:
                    res_long = self._api.set_leverage(
                        leverage=lev_int, symbol=pair, params={"side": "LONG"}
                    )
                    res_short = self._api.set_leverage(
                        leverage=lev_int, symbol=pair, params={"side": "SHORT"}
                    )
                    self._log_exchange_response("set_leverage_long", res_long)
                    self._log_exchange_response("set_leverage_short", res_short)
            except Exception as e:
                if not accept_fail:
                    raise
                logger.warning(f"Could not set leverage on BingX: {e}")

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
        Calculate liquidation price for dry run on BingX.
        Assuming standard USDT-M futures formulas.
        """
        mm_ratio = 0.005  # 0.5% approximation for maintenance margin
        if is_short:
            return open_rate * (1 + 1 / leverage - mm_ratio)
        else:
            return open_rate * (1 - 1 / leverage + mm_ratio)

    def load_leverage_tiers(self) -> dict[str, list[dict]]:
        """
        BingX API via CCXT currently lacks fetchLeverageTiers support.
        We mock a basic leverage tier for all futures markets.
        """
        tiers = {}
        for symbol, market in self.markets.items():
            if self.market_is_future(market):
                tiers[symbol] = [
                    {
                        "tier": 1,
                        "minNotional": 0.0,
                        "maxNotional": 999999999.0,
                        "maintenanceMarginRate": 0.005,
                        "maxLeverage": 125.0,
                        "info": {},
                    }
                ]
        return tiers

    def fetch_l2_order_book(self, pair: str, limit: int = 100) -> OrderBook:
        """
        BingX strictly enforces orderbook limits (5, 10, 20, 50, 100, 500, 1000).
        Freqtrade frequently requests limit=1 for dry-run order filling checks.
        We override this to ensure limit is at least 5.
        """
        if limit < 5:
            limit = 5
        return super().fetch_l2_order_book(pair, limit)

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
        if self.trading_mode == TradingMode.FUTURES:
            hedge_mode = getattr(self, "_bingx_hedge_mode", True)
            if hedge_mode:
                # BingX requires positionSide for Hedge mode.
                is_buy = side == "buy"
                is_long_position = (is_buy and not reduceOnly) or (not is_buy and reduceOnly)
                params["positionSide"] = "LONG" if is_long_position else "SHORT"
        return params

    def _get_stop_params(self, side: BuySell, ordertype: str, stop_price: float) -> dict:
        params = super()._get_stop_params(
            side=side, ordertype=ordertype, stop_price=stop_price
        )
        if self.trading_mode == TradingMode.FUTURES:
            hedge_mode = getattr(self, "_bingx_hedge_mode", True)
            if hedge_mode:
                is_buy = side == "buy"
                # Stoploss orders are reduceOnly by definition in Freqtrade.
                params["positionSide"] = "SHORT" if is_buy else "LONG"
        return params

    def create_order(
        self,
        pair: str,
        ordertype: str,
        side: BuySell,
        amount: float,
        rate: float,
        leverage: float = 1.0,
        reduceOnly: bool = False,
        time_in_force: str = "GTC",
        **kwargs,
    ) -> dict:
        params = self._get_params(
            side=side,
            ordertype=ordertype,
            leverage=leverage,
            reduceOnly=reduceOnly,
            time_in_force=time_in_force,
        )
        logger.info(
            f"BINGX DEBUG CREATE_ORDER: pair={pair} type={ordertype} side={side} "
            f"amount={amount} rate={rate} params={params} kwargs={kwargs}"
        )
        return super().create_order(
            pair=pair,
            ordertype=ordertype,
            side=side,
            amount=amount,
            rate=rate,
            leverage=leverage,
            reduceOnly=reduceOnly,
            time_in_force=time_in_force,
            **kwargs,
        )
