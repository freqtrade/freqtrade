"""Gate.io exchange subclass"""

import logging
from datetime import datetime

import ccxt

from freqtrade.constants import BuySell
from freqtrade.enums import MarginMode, PriceType, TradingMode
from freqtrade.exceptions import DDosProtection, OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Gate(Exchange):
    """Gate.io exchange class.
    Contains adjustments needed for Freqtrade to work with this exchange.
    """

    unified_account = False

    _ft_has: FtHas = {
        "order_time_in_force": ["GTC", "IOC"],
        "stoploss_on_exchange": True,
        "stoploss_order_types": {"limit": "limit"},
        "stop_price_param": "stopPrice",
        "stop_price_prop": "stopPrice",
        "stoploss_query_requires_stop_flag": True,
        "stoploss_algo_order_info_id": "fired_order_id",
        "l2_limit_upper": 1000,
        "marketOrderRequiresPrice": True,
        "trades_has_history": False,  # Endpoint would support this - but ccxt doesn't.
    }

    _ft_has_futures: FtHas = {
        "needs_trading_fees": True,
        "marketOrderRequiresPrice": False,
        "funding_fee_candle_limit": 90,
        "stop_price_type_field": "price_type",
        "l2_limit_upper": 300,
        "stoploss_blocks_assets": False,
        "stoploss_algo_order_info_id": "trade_id",
        "stop_price_type_value_mapping": {
            PriceType.LAST: 0,
            PriceType.MARK: 1,
            PriceType.INDEX: 2,
        },
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.SPOT, MarginMode.NONE),
        # (TradingMode.MARGIN, MarginMode.CROSS),
        # (TradingMode.FUTURES, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.ISOLATED),
    ]

    @retrier
    def additional_exchange_init(self) -> None:
        """
        Additional exchange initialization logic.
        .api will be available at this point.
        Must be overridden in child methods if required.
        """
        try:
            if not self._config["dry_run"]:
                self._api.load_unified_status()
                is_unified = self._api.options.get("unifiedAccount")

                # Returns a tuple of bools, first for margin, second for Account
                if is_unified:
                    self.unified_account = True
                    logger.info("Gate: Unified account.")
                else:
                    self.unified_account = False
                    logger.info("Gate: Classic account.")
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"Error in additional_exchange_init due to {e.__class__.__name__}. Message: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

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
        if ordertype == "market" and self.trading_mode == TradingMode.FUTURES:
            params["type"] = "market"
            params.update({"timeInForce": "IOC"})
        return params

    def get_trades_for_order(
        self, order_id: str, pair: str, since: datetime, params: dict | None = None
    ) -> list:
        trades = super().get_trades_for_order(order_id, pair, since, params)

        if self.trading_mode == TradingMode.FUTURES:
            # Futures usually don't contain fees in the response.
            # As such, futures orders on gate will not contain a fee, which causes
            # a repeated "update fee" cycle and wrong calculations.
            # Therefore we patch the response with fees if it's not available.
            # An alternative also containing fees would be
            # privateFuturesGetSettleAccountBook({"settle": "usdt"})
            pair_fees = self._trading_fees.get(pair, {})
            if pair_fees:
                for idx, trade in enumerate(trades):
                    fee = trade.get("fee", {})
                    if fee and fee.get("cost") is None:
                        takerOrMaker = trade.get("takerOrMaker", "taker")
                        if pair_fees.get(takerOrMaker) is not None:
                            trades[idx]["fee"] = {
                                "currency": self.get_pair_quote_currency(pair),
                                "cost": trade["cost"] * pair_fees[takerOrMaker],
                                "rate": pair_fees[takerOrMaker],
                            }
        return trades
