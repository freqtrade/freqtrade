"""Kraken exchange subclass"""

import logging
from typing import Any

import ccxt

from freqtrade.constants import BuySell
from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exceptions import DDosProtection, OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier
from freqtrade.exchange.exchange_types import CcxtBalances, FtHas


logger = logging.getLogger(__name__)


class Kraken(Exchange):
    """Kraken exchange class.
    Contains adjustments needed for Freqtrade to work with this exchange.
    """

    _params: dict = {"trading_agreement": "agree"}
    _ft_has: FtHas = {
        "stoploss_on_exchange": True,
        "stop_price_param": "stopLossPrice",
        "stop_price_prop": "stopLossPrice",
        "stoploss_order_types": {"limit": "limit", "market": "market"},
        "order_time_in_force": ["GTC", "IOC", "PO"],
        "ohlcv_has_history": False,
        "trades_pagination": "id",
        "trades_pagination_arg": "since",
        "trades_pagination_overlap": False,
        "trades_has_history": True,
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.SPOT, MarginMode.NONE),
        # (TradingMode.MARGIN, MarginMode.CROSS),
    ]

    def market_is_tradable(self, market: dict[str, Any]) -> bool:
        """
        Check if the market symbol is tradable by Freqtrade.
        Default checks + check if pair is darkpool pair.
        """
        parent_check = super().market_is_tradable(market)

        return parent_check and market.get("darkpool", False) is False

    def consolidate_balances(self, balances: CcxtBalances) -> CcxtBalances:
        """
        Consolidate balances for the same currency.
        Kraken returns ".F" balances if rewards is enabled.
        """
        consolidated: CcxtBalances = {}
        for currency, balance in balances.items():
            base_currency = currency[:-2] if currency.endswith(".F") else currency

            if base_currency in consolidated:
                consolidated[base_currency]["free"] += balance["free"]
                consolidated[base_currency]["used"] += balance["used"]
                consolidated[base_currency]["total"] += balance["total"]
            else:
                consolidated[base_currency] = balance
        return consolidated

    @retrier
    def get_balances(self, params: dict | None = None) -> CcxtBalances:
        if self._config["dry_run"]:
            return {}

        try:
            balances = self._api.fetch_balance()
            # Remove additional info from ccxt results
            balances.pop("info", None)
            balances.pop("free", None)
            balances.pop("total", None)
            balances.pop("used", None)
            self._log_exchange_response("fetch_balance", balances)

            # Consolidate balances
            balances = self.consolidate_balances(balances)

            orders = self._api.fetch_open_orders()
            order_list = [
                (
                    x["symbol"].split("/")[0 if x["side"] == "sell" else 1],
                    x["remaining"] if x["side"] == "sell" else x["remaining"] * x["price"],
                    # Don't remove the below comment, this can be important for debugging
                    # x["side"], x["amount"],
                )
                for x in orders
                if x["remaining"] is not None and (x["side"] == "sell" or x["price"] is not None)
            ]
            for bal in balances:
                if not isinstance(balances[bal], dict):
                    continue
                balances[bal]["used"] = sum(order[1] for order in order_list if order[0] == bal)
                balances[bal]["free"] = balances[bal]["total"] - balances[bal]["used"]

            self._log_exchange_response("fetch_balance2", balances)
            return balances
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"Could not get balance due to {e.__class__.__name__}. Message: {e}"
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
        if leverage > 1.0:
            params["leverage"] = round(leverage)
        if time_in_force == "PO":
            params.pop("timeInForce", None)
            params["postOnly"] = True
        return params

    def _get_trade_pagination_next_value(self, trades: list[dict]):
        """
        Extract pagination id for the next "from_id" value
        Applies only to fetch_trade_history by id.
        """
        if len(trades) > 0:
            if isinstance(trades[-1].get("info"), list) and len(trades[-1].get("info", [])) > 7:
                # Trade response's "last" value.
                return trades[-1].get("info", [])[-1]
            # Fall back to timestamp if info is somehow empty.
            return trades[-1].get("timestamp")
        return None

    def _valid_trade_pagination_id(self, pair: str, from_id: str) -> bool:
        """
        Verify trade-pagination id is valid.
        Workaround for odd Kraken issue where ID is sometimes wrong.
        """
        # Regular id's are in timestamp format 1705443695120072285
        # If the id is smaller than 19 characters, it's not a valid timestamp.
        if len(from_id) >= 19:
            return True
        logger.debug(f"{pair} - trade-pagination id is not valid. Fallback to timestamp.")
        return False
