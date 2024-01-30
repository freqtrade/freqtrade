""" Kraken exchange subclass """
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import ccxt
from pandas import DataFrame

from freqtrade.constants import BuySell
from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exceptions import DDosProtection, OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier
from freqtrade.exchange.types import Tickers


logger = logging.getLogger(__name__)


class Kraken(Exchange):

    _params: Dict = {"trading_agreement": "agree"}
    _ft_has: Dict = {
        "stoploss_on_exchange": True,
        "stop_price_param": "stopLossPrice",
        "stop_price_prop": "stopLossPrice",
        "stoploss_order_types": {"limit": "limit", "market": "market"},
        "order_time_in_force": ["GTC", "IOC", "PO"],
        "ohlcv_candle_limit": 720,
        "ohlcv_has_history": False,
        "trades_pagination": "id",
        "trades_pagination_arg": "since",
        "trades_pagination_overlap": False,
        "mark_ohlcv_timeframe": "4h",
    }

    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.MARGIN, MarginMode.CROSS),
        # (TradingMode.FUTURES, MarginMode.CROSS)
    ]

    def market_is_tradable(self, market: Dict[str, Any]) -> bool:
        """
        Check if the market symbol is tradable by Freqtrade.
        Default checks + check if pair is darkpool pair.
        """
        parent_check = super().market_is_tradable(market)

        return (parent_check and
                market.get('darkpool', False) is False)

    def get_tickers(self, symbols: Optional[List[str]] = None, cached: bool = False) -> Tickers:
        # Only fetch tickers for current stake currency
        # Otherwise the request for kraken becomes too large.
        symbols = list(self.get_markets(quote_currencies=[self._config['stake_currency']]))
        return super().get_tickers(symbols=symbols, cached=cached)

    @retrier
    def get_balances(self) -> dict:
        if self._config['dry_run']:
            return {}

        try:
            balances = self._api.fetch_balance()
            # Remove additional info from ccxt results
            balances.pop("info", None)
            balances.pop("free", None)
            balances.pop("total", None)
            balances.pop("used", None)

            orders = self._api.fetch_open_orders()
            order_list = [(x["symbol"].split("/")[0 if x["side"] == "sell" else 1],
                           x["remaining"] if x["side"] == "sell" else x["remaining"] * x["price"],
                           # Don't remove the below comment, this can be important for debugging
                           # x["side"], x["amount"],
                           ) for x in orders]
            for bal in balances:
                if not isinstance(balances[bal], dict):
                    continue
                balances[bal]['used'] = sum(order[1] for order in order_list if order[0] == bal)
                balances[bal]['free'] = balances[bal]['total'] - balances[bal]['used']

            return balances
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get balance due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _set_leverage(
        self,
        leverage: float,
        pair: Optional[str] = None,
        accept_fail: bool = False,
    ):
        """
        Kraken set's the leverage as an option in the order object, so we need to
        add it to params
        """
        return

    def _get_params(
        self,
        side: BuySell,
        ordertype: str,
        leverage: float,
        reduceOnly: bool,
        time_in_force: str = 'GTC'
    ) -> Dict:
        params = super()._get_params(
            side=side,
            ordertype=ordertype,
            leverage=leverage,
            reduceOnly=reduceOnly,
            time_in_force=time_in_force,
        )
        if leverage > 1.0:
            params['leverage'] = round(leverage)
        if time_in_force == 'PO':
            params.pop('timeInForce', None)
            params['postOnly'] = True
        return params

    def calculate_funding_fees(
        self,
        df: DataFrame,
        amount: float,
        is_short: bool,
        open_date: datetime,
        close_date: datetime,
        time_in_ratio: Optional[float] = None
    ) -> float:
        """
        # ! This method will always error when run by Freqtrade because time_in_ratio is never
        # ! passed to _get_funding_fee. For kraken futures to work in dry run and backtesting
        # ! functionality must be added that passes the parameter time_in_ratio to
        # ! _get_funding_fee when using Kraken
        calculates the sum of all funding fees that occurred for a pair during a futures trade
        :param df: Dataframe containing combined funding and mark rates
                   as `open_fund` and `open_mark`.
        :param amount: The quantity of the trade
        :param is_short: trade direction
        :param open_date: The date and time that the trade started
        :param close_date: The date and time that the trade ended
        :param time_in_ratio: Not used by most exchange classes
        """
        if not time_in_ratio:
            raise OperationalException(
                f"time_in_ratio is required for {self.name}._get_funding_fee")
        fees: float = 0

        if not df.empty:
            df = df[(df['date'] >= open_date) & (df['date'] <= close_date)]
            fees = sum(df['open_fund'] * df['open_mark'] * amount * time_in_ratio)

        return fees if is_short else -fees

    def _get_trade_pagination_next_value(self, trades: List[Dict]):
        """
        Extract pagination id for the next "from_id" value
        Applies only to fetch_trade_history by id.
        """
        if len(trades) > 0:
            if (
                isinstance(trades[-1].get('info'), list)
                and len(trades[-1].get('info', [])) > 7
            ):
                # Trade response's "last" value.
                return trades[-1].get('info', [])[-1]
            # Fall back to timestamp if info is somehow empty.
            return trades[-1].get('timestamp')
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
