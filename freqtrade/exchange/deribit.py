"""Deribit exchange subclass"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from pandas import DataFrame

from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


class Deribit(Exchange):
    _params: Dict = {"trading_agreement": "agree"}
    _ft_has: Dict = {
        "stoploss_on_exchange": True,
        "stop_price_param": "triggerPrice",
        "stop_price_prop": "triggerPrice",
        "stoploss_order_types": {"limit": "limit", "market": "market"},
        "order_time_in_force": ["GTC", "IOC", "FOK", "PO"],
        "ohlcv_candle_limit": 720,  # TODO
        "ohlcv_has_history": False,  # TODO
        "trades_pagination": "id",  # TODO
        "trades_pagination_arg": "since",  # TODO
        "trades_pagination_overlap": False,  # TODO
        "mark_ohlcv_timeframe": "4h",  # TODO
    }

    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.MARGIN, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.ISOLATED)
    ]

    def _set_leverage(
        self,
        leverage: float,
        pair: Optional[str] = None,
        accept_fail: bool = False,
    ):
        # TODO
        return

    def calculate_funding_fees(
        self,
        df: DataFrame,
        amount: float,
        is_short: bool,
        open_date: datetime,
        close_date: datetime,
        time_in_ratio: Optional[float] = None,
    ) -> float:
        """
        calculates the sum of all funding fees that occurred for a pair during a futures trade
        :param df: Dataframe containing combined funding and mark rates
                   as `open_fund` and `open_mark`.
        :param amount: The quantity of the trade
        :param is_short: trade direction
        :param open_date: The date and time that the trade started
        :param close_date: The date and time that the trade ended
        :param time_in_ratio: Not used by most exchange classes
        """
        # TODO
        return 0
