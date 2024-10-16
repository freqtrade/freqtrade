"""Deribit exchange subclass"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from pandas import DataFrame

from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Deribit(Exchange):
    _params: Dict = {"trading_agreement": "agree"}
    _ft_has: FtHas = {
        "stoploss_on_exchange": True,
        "stop_price_param": "triggerPrice",
        "stop_price_prop": "triggerPrice",
        "stoploss_order_types": {"limit": "limit", "market": "market"},
        "order_time_in_force": ["GTC", "IOC", "FOK", "PO"],
        "ohlcv_candle_limit": 5000,
        "ohlcv_has_history": False,
        "mark_ohlcv_timeframe": "8h",
    }

    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.MARGIN, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.ISOLATED)
    ]

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
        TODO
        see https://www.deribit.com/kb/deribit-linear-perpetual
        calculates the sum of all funding fees that occurred for a pair during a futures trade
        :param df: Dataframe containing combined funding and mark rates
                   as `open_fund` and `open_mark`.
        :param amount: The quantity of the trade
        :param is_short: trade direction
        :param open_date: The date and time that the trade started
        :param close_date: The date and time that the trade ended
        :param time_in_ratio: Not used by most exchange classes
        """
        # Sequentially, the funding rate is derived from the premium rate by
        # applying a damper.
        # If the premium rate is within -0.025% and 0.025% range, the actual
        # funding rate will be reduced to 0.00%.
        # If the premium rate is lower than -0.025%, then the actual funding
        # rate will be the premium rate + 0.025%.
        # If the premium rate is higher than 0.025%, then the actual funding
        # rate will be the premium rate - 0.025%.
        # Additionally, the funding rate is capped at +/ - 5% for all linear
        # USDC perpetuals.

        # Premium Rate = ((Mark Price - Deribit Index) / Deribit Index) * 100%
        # funding_rate = Maximum (0.025%, Premium Rate) + Minimum (-0.025%, Premium Rate)
        # time_fraction = funding_rate_time_period / 8 hours
        # return funding_rate * position_size * time_fraction
        return 0
