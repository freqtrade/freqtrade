"""WhiteBit exchange subclass"""

import logging
from datetime import datetime

from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exceptions import ExchangeError
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Whitebit(Exchange):
    """WhiteBit exchange class.
    Contains adjustments needed for Freqtrade to work with this exchange.
    """

    _ft_has: FtHas = {
        "trades_has_history": False,
    }
    _ft_has_futures: FtHas = {
        "uses_leverage_tiers": False,
        "mark_ohlcv_price": "futures",
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.SPOT, MarginMode.NONE),
        (TradingMode.FUTURES, MarginMode.ISOLATED),
    ]

    def get_max_leverage(self, pair: str, stake_amount: float | None) -> float:
        if self.trading_mode == TradingMode.FUTURES:
            return self.markets[pair]["limits"]["leverage"]["max"]
        else:
            return 1.0

    def get_funding_fees(
        self, pair: str, amount: float, is_short: bool, open_date: datetime
    ) -> float:
        """
        Fetch funding fees, either from the exchange (live) or calculates them
        based on funding rate/mark price history.
        WhiteBit does not support fetchFundingRateHistory, so fall back to
        _fetch_and_calculate_funding_fees and return 0.0 if unavailable.
        """
        if self.trading_mode == TradingMode.FUTURES:
            try:
                return self._fetch_and_calculate_funding_fees(pair, amount, is_short, open_date)
            except ExchangeError:
                logger.warning(f"Could not update funding fees for {pair}.")
        return 0.0
