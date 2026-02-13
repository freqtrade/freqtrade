"""WhiteBit exchange subclass"""

import logging

from freqtrade.enums import MarginMode, TradingMode
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
