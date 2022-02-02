import logging
from typing import Dict, List, Tuple

from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


class Okex(Exchange):
    """Okex exchange class.

    Contains adjustments needed for Freqtrade to work with this exchange.
    """

    _ft_has: Dict = {
        "ohlcv_candle_limit": 300,
        "mark_ohlcv_timeframe": "4h",
        "funding_fee_timeframe": "8h",
    }

    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.MARGIN, MarginMode.CROSS),
        # (TradingMode.FUTURES, MarginMode.CROSS),
        # (TradingMode.FUTURES, MarginMode.ISOLATED)
    ]

    def _lev_prep(
        self,
        pair: str,
        leverage: float,
        side: str  # buy or sell
    ):
        if self.trading_mode != TradingMode.SPOT:
            if self.margin_mode is None:
                raise OperationalException(
                    f"{self.name}.margin_mode must be set for {self.trading_mode.value}"
                )
            self._api.set_leverage(
                leverage,
                pair,
                params={
                    "mgnMode": self.margin_mode.value,
                    "posSide": "long" if side == "buy" else "short",
                })
