""" Bybit exchange subclass """
import logging
from typing import Dict, List, Optional, Tuple

from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_utils import timeframe_to_msecs


logger = logging.getLogger(__name__)


class Bybit(Exchange):
    """
    Bybit exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Freqtrade development team. So some features
    may still not work as expected.
    """

    _ft_has: Dict = {
        "ohlcv_candle_limit": 1000,
        "ccxt_futures_name": "linear",
        "ohlcv_has_history": False,
    }
    _ft_has_futures: Dict = {
        "ohlcv_candle_limit": 200,
        "ohlcv_has_history": True,
        "mark_ohlcv_timeframe": "4h",
    }

    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.FUTURES, MarginMode.CROSS),
        # (TradingMode.FUTURES, MarginMode.ISOLATED)
    ]

    @property
    def _ccxt_config(self) -> Dict:
        # Parameters to add directly to ccxt sync/async initialization.
        # ccxt defaults to swap mode.
        config = {}
        if self.trading_mode == TradingMode.SPOT:
            config.update({
                "options": {
                    "defaultType": "spot"
                }
            })
        config.update(super()._ccxt_config)
        return config

    async def _fetch_funding_rate_history(
        self,
        pair: str,
        timeframe: str,
        limit: int,
        since_ms: Optional[int] = None,
    ) -> List[List]:
        """
        Fetch funding rate history
        Necessary workaround until https://github.com/ccxt/ccxt/issues/15990 is fixed.
        """
        params = {}
        if since_ms:
            until = since_ms + (timeframe_to_msecs(timeframe) * self._ft_has['ohlcv_candle_limit'])
            params.update({'until': until})
        # Funding rate
        data = await self._api_async.fetch_funding_rate_history(
            pair, since=since_ms,
            params=params)
        # Convert funding rate to candle pattern
        data = [[x['timestamp'], x['fundingRate'], 0, 0, 0, 0] for x in data]
        return data
