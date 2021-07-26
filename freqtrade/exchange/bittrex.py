""" Bittrex exchange subclass """
import logging
from typing import Dict, Optional

from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


class Bittrex(Exchange):
    """
    Bittrex exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.
    """

    _ft_has: Dict = {
        "ohlcv_candle_limit_per_timeframe": {
            '1m': 1440,
            '5m': 288,
            '1h': 744,
            '1d': 365,
        },
        "l2_limit_range": [1, 25, 500],
    }

    def setup_leveraged_enter(
        self,
        pair: str,
        leverage: float,
        amount: float,
        quote_currency: Optional[str],
        is_short: Optional[bool]
    ):
        raise OperationalException("Bittrex does not support leveraged trading")

    def complete_leveraged_exit(
        self,
        pair: str,
        leverage: float,
        amount: float,
        quote_currency: Optional[str],
        is_short: Optional[bool]
    ):
        raise OperationalException("Bittrex does not support leveraged trading")

    def get_isolated_liq(self, pair: str, open_rate: float,
                         amount: float, leverage: float, is_short: bool) -> float:
        # TODO-mg: implement
        raise OperationalException("Bittrex does not support margin trading")
