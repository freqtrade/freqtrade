""" Waves exchange subclass """
import logging
from typing import Dict, Optional

from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


class Waves(Exchange):
    """
    Waves exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Freqtrade development team. So some features
    may still not work as expected.
    """

    _ft_has: Dict = {
        "ohlcv_candle_limit": 1440,
    }

    # There seems to be no minumum?
    def get_min_pair_stake_amount(self, pair: str, price: float,
                                  stoploss: float) -> Optional[float]:
        return 0