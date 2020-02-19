""" FTX exchange subclass """
import logging
from typing import Dict

from freqtrade.exchange import Exchange

logger = logging.getLogger(__name__)


class Ftx(Exchange):

    _ft_has: Dict = {
        "ohlcv_candle_limit": 1500,
    }
