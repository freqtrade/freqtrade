""" BitMEX Exchange subclass """
import logging
from typing import Dict

from freqtrade.exchange import Exchange

logger = logging.getLogger(__name__)


class Bitmex(Exchange):

    _params: Dict = {}
    _exchange_internals_options: Dict = {
        'candles_have_close_time': True,
        'default_api_request_count': 100,
        'max_api_request_count': 500,
    }
