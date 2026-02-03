import logging

from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Luno(Exchange):
    """
    Luno exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Freqtrade development team. So some features
    may still not work as expected.
    """

    _ft_has: FtHas = {
        "ohlcv_has_history": False,  # Only provides the last 1000 candles
        "always_require_api_keys": True,  # Requires API keys to fetch candles
        "trades_has_history": False,  # Only the last 24h are available
    }
