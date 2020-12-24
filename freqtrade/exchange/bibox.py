""" Bibox exchange subclass """
import logging
from typing import Dict

from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


class Bibox(Exchange):
    """
    Bibox exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Freqtrade development team. So some features
    may still not work as expected.
    """

    # fetchCurrencies API point requires authentication for Bibox,
    # so switch it off for Freqtrade load_markets()
    _ccxt_config: Dict = {"has": {"fetchCurrencies": False}}
