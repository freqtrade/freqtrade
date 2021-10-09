""" Bibox exchange subclass """
import logging
from typing import Dict, List

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
    @property
    def _ccxt_config(self) -> Dict:
        # Parameters to add directly to ccxt sync/async initialization.
        return {"has": {"fetchCurrencies": False}}

    funding_fee_times: List[int] = [0, 8, 16]  # hours of the day
