""" Kraken exchange subclass """
import logging
from typing import Dict

from freqtrade.exchange import Exchange

logger = logging.getLogger(__name__)


class Kraken(Exchange):

    _params: Dict = {"trading_agreement": "agree"}
