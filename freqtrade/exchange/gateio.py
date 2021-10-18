""" Gate.io exchange subclass """
import logging
from typing import Dict, List

from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


class Gateio(Exchange):
    """
    Gate.io exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Freqtrade development team. So some features
    may still not work as expected.
    """

    _ft_has: Dict = {
        "ohlcv_candle_limit": 1000,
    }

    _headers = {'X-Gate-Channel-Id': 'freqtrade'}

    funding_fee_times: List[int] = [0, 8, 16]  # hours of the day

    def validate_ordertypes(self, order_types: Dict) -> None:
        super().validate_ordertypes(order_types)

        if any(v == 'market' for k, v in order_types.items()):
            raise OperationalException(
                f'Exchange {self.name} does not support market orders.')

    def get_funding_rate_history(
        self,
        start: int,
        end: int
    ) -> Dict:
        '''
            :param start: timestamp in ms of the beginning time
            :param end: timestamp in ms of the end time
        '''
        # TODO-lev: Has a max limit into the past of 333 days
        return super().get_funding_rate_history(start, end)
