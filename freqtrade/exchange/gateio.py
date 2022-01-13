""" Gate.io exchange subclass """
import logging
from typing import Dict, List, Optional, Tuple

from freqtrade.enums import Collateral, TradingMode
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
        "ohlcv_volume_currency": "quote",
    }

    _headers = {'X-Gate-Channel-Id': 'freqtrade'}

    _supported_trading_mode_collateral_pairs: List[Tuple[TradingMode, Collateral]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.MARGIN, Collateral.CROSS),
        # (TradingMode.FUTURES, Collateral.CROSS),
        (TradingMode.FUTURES, Collateral.ISOLATED)
    ]

    def validate_ordertypes(self, order_types: Dict) -> None:
        super().validate_ordertypes(order_types)

        if any(v == 'market' for k, v in order_types.items()):
            raise OperationalException(
                f'Exchange {self.name} does not support market orders.')

    def get_maintenance_ratio_and_amt(
        self,
        pair: str,
        nominal_value: Optional[float] = 0.0,
    ):
        info = self.markets[pair]['info']
        if 'maintenance_rate' in info:
            return [float(info['maintenance_rate']), None]
        else:
            return [None, None]
