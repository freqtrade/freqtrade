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
    ) -> Tuple[float, Optional[float]]:
        """
        :return: The maintenance margin ratio and maintenance amount
        """
        info = self.markets[pair]['info']
        return (float(info['maintenance_rate']), None)

    def liquidation_price_helper(
        self,
        open_rate: float,   # Entry price of position
        is_short: bool,
        leverage: float,
        mm_ratio: float,
        position: float,  # Absolute value of position size
        trading_mode: TradingMode,
        collateral: Collateral,
        maintenance_amt: Optional[float] = None,  # (Binance)
        wallet_balance: Optional[float] = None,  # (Binance and Gateio)
        taker_fee_rate: Optional[float] = None,  # (Gateio & Okex)
        mm_ex_1: Optional[float] = 0.0,  # (Binance) Cross only
        upnl_ex_1: Optional[float] = 0.0,  # (Binance) Cross only
    ) -> Optional[float]:
        """
        PERPETUAL: https://www.gate.io/help/futures/perpetual/22160/calculation-of-liquidation-price

        :param exchange_name:
        :param open_rate: Entry price of position
        :param is_short: True if the trade is a short, false otherwise
        :param leverage: The amount of leverage on the trade
        :param position: Absolute value of position size (in base currency)
        :param mm_ratio:
        :param trading_mode: SPOT, MARGIN, FUTURES, etc.
        :param collateral: Either ISOLATED or CROSS
        :param maintenance_amt: # * Not required by Gateio
        :param wallet_balance:
            Cross-Margin Mode: crossWalletBalance
            Isolated-Margin Mode: isolatedWalletBalance
        :param taker_fee_rate:

        # * Not required by Gateio
        :param mm_ex_1:
        :param upnl_ex_1:
        """
        if trading_mode == TradingMode.SPOT:
            return None

        if not collateral:
            raise OperationalException(
                "Parameter collateral is required by liquidation_price when trading_mode is "
                f"{trading_mode}"
            )

        if (not wallet_balance or not position or not taker_fee_rate):
            raise OperationalException(
                "Parameters wallet_balance, position, taker_fee_rate"
                "are required by Gateio.liquidation_price"
            )

        if trading_mode == TradingMode.FUTURES and collateral == Collateral.ISOLATED:
            # if is_inverse:
            #     # ! Not implemented
            #     raise OperationalException(
            #         "Freqtrade does not support inverse contracts at the moment")

            value = wallet_balance / position

            mm_ratio_taker = (mm_ratio + taker_fee_rate)
            if is_short:
                return (open_rate + value) / (1 + mm_ratio_taker)
            else:
                return (open_rate - value) / (1 - mm_ratio_taker)
        else:
            raise OperationalException(
                f"Gateio does not support {collateral.value} Mode {trading_mode.value} trading ")
