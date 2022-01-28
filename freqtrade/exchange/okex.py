import logging
from typing import Dict, List, Optional, Tuple

from freqtrade.enums import Collateral, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


class Okex(Exchange):
    """Okex exchange class.

    Contains adjustments needed for Freqtrade to work with this exchange.
    """

    _ft_has: Dict = {
        "ohlcv_candle_limit": 300,
        "mark_ohlcv_timeframe": "4h",
        "funding_fee_timeframe": "8h",
    }

    _supported_trading_mode_collateral_pairs: List[Tuple[TradingMode, Collateral]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.MARGIN, Collateral.CROSS),
        # (TradingMode.FUTURES, Collateral.CROSS),
        # (TradingMode.FUTURES, Collateral.ISOLATED)
    ]

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
        PERPETUAL: https://www.okex.com/support/hc/en-us/articles/
        360053909592-VI-Introduction-to-the-isolated-mode-of-Single-Multi-currency-Portfolio-margin

        :param exchange_name:
        :param open_rate: (EP1) Entry price of position
        :param is_short: True if the trade is a short, false otherwise
        :param leverage: The amount of leverage on the trade
        :param position: Absolute value of position size (in base currency)
        :param mm_ratio:
            Okex: [assets in the position - (liability +interest) * mark price] /
                (maintenance margin + liquidation fee)
        :param trading_mode: SPOT, MARGIN, FUTURES, etc.
        :param collateral: Either ISOLATED or CROSS
        :param maintenance_amt: # * Not required by Okex
        :param wallet_balance: # * margin_balance?
        :param taker_fee_rate:
        :param mm_ex_1: # * Not required by Okex
        :param upnl_ex_1: # * Not required by Okex
        """

        if (not taker_fee_rate):
            raise OperationalException(
                "Parameter taker_fee_rate is required by Okex.liquidation_price"
            )

        if trading_mode == TradingMode.FUTURES and collateral == Collateral.ISOLATED:

            if is_short:
                return (margin_balance + (face_value * number_of_contracts * open_price)) / [face_value * number_of_contracts * (mm_ratio + taker_fee_rate + 1)]
            else:
                return (margin_balance - (face_value * number_of_contracts * open_price)) / [face_value * number_of_contracts * (mm_ratio + taker_fee_rate - 1)]
        else:
            raise OperationalException(
                f"Okex does not support {collateral.value} Mode {trading_mode.value} trading")
