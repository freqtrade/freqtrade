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
        liability: Optional[float] = None,  # (Okex)
        interest: Optional[float] = None,  # (Okex)
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
        :param position:
            Total position assets – on-hold by pending order
        :param trading_mode: SPOT, MARGIN, FUTURES, etc.
        :param collateral: Either ISOLATED or CROSS
        :param maintenance_amt: # * Not required by Okex
        :param wallet_balance: # * Not required by Okex
        :param taker_fee_rate:
        :param liability:
            Initial liabilities + deducted interest
                • Long positions: Liability is calculated in quote currency.
                • Short positions: Liability is calculated in trading currency.
        :param interest: Interest that has not been deducted yet.
        :param mm_ex_1: # * Not required by Okex
        :param upnl_ex_1: # * Not required by Okex
        """

        if (not liability or not interest or not taker_fee_rate or not position_assets):
            raise OperationalException(
                "Parameters liability, interest, taker_fee_rate, position_assets"
                "are required by Okex.liquidation_price"
            )

        if trading_mode == TradingMode.FUTURES and collateral == Collateral.ISOLATED:
            if is_short:
                return (liability + interest) * (1 + mm_ratio) * (1 + taker_fee_rate)
            else:
                return (
                    (liability + interest) * (1 + mm_ratio) * (1 + taker_fee_rate) /
                    position_assets
                )
        else:
            raise OperationalException(
                f"Okex does not support {collateral.value} Mode {trading_mode.value} trading")
