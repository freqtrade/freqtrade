from enum import Enum
from freqtrade.exceptions import OperationalException


class MaintenanceMarginFormula(Enum):
    """Equations to calculate maintenance margin"""

    BINANCE = "BINANCE"
    FTX = "FTX"
    KRAKEN = "KRAKEN"

    # TODO: Add arguments
    def __call__(self):
        if self.name == "BINANCE":
            raise OperationalException("Cross margin not available on this exchange with freqtrade")
            # TODO: return This formula
            # https://www.binance.com/en/support/faq/f6b010588e55413aa58b7d63ee0125ed
        elif self.name == "FTX":
            # TODO: Implement
            raise OperationalException("Cross margin not available on this exchange with freqtrade")
        elif self.name == "KRAKEN":
            # TODO: Implement
            raise OperationalException("Cross margin not available on this exchange with freqtrade")
            # https://support.kraken.com/hc/en-us/articles/203325763-Margin-Call-Level-and-Margin-Liquidation-Level
        else:
            raise OperationalException("Cross margin not available on this exchange with freqtrade")
