from typing import List

from freqtrade.enums import LiqFormula, TradingMode
from freqtrade.persistence import Trade


class MaintenanceMargin:

    trades: List[Trade]
    liq_formula: LiqFormula
    trading_mode: TradingMode

    @property
    def margin_level(self):
        return self.liq_formula(
            trading_mode=self.trading_mode
            # TODO: Add args to formula
        )

    @property
    def liq_level(self):    # This may be a constant value and may not need a function
        return              # If constant, would need to be recalculated after each new trade

    def __init__(self, liq_formula: LiqFormula, trading_mode: TradingMode):
        self.liq_formula = liq_formula
        self.trading_mode = trading_mode

    def add_new_trade(self, trade):
        return

    def remove_trade(self, trade):
        return

    # ? def update_trade_pric(self):

    def sell_all(self):
        return

    def run(self):
        # TODO-mg: implement a thread that constantly updates with every price change,
        # TODO-mg: must update at least every second
        # while true:
        #   if self.margin_level <= self.liq_level:
        #       self.sell_all()
        return
